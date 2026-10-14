"""Measure frozen Generator directions against E_D class margins."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    check_resume_compatibility,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as base
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as parent


TASK_ID = "M0_GENERATOR_SEMANTIC_DIRECTION_ADAM_DIAGNOSTIC_V1"
REPORT_SCHEMA = "m0_generator_semantic_direction_adam_report.v2"
PREFLIGHT_SCHEMA = "m0_generator_semantic_direction_adam_preflight.v2"
BLOCK_SCHEMA = "m0_generator_semantic_direction_adam_block.v2"
MANIFEST_SCHEMA = "m0_generator_semantic_direction_adam_manifest.v2"
MARGIN_DEFINITION = "true_logit_minus_unique_max_wrong_logit_full_autograd.v2"
FIT_ROWS = 200
BATCH_SIZE = 10
BATCH_COUNT = 20
ENDPOINTS = ("step2200", "A_feature", "B_class_ce")
ENDPOINT_FILES = {
    "step2200": "arm_a_step0000.pt",
    "A_feature": "arm_a_step0300.pt",
    "B_class_ce": "arm_b_step0300.pt",
}
READOUT_FILES = {
    "step2200": "readouts_step2200.pt",
    "A_feature": "readouts_A_feature.pt",
    "B_class_ce": "readouts_B_class_ce.pt",
}
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
ROW_CLASSES = (*FOCUS_CLASSES, *CONTROL_CLASSES)
EVALUATORS = ("E_S", "E_I", "E_C")
ALL_EVALUATORS = (*EVALUATORS, "E_D")
DELTA_SIC = 1.365988374680088e-5
KAPPA = 0.6936758419764932
ALPHA = 0.01596223111734585
BETA = 2.4509953404250195
GAMMA = 0.5330424284576345
COSINE_BOUNDARY = 0.05
STABLE_ROWS = 16
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7
MIN_FREE_BYTES = 1_500_000_000
ESTIMATED_OUTPUT_BYTES = 400_000_000
RESOURCE_NAME = "resource_telemetry_generator_direction.jsonl"
DEFAULT_ARTIFACTS = Path("outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01")
DEFAULT_REVIEW = Path("outputs/20261004_m0_semantic_target_evidence_review_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_04_margin_gradient_repair")


class DirectionDiagnosticError(RuntimeError):
    """Invalid input, runtime invariant, or incomplete diagnostic."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectionDiagnosticError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _state_snapshot(module: torch.nn.Module) -> dict[str, Any]:
    return {
        "state_dict": {name: value.detach().cpu().clone() for name, value in module.state_dict().items()},
        "named_parameters": {name: value.detach().cpu().clone() for name, value in module.named_parameters()},
        "named_buffers": {name: value.detach().cpu().clone() for name, value in module.named_buffers()},
    }


def _write_json_new(path: Path, payload: Mapping[str, Any], label: str) -> None:
    parent.diagnostic._write_json_new(path, payload, label)


def _write_torch_new(path: Path, payload: Mapping[str, Any], label: str) -> None:
    parent.diagnostic._write_torch_new(path, payload, label)


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically advance the one task-owned resume pointer."""
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _path_inputs(root: Path, artifact_dir: Path, review_dir: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], dict[str, Any]]:
    population_path = artifact_dir / "input_population.json"
    report_path = artifact_dir / "report.json"
    review_path = review_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "semantic-target comparison input lacks its population or final report")
    _require(review_path.is_file() and (review_dir / "training_log_source_row_mapping.csv").is_file(), "completed zero-update review lacks its report or mapped-row evidence")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    _require(population.get("task_id") == parent.TASK_ID, "parent population belongs to a different task")
    _require(parent_report.get("task_id") == parent.TASK_ID and parent_report.get("status") == "VALID_FAILED" and parent_report.get("classification") == "NOT_SUPPORTED_WITHIN_SCOPE", "parent experiment terminal state changed")
    _require(review.get("task_id") == "M0_SEMANTIC_TARGET_EVIDENCE_REVIEW_V1" and review.get("status") == "EVIDENCE_REVIEW_COMPLETED", "evidence review is not complete")
    _require(review.get("parent_status_preserved") == "VALID_FAILED" and review.get("parent_classification_preserved") == "NOT_SUPPORTED_WITHIN_SCOPE", "review did not preserve the parent classification")
    _require(review.get("training_recommendation") == "NONE" and review.get("next_action") == "NONE", "review contains an unexpected follow-on authorization")
    declared_paths = population.get("input_paths")
    _require(isinstance(declared_paths, Mapping), "parent population omitted explicit inputs")
    required = ("step1200", "step2200", "training_data", "e_s", "e_i", "e_c", "e_d")
    paths = {name: _resolve(root, declared_paths[name]) for name in required}
    for name, path in paths.items():
        _require(path.is_file(), f"explicit parent input is missing for {name}: {path}")
    checkpoint_paths = {name: artifact_dir / ENDPOINT_FILES[name] for name in ENDPOINTS}
    readout_paths = {name: artifact_dir / READOUT_FILES[name] for name in ENDPOINTS}
    _require(all(path.is_file() for path in (*checkpoint_paths.values(), *readout_paths.values())), "one of the three explicit checkpoint/readout pairs is missing")
    paths.update({f"checkpoint_{name}": path.resolve() for name, path in checkpoint_paths.items()})
    paths.update({f"readout_{name}": path.resolve() for name, path in readout_paths.items()})
    paths.update({"parent_population": population_path.resolve(), "parent_report": report_path.resolve(), "review_report": review_path.resolve(), "review_mapping": (review_dir / "training_log_source_row_mapping.csv").resolve()})
    return paths, population, parent_report, review


def _validate_optimizer(payload: Mapping[str, Any], endpoint: str, parameters: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    expected_step = 2200 if endpoint == "step2200" else 2500
    _require(payload.get("optimizer_step_by_parameter") == {"encoder.phase": 2200, "generator.phase": expected_step}, f"{endpoint} declared per-parameter Adam steps mismatch")
    optimizer = payload.get("optimizer_state")
    _require(isinstance(optimizer, Mapping), f"{endpoint} optimizer state is missing")
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and isinstance(states, Mapping), f"{endpoint} Adam structure mismatch")
    group = groups[0]
    _require(group.get("params") == [0, 1], f"{endpoint} Adam parameter order mismatch")
    for key, value in (("lr", 1e-3), ("betas", (0.9, 0.999)), ("eps", 1e-8), ("weight_decay", 0.0), ("amsgrad", False), ("maximize", False)):
        _require(group.get(key) == value, f"{endpoint} Adam setting {key} changed")
    _require(set(states) == {0, 1}, f"{endpoint} Adam state does not contain both saved phase groups")
    for index, name in enumerate(("encoder.phase", "generator.phase")):
        state = states[index]
        _require(isinstance(state, Mapping) and set(state) >= {"step", "exp_avg", "exp_avg_sq"}, f"{endpoint} Adam moments are incomplete for {name}")
        _require(float(state["step"].item()) == (2200 if index == 0 else expected_step), f"{endpoint} internal Adam step changed for {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = state[key]
            _require(torch.is_tensor(value) and value.shape == parameters[name].shape and value.dtype == parameters[name].dtype, f"{endpoint} Adam {key} geometry changed for {name}")
            _finite(value, f"{endpoint} {name} {key}")
        _require(bool((state["exp_avg_sq"] >= 0).all().item()), f"{endpoint} second moments contain a negative value")
    return dict(optimizer)


def _validate_population_vectors(population: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.tensor(population["source_row_indices"], dtype=torch.long)
    labels = torch.tensor(population["labels"], dtype=torch.long)
    _require(rows.shape == (600,) and labels.shape == (600,), "parent saved population must contain the declared 600 rows")
    _require(torch.unique(rows).numel() == 600, "parent population row indices are not unique")
    _require(bool(((labels >= 0) & (labels <= 9)).all().item()), "parent population contains an invalid MNIST label")
    _require(torch.equal(labels[:FIT_ROWS].reshape(BATCH_COUNT, BATCH_SIZE), torch.arange(10).expand(BATCH_COUNT, 10)), "parent fit order is not the original 20 balanced batches")
    _require(torch.bincount(labels[200:300], minlength=10).tolist() == [10] * 10, "parent historic-readout role is not ten rows per class")
    _require(torch.bincount(labels[300:], minlength=10).tolist() == [30] * 10, "parent confirmation-readout role is not thirty rows per class")
    saved_roles = population.get("roles")
    _require(isinstance(saved_roles, list) and len(saved_roles) == 600, "parent row-role vector is missing or has the wrong length")
    _require(saved_roles[:200] == ["fit"] * 200 and saved_roles[200:300] == ["historic_holdout"] * 100 and saved_roles[300:] == ["confirmation"] * 300, "parent data roles changed")
    _require(population.get("original_fit_rows") == 200 and population.get("historic_holdout_rows") == 100 and population.get("new_confirmation_rows") == 300, "parent population role sizes changed")
    _require(population.get("focus_classes") == [1, 7] and population.get("confirmation_indices") == rows[300:].tolist(), "parent fixed population metadata changed")
    _require(not set(rows[:FIT_ROWS].tolist()).intersection(rows[FIT_ROWS:].tolist()), "parent fit rows overlap a non-fit role")
    return rows, labels


def _fit_batch_slice(batch_index: int) -> tuple[int, int]:
    _require(isinstance(batch_index, int) and 0 <= batch_index < BATCH_COUNT, "batch index is outside the fixed TRAIN fit population")
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    _require(stop <= FIT_ROWS, "non-fit population rows cannot enter diagnostic gradients")
    return start, stop


def _fixed_fit_population_matches(inputs: Mapping[str, Any]) -> bool:
    population = inputs["population"]
    roles = population.get("roles", [])
    rows = population.get("source_row_indices", [])
    labels = population.get("labels", [])
    return (
        inputs["images"].shape == (FIT_ROWS, 1, 28, 28)
        and torch.bincount(inputs["labels"], minlength=10).tolist() == [20] * 10
        and isinstance(roles, list)
        and len(roles) == 600
        and roles[:FIT_ROWS] == ["fit"] * FIT_ROWS
        and roles[FIT_ROWS:300] == ["historic_holdout"] * 100
        and roles[300:] == ["confirmation"] * 300
        and len(rows) == 600
        and len(labels) == 600
        and inputs["source_indices"].tolist() == rows[:FIT_ROWS]
        and inputs["labels"].tolist() == labels[:FIT_ROWS]
        and not bool(set(rows[:FIT_ROWS]).intersection(rows[FIT_ROWS:]))
    )


def _load_inputs(root: Path, paths: Mapping[str, Path], population: Mapping[str, Any], artifact_dir: Path) -> dict[str, Any]:
    payloads = {name: torch.load(paths[f"checkpoint_{name}"], map_location="cpu", weights_only=True) for name in ENDPOINTS}
    readouts = {name: torch.load(paths[f"readout_{name}"], map_location="cpu", weights_only=True) for name in ENDPOINTS}
    original_rows, original_labels = _validate_population_vectors(population)

    data_payload = torch.load(paths["training_data"], map_location="cpu", weights_only=True, mmap=True)
    source_images, source_labels = base._read_image_labels(data_payload)
    _require(torch.equal(source_labels.index_select(0, original_rows), original_labels), "parent population labels differ from canonical source labels")
    train_indices, _ = base.stratified_mnist_indices(source_labels, seed=1234, validation_per_label=500)
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices] = True
    _require(bool(train_mask.index_select(0, original_rows[:FIT_ROWS]).all().item()), "a gradient row is outside canonical TRAIN")
    images = source_images.index_select(0, original_rows[:FIT_ROWS]).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    labels = source_labels.index_select(0, original_rows[:FIT_ROWS]).contiguous()
    _require(tuple(images.shape) == (FIT_ROWS, 1, 28, 28) and torch.bincount(labels, minlength=10).tolist() == [20] * 10, "materialized fit images or labels changed")
    _finite(images, "materialized TRAIN fit images")
    del source_images, data_payload

    endpoint_parameters: dict[str, dict[str, torch.Tensor]] = {}
    endpoint_optimizers: dict[str, dict[str, Any]] = {}
    for endpoint, payload in payloads.items():
        _require(isinstance(payload, Mapping) and payload.get("task_id") == parent.TASK_ID and payload.get("source_checkpoint_step") == 2200, f"{endpoint} checkpoint task/source step changed")
        _require(payload.get("arm") == {"step2200": "A_feature", "A_feature": "A_feature", "B_class_ce": "B_class_ce"}[endpoint], f"{endpoint} checkpoint arm changed")
        _require(payload.get("completed_updates") == (0 if endpoint == "step2200" else 300), f"{endpoint} checkpoint update count changed")
        saved_population = payload.get("population")
        _require(isinstance(saved_population, Mapping) and saved_population == population, f"{endpoint} checkpoint population/roles differ from explicit parent population")
        parameters = payload.get("trainable_parameter_state")
        _require(isinstance(parameters, Mapping) and set(parameters) == {"encoder.phase", "generator.phase"}, f"{endpoint} phase parameters are missing")
        for name in ("encoder.phase", "generator.phase"):
            value = parameters[name]
            _require(torch.is_tensor(value) and value.shape == (2, 1, 100, 100) and value.dtype == torch.float32, f"{endpoint} {name} shape/dtype changed")
            _finite(value, f"{endpoint} {name}")
        endpoint_parameters[endpoint] = dict(parameters)
        endpoint_optimizers[endpoint] = _validate_optimizer(payload, endpoint, parameters)
        for key in ("references", "weights"):
            _require(_tree_equal(payload.get(key), payloads["step2200"].get(key)), f"{endpoint} inherited shared {key} changed")
        _require(_tree_equal(payload.get("frozen_runtime_state"), payloads["step2200"].get("frozen_runtime_state")), f"{endpoint} frozen runtime state changed")

        saved_readout = readouts[endpoint]
        _require(isinstance(saved_readout, Mapping), f"{endpoint} saved readout is invalid")
        _require(torch.equal(saved_readout.get("source_indices"), original_rows), f"{endpoint} saved readout row order changed")
        _require(torch.equal(saved_readout.get("labels"), original_labels), f"{endpoint} saved readout labels changed")
        for field, shape in (("z", (600, 100)), ("raw_cmos", (600, 1, 200, 200)), ("target_slm_intensity", (600, 1, 200, 200)), ("power_by_plane", (600, 5)), ("e_s_logits", (600, 10)), ("e_i_logits", (600, 10)), ("e_c_logits", (600, 10)), ("e_d_logits", (600, 10))):
            value = saved_readout.get(field)
            _require(torch.is_tensor(value) and tuple(value.shape) == shape, f"{endpoint} saved {field} shape changed")
            _finite(value.real if torch.is_complex(value) else value, f"{endpoint} saved {field}")

    mapped_rows = 0
    position_updates = {"A_feature": [0] * FIT_ROWS, "B_class_ce": [0] * FIT_ROWS}
    with paths["review_mapping"].open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            arm = row.get("arm")
            _require(arm in position_updates, "review mapping contains an unexpected training arm")
            update = int(row["update"])
            position = int(row["fit_position"])
            batch_index = int(row["batch_index"])
            within_batch = int(row["within_batch"])
            _require(1 <= update <= 300 and 0 <= position < FIT_ROWS, "review mapping contains an out-of-scope update or fit position")
            _require(batch_index == (update - 1) % BATCH_COUNT and within_batch == position % BATCH_SIZE and position == batch_index * BATCH_SIZE + within_batch, "review mapping batch/position relation changed")
            _require(int(row["mapped_source_row_index"]) == int(original_rows[position]), f"review mother-row mapping differs at fit position {position}")
            _require(int(row["label"]) == int(original_labels[position]) and row["role"] == "fit", f"review mapping label/role differs at fit position {position}")
            position_updates[arm][position] += 1
            mapped_rows += 1
    _require(mapped_rows == 2 * 300 * BATCH_SIZE, "zero-update review mapping does not cover both arms and all 300 updates")
    _require(all(count == 15 for counts in position_updates.values() for count in counts), "derived review mapping does not show exactly 15 updates per row and arm")

    return {
        "payloads": payloads,
        "parameters": endpoint_parameters,
        "optimizers": endpoint_optimizers,
        "readouts": readouts,
        "source_indices": original_rows[:FIT_ROWS].clone(),
        "labels": labels,
        "images": images,
        "references": payloads["step2200"]["references"],
        "weights": payloads["step2200"]["weights"],
        "population": dict(population),
        "input_paths": {name: path.as_posix() for name, path in paths.items()},
        "review_rows_mapped": mapped_rows,
    }


def _load_evaluators(paths: Mapping[str, Path], device: torch.device) -> dict[str, torch.nn.Module]:
    return base.load_evaluators(paths, device)


def _new_consumer(inputs: Mapping[str, Any], endpoint: str, device: torch.device) -> tuple[Any, dict[str, torch.nn.Parameter]]:
    payload = inputs["payloads"][endpoint]
    consumer = bogt.LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=bogt.PRODUCTION_PROPAGATION_GRID)
    references = bogt.FrozenBogtReferences.from_payload(payload["references"])
    weights = bogt.BogtLossWeights.from_payload(payload["weights"])
    _require(abs(weights.alpha - ALPHA) < 1e-15 and abs(weights.beta - BETA) < 1e-15 and abs(weights.gamma - GAMMA) < 1e-15, f"{endpoint} BOGT weights changed")
    consumer.bind_references(references, for_scientific=True)
    consumer.bind_weights(weights, for_scientific=True)
    consumer.load_trainable_state_dict(payload["trainable_parameter_state"])
    consumer.runtime.eval()
    _require(consumer.runtime.config.propagation_grid_size == 2048, "diagnostic requires the complete production propagation grid")
    parameters = dict(consumer.runtime.named_parameters())
    _require(tuple(parameters) == ("encoder.phase", "generator.phase"), "runtime phase parameter order changed")
    _require(all(parameter.grad is None for parameter in parameters.values()), "new frozen endpoint begins with parameter gradients")
    return consumer, parameters


def _consumer_frozen_inputs_snapshot(consumer: Any) -> dict[str, Any]:
    references = consumer._references
    weights = consumer._weights
    _require(references is not None and weights is not None, "diagnostic consumer lost its frozen BOGT inputs")
    return {
        "references": {
            "scale": references.scale.detach().cpu().clone(),
            "power_reference": references.power_reference.detach().cpu().clone(),
            "data_role": references.data_role,
            "count": references.count,
            "engineering_only": references.engineering_only,
            "theta0_identity": dict(references.theta0_identity),
            "data_identity": dict(references.data_identity),
        },
        "weights": {
            "alpha": weights.alpha,
            "beta": weights.beta,
            "gamma": weights.gamma,
            "gradient_norms": dict(weights.gradient_norms),
            "data_role": weights.data_role,
            "engineering_only": weights.engineering_only,
            "calibration_batch_identity": dict(weights.calibration_batch_identity),
            "references_identity_sha256": weights.references_identity_sha256,
        },
    }


def _target_control(images: torch.Tensor, labels: torch.Tensor, evaluators: Mapping[str, torch.nn.Module], device: torch.device) -> dict[str, Any]:
    target = canonical.cmos_view(images.to(device))
    result: dict[str, Any] = {}
    for name, evaluator in evaluators.items():
        with torch.no_grad():
            output = evaluator(target)
        logits = output[1] if name in EVALUATORS else output
        _finite(logits, f"target {name} logits")
        predicted = logits.argmax(dim=1).cpu()
        class_correct = [int(((labels == cls) & (predicted == labels)).sum().item()) for cls in range(10)]
        class_counts = torch.bincount(labels, minlength=10).tolist()
        macro = sum(correct / count for correct, count in zip(class_correct, class_counts, strict=True)) / 10
        result[name] = {"rows": int(labels.numel()), "macro_recall": macro, "per_class_correct": class_correct, "per_class_rows": class_counts, "prediction_counts": torch.bincount(predicted, minlength=10).tolist()}
    return result


def _control_qualified(control: Mapping[str, Any]) -> bool:
    return all(
        control[name]["macro_recall"] >= 0.90
        and control[name]["per_class_correct"][1] >= 16
        and control[name]["per_class_correct"][7] >= 16
        for name in ALL_EVALUATORS
    )


def _flatten_phase(encoder: torch.Tensor, generator: torch.Tensor) -> torch.Tensor:
    return torch.cat((encoder.reshape(-1), generator.reshape(-1)))


def _canonical_semantic(gradients: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    canonical_inputs: dict[str, Mapping[str, torch.Tensor]] = {}
    per_evaluator: dict[str, Any] = {}
    for name in EVALUATORS:
        encoder, generator = gradients[name]["encoder.phase"], gradients[name]["generator.phase"]
        vector = _flatten_phase(encoder, generator)
        norm = float(vector.detach().to(torch.float64).norm().cpu())
        if not math.isfinite(norm) or norm <= 0:
            return None, {"status": "UNAVAILABLE_ZERO_OR_INVALID_EVALUATOR_GRADIENT", "evaluator": name, "norm": norm}
        canonical_inputs[f"L_{name[-1]}"] = gradients[name]
        per_evaluator[name] = {"joint_l2_norm": norm, "encoder_l2_norm": float(encoder.detach().double().norm().cpu()), "generator_l2_norm": float(generator.detach().double().norm().cpu()), "generator_share_of_joint_l2": float(generator.detach().double().norm().cpu()) / norm}
    combined_direction, unavailable = base._safe_canonical_gradient(canonical_inputs)
    if combined_direction is None:
        return None, {"status": "UNAVAILABLE_CANONICAL_COMBINATION", "reason": unavailable, "evaluator_norms": per_evaluator}
    normalized = []
    for name in EVALUATORS:
        vector = _flatten_phase(gradients[name]["encoder.phase"], gradients[name]["generator.phase"])
        normalized.append(vector / float(vector.detach().to(torch.float64).norm().cpu()))
    combined = sum(normalized)
    scale = float(combined.detach().to(torch.float64).square().mean().sqrt().cpu())
    if not math.isfinite(scale) or scale <= 0:
        return None, {"status": "UNAVAILABLE_ZERO_OR_INVALID_EQUAL_WEIGHT_SUM", "evaluator_norms": per_evaluator}
    _require(abs(base.DELTA_SIC - DELTA_SIC) < 1e-18, "canonical semantic-gradient scale changed")
    return combined_direction, {"status": "AVAILABLE", "delta_SIC": DELTA_SIC, "evaluator_normalization": per_evaluator, "equal_weight_joint_rms_before_scale": scale}


def _adam_candidate(gradient: torch.Tensor, optimizer_state: Mapping[str, Any], *, variant: str) -> torch.Tensor:
    """Read-only Adam formula for the saved Generator slot."""
    _require(variant in {"saved", "zero_first", "zero_all"}, f"unknown Adam diagnostic variant: {variant}")
    state = optimizer_state["state"][1]
    beta1, beta2 = optimizer_state["param_groups"][0]["betas"]
    gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
    _require(gradient.shape == state["exp_avg"].shape, "Adam candidate gradient shape changed")
    if variant == "zero_all":
        first, second, step = torch.zeros_like(gradient), torch.zeros_like(gradient), 1
    else:
        first = torch.zeros_like(state["exp_avg"]) if variant == "zero_first" else state["exp_avg"].detach().cpu()
        second = state["exp_avg_sq"].detach().cpu()
        step = int(float(state["step"].item())) + 1
    first_next = beta1 * first + (1.0 - beta1) * gradient
    second_next = beta2 * second + (1.0 - beta2) * gradient.square()
    bias1 = 1.0 - beta1**step
    bias2 = 1.0 - beta2**step
    candidate = -(float(optimizer_state["param_groups"][0]["lr"]) / bias1) * first_next / ((second_next.sqrt() / math.sqrt(bias2)).add(float(optimizer_state["param_groups"][0]["eps"])))
    _finite(candidate, f"{variant} Adam candidate")
    return candidate


def _adam_formula_cpu_reference_check() -> dict[str, Any]:
    """Compare the read-only formula with independent CPU Adam toy updates."""
    encoder = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32))
    generator = torch.nn.Parameter(torch.tensor([0.4, -0.2, 0.1], dtype=torch.float32))
    source = torch.optim.Adam([encoder, generator], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    for encoder_gradient, generator_gradient in (
        (torch.tensor([0.2]), torch.tensor([0.4, -0.3, 0.1])),
        (torch.tensor([-0.1]), torch.tensor([0.2, 0.1, -0.5])),
        (torch.tensor([0.05]), torch.tensor([-0.2, 0.4, 0.3])),
    ):
        encoder.grad = encoder_gradient.clone()
        generator.grad = generator_gradient.clone()
        source.step()
        source.zero_grad(set_to_none=True)
    saved = source.state_dict()
    next_gradient = torch.tensor([0.15, -0.25, 0.35], dtype=torch.float32)
    variants = ("saved", "zero_first", "zero_all")
    maximum_differences: dict[str, float] = {}
    for variant in variants:
        reference_encoder = torch.nn.Parameter(encoder.detach().clone())
        reference_generator = torch.nn.Parameter(generator.detach().clone())
        reference_optimizer = torch.optim.Adam([reference_encoder, reference_generator], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, foreach=False)
        reference_optimizer.load_state_dict(_cpu_tree(saved))
        reference_state = reference_optimizer.state[reference_generator]
        if variant in {"zero_first", "zero_all"}:
            reference_state["exp_avg"].zero_()
        if variant == "zero_all":
            reference_state["exp_avg_sq"].zero_()
            reference_state["step"].zero_()
        before = reference_generator.detach().clone()
        reference_encoder.grad = torch.zeros_like(reference_encoder)
        reference_generator.grad = next_gradient.clone()
        reference_optimizer.step()
        observed = reference_generator.detach() - before
        predicted = _adam_candidate(next_gradient, saved, variant=variant)
        _require(torch.allclose(observed, predicted, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"read-only {variant} Adam formula differs from the independent CPU optimizer reference")
        maximum_differences[variant] = float((observed - predicted).abs().max())
    _require(encoder.grad is None and generator.grad is None, "CPU Adam formula check left gradients on its source toy parameters")
    return {"passed": True, "reference": "torch.optim.Adam CPU toy with two parameter slots", "variants": list(variants), "maximum_absolute_difference": maximum_differences}


def _ed_margin_components(logits: torch.Tensor, true_class: int) -> dict[str, Any]:
    _require(logits.ndim == 1 and 0 <= true_class < logits.numel(), "E_D margin expects one valid class-logit vector")
    wrong = logits.detach().clone()
    wrong[true_class] = -torch.inf
    best_wrong_value, best_wrong_index = wrong.max(dim=0)
    tied_classes = torch.nonzero(wrong == best_wrong_value, as_tuple=False).reshape(-1)
    selected_class = int(best_wrong_index.item())
    margin = logits[true_class] - logits[selected_class]
    return {
        "margin": margin,
        "best_wrong_logit": best_wrong_value,
        "best_wrong_class": selected_class,
        "best_wrong_tied_classes": tuple(int(value) for value in tied_classes.tolist()),
        "tied_maximum": tied_classes.numel() > 1,
    }


def _margin_derivative_cpu_reference_check() -> dict[str, Any]:
    """Verify the production margin expression against independent derivatives and finite differences."""
    epsilon = 1e-5
    cases = (
        ("sign_reversal", (1.0, 2.0, 0.0), (0.0, 1.0, -5.0), -1.0),
        ("common_logit_shift", (1.0, 1.0, 1.0), (0.0, 1.0, -5.0), 0.0),
    )
    records: dict[str, Any] = {}
    for name, slopes, offsets, expected in cases:
        phi = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        slope_tensor = torch.tensor(slopes, dtype=torch.float64)
        offset_tensor = torch.tensor(offsets, dtype=torch.float64)
        logits = phi * slope_tensor + offset_tensor
        components = _ed_margin_components(logits, true_class=0)
        true_gradient = torch.autograd.grad(logits[0], phi, retain_graph=True)[0]
        wrong_gradient = torch.autograd.grad(logits[components["best_wrong_class"]], phi, retain_graph=True)[0]
        observed_gradient = torch.autograd.grad(components["margin"], phi)[0]

        def margin_at(value: float) -> float:
            values = value * slope_tensor + offset_tensor
            return float(_ed_margin_components(values, true_class=0)["margin"])

        finite_difference = (margin_at(epsilon) - margin_at(-epsilon)) / (2.0 * epsilon)
        independent_gradient = float((true_gradient - wrong_gradient).item())
        observed = float(observed_gradient.item())
        _require(not components["tied_maximum"], f"{name} CPU reference unexpectedly selected a tied wrong class")
        _require(abs(observed - expected) <= 1e-12, f"{name} production margin derivative differs from its analytic value")
        _require(abs(observed - independent_gradient) <= 1e-12, f"{name} production margin derivative omits a class-logit gradient")
        _require(abs(observed - finite_difference) <= 1e-9, f"{name} production margin derivative differs from central finite differences")
        records[name] = {
            "margin": float(components["margin"].detach()),
            "best_wrong_class": components["best_wrong_class"],
            "expected_derivative": expected,
            "observed_derivative": observed,
            "independent_component_derivative": independent_gradient,
            "central_finite_difference": finite_difference,
        }

    tied_logits = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
    tied = _ed_margin_components(tied_logits, true_class=0)
    _require(tied["tied_maximum"] and tied["best_wrong_tied_classes"] == (1, 2), "CPU tie reference did not preserve all maximizing wrong classes")
    records["exact_tie"] = {
        "margin": float(tied["margin"].detach()),
        "best_wrong_tied_classes": list(tied["best_wrong_tied_classes"]),
        "direction_status": "UNDECIDABLE_TIED_MAXIMUM",
    }
    return {"passed": True, "reference": "independent autograd components plus central finite differences in CPU float64", "epsilon": epsilon, "cases": records}


def _direction_record(margin_gradient: torch.Tensor | None, direction: torch.Tensor, *, tied_maximum: bool = False) -> dict[str, Any]:
    if tied_maximum:
        return {"status": "UNDECIDABLE_TIED_MAXIMUM", "dot": None, "cosine": None, "angle_degrees": None}
    if margin_gradient is None:
        return {"status": "UNDECIDABLE_MISSING_MARGIN_GRADIENT", "dot": None, "cosine": None, "angle_degrees": None}
    margin_norm = float(margin_gradient.detach().to(torch.float64).norm().cpu())
    direction_norm = float(direction.detach().to(torch.float64).norm().cpu())
    if margin_norm == 0 or direction_norm == 0:
        return {"status": "UNDECIDABLE_ZERO_GRADIENT", "dot": None, "cosine": None, "angle_degrees": None}
    dot = float((margin_gradient.detach().to(torch.float64) * direction.detach().to(torch.float64)).sum().cpu())
    cosine = dot / (margin_norm * direction_norm)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    status = "BOUNDARY" if abs(cosine) < COSINE_BOUNDARY else ("BENEFICIAL" if cosine > 0 else "HARMFUL")
    return {"status": status, "dot": dot, "cosine": cosine, "angle_degrees": angle}


def _project_auxiliary(auxiliary: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    denominator = float(semantic.detach().to(torch.float64).square().sum().cpu())
    if denominator <= 0:
        return torch.zeros_like(auxiliary), {"available": False, "reason": "semantic Generator direction is zero"}
    if float(auxiliary.detach().to(torch.float64).norm().cpu()) == 0:
        return torch.zeros_like(auxiliary), {"available": True, "projected": False, "reason": "auxiliary Generator gradient is zero", "dot_before": 0.0, "dot_after": 0.0}
    flattened, record = canonical.project_auxiliary_gradient(auxiliary.reshape(-1), semantic.reshape(-1), "Generator auxiliary gradient")
    result = flattened.reshape_as(auxiliary)
    _finite(result, "projected Generator auxiliary gradient")
    return result, {"available": True, **record}


def _gradient(loss: torch.Tensor, parameters: Mapping[str, torch.nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    values = torch.autograd.grad(loss, (parameters["encoder.phase"], parameters["generator.phase"]), retain_graph=retain_graph, allow_unused=True)
    return {name: torch.zeros_like(parameters[name]) if value is None else value.detach() for name, value in zip(("encoder.phase", "generator.phase"), values, strict=True)}


def _vector_similarity(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_vector = left.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right_vector = right.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    left_norm = float(left_vector.norm())
    right_norm = float(right_vector.norm())
    if left_norm == 0 or right_norm == 0:
        return {"status": "UNDECIDABLE_ZERO_GRADIENT", "cosine": None, "angle_degrees": None}
    cosine = float(torch.dot(left_vector, right_vector) / (left_norm * right_norm))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return {"status": "AVAILABLE", "cosine": cosine, "angle_degrees": angle}


def _gradient_similarity(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], group: str) -> dict[str, Any]:
    return _vector_similarity(left[group], right[group])


def _record_batch(
    consumer: Any,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    optimizer_state: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
    batch_index: int,
    endpoint: str,
    *,
    include_negative_control: bool = False,
    include_margin_reference: bool = False,
) -> dict[str, Any]:
    start, stop = _fit_batch_slice(batch_index)
    image = images[start:stop].to(consumer.device)
    label = labels[start:stop].to(consumer.device)
    result = consumer.forward_loss(image, data_role=bogt.TRAIN_ROLE)
    raw = result.result.raw_intensity
    raw_target = result.result.image_input.slm_intensity
    view = canonical.cmos_view(raw)
    target_view = canonical.cmos_view(image)
    losses: dict[str, torch.Tensor] = {"L_dist": result.L_dist, "L_rec": (raw - raw_target).square().mean(), "L_op": result.L_op}
    evaluator_features: dict[str, torch.Tensor] = {}
    evaluator_logits: dict[str, torch.Tensor] = {}
    target_metrics: dict[str, Any] = {}
    for name in EVALUATORS:
        generated_features, generated_logits = evaluators[name](view)
        with torch.no_grad():
            target_features, target_logits = evaluators[name](target_view)
        losses[f"feature_{name}"] = canonical.normalized_feature_loss(generated_features, target_features, f"{endpoint}/{name} feature loss")
        losses[f"ce_{name}"] = F.cross_entropy(generated_logits, label)
        evaluator_features[name] = generated_features
        evaluator_logits[name] = generated_logits
        target_metrics[name] = target_logits.detach()
    ed_logits = evaluators["E_D"](view)
    with torch.no_grad():
        target_metrics["E_D"] = evaluators["E_D"](target_view).detach()
    _finite(raw, f"{endpoint} raw CMOS")
    _finite(ed_logits, f"{endpoint} E_D logits")

    train_gradients: dict[str, dict[str, torch.Tensor]] = {}
    for name, loss in losses.items():
        train_gradients[name] = _gradient(loss, parameters, retain_graph=True)

    wrong_label_control: dict[str, Any] | None = None
    if include_negative_control:
        wrong_labels = (label + 1).remainder(10)
        wrong_ce = F.cross_entropy(evaluator_logits["E_S"], wrong_labels)
        wrong_gradient = _gradient(wrong_ce, parameters, retain_graph=True)
        true_vector = _flatten_phase(train_gradients["ce_E_S"]["encoder.phase"], train_gradients["ce_E_S"]["generator.phase"]).detach().cpu()
        wrong_vector = _flatten_phase(wrong_gradient["encoder.phase"], wrong_gradient["generator.phase"]).detach().cpu()
        delta_norm = float((true_vector - wrong_vector).double().norm().cpu())
        wrong_label_control = {"different_generator_and_encoder_gradient": not torch.allclose(true_vector, wrong_vector, rtol=0.0, atol=1e-8), "l2_difference": delta_norm, "wrong_labels": wrong_labels.cpu().tolist()}
        _require(wrong_label_control["different_generator_and_encoder_gradient"], "wrong-label CE negative control did not change the joint gradient")

    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for evaluator in EVALUATORS:
        gradients[evaluator] = train_gradients[f"ce_{evaluator}"]
    feature_gradients = {evaluator: train_gradients[f"feature_{evaluator}"] for evaluator in EVALUATORS}
    feature_semantic, feature_norms = _canonical_semantic(feature_gradients)
    ce_semantic, ce_norms = _canonical_semantic(gradients)
    dist_generator_norm = float(train_gradients["L_dist"]["generator.phase"].detach().double().norm().cpu())
    _require(dist_generator_norm == 0.0, "L_dist no longer has the declared structural-zero Generator gradient")
    auxiliary = ALPHA * train_gradients["L_dist"]["generator.phase"] + BETA * train_gradients["L_rec"]["generator.phase"] + GAMMA * train_gradients["L_op"]["generator.phase"]
    feature_total = None if feature_semantic is None else auxiliary + KAPPA * feature_semantic["generator.phase"]
    ce_total = None if ce_semantic is None else auxiliary + KAPPA * ce_semantic["generator.phase"]
    projected_feature_aux: torch.Tensor | None = None
    projected_ce_aux: torch.Tensor | None = None
    feature_projection = {"available": False}
    ce_projection = {"available": False}
    if feature_semantic is not None:
        projected_feature_aux, feature_projection = _project_auxiliary(auxiliary, feature_semantic["generator.phase"])
    if ce_semantic is not None:
        projected_ce_aux, ce_projection = _project_auxiliary(auxiliary, ce_semantic["generator.phase"])
    feature_projected = None if projected_feature_aux is None or feature_semantic is None else projected_feature_aux + KAPPA * feature_semantic["generator.phase"]
    ce_projected = None if projected_ce_aux is None or ce_semantic is None else projected_ce_aux + KAPPA * ce_semantic["generator.phase"]

    candidate_vectors: dict[str, torch.Tensor] = {}
    component_similarities: dict[str, Any] = {}
    for family, components in (("feature", feature_gradients), ("ce", gradients)):
        for name, value in components.items():
            candidate_vectors[f"{family}_component_gradient_descent_{name}"] = -value["generator.phase"].detach().cpu()
        component_similarities[family] = {}
        for left_index, left_name in enumerate(EVALUATORS):
            for right_name in EVALUATORS[left_index + 1 :]:
                pair = f"{left_name}_vs_{right_name}"
                component_similarities[family][pair] = {
                    "generator_only": _gradient_similarity(components[left_name], components[right_name], "generator.phase"),
                }
                joint_left = torch.cat((components[left_name]["encoder.phase"].reshape(-1), components[left_name]["generator.phase"].reshape(-1)))
                joint_right = torch.cat((components[right_name]["encoder.phase"].reshape(-1), components[right_name]["generator.phase"].reshape(-1)))
                component_similarities[family][pair]["joint_encoder_generator"] = _vector_similarity(joint_left, joint_right)
    for family, total, projected in (("feature", feature_total, feature_projected), ("ce", ce_total, ce_projected)):
        if total is None:
            continue
        semantic = feature_semantic if family == "feature" else ce_semantic
        candidate_vectors[f"{family}_semantic_gradient_descent"] = -KAPPA * semantic["generator.phase"].detach().cpu()
        candidate_vectors[f"{family}_gradient_descent"] = -total.detach().cpu()
        candidate_vectors[f"{family}_bogt_auxiliary_gradient_descent"] = -auxiliary.detach().cpu()
        candidate_vectors[f"{family}_adam_saved"] = _adam_candidate(total, optimizer_state, variant="saved")
        candidate_vectors[f"{family}_adam_zero_first"] = _adam_candidate(total, optimizer_state, variant="zero_first")
        candidate_vectors[f"{family}_adam_zero_history"] = _adam_candidate(total, optimizer_state, variant="zero_all")
        if projected is not None:
            candidate_vectors[f"{family}_projected_gradient_descent"] = -projected.detach().cpu()
            candidate_vectors[f"{family}_projected_adam_saved"] = _adam_candidate(projected, optimizer_state, variant="saved")

    relevant_positions = [offset for offset in range(BATCH_SIZE) if int(labels[start + offset]) in ROW_CLASSES]
    margin_components = {
        offset: _ed_margin_components(ed_logits[offset], int(label[offset].item()))
        for offset in relevant_positions
    }
    differentiable_positions = [offset for offset in relevant_positions if not margin_components[offset]["tied_maximum"]]
    row_records_by_offset: dict[int, dict[str, Any]] = {}
    margin_reference_rows: list[dict[str, Any]] = []
    margin_reference_tied_rows = [
        {"fit_position": start + offset, "source_row_index": int(source_indices[start + offset]), "label": int(label[offset].item()), "best_wrong_tied_classes": list(margin_components[offset]["best_wrong_tied_classes"]), "status": "UNDECIDABLE_TIED_MAXIMUM"}
        for offset in relevant_positions
        if margin_components[offset]["tied_maximum"]
    ]
    for ordinal, offset in enumerate(differentiable_positions):
        logits = ed_logits[offset]
        true_class = int(label[offset].item())
        components = margin_components[offset]
        row_gradients = torch.autograd.grad(
            components["margin"],
            (parameters["encoder.phase"], parameters["generator.phase"]),
            retain_graph=include_margin_reference or ordinal < len(differentiable_positions) - 1,
            allow_unused=True,
        )
        generator_gradient = torch.zeros_like(parameters["generator.phase"]) if row_gradients[1] is None else row_gradients[1].detach()
        encoder_gradient = torch.zeros_like(parameters["encoder.phase"]) if row_gradients[0] is None else row_gradients[0].detach()
        _finite(generator_gradient, f"{endpoint} E_D margin Generator gradient")
        _finite(encoder_gradient, f"{endpoint} E_D margin Encoder gradient")

        if include_margin_reference:
            true_components = torch.autograd.grad(
                logits[true_class],
                (parameters["encoder.phase"], parameters["generator.phase"]),
                retain_graph=True,
                allow_unused=True,
            )
            wrong_class = int(components["best_wrong_class"])
            wrong_components = torch.autograd.grad(
                logits[wrong_class],
                (parameters["encoder.phase"], parameters["generator.phase"]),
                retain_graph=True,
                allow_unused=True,
            )
            true_generator = torch.zeros_like(parameters["generator.phase"]) if true_components[1] is None else true_components[1].detach()
            wrong_generator = torch.zeros_like(parameters["generator.phase"]) if wrong_components[1] is None else wrong_components[1].detach()
            reference = true_generator.detach().to(device="cpu", dtype=torch.float64) - wrong_generator.detach().to(device="cpu", dtype=torch.float64)
            observed = generator_gradient.detach().to(device="cpu", dtype=torch.float64)
            difference_l2 = float((observed - reference).norm())
            true_norm = float(true_generator.detach().to(device="cpu", dtype=torch.float64).norm())
            wrong_norm = float(wrong_generator.detach().to(device="cpu", dtype=torch.float64).norm())
            tolerance_l2 = 2e-5 * (true_norm + wrong_norm) + 1e-12
            passed = difference_l2 <= tolerance_l2
            reference_record = {
                "fit_position": start + offset,
                "source_row_index": int(source_indices[start + offset]),
                "label": true_class,
                "best_wrong_class": wrong_class,
                "true_logit_gradient_generator_l2": true_norm,
                "best_wrong_logit_gradient_generator_l2": wrong_norm,
                "difference_l2": difference_l2,
                "tolerance_l2": tolerance_l2,
                "passed": passed,
            }
            _require(passed, f"{endpoint} E_D margin gradient differs from independent logit-component gradients at fit position {start + offset}")
            margin_reference_rows.append(reference_record)

        row_records_by_offset[offset] = {
            "fit_position": start + offset,
            "batch_index": batch_index,
            "within_batch": offset,
            "source_row_index": int(source_indices[start + offset]),
            "label": true_class,
            "ed_prediction": int(logits.detach().argmax().item()),
            "best_wrong_class": int(components["best_wrong_class"]),
            "best_wrong_tied_classes": list(components["best_wrong_tied_classes"]),
            "ed_true_logit": float(logits[true_class].detach().cpu()),
            "ed_best_wrong_logit": float(components["best_wrong_logit"].cpu()),
            "ed_margin": float(components["margin"].detach().cpu()),
            "best_wrong_exact_tie": bool(components["tied_maximum"]),
            "margin_gradient_generator_l2": float(generator_gradient.detach().double().norm().cpu()),
            "margin_gradient_encoder_l2_read_only": float(encoder_gradient.detach().double().norm().cpu()),
            "directions": {name: _direction_record(generator_gradient.reshape(-1).cpu(), vector.reshape(-1)) for name, vector in candidate_vectors.items()},
            "margin_gradient_generator": generator_gradient.reshape(-1).cpu(),
        }

    active_records: list[dict[str, Any]] = []
    for offset in relevant_positions:
        if offset in row_records_by_offset:
            active_records.append(row_records_by_offset[offset])
            continue
        logits = ed_logits[offset]
        true_class = int(label[offset].item())
        components = margin_components[offset]
        active_records.append({
            "fit_position": start + offset,
            "batch_index": batch_index,
            "within_batch": offset,
            "source_row_index": int(source_indices[start + offset]),
            "label": true_class,
            "ed_prediction": int(logits.detach().argmax().item()),
            "best_wrong_class": int(components["best_wrong_class"]),
            "best_wrong_tied_classes": list(components["best_wrong_tied_classes"]),
            "ed_true_logit": float(logits[true_class].detach().cpu()),
            "ed_best_wrong_logit": float(components["best_wrong_logit"].cpu()),
            "ed_margin": float(components["margin"].detach().cpu()),
            "best_wrong_exact_tie": True,
            "margin_gradient_generator_l2": None,
            "margin_gradient_encoder_l2_read_only": None,
            "directions": {name: _direction_record(None, vector.reshape(-1), tied_maximum=True) for name, vector in candidate_vectors.items()},
            "margin_gradient_generator": None,
        })
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only autograd populated optical parameter .grad buffers")
    _require(all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters()), "a frozen evaluator parameter received a gradient")

    semantic_norm_summary = {"feature": feature_norms, "ce": ce_norms}
    gradient_summaries: dict[str, Any] = {}
    for name, values in train_gradients.items():
        gradient_summaries[name] = {group: {"l2_norm": float(value.detach().double().norm().cpu()), "rms": float(value.detach().double().square().mean().sqrt().cpu())} for group, value in values.items()}
    loss_summary = {name: float(value.detach().cpu()) for name, value in losses.items()}
    readout = {
        "source_indices": source_indices[start:stop].clone(),
        "labels": labels[start:stop].clone(),
        "z": result.result.z_ols.coefficients.detach().cpu().clone(),
        "raw_cmos": raw.detach().cpu().clone(),
        "target_slm_intensity": raw_target.detach().cpu().clone(),
        "power_by_plane": bogt.contracted_plane_powers(result.result).detach().cpu().clone(),
        "evaluator_logits": {name: evaluator_logits[name].detach().cpu() for name in EVALUATORS},
        "e_d_logits": ed_logits.detach().cpu().clone(),
        "target_logits": {name: value.cpu() for name, value in target_metrics.items()},
    }
    return {
        "losses": loss_summary,
        "margin_derivative_definition": MARGIN_DEFINITION,
        "gradient_summaries": gradient_summaries,
        "semantic_normalization": semantic_norm_summary,
        "evaluator_gradient_pairwise_similarity": component_similarities,
        "structural_zero_l_dist_generator_norm": dist_generator_norm,
        "bo_gt_auxiliary_generator_norm": float(auxiliary.detach().double().norm().cpu()),
        "projection": {"feature": feature_projection, "ce": ce_projection},
        "rows": [{key: value for key, value in row.items() if key != "margin_gradient_generator"} for row in active_records],
        "margin_gradients_generator": torch.stack([row["margin_gradient_generator"] for row in active_records if row["margin_gradient_generator"] is not None]) if differentiable_positions else torch.empty((0, parameters["generator.phase"].numel()), dtype=torch.float32),
        "margin_gradient_fit_positions": [int(active_records[index]["fit_position"]) for index in range(len(active_records)) if active_records[index]["margin_gradient_generator"] is not None],
        "candidate_vectors_generator": candidate_vectors,
        "readouts": readout,
        "negative_control": wrong_label_control,
        "margin_derivative_reference": {"passed": all(row["passed"] for row in margin_reference_rows), "rows": margin_reference_rows, "undecidable_tied_rows": margin_reference_tied_rows} if include_margin_reference else None,
        "gradient_call_count": len(losses) + len(differentiable_positions) + 2 * len(margin_reference_rows) + int(include_negative_control),
    }


def _class_direction_summary(rows: Sequence[Mapping[str, Any]], class_id: int) -> dict[str, Any]:
    selected = [row for row in rows if int(row["label"]) == class_id]
    candidate_names = sorted({name for row in selected for name in row["directions"]})
    summary: dict[str, Any] = {}
    for candidate in candidate_names:
        counts = {status: sum(row["directions"].get(candidate, {}).get("status") == status for row in selected) for status in ("BENEFICIAL", "HARMFUL", "BOUNDARY", "UNDECIDABLE_ZERO_GRADIENT", "UNDECIDABLE_TIED_MAXIMUM", "UNDECIDABLE_MISSING_MARGIN_GRADIENT")}
        beneficial = counts["BENEFICIAL"]
        harmful = counts["HARMFUL"]
        mode = "BENEFICIAL" if beneficial >= STABLE_ROWS else ("HARMFUL" if harmful >= STABLE_ROWS else None)
        summary[candidate] = {"rows": len(selected), **counts, "stable": mode is not None, "stable_mode": mode}
    return summary


def _stable_paired(rows: Sequence[Mapping[str, Any]], class_id: int, predicate: Any) -> int:
    return sum(bool(predicate(row["directions"])) for row in rows if int(row["label"]) == class_id)


def _decision(rows_by_endpoint: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    primary = rows_by_endpoint["B_class_ce"]
    summaries = {endpoint: {str(class_id): _class_direction_summary(rows_by_endpoint[endpoint], class_id) for class_id in (*FOCUS_CLASSES, *CONTROL_CLASSES)} for endpoint in ENDPOINTS}
    optimizer_counts: dict[str, int] = {}
    auxiliary_counts: dict[str, int] = {}
    for class_id in FOCUS_CLASSES:
        optimizer_counts[str(class_id)] = _stable_paired(primary, class_id, lambda d: d.get("ce_gradient_descent", {}).get("status") == "BENEFICIAL" and d.get("ce_adam_saved", {}).get("status") == "HARMFUL" and d.get("ce_adam_zero_history", {}).get("status") == "BENEFICIAL")
        auxiliary_counts[str(class_id)] = _stable_paired(primary, class_id, lambda d: d.get("ce_semantic_gradient_descent", {}).get("status") == "BENEFICIAL" and d.get("ce_gradient_descent", {}).get("status") == "HARMFUL" and d.get("ce_adam_saved", {}).get("status") == "HARMFUL" and d.get("ce_projected_adam_saved", {}).get("status") == "BENEFICIAL")
    controls_no_stable_harm_adam = all(summaries["B_class_ce"][str(class_id)].get("ce_adam_zero_history", {}).get("stable_mode") != "HARMFUL" for class_id in CONTROL_CLASSES)
    controls_no_stable_harm_projection = all(summaries["B_class_ce"][str(class_id)].get("ce_projected_adam_saved", {}).get("stable_mode") != "HARMFUL" for class_id in CONTROL_CLASSES)
    focus_total_gradient_stable = all(summaries["B_class_ce"][str(class_id)].get("ce_gradient_descent", {}).get("stable_mode") == "BENEFICIAL" for class_id in FOCUS_CLASSES)
    focus_semantic_stable = all(summaries["B_class_ce"][str(class_id)].get("ce_semantic_gradient_descent", {}).get("stable_mode") == "BENEFICIAL" for class_id in FOCUS_CLASSES)
    reset_stable = all(summaries["B_class_ce"][str(class_id)].get("ce_adam_zero_history", {}).get("stable_mode") == "BENEFICIAL" for class_id in FOCUS_CLASSES)
    projection_stable = all(summaries["B_class_ce"][str(class_id)].get("ce_projected_adam_saved", {}).get("stable_mode") == "BENEFICIAL" for class_id in FOCUS_CLASSES)
    optimizer_supported = focus_total_gradient_stable and reset_stable and max(optimizer_counts.values(), default=0) >= STABLE_ROWS and controls_no_stable_harm_adam
    auxiliary_supported = focus_semantic_stable and projection_stable and max(auxiliary_counts.values(), default=0) >= STABLE_ROWS and controls_no_stable_harm_projection
    if optimizer_supported:
        classification = "OPTIMIZER_HISTORY_DIRECTION_CANDIDATE"
        recommendation = "提出一个新任务：在step2200固定模型/目标/数据下，对比保留与重置Generator Adam状态；本次诊断不授权执行。"
    elif auxiliary_supported:
        classification = "AUXILIARY_INTERFERENCE_DIRECTION_CANDIDATE"
        recommendation = "提出一个新任务：在同一固定模型/目标/数据下，对比原BOGT规则与Generator辅助梯度投影规则；本次诊断不授权执行。"
    else:
        classification = "MIXED_OR_INSUFFICIENT"
        recommendation = "NONE"
    return {
        "classification": classification,
        "training_recommendation": recommendation,
        "next_action": "NONE",
        "stable_direction_rows_required": STABLE_ROWS,
        "cosine_boundary_absolute": COSINE_BOUNDARY,
        "paired_row_counts": {"optimizer_reversal": optimizer_counts, "auxiliary_interference": auxiliary_counts},
        "focus_total_gradient_descent_stable_beneficial": focus_total_gradient_stable,
        "focus_semantic_descent_stable_beneficial": focus_semantic_stable,
        "zero_history_adam_stable_beneficial_both_focus_classes": reset_stable,
        "projected_saved_adam_stable_beneficial_both_focus_classes": projection_stable,
        "control_classes_no_stable_harm": {"zero_history_adam": controls_no_stable_harm_adam, "projected_saved_adam": controls_no_stable_harm_projection},
        "endpoint_class_summaries": summaries,
        "optimizer_candidate_supported": optimizer_supported,
        "auxiliary_candidate_supported": auxiliary_supported,
    }


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    used = int(total_bytes - free_bytes)
    _require(used / total_bytes <= 0.90, "PREFLIGHT_BLOCKED(resource): whole-device memory exceeds the safety line")
    return {"device": str(device), "total_bytes": int(total_bytes), "free_bytes": int(free_bytes), "global_used_bytes": used, "global_used_fraction": used / total_bytes, "pytorch_allocated_bytes": allocated, "pytorch_reserved_bytes": reserved}


def _disk_snapshot(target: Path) -> dict[str, Any]:
    probe = target
    while not probe.exists():
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    _require(free >= ESTIMATED_OUTPUT_BYTES + MIN_FREE_BYTES, "PREFLIGHT_BLOCKED(resource): estimated artifacts plus 1.5 GiB output reserve do not fit")
    return {"checked_path": probe.as_posix(), "free_bytes": free, "estimated_output_bytes": ESTIMATED_OUTPUT_BYTES, "required_remaining_bytes": MIN_FREE_BYTES}


def _resource_log(target: Path, phase: str, progress: Mapping[str, Any] | None = None, started: float | None = None) -> dict[str, Any]:
    snapshot = {"task_id": TASK_ID, "phase": phase, "progress": dict(progress or {}), "elapsed_seconds_observation_only": None if started is None else time.perf_counter() - started, "resource": _resource_snapshot(torch.device("cuda:0"))}
    with (target / RESOURCE_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
        handle.flush()
    return snapshot


def _runtime_dll_probe() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "not_applicable", "scope": "current_process"}
    build = torch.version.cuda
    _require(isinstance(build, str) and len(build.split(".")) == 2 and all(part.isdigit() for part in build.split(".")), "PyTorch CUDA build cannot identify the environment NVRTC DLL")
    major, minor = build.split(".")
    environment_bin = (Path(sys.prefix) / "bin").resolve()
    dll_path = environment_bin / f"nvrtc-builtins64_{major}{minor}.dll"
    _require(dll_path.is_file(), f"the installed environment is missing {dll_path.name}")
    current_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    normalized = os.path.normcase(os.path.normpath(str(environment_bin)))
    present = any(os.path.normcase(os.path.normpath(entry)) == normalized for entry in current_entries)
    if not present:
        os.environ["PATH"] = str(environment_bin) + (os.pathsep + os.environ.get("PATH", "") if os.environ.get("PATH") else "")
    try:
        import ctypes

        loader = getattr(ctypes, "WinDLL", None)
        if loader is not None:
            loader(str(dll_path))
    except OSError as error:
        raise DirectionDiagnosticError(f"CUDA NVRTC DLL load failed in the supported process environment: {error}") from error
    return {"status": "already_present" if present else "prepended", "scope": "current_process_only", "cuda_build": build, "environment_bin": str(environment_bin), "dll": dll_path.name}


def _preflight(inputs: Mapping[str, Any], paths: Mapping[str, Path], workspace: Path, population: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    margin_cpu_reference = _margin_derivative_cpu_reference_check()
    adam_formula_check = _adam_formula_cpu_reference_check()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "CUDA is unavailable for the declared full-production diagnostic")
    device = torch.device("cuda:0")
    runtime_dll = _runtime_dll_probe()
    resource_before = _resource_snapshot(device)
    disk = _disk_snapshot(workspace)
    evaluators = _load_evaluators(paths, device)
    evaluator_state = {name: _state_snapshot(model) for name, model in evaluators.items()}
    target_control = _target_control(inputs["images"], inputs["labels"], evaluators, device)
    qualified = _control_qualified(target_control)
    if not qualified:
        return {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "status": "INVALID_HARNESS", "reason": "one or more fixed TRAIN target positive controls failed qualification", "target_positive_control": target_control, "qualified": False, "runtime_dll": runtime_dll, "cuda_before": resource_before, "disk": disk, "elapsed_seconds_observation_only": time.perf_counter() - started}
    all_records: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        consumer, parameters = _new_consumer(inputs, endpoint, device)
        runtime_before = _state_snapshot(consumer.runtime)
        frozen_inputs_before = _consumer_frozen_inputs_snapshot(consumer)
        source_adam_before = _cpu_tree(inputs["optimizers"][endpoint])
        first_images = inputs["images"][:BATCH_SIZE].to(device)
        with torch.no_grad():
            no_grad_result = consumer.forward_loss(first_images, data_role=bogt.TRAIN_ROLE)
        no_grad_readout = {"z": no_grad_result.result.z_ols.coefficients.detach().cpu().clone(), "raw_cmos": no_grad_result.result.raw_intensity.detach().cpu().clone()}
        _finite(no_grad_readout["z"], f"{endpoint} no-grad OLS replay")
        _finite(no_grad_readout["raw_cmos"], f"{endpoint} no-grad CMOS replay")
        batch = _record_batch(consumer, parameters, evaluators, inputs["optimizers"][endpoint], inputs["images"], inputs["labels"], inputs["source_indices"], 0, endpoint, include_negative_control=endpoint == "B_class_ce", include_margin_reference=True)
        margin_reference = batch["margin_derivative_reference"]
        _require(isinstance(margin_reference, Mapping) and margin_reference["passed"] is True, f"{endpoint} GPU margin-derivative reference did not pass")
        reference_labels = {int(row["label"]) for row in margin_reference["rows"]} | {int(row["label"]) for row in margin_reference["undecidable_tied_rows"]}
        _require(reference_labels == set(ROW_CLASSES), f"{endpoint} GPU margin-derivative reference did not cover all focus and control classes")
        torch.cuda.synchronize(device)
        _require(torch.allclose(no_grad_readout["z"], batch["readouts"]["z"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{endpoint} no-grad then autograd OLS outputs differ")
        _require(torch.allclose(no_grad_readout["raw_cmos"], batch["readouts"]["raw_cmos"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{endpoint} no-grad then autograd CMOS outputs differ")
        parent_replay = _compare_parent_readout(batch, inputs, endpoint, 0)
        _require(_tree_equal(runtime_before, _state_snapshot(consumer.runtime)), f"{endpoint} preflight changed its runtime state")
        _require(_tree_equal(frozen_inputs_before, _consumer_frozen_inputs_snapshot(consumer)), f"{endpoint} preflight changed frozen references or loss weights")
        _require(_tree_equal(source_adam_before, _cpu_tree(inputs["optimizers"][endpoint])), f"{endpoint} preflight changed saved Adam tensors")
        _require(all(parameter.grad is None for parameter in parameters.values()), f"{endpoint} preflight left a model .grad buffer")
        all_records[endpoint] = {"batch": batch, "no_grad_then_autograd_equal": True, "parent_readout_replay": parent_replay, "runtime_state_unchanged": True, "references_and_weights_unchanged": True, "source_adam_unchanged": True, "parameter_grad_buffers_empty": all(parameter.grad is None for parameter in parameters.values())}
        del no_grad_result, no_grad_readout, first_images, batch, consumer, parameters
        torch.cuda.empty_cache()
    for name, model in evaluators.items():
        _require(_tree_equal(evaluator_state[name], _state_snapshot(model)), f"preflight mutated frozen evaluator {name}")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight evaluator {name} crossed its frozen boundary")
    resource_after = _resource_snapshot(device)
    _require(all(record["parameter_grad_buffers_empty"] for record in all_records.values()), "preflight left a model .grad buffer")
    _require(all(record["source_adam_unchanged"] for record in all_records.values()), "preflight changed saved Adam tensors")
    _require(all(record["runtime_state_unchanged"] and record["references_and_weights_unchanged"] for record in all_records.values()), "preflight changed frozen optical state")
    _require(all(_tree_equal(evaluator_state[name], _state_snapshot(model)) for name, model in evaluators.items()), "preflight changed frozen evaluator state")
    fixed_fit_rows_and_roles = _fixed_fit_population_matches(inputs)
    _require(fixed_fit_rows_and_roles, "preflight fit-row population or data roles changed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "status": "READY",
        "cuda_runtime": runtime_dll,
        "propagation_grid": 2048,
        "scientific_updates": 0,
        "gradient_call_count": sum(record["batch"]["gradient_call_count"] for record in all_records.values()),
        "margin_derivative_definition": MARGIN_DEFINITION,
        "margin_derivative_cpu_reference": margin_cpu_reference,
        "cpu_toy_adam_formula_reference": adam_formula_check,
        "target_positive_control": target_control,
        "target_positive_control_qualified": qualified,
        "negative_label_control": all_records["B_class_ce"]["batch"]["negative_control"],
        "representative_autograd_batches": {name: {"batch_index": 0, "readout_shapes": {key: list(value.shape) for key, value in record["batch"]["readouts"].items() if torch.is_tensor(value)}, "no_grad_then_autograd_equal": record["no_grad_then_autograd_equal"], "parent_readout_replay": record["parent_readout_replay"], "runtime_state_unchanged": record["runtime_state_unchanged"], "references_and_weights_unchanged": record["references_and_weights_unchanged"], "source_adam_unchanged": record["source_adam_unchanged"], "parameter_grad_buffers_empty": record["parameter_grad_buffers_empty"], "margin_derivative_reference": record["batch"]["margin_derivative_reference"]} for name, record in all_records.items()},
        "fixed_fit_rows_and_roles": fixed_fit_rows_and_roles,
        "parameter_grad_buffers_empty": all(record["parameter_grad_buffers_empty"] for record in all_records.values()),
        "evaluator_weights_frozen": all(_tree_equal(evaluator_state[name], _state_snapshot(model)) and all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()) for name, model in evaluators.items()),
        "optical_source_adam_object_instantiated": False,
        "source_adam_tensor_state_modified": not all(record["source_adam_unchanged"] for record in all_records.values()),
        "cpu_toy_adam_reference_created": True,
        "candidate_adam_formula_cpu_reference_checked": adam_formula_check["passed"],
        "direction_sign_control_checked": margin_cpu_reference["passed"],
        "gpu_margin_derivative_references": {name: record["batch"]["margin_derivative_reference"] for name, record in all_records.items()},
        "disk_observation": disk,
        "cuda_before": resource_before,
        "cuda_after": resource_after,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "preflight_vectors": {name: record["batch"] for name, record in all_records.items()},
    }


def _identity(inputs: Mapping[str, Any], population_record: Mapping[str, Any], output_dir: Path) -> ResearchRunIdentity:
    sample = inputs["parameters"]["B_class_ce"]
    attempt_name = output_dir.name.replace("_", "-")
    attempt_id = f"{TASK_ID}:{attempt_name}"
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-{attempt_name}",
        trajectory_id="frozen-step2200-feature-and-class-directions-adam-e-d-margin",
        semantic_authority_id=parent.TASK_ID,
        data_role="200 original canonical TRAIN fit rows; per-batch gradients; no observation or confirmation images read",
        config_identity={"propagation_grid": 2048, "source_checkpoint_step": 2200, "fit_rows": FIT_ROWS, "batch_size": BATCH_SIZE, "delta_SIC": DELTA_SIC, "kappa": KAPPA, "alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "evaluators": list(ALL_EVALUATORS)},
        model_identity={"parameter_shapes": {key: list(value.shape) for key, value in sample.items()}, "endpoint_steps": {"step2200": [2200, 2200], "A_feature": [2200, 2500], "B_class_ce": [2200, 2500]}},
        optimizer_identity={"class": "saved optical Adam tensors evaluated read-only; CPU toy Adam used only as an independent formula reference", "parameter_order": ["encoder.phase", "generator.phase"], "variants": ["saved", "zero_first_keep_second_and_step", "zero_all_at_step_one"]},
        sampler_identity={"source_indices": population_record["source_row_indices"], "labels": population_record["labels"], "roles": "fit-only first200; original 20 batches unchanged"},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=attempt_id,
    )


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    margin_cpu_passed = preflight.get("margin_derivative_cpu_reference", {}).get("passed") is True
    gpu_margin_references = preflight.get("gpu_margin_derivative_references", {})
    margin_gpu_passed = set(gpu_margin_references) == set(ENDPOINTS) and all(
        isinstance(gpu_margin_references[name], Mapping)
        and gpu_margin_references[name].get("passed") is True
        and ({int(row["label"]) for row in gpu_margin_references[name].get("rows", [])} | {int(row["label"]) for row in gpu_margin_references[name].get("undecidable_tied_rows", [])}) == set(ROW_CLASSES)
        for name in ENDPOINTS
    )
    positive_control_passed = preflight.get("target_positive_control_qualified") is True and _control_qualified(preflight["target_positive_control"])
    negative_control_passed = preflight.get("negative_label_control", {}).get("different_generator_and_encoder_gradient") is True
    frozen_state_passed = all(
        record.get("runtime_state_unchanged") is True
        and record.get("references_and_weights_unchanged") is True
        and record.get("source_adam_unchanged") is True
        for record in preflight.get("representative_autograd_batches", {}).values()
    )
    representative_path_passed = set(preflight.get("representative_autograd_batches", {})) == set(ENDPOINTS) and all(
        record.get("no_grad_then_autograd_equal") is True
        and record.get("parent_readout_replay", {}).get("passed") is True
        for record in preflight["representative_autograd_batches"].values()
    )
    fixed_fit_population_passed = preflight.get("fixed_fit_rows_and_roles") is True
    correctness_passed = all((margin_cpu_passed, margin_gpu_passed, positive_control_passed, negative_control_passed, frozen_state_passed, fixed_fit_population_passed))
    implementation_passed = representative_path_passed and margin_cpu_passed and margin_gpu_passed and preflight.get("parameter_grad_buffers_empty") is True and preflight.get("evaluator_weights_frozen") is True
    runtime_status = preflight.get("cuda_runtime", {}).get("status")
    runtime_passed = runtime_status in {"already_present", "prepended", "not_applicable"}
    resource_snapshots = (preflight.get("cuda_before", {}), preflight.get("cuda_after", {}))
    whole_device_safety_passed = all(float(snapshot.get("global_used_fraction", 1.0)) <= 0.90 for snapshot in resource_snapshots)
    disk = preflight.get("disk_observation", {})
    disk_passed = int(disk.get("free_bytes", 0)) >= int(disk.get("estimated_output_bytes", 0)) + int(disk.get("required_remaining_bytes", MIN_FREE_BYTES))
    resource_passed = runtime_passed and whole_device_safety_passed and disk_passed
    return control.make_ready_receipt(
        correctness={"passed": correctness_passed, "status": "PASS" if correctness_passed else "BLOCKED", "fixed_fit_rows_and_roles": fixed_fit_population_passed, "margin_derivative_cpu_reference": margin_cpu_passed, "margin_derivative_gpu_references": margin_gpu_passed, "target_positive_controls": positive_control_passed, "wrong_label_negative_control": negative_control_passed, "freeze_and_readonly_adam_checks": frozen_state_passed},
        resource={"passed": resource_passed, "status": "PASS" if resource_passed else "BLOCKED", "cuda_runtime": runtime_passed, "whole_device_safety_line": 0.90, "whole_device_safety_passed": whole_device_safety_passed, "observations": {"before": preflight.get("cuda_before"), "after": preflight.get("cuda_after")}, "disk_passed": disk_passed, "disk": disk},
        implementation={"passed": implementation_passed, "status": "PASS" if implementation_passed else "BLOCKED", "selected_engine": "cuda_full_2048_frozen_gradient_diagnostic", "representative_no_grad_to_autograd": representative_path_passed, "three_endpoint_state_frozen": frozen_state_passed, "adam_candidates_read_only": preflight.get("source_adam_tensor_state_modified") is False, "margin_derivative_reference": margin_cpu_passed and margin_gpu_passed},
        selected_engine="cuda_full_2048_frozen_gradient_diagnostic",
        resource_envelope={"cuda": preflight["cuda_after"], "disk": preflight["disk_observation"]},
    )


def _population_record(paths: Mapping[str, Path], population: Mapping[str, Any], inputs: Mapping[str, Any], artifact_dir: Path, review_dir: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "artifact_dir": artifact_dir.as_posix(),
        "review_dir": review_dir.as_posix(),
        "source_paths": {key: paths[key].as_posix() for key in sorted(paths)},
        "parent_source_indices": [int(value) for value in population["source_row_indices"]],
        "parent_labels": [int(value) for value in population["labels"]],
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "fit_rows": FIT_ROWS,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "focus_classes": list(FOCUS_CLASSES),
        "control_classes": list(CONTROL_CLASSES),
        "roles": {"fit": [0, 200], "historic_holdout": "readout provenance only", "confirmation": "readout provenance only"},
        "e_d_role": "frozen read-only input-gradient evaluator; used as a direction diagnostic, not a blind future confirmation",
        "source_row_mapping_verified": inputs["review_rows_mapped"],
    }


def _manifest_seed(identity: ResearchRunIdentity, population: Mapping[str, Any], receipt: PreflightReceipt, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": MANIFEST_SCHEMA, "task_id": TASK_ID, "margin_derivative_definition": MARGIN_DEFINITION, "identity": identity.to_dict(), "population": dict(population), "receipt": receipt.to_dict(), "preflight": {key: value for key, value in preflight.items() if key != "preflight_vectors"}, "preflight_replay_artifact": "preflight_replay.pt", "completed_blocks": [], "status": "RUNNING"}


def _block_name(endpoint: str, batch_index: int) -> str:
    return f"block_{endpoint}_batch{batch_index:02d}.pt"


def _compare_preflight_batch(actual: Mapping[str, Any], expected: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    left, right = actual["candidate_vectors_generator"], expected["candidate_vectors_generator"]
    _require(set(left) == set(right), f"{endpoint} repeated preflight candidate set changed")
    differences = {}
    within_tolerance: dict[str, bool] = {}
    for name in left:
        _require(left[name].shape == right[name].shape, f"{endpoint} preflight candidate shape differs for {name}")
        within_tolerance[name] = bool(torch.allclose(left[name], right[name], rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
        differences[name] = float((left[name] - right[name]).abs().max().item())
    _require(_tree_equal(actual["readouts"]["source_indices"], expected["readouts"]["source_indices"]), f"{endpoint} preflight row indices changed")
    _require(_tree_equal(actual["readouts"]["labels"], expected["readouts"]["labels"]), f"{endpoint} preflight labels changed")
    for field in ("z", "raw_cmos", "target_slm_intensity", "power_by_plane", "e_d_logits"):
        _require(torch.allclose(actual["readouts"][field], expected["readouts"][field], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{endpoint} preflight readout replay differs for {field}")
    for name in EVALUATORS:
        _require(torch.allclose(actual["readouts"]["evaluator_logits"][name], expected["readouts"]["evaluator_logits"][name], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{endpoint} preflight evaluator replay differs for {name}")
    return {"readout_replay_passed": True, "candidate_comparison_is_observation_only": True, "candidate_vector_within_tolerance": within_tolerance, "tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL}, "candidate_max_absolute_difference": differences}


def _compare_parent_readout(block: Mapping[str, Any], inputs: Mapping[str, Any], endpoint: str, batch_index: int) -> dict[str, Any]:
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    actual = block["readouts"]
    saved = inputs["readouts"][endpoint]
    field_map = {"z": "z", "raw_cmos": "raw_cmos", "target_slm_intensity": "target_slm_intensity", "power_by_plane": "power_by_plane", "e_d_logits": "e_d_logits"}
    for name in EVALUATORS:
        field_map[f"{name}_logits"] = {"E_S": "e_s_logits", "E_I": "e_i_logits", "E_C": "e_c_logits"}[name]
    max_difference: dict[str, float] = {}
    prediction_flips: dict[str, int] = {}
    for actual_name, saved_name in field_map.items():
        actual_value = actual["evaluator_logits"][actual_name[0:-7]] if actual_name.endswith("_logits") and actual_name[0:-7] in EVALUATORS else actual.get(actual_name)
        expected = saved[saved_name][start:stop]
        _require(torch.is_tensor(actual_value) and actual_value.shape == expected.shape, f"{endpoint} replay shape mismatch for {actual_name}")
        difference = (actual_value - expected).abs()
        _require(torch.allclose(actual_value, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{endpoint} frozen replay differs from parent output for {actual_name}")
        max_difference[actual_name] = float(difference.max().item()) if difference.numel() else 0.0
        if actual_name == "e_d_logits":
            prediction_flips["E_D"] = int((actual_value.argmax(1) != expected.argmax(1)).sum().item())
        elif actual_name.endswith("_logits"):
            evaluator = actual_name[:-7]
            prediction_flips[evaluator] = int((actual_value.argmax(1) != expected.argmax(1)).sum().item())
    _require(torch.equal(actual["source_indices"], saved["source_indices"][start:stop]), f"{endpoint} replay source rows differ from parent output")
    _require(torch.equal(actual["labels"], saved["labels"][start:stop]), f"{endpoint} replay labels differ from parent output")
    return {"passed": True, "tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL}, "maximum_absolute_difference_by_field": max_difference, "prediction_boundary_flips": prediction_flips}


def _write_block(target: Path, endpoint: str, batch_index: int, block: Mapping[str, Any]) -> str:
    path = target / _block_name(endpoint, batch_index)
    payload = {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "margin_derivative_definition": MARGIN_DEFINITION, "endpoint": endpoint, "batch_index": batch_index, "block": dict(block)}
    _write_torch_new(path, payload, "immutable frozen Generator direction measurement block")
    return path.name


def _read_orphan_block(path: Path, endpoint: str, batch_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping) and payload.get("schema") == BLOCK_SCHEMA and payload.get("task_id") == TASK_ID and payload.get("margin_derivative_definition") == MARGIN_DEFINITION, "uncommitted existing block has invalid task or derivative semantics")
    _require(payload.get("endpoint") == endpoint and payload.get("batch_index") == batch_index, "uncommitted block does not match its explicit name")
    block = payload.get("block")
    _require(isinstance(block, Mapping) and block.get("margin_derivative_definition") == MARGIN_DEFINITION, "uncommitted block derivative semantics are invalid")
    expected_rows = inputs["source_indices"][batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
    _require(torch.equal(block["readouts"]["source_indices"], expected_rows), "uncommitted block source rows differ from fixed fit batch")
    _require(torch.equal(block["readouts"]["labels"], inputs["labels"][batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]), "uncommitted block labels differ from fixed fit batch")
    return block


def _class_rows(blocks: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {endpoint: [{**dict(row), "endpoint": endpoint} for block in values for row in block["rows"]] for endpoint, values in blocks.items()}


def _finalize(target: Path, inputs: Mapping[str, Any], population: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, identity: ResearchRunIdentity, blocks: Mapping[str, Sequence[Mapping[str, Any]]], started: float) -> dict[str, Any]:
    flat_rows = _class_rows(blocks)
    report_rows: dict[str, list[dict[str, Any]]] = {}
    for endpoint, rows in flat_rows.items():
        report_rows[endpoint] = [{key: value for key, value in row.items()} for row in rows]
    row_directions_path = target / "row_direction_metrics.csv"
    columns = ("endpoint", "fit_position", "batch_index", "source_row_index", "label", "ed_prediction", "best_wrong_class", "best_wrong_tied_classes_json", "ed_true_logit", "ed_best_wrong_logit", "ed_margin", "best_wrong_exact_tie", "margin_gradient_generator_l2", "directions_json")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for endpoint, rows in report_rows.items():
        for row in rows:
            writer.writerow({**{key: row[key] for key in columns if key not in {"best_wrong_tied_classes_json", "directions_json"}}, "best_wrong_tied_classes_json": json.dumps(row["best_wrong_tied_classes"]), "directions_json": json.dumps(row["directions"], sort_keys=True)})
    csv_content = buffer.getvalue().encode("utf-8")
    if row_directions_path.exists():
        _require(row_directions_path.read_bytes() == csv_content, "existing row-direction table differs from the immutable recovered blocks")
    else:
        with row_directions_path.open("xb") as handle:
            handle.write(csv_content)
    decision = _decision(flat_rows)
    evaluator_metrics: dict[str, Any] = {}
    for endpoint, endpoint_blocks in blocks.items():
        scores = {name: torch.cat([block["readouts"]["e_d_logits"] if name == "E_D" else block["readouts"]["evaluator_logits"][name] for block in endpoint_blocks]) for name in ALL_EVALUATORS}
        metrics = {}
        for name, logits in scores.items():
            predicted = logits.argmax(1)
            class_counts = torch.bincount(inputs["labels"], minlength=10).tolist()
            correct = [int(((inputs["labels"] == cls) & (predicted == inputs["labels"])).sum()) for cls in range(10)]
            metrics[name] = {"rows": FIT_ROWS, "macro_recall": sum(correct[i] / class_counts[i] for i in range(10)) / 10, "per_class_correct": correct, "per_class_rows": class_counts, "prediction_counts": torch.bincount(predicted, minlength=10).tolist(), "confusion_matrix_actual_rows_predicted_columns": [[int(((inputs["labels"] == actual) & (predicted == predicted_class)).sum()) for predicted_class in range(10)] for actual in range(10)]}
        evaluator_metrics[endpoint] = metrics
    moment_decay = {"beta1_direct_source_moment_residual_after_300_updates": 0.9**300, "beta2_old_component_coefficient_after_300_updates": 0.999**300, "interpretation": "recurrence coefficients only; they do not measure current second-moment share and do not erase effects of earlier parameter trajectory"}
    replay_comparisons = {endpoint: [block["parent_readout_comparison"] for block in blocks[endpoint]] for endpoint in ENDPOINTS}
    preflight_candidate_replay = {endpoint: blocks[endpoint][0].get("preflight_candidate_replay") for endpoint in ENDPOINTS}
    prior_failure_path = target / "failure.json"
    prior_engineering_failure = json.loads(prior_failure_path.read_text(encoding="utf-8")) if prior_failure_path.is_file() else None
    engineering_events_path = target / "engineering_events.jsonl"
    engineering_events = [json.loads(line) for line in engineering_events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if engineering_events_path.is_file() else []
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "margin_derivative_definition": MARGIN_DEFINITION,
        "predecessor_v1_direction_classification_qualified": False,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "training_updates": 0,
        "optical_optimizer_objects_created": 0,
        "optical_forward_batch_count": BATCH_COUNT * len(ENDPOINTS) + 2 * len(ENDPOINTS),
        "optical_forward_row_count": FIT_ROWS * len(ENDPOINTS) + 2 * len(ENDPOINTS) * BATCH_SIZE,
        "measurement_gradient_calls": sum(block["gradient_call_count"] for values in blocks.values() for block in values),
        "parent_training_status_preserved": "VALID_FAILED / NOT_SUPPORTED_WITHIN_SCOPE",
        "training_recommendation": decision["training_recommendation"],
        "next_action": "NONE",
        "stop": True,
        "scope": {"fit_rows": FIT_ROWS, "batch_size": BATCH_SIZE, "batch_count": BATCH_COUNT, "focus_classes": list(FOCUS_CLASSES), "control_classes": list(CONTROL_CLASSES), "all_fit_rows_canonical_train": True, "historic_holdout_and_confirmation_images_opened": False, "e_d_weights_frozen": True, "e_d_used_for_direction_diagnostic": True, "future_blind_e_d_confirmation_claimed": False, "model_ols_geometry_references_weights_evaluators_frozen": True},
        "population": dict(population),
        "endpoint_adam_steps": {endpoint: inputs["payloads"][endpoint]["optimizer_step_by_parameter"] for endpoint in ENDPOINTS},
        "loss_weights": {"alpha": ALPHA, "beta": BETA, "gamma": GAMMA, "delta_SIC": DELTA_SIC, "kappa": KAPPA},
        "adam_moment_decay_observation": moment_decay,
        "target_positive_control": preflight["target_positive_control"],
        "target_positive_control_qualified": True,
        "endpoint_evaluator_metrics_fit_rows": evaluator_metrics,
        "parent_saved_readout_replay": replay_comparisons,
        "preflight_candidate_replay_observation": preflight_candidate_replay,
        "prior_engineering_failure_artifact": None if prior_engineering_failure is None else {"artifact": prior_failure_path.name, "status": prior_engineering_failure.get("status"), "reason": prior_engineering_failure.get("reason"), "scientific_classification": None},
        "prior_engineering_events": engineering_events,
        "direction_summary": decision,
        "evidence_artifacts": {"direction_csv": row_directions_path.name, "readout_blocks": sorted(path.name for path in target.glob("block_*.pt")), "class_candidate_summary": "report.json direction_summary.endpoint_class_summaries"},
        "ready_receipt": receipt.to_dict(),
        "research_run_identity": identity.to_dict(),
        "preflight": {key: value for key, value in preflight.items() if key != "preflight_vectors"},
        "resource_telemetry_artifact": RESOURCE_NAME,
        "preflight_gradient_calls": preflight["gradient_call_count"],
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args()


def _validate_resume_manifest(path: Path, target: Path, population: Mapping[str, Any], identity: ResearchRunIdentity) -> dict[str, Any]:
    _require(path.resolve() == (target / "resume_manifest.json").resolve() and path.is_file(), "resume requires the task-owned explicit resume_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID and manifest.get("margin_derivative_definition") == MARGIN_DEFINITION, "resume manifest task/schema/derivative semantics mismatch")
    _require(manifest.get("population") == dict(population), "resume population, rows, labels, or roles changed")
    check_resume_compatibility(identity, manifest["identity"])
    _require(manifest.get("status") in {"RUNNING", "INTERRUPTED", "COMPLETED"}, "resume manifest state is invalid")
    return manifest


def _terminal_report_if_present(target: Path, manifest: dict[str, Any], identity: ResearchRunIdentity) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if manifest.get("status") == "COMPLETED":
        _require(report_path.is_file(), "completed manifest lacks its final report")
    elif not report_path.is_file():
        return None
    expected_blocks = {_block_name(endpoint, batch_index) for endpoint in ENDPOINTS for batch_index in range(BATCH_COUNT)}
    _require(expected_blocks.issubset(set(manifest.get("completed_blocks", []))) and all((target / name).is_file() for name in expected_blocks), "final report exists before all immutable readout blocks are recoverably complete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == REPORT_SCHEMA and report.get("task_id") == TASK_ID and report.get("margin_derivative_definition") == MARGIN_DEFINITION and report.get("status") == "VALID_COMPLETED" and report.get("research_run_identity") == identity.to_dict(), "saved terminal report is invalid or belongs to another derivative contract")
    if manifest.get("status") != "COMPLETED":
        manifest["status"] = "COMPLETED"
        _write_json_replace(target / "resume_manifest.json", manifest)
    return report


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    artifact_dir = _resolve(root, arguments.artifact_dir)
    review_dir = _resolve(root, arguments.review_dir)
    target = _resolve(root, arguments.output_dir)
    output_root = (root / "outputs").resolve()
    _require(target != output_root and output_root in target.parents, "output must be a named child of repository outputs/")
    _require(not (arguments.preflight_only and arguments.resume_manifest), "--preflight-only cannot resume")
    _require(arguments.resume_manifest is not None or not target.exists(), "output exists; provide its explicit --resume-manifest to continue")
    paths, parent_population, parent_report, review = _path_inputs(root, artifact_dir, review_dir)
    inputs = _load_inputs(root, paths, parent_population, artifact_dir)
    population = _population_record(paths, parent_population, inputs, artifact_dir, review_dir)
    identity = _identity(inputs, population, target)
    existing_manifest: dict[str, Any] | None = None
    if arguments.resume_manifest is not None:
        _require(target.is_dir(), "resume output directory is missing")
        existing_manifest = _validate_resume_manifest(_resolve(root, arguments.resume_manifest), target, population, identity)
        report = _terminal_report_if_present(target, existing_manifest, identity)
        if report is not None:
            return report
    started = time.perf_counter()
    if arguments.preflight_only:
        _require(not target.exists(), "preflight-only output directory must be new and explicit")
        target.mkdir(parents=True, exist_ok=False)
        workspace = target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        workspace = target.parent
    if existing_manifest is None:
        if arguments.preflight_only:
            preflight = _preflight(inputs, paths, workspace, population)
        else:
            with tempfile.TemporaryDirectory(prefix="lightgen_semantic_direction_preflight_", dir=workspace) as temp:
                preflight = _preflight(inputs, paths, Path(temp), population)
        receipt_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
        if preflight.get("status") != "READY":
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": preflight.get("reason"), "training_updates": 0, "next_action": "NONE", "preflight": {key: value for key, value in preflight.items() if key != "preflight_vectors"}}
            if not arguments.preflight_only:
                target.mkdir(parents=True, exist_ok=False)
                _write_json_new(target / "failure.json", failure, "unqualified frozen E_D direction preflight")
            else:
                _write_json_new(target / "failure.json", failure, "unqualified frozen E_D direction preflight")
            return failure
        receipt = _ready_receipt(receipt_control, preflight)
        receipt_control.mark_preflight_ready(receipt)
        if arguments.preflight_only:
            _write_json_new(target / "input_population.json", population, "fixed frozen direction diagnostic population")
            _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": {key: value for key, value in preflight.items() if key != "preflight_vectors"}}, "standalone CUDA direction preflight")
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}
        receipt_control.start_execution()
        receipt_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
        _write_json_new(target / "input_population.json", population, "fixed TRAIN fit rows and explicit endpoint paths")
        _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": {key: value for key, value in preflight.items() if key != "preflight_vectors"}}, "READY receipt and full CUDA zero-update preflight")
        manifest = _manifest_seed(identity, population, receipt, preflight)
        _write_torch_new(target / manifest["preflight_replay_artifact"], {"schema": PREFLIGHT_SCHEMA, "task_id": TASK_ID, "margin_derivative_definition": MARGIN_DEFINITION, "endpoint_vectors": preflight["preflight_vectors"]}, "immutable preflight replay vectors for recovery verification")
        _write_json_replace(target / "resume_manifest.json", manifest)
    else:
        preflight_file = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
        _require(preflight_file.get("task_id") == TASK_ID and preflight_file.get("status") == "READY", "saved preflight receipt is not READY")
        receipt = PreflightReceipt.from_mapping(preflight_file["receipt"])
        preflight = existing_manifest["preflight"]
        receipt_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
        receipt_control.mark_preflight_ready(receipt)
        receipt_control.start_execution()
        receipt_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=True)
        manifest = existing_manifest
        manifest["status"] = "RUNNING"
        _write_json_replace(target / "resume_manifest.json", manifest)
    replay_path = target / manifest.get("preflight_replay_artifact", "")
    _require(replay_path.name == "preflight_replay.pt" and replay_path.is_file(), "resume lacks its explicit saved preflight replay-vector artifact")
    replay_payload = torch.load(replay_path, map_location="cpu", weights_only=True)
    _require(replay_payload.get("schema") == PREFLIGHT_SCHEMA and replay_payload.get("task_id") == TASK_ID and replay_payload.get("margin_derivative_definition") == MARGIN_DEFINITION, "saved preflight replay-vector artifact is invalid")
    replay_vectors = replay_payload.get("endpoint_vectors")
    _require(isinstance(replay_vectors, Mapping) and set(replay_vectors) == set(ENDPOINTS), "saved preflight replay vectors do not cover every endpoint")

    execution_runtime_dll = _runtime_dll_probe()
    preflight["execution_runtime_dll_probe"] = execution_runtime_dll
    device = torch.device("cuda:0")
    _resource_log(target, "measurement_start", {"completed_blocks": len(manifest.get("completed_blocks", []))}, started)
    evaluators = _load_evaluators(paths, device)
    evaluator_snapshots = {name: _state_snapshot(model) for name, model in evaluators.items()}
    blocks_by_endpoint: dict[str, list[Mapping[str, Any]]] = {endpoint: [] for endpoint in ENDPOINTS}
    completed = set(manifest.get("completed_blocks", []))
    try:
        for endpoint in ENDPOINTS:
            consumer, parameters = _new_consumer(inputs, endpoint, device)
            runtime_initial = _state_snapshot(consumer.runtime)
            frozen_inputs_initial = _consumer_frozen_inputs_snapshot(consumer)
            source_adam_initial = _cpu_tree(inputs["optimizers"][endpoint])
            for batch_index in range(BATCH_COUNT):
                name = _block_name(endpoint, batch_index)
                path = target / name
                if name in completed:
                    _require(path.is_file(), f"resume manifest declares missing block: {name}")
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    _require(payload.get("schema") == BLOCK_SCHEMA and payload.get("margin_derivative_definition") == MARGIN_DEFINITION and payload.get("endpoint") == endpoint and payload.get("batch_index") == batch_index, f"saved block does not match resume manifest derivative semantics: {name}")
                    block = payload["block"]
                elif path.exists():
                    block = _read_orphan_block(path, endpoint, batch_index, inputs)
                    completed.add(name)
                    manifest["completed_blocks"] = sorted(completed)
                    _write_json_replace(target / "resume_manifest.json", manifest)
                else:
                    _resource_log(target, "batch_start", {"endpoint": endpoint, "batch_index": batch_index}, started)
                    block = _record_batch(consumer, parameters, evaluators, inputs["optimizers"][endpoint], inputs["images"], inputs["labels"], inputs["source_indices"], batch_index, endpoint)
                    if batch_index == 0:
                        reference = replay_vectors.get(endpoint)
                        _require(isinstance(reference, Mapping), f"{endpoint} preflight replay vector reference is missing")
                        block["preflight_candidate_replay"] = _compare_preflight_batch(block, reference, endpoint)
                    block["parent_readout_comparison"] = _compare_parent_readout(block, inputs, endpoint, batch_index)
                    name = _write_block(target, endpoint, batch_index, block)
                    completed.add(name)
                    manifest["completed_blocks"] = sorted(completed)
                    _write_json_replace(target / "resume_manifest.json", manifest)
                parent_replay = _compare_parent_readout(block, inputs, endpoint, batch_index)
                _require(block.get("parent_readout_comparison") == parent_replay, f"{endpoint} saved per-batch replay comparison changed")
                expected_ids = inputs["source_indices"][batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
                _require(torch.equal(block["readouts"]["source_indices"], expected_ids), f"{endpoint} batch {batch_index} source-row order changed")
                _require(torch.equal(block["readouts"]["labels"], inputs["labels"][batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]), f"{endpoint} batch {batch_index} labels changed")
                blocks_by_endpoint[endpoint].append(block)
                if batch_index % 4 == 3:
                    _resource_log(target, "batch_progress", {"endpoint": endpoint, "completed_batches": batch_index + 1}, started)
                del block
            _require(_tree_equal(runtime_initial, _state_snapshot(consumer.runtime)), f"{endpoint} full endpoint measurement changed model parameters or scientific buffers")
            _require(_tree_equal(frozen_inputs_initial, _consumer_frozen_inputs_snapshot(consumer)), f"{endpoint} full endpoint measurement changed frozen references or weights")
            _require(_tree_equal(source_adam_initial, _cpu_tree(inputs["optimizers"][endpoint])), f"{endpoint} full endpoint measurement changed source Adam state")
            _require(all(parameter.grad is None for parameter in parameters.values()), f"{endpoint} ended with a nonempty parameter .grad buffer")
            del consumer, parameters
            torch.cuda.empty_cache()
            for evaluator_name, model in evaluators.items():
                _require(_tree_equal(evaluator_snapshots[evaluator_name], _state_snapshot(model)), f"{endpoint} changed frozen evaluator {evaluator_name}")
                _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{endpoint} evaluator {evaluator_name} crossed its frozen boundary")
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        _write_json_replace(target / "resume_manifest.json", manifest)
        receipt_control.interrupted()
        raise
    report = _finalize(target, inputs, population, preflight, receipt, identity, blocks_by_endpoint, started)
    _write_json_new(target / "report.json", report, "final frozen Generator direction decision")
    manifest["status"] = "COMPLETED"
    _write_json_replace(target / "resume_manifest.json", manifest)
    receipt_control.complete()
    _resource_log(target, "measurement_complete", {"completed_blocks": len(completed), "classification": report["classification"]}, started)
    return report


def main() -> int:
    arguments = _arguments()
    try:
        result = execute(arguments)
    except (DirectionDiagnosticError, base.DiagnosticError, parent.SemanticTargetRunError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        target = _resolve(_root(), arguments.output_dir)
        failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "training_updates": 0, "reason": f"{type(error).__name__}: {error}", "next_action": "NONE"}
        try:
            output_root = (_root() / "outputs").resolve()
            safe_new_target = target != output_root and output_root in target.parents and not target.exists()
            task_output_in_progress = target.is_dir() and (target / "input_population.json").is_file() and (target / "preflight.json").is_file()
            if safe_new_target:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir(exist_ok=False)
            if (safe_new_target or task_output_in_progress) and not (target / "report.json").exists() and not (target / "failure.json").exists():
                _write_json_new(target / "failure.json", failure, "engineering failure; no scientific classification")
            if task_output_in_progress:
                with (target / "engineering_events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "INVALID_HARNESS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
