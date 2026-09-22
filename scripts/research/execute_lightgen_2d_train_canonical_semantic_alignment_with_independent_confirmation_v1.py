"""Train the clean 2L Generator with canonical Smooth alignment.

This consumer starts from the accepted clean 2L Generator step-0 state.  It
updates only ``generator.phase`` with the frozen Smooth target-image feature
objective, while a distinct accepted FrozenMNISTEvaluator is reserved for
step-0/step-300 confirmation only.  The training oracle, optimizer, data
order, auxiliary losses, and conflict-aware gradient composition are fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen_2d_chen_bogt_complex_ols_training import (
    ADAM_BETAS,
    ADAM_EPSILON,
    AMSGRAD,
    CHECKPOINT_STEPS,
    DataReadLedger,
    DeterministicTrainRoleSampler,
    SCIENTIFIC_UPDATES,
    TRAIN_ROLE,
    WEIGHT_DECAY,
    contracted_plane_powers,
    optical_power_loss_from_powers,
    _rng_state_payload,
)
from ldm_project.v2.application.research.research_run_control import (
    HeartbeatWriter,
    InterruptController,
    RecoveryCheckpointManager,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUNNING,
    RUN_PREPARING,
    capture_execution_source_manifest,
    recovery_due,
)
from ldm_project.v2.evidence import (
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
)

from scripts.research import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research import execute_lightgen_2d_generator_target_image_semantic_alignment_v1 as prior_alignment
from scripts.research import execute_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1 as realization


TASK_ID = "TRAIN_CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_TRAIN_CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "canonical-semantic-alignment-independent-confirmation-v1"
TRAJECTORY_ID = "canonical-semantic-alignment-independent-confirmation-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = prior_alignment.EXPECTED_BRANCH
EXPECTED_HEAD = prior_alignment.EXPECTED_HEAD

PRIOR_RUN = prior_alignment.PRIOR_RUN
PRIOR_REPORT = prior_alignment.PRIOR_REPORT
PRIOR_STEP_0 = prior_alignment.PRIOR_STEP_0
EXPECTED_PRIOR_REPORT_SHA256 = prior_alignment.EXPECTED_PRIOR_REPORT_SHA256
EXPECTED_PRIOR_STEP_0_SHA256 = prior_alignment.EXPECTED_PRIOR_STEP_0_SHA256
TRAINING_SOURCE_SHA256 = successor.TRAINING_SOURCE_SHA256
SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
CURRENT_TEACHER_PATH = producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
CURRENT_TEACHER_SHA256 = conflict.EXPECTED_TRAIN_EVALUATOR_SHA256

INDEPENDENT_RUN = Path(
    "outputs/20260724_v2/runs/"
    "20260724_114819_935841_formal_v2_amplitude_finite_crop_identity"
)
INDEPENDENT_CHECKPOINT = INDEPENDENT_RUN / "frozen_mnist_evaluator.pt"
INDEPENDENT_QUALIFICATION = INDEPENDENT_RUN / "evaluator_qualification_evidence.json"
INDEPENDENT_SPLIT = INDEPENDENT_RUN / "mnist_split_manifest.json"
EXPECTED_INDEPENDENT_CHECKPOINT_SHA256 = "16a42fad849ed9af249fa65bed9567a8fabc011fcc9d107061f69cdb27f21684"
EXPECTED_INDEPENDENT_QUALIFICATION_SHA256 = "e5ab7ea262756bb50badbd810ce6aff165b8b9ca84f0c8e3a04424474c78aa29"
EXPECTED_INDEPENDENT_SPLIT_SHA256 = "3ac59d7c2ccdbc42d46485250afd50bb863a19e78ad0706ec00e22252012dbaa"
INDEPENDENT_VALIDATION_ACCURACY = 0.9893999695777893

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_canonical_semantic_alignment_with_independent_confirmation_v1/"
    "runs/attempt_01"
)

LOGICAL_BATCH_SIZE = successor.LOGICAL_BATCH_SIZE
BOOTSTRAP_REPLICATES = successor.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = successor.BOOTSTRAP_SEED
RECOVERY_INTERVAL_STEPS = successor.RECOVERY_INTERVAL_STEPS
RECONSTRUCTION_TOLERANCE = successor.RECONSTRUCTION_TOLERANCE
FEATURE_EPS = 1.0e-12
EPS_PROJ = 1.0e-12
EXPECTED_SMOOTH_STEP0_MACRO = 0.1030
EXPECTED_CANONICAL_GRADIENT_NORM = 0.0019317994

REPORT_SCHEMA = "canonical_semantic_alignment_independent_confirmation_report.v1"
PREFLIGHT_SCHEMA = "canonical_semantic_alignment_independent_confirmation_preflight.v1"
CHECKPOINT_SCHEMA = "canonical_semantic_alignment_independent_confirmation_checkpoint.v1"
DEV_SCHEMA = "canonical_semantic_alignment_independent_confirmation_dev_diagnostic.v1"
BOOTSTRAP_SCHEMA = "canonical_semantic_alignment_independent_confirmation_paired_bootstrap.v1"

SUPPORTED_CLASSIFICATION = "CANONICAL_SEMANTIC_ALIGNMENT_INDEPENDENTLY_SUPPORTED"
SMOOTH_ONLY_CLASSIFICATION = "SMOOTH_GUIDED_ALIGNMENT_WITHOUT_INDEPENDENT_CONFIRMATION"
LOCAL_FAILURE_CLASSIFICATION = "LOCAL_CANONICAL_DIRECTION_WITH_GLOBAL_TRAJECTORY_FAILURE"
TRADEOFF_CLASSIFICATION = "CANONICAL_SEMANTIC_RECOVERY_RECONSTRUCTION_TRADEOFF"
NOT_SUPPORTED_CLASSIFICATION = "CANONICAL_ALIGNMENT_TRAINING_NOT_SUPPORTED"
BLOCKED_EVALUATOR_CLASSIFICATION = "BLOCKED_INDEPENDENT_CONFIRMATION_EVALUATOR_UNRESOLVED"


ExecutionError = conflict.ConflictAwareExecutionError
_require = conflict._require
_mapping = conflict._mapping
_tensor = conflict._tensor
_finite = conflict._finite
_artifact_ref = conflict._artifact_ref
_write_json = conflict._write_json
_write_torch = conflict._write_torch
_state_digest = conflict._state_digest
_frozen_state = conflict._frozen_state
_generator = conflict._generator
_freeze_generator_only = conflict._freeze_generator_only
_validate_generator_optimizer = conflict._validate_generator_optimizer
_project_auxiliary_gradient = conflict._project_auxiliary_gradient
_gradient_stats = conflict._gradient_stats
_cosine = conflict._cosine
_dot = conflict._dot
_norm = conflict._norm
_rng_snapshot = conflict._rng_snapshot
_restore_rng = conflict._restore_rng
_repo_root = conflict._repo_root
_relative = conflict._relative


def _load_independent_authority(root: Path) -> dict[str, Any]:
    checkpoint = root / INDEPENDENT_CHECKPOINT
    qualification_path = root / INDEPENDENT_QUALIFICATION
    split_path = root / INDEPENDENT_SPLIT
    _require(file_sha256(checkpoint) == EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, "independent evaluator checkpoint identity changed")
    _require(file_sha256(qualification_path) == EXPECTED_INDEPENDENT_QUALIFICATION_SHA256, "independent evaluator qualification identity changed")
    _require(file_sha256(split_path) == EXPECTED_INDEPENDENT_SPLIT_SHA256, "independent evaluator split identity changed")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    _require(isinstance(qualification, Mapping), "independent evaluator qualification is malformed")
    _require(isinstance(split, Mapping), "independent evaluator split is malformed")
    _require(qualification.get("status") == "passed", "independent evaluator qualification did not pass")
    _require(qualification.get("tier") == "acceptance", "independent evaluator tier changed")
    gates = qualification.get("gates")
    _require(isinstance(gates, list) and len(gates) == 1, "independent qualification gates are malformed")
    metrics = _mapping(gates[0], "independent qualification gate")
    metric_values = _mapping(metrics.get("metrics"), "independent qualification metrics")
    _require(math.isclose(float(metric_values.get("validation_accuracy")), INDEPENDENT_VALIDATION_ACCURACY, rel_tol=0.0, abs_tol=1.0e-12), "independent qualification accuracy changed")
    _require(split.get("train_count") == 55000 and split.get("validation_count") == 5000, "independent evaluator split counts changed")
    _require(set(split) == {"dataset_root", "manifest_hash", "schema_version", "seed", "train_count", "validation_count", "train_indices_hash", "validation_indices_hash"}, "independent evaluator split includes an unbound data role")
    payload = safe_torch_load(checkpoint, expected_sha256=EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, map_location="cpu")
    payload_map = _mapping(payload, "independent evaluator checkpoint")
    _require(payload_map.get("architecture_kind") == "frozen_mnist_evaluator_v1", "independent evaluator architecture changed")
    _require(math.isclose(float(payload_map.get("validation_accuracy")), INDEPENDENT_VALIDATION_ACCURACY, rel_tol=0.0, abs_tol=1.0e-12), "independent checkpoint accuracy changed")
    return {
        "checkpoint": _artifact_ref(root, checkpoint, role="frozen independent canonical confirmation evaluator"),
        "qualification": _artifact_ref(root, qualification_path, role="accepted independent evaluator qualification"),
        "split": _artifact_ref(root, split_path, role="independent evaluator TRAIN/validation split"),
        "architecture": "FrozenMNISTEvaluator v1",
        "feature_layer": "embedding output immediately before classifier",
        "feature_dimension": 128,
        "class_order": list(range(10)),
        "validation_accuracy": INDEPENDENT_VALIDATION_ACCURACY,
        "train_count": 55000,
        "validation_count": 5000,
        "data_roles": {"train": "canonical MNIST train rows 0:55000", "validation": "canonical MNIST train rows 55000:60000", "test": 0, "protected": 0, "sealed": 0},
        "selection_rule": "sole eligible pre-existing authority after excluding current training teacher, Smooth training oracle, failed qualification, and excluded Shape candidate; no Generator-response selection",
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    authorities = conflict._load_authorities(root)
    clean_report_path = root / PRIOR_REPORT
    _require(file_sha256(clean_report_path) == EXPECTED_PRIOR_REPORT_SHA256, "accepted clean conflict-aware report identity changed")
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
    _require(clean_report.get("result") == "VALID_COMPLETED", "accepted clean conflict-aware result is not valid completed")
    _require(clean_report.get("outer_gate") == conflict.OUTER_GATE, "accepted clean conflict-aware gate changed")
    _require(clean_report.get("classification") == conflict.NOT_SUPPORTED_CLASSIFICATION, "accepted clean conflict-aware classification changed")
    _require(clean_report.get("training_trajectory", {}).get("updates") == SCIENTIFIC_UPDATES, "accepted clean conflict-aware update count changed")
    clean_step0 = safe_torch_load(root / PRIOR_STEP_0, expected_sha256=EXPECTED_PRIOR_STEP_0_SHA256, map_location="cpu")
    _require(isinstance(clean_step0, Mapping), "accepted clean conflict-aware step-0 checkpoint is malformed")
    _require(clean_step0.get("completed_updates") == 0, "accepted clean conflict-aware step-0 is not step 0")
    _require(clean_step0.get("scientific_artifact") is True and clean_step0.get("operational_artifact") is False, "accepted clean conflict-aware step-0 role changed")
    _require(clean_step0.get("outer_gate") == conflict.OUTER_GATE, "accepted clean conflict-aware step-0 gate changed")
    _require(clean_step0.get("trainable_parameter_names") == ["generator.phase"], "accepted clean conflict-aware trainable boundary changed")
    _finite(_tensor(clean_step0.get("generator_state"), "accepted clean conflict-aware generator"), "accepted clean conflict-aware generator")
    _finite(_tensor(clean_step0.get("encoder_state"), "accepted clean conflict-aware encoder"), "accepted clean conflict-aware encoder")
    authorities["accepted_report"] = clean_report
    authorities["step0"] = dict(clean_step0)
    gradient_composition = _mapping(clean_report.get("gradient_composition"), "accepted gradient composition")
    trajectory = _mapping(gradient_composition.get("trajectory"), "accepted gradient trajectory")
    step0_composition = _mapping(trajectory.get("0"), "accepted step-0 gradient composition")
    gradient_norms = _mapping(step0_composition.get("gradient_norms"), "accepted step-0 gradient norms")
    semantic_norm = _mapping(gradient_norms.get("g_sem"), "accepted semantic gradient norm").get("norm")
    _require(math.isfinite(float(semantic_norm)) and float(semantic_norm) > 0.0, "accepted semantic gradient scale is invalid")
    authorities["historical_gradient_norm"] = float(semantic_norm)
    authorities["authority_hashes"].update({"accepted_clean_conflict_report": EXPECTED_PRIOR_REPORT_SHA256, "accepted_clean_step0": EXPECTED_PRIOR_STEP_0_SHA256})
    authorities["independent"] = _load_independent_authority(root)
    return authorities


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, FrozenMNISTEvaluator, SmoothMNISTEvaluator, FrozenMNISTEvaluator, dict[str, Any]]:
    consumer, current_teacher, smooth, identity = prior_alignment._load_models(root, authorities, device)
    confirmation, payload = load_frozen_evaluator(
        root / INDEPENDENT_CHECKPOINT,
        device=device,
        min_validation_accuracy=0.985,
    )
    _require(payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "confirmation evaluator architecture changed")
    _require(all(not parameter.requires_grad for parameter in confirmation.parameters()), "confirmation evaluator is not frozen")
    _require(file_sha256(root / INDEPENDENT_CHECKPOINT) != CURRENT_TEACHER_SHA256, "confirmation evaluator reuses current teacher checkpoint")
    _require(_state_digest(confirmation.state_dict()) != _state_digest(smooth.state_dict()), "confirmation evaluator reuses Smooth state")
    _require(_state_digest(confirmation.state_dict()) != _state_digest(current_teacher.state_dict()), "confirmation evaluator is not distinct from current teacher")
    identity = dict(identity)
    identity.update({"confirmation": authorities["independent"], "confirmation_in_backward": False, "current_teacher_in_backward": False})
    return consumer, current_teacher, smooth, confirmation, identity


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    _require(value.ndim == 4 and value.shape[1] == 1 and tuple(value.shape[-2:]) in {(28, 28), (200, 200)}, "canonical CMOS view shape changed")
    _finite(value, "canonical CMOS view input")
    view = F.adaptive_avg_pool2d(value.clamp(0.0, 1.0), (28, 28))
    _finite(view, "canonical CMOS view")
    return view


def _normalized_feature_loss(cmos_features: torch.Tensor, target_features: torch.Tensor, name: str) -> torch.Tensor:
    _require(cmos_features.ndim == target_features.ndim == 2 and cmos_features.shape == target_features.shape and cmos_features.shape[1] == 128, f"{name} feature shape changed")
    loss = (F.normalize(cmos_features, dim=1, eps=FEATURE_EPS) - F.normalize(target_features.detach(), dim=1, eps=FEATURE_EPS)).square().sum(dim=1).mean()
    _finite(loss, name)
    return loss


def _canonical_components(consumer: Any, smooth: SmoothMNISTEvaluator, batch: torch.Tensor, references: Any) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "canonical real path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "canonical raw/target geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "canonical raw/target dtype changed")
    _finite(raw, "canonical raw CMOS")
    _finite(target_image, "canonical target image")
    with torch.no_grad():
        target_features, _ = smooth(_cmos_view(batch))
    cmos_features, _ = smooth(_cmos_view(raw))
    l_can = _normalized_feature_loss(cmos_features, target_features, "L_canonical")
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    for name, value in (("L_canonical", l_can), ("L_rec", l_rec), ("L_op", l_op)):
        _finite(value, name)
    return {"result": result, "raw": raw, "target": target_image, "target_features": target_features.detach(), "cmos_features": cmos_features, "L_canonical": l_can, "L_rec": l_rec, "L_op": l_op}


def _compose_gradients(consumer: Any, smooth: SmoothMNISTEvaluator, batch: torch.Tensor, references: Any, *, beta: float, gamma: float, delta_can: float) -> dict[str, Any]:
    generator = _generator(consumer)
    components = _canonical_components(consumer, smooth, batch, references)
    g_rec = torch.autograd.grad(beta * components["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
    g_op = torch.autograd.grad(gamma * components["L_op"], generator, retain_graph=True, allow_unused=False)[0]
    g_can = torch.autograd.grad(delta_can * components["L_canonical"], generator, retain_graph=False, allow_unused=False)[0]
    for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_can", g_can)):
        _finite(value, name)
    _norm(g_can, "g_can")
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec, g_can, "g_rec")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op, g_can, "g_op")
    g_total = g_can + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_can) > 0.0, "composed gradient lacks a positive canonical component")
    return {
        "components": components,
        "gradients": {"g_rec": g_rec, "g_op": g_op, "g_can": g_can, "g_rec_safe": g_rec_safe, "g_op_safe": g_op_safe, "g_total": g_total},
        "gradient_norms": {name: _gradient_stats(value, name) for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_can", g_can), ("g_rec_safe", g_rec_safe), ("g_op_safe", g_op_safe), ("g_total", g_total))},
        "gradient_cosines": {"cos_g_rec_g_can": _cosine(g_rec, g_can), "cos_g_op_g_can": _cosine(g_op, g_can), "cos_g_rec_safe_g_can": _cosine(g_rec_safe, g_can), "cos_g_op_safe_g_can": _cosine(g_op_safe, g_can), "cos_g_total_g_can": _cosine(g_total, g_can)},
        "projection": {"rec": rec_projection, "op": op_projection},
        "smooth_in_backward": True,
        "confirmation_in_backward": False,
        "label_cross_entropy_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    components = _mapping(bundle["components"], "canonical components")
    return {"L_canonical": float(components["L_canonical"].detach().cpu()), "L_rec": float(components["L_rec"].detach().cpu()), "L_op": float(components["L_op"].detach().cpu()), "gradient_norms": bundle["gradient_norms"], "gradient_cosines": bundle["gradient_cosines"], "projection": bundle["projection"], "smooth_in_backward": True, "confirmation_in_backward": False, "label_cross_entropy_in_objective": False, "scalar_weighted_sum_backward": False}


def _diagnose_composition(consumer: Any, smooth: SmoothMNISTEvaluator, batch: torch.Tensor, references: Any, *, beta: float, gamma: float, delta_can: float) -> dict[str, Any]:
    before = _state_digest(consumer.runtime.state_dict())
    bundle = _compose_gradients(consumer, smooth, batch, references, beta=beta, gamma=gamma, delta_can=delta_can)
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "diagnostic composition populated runtime gradients")
    successor._clear_grads(consumer)
    _require(before == _state_digest(consumer.runtime.state_dict()), "diagnostic composition changed runtime state")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "Smooth received a gradient")
    return _composition_record(bundle)


def _apply_update(consumer: Any, smooth: SmoothMNISTEvaluator, optimizer: torch.optim.Adam, batch: torch.Tensor, references: Any, *, beta: float, gamma: float, delta_can: float, step: int, ledger: DataReadLedger | None) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    successor._clear_grads(consumer)
    bundle = _compose_gradients(consumer, smooth, batch, references, beta=beta, gamma=gamma, delta_can=delta_can)
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "runtime gradient populated before assignment")
    generator.grad = bundle["gradients"]["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    _require(_state_digest(before_frozen) == _state_digest(_frozen_state(consumer)), "frozen runtime state changed")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "Smooth received an optimizer gradient")
    update_norm = float((generator.detach() - before_generator).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "generator.phase did not update")
    successor._clear_grads(consumer)
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    record = _composition_record(bundle)
    record.update({"step": step, "generator_update_norm": update_norm, "encoder_unchanged": True, "frozen_runtime_unchanged": True, "trainable_parameter_names": ["generator.phase"]})
    return record


def _calibrate_delta_can(consumer: Any, smooth: SmoothMNISTEvaluator, batch: torch.Tensor, references: Any, target_norm: float) -> dict[str, Any]:
    before = _state_digest(consumer.runtime.state_dict())
    components = _canonical_components(consumer, smooth, batch, references)
    raw_gradient = torch.autograd.grad(components["L_canonical"], _generator(consumer), retain_graph=False, allow_unused=False)[0]
    raw_norm = _norm(raw_gradient, "unweighted canonical gradient")
    _require(math.isfinite(target_norm) and target_norm > 0.0, "historical canonical gradient scale is invalid")
    delta_can = target_norm / raw_norm
    weighted_norm = _norm(delta_can * raw_gradient, "calibrated canonical gradient")
    _require(math.isclose(weighted_norm, target_norm, rel_tol=2.0e-5, abs_tol=1.0e-12), "canonical gradient calibration missed fixed target scale")
    successor._clear_grads(consumer)
    _require(before == _state_digest(consumer.runtime.state_dict()), "canonical calibration changed runtime state")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "canonical calibration populated Smooth gradients")
    return {"method": "one TRAIN-only step-0 calibration batch", "target_gradient_norm": target_norm, "unweighted_gradient_norm": raw_norm, "delta_can": delta_can, "weighted_gradient_norm": weighted_norm, "sweep": False, "tuning": False}


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    flat = raw.detach().to(torch.float64).flatten(1)
    std = flat.std(dim=1, unbiased=False)
    return {"finite": bool(torch.isfinite(raw).all().item()), "nonconstant_fraction": float((std > 0.0).to(torch.float64).mean().cpu()), "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item())}


def _class_metrics(correct: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(correct.shape == labels.shape == (5000,), "DEV correctness shape changed")
    per_class = []
    for digit in range(10):
        mask = labels == digit
        _require(int(mask.sum()) == 500, f"DEV class count changed for {digit}")
        per_class.append(float(correct[mask].to(torch.float64).mean()))
    return {"macro_recall": float(sum(per_class) / 10.0), "accuracy": float(correct.to(torch.float64).mean()), "per_class_recall": per_class, "rows": 5000}


@torch.no_grad()
def _evaluate_dev(
    consumer: Any,
    smooth: SmoothMNISTEvaluator,
    confirmation: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: Any,
    ledger: DataReadLedger,
    *,
    step: int,
    include_confirmation: bool,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    smooth.eval()
    confirmation.eval()
    labels_dev = labels.index_select(0, development_indices)
    smooth_correct: list[torch.Tensor] = []
    confirmation_correct: list[torch.Tensor] = []
    smooth_features: list[torch.Tensor] = []
    smooth_target_features: list[torch.Tensor] = []
    smooth_logits_parts: list[torch.Tensor] = []
    canonical_total = 0.0
    reconstruction_total = 0.0
    optical_total = 0.0
    raw_parts: list[torch.Tensor] = []
    try:
        for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
            indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
            batch = images.index_select(0, indices).to(consumer.device)
            batch_labels = labels.index_select(0, indices)
            result = consumer.runtime.forward_from_image(batch)
            _require(result.image_input is not None and result.ols_input_field is not None, "DEV real path is incomplete")
            raw = result.raw_intensity
            target = result.image_input.slm_intensity
            _finite(raw, "DEV raw CMOS")
            _finite(target, "DEV target image")
            raw_parts.append(raw.cpu())
            smooth_target_batch, _ = smooth(_cmos_view(batch))
            smooth_batch, smooth_logits = smooth(_cmos_view(raw))
            _finite(smooth_target_batch, "DEV Smooth target features")
            _finite(smooth_batch, "DEV Smooth CMOS features")
            _finite(smooth_logits, "DEV Smooth logits")
            canonical_total += float((F.normalize(smooth_batch, dim=1, eps=FEATURE_EPS) - F.normalize(smooth_target_batch, dim=1, eps=FEATURE_EPS)).square().sum(dim=1).sum().cpu())
            reconstruction_total += float((raw - target).square().sum().cpu())
            powers = contracted_plane_powers(result)
            ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
            optical_total += float(torch.log(ratios).square().sum().cpu())
            smooth_cpu = smooth_logits.cpu()
            smooth_logits_parts.append(smooth_cpu)
            smooth_correct.append((smooth_cpu.argmax(dim=1) == batch_labels).contiguous())
            smooth_features.append(smooth_batch.cpu())
            smooth_target_features.append(smooth_target_batch.cpu())
            if include_confirmation:
                _, confirmation_logits = confirmation(_cmos_view(raw))
                _finite(confirmation_logits, "DEV confirmation logits")
                confirmation_correct.append((confirmation_logits.cpu().argmax(dim=1) == batch_labels).contiguous())
            ledger.record("dev", int(indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    raw_all = torch.cat(raw_parts, dim=0)
    smooth_correct_all = torch.cat(smooth_correct)
    smooth_features_all = torch.cat(smooth_features)
    smooth_target_all = torch.cat(smooth_target_features)
    smooth_logits_all = torch.cat(smooth_logits_parts)
    raw_stats = _raw_validity(raw_all)
    _require(raw_stats["finite"] and raw_stats["nonconstant_fraction"] == 1.0 and raw_stats["zero_image_count"] == 0, f"DEV raw validity failed at step {step}")
    result: dict[str, Any] = {
        "step": step,
        "rows": 5000,
        "role": "DEV diagnostic; labels used only for fixed descriptive metrics and confirmation endpoint",
        "smooth": {"architecture": "SmoothMNISTEvaluator v2.1", **_class_metrics(smooth_correct_all, labels_dev), "feature_layer": "embedding output immediately before classifier"},
        "canonical_loss": canonical_total / 5000.0,
        "reconstruction": {"L_rec": reconstruction_total / float(raw_all.numel())},
        "optical": {"L_op": optical_total / float(5000 * 5)},
        "raw_cmos": raw_stats,
        "smooth_features": smooth_features_all,
        "smooth_target_features": smooth_target_all,
        "smooth_logits": smooth_logits_all,
        "_smooth_correct": smooth_correct_all,
        "_smooth_target_correct": None,
        "evaluator_parameter_gradients_none": bool(all(parameter.grad is None for parameter in smooth.parameters()) and all(parameter.grad is None for parameter in confirmation.parameters())),
    }
    if include_confirmation:
        confirmation_correct_all = torch.cat(confirmation_correct)
        result["confirmation"] = {"architecture": "FrozenMNISTEvaluator v1", **_class_metrics(confirmation_correct_all, labels_dev), "checkpoint_sha256": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256}
        result["_confirmation_correct"] = confirmation_correct_all
    _require(result["evaluator_parameter_gradients_none"], "DEV evaluator received a gradient")
    return result


def _public_alignment(value: Mapping[str, Any]) -> dict[str, Any]:
    return prior_alignment._public_alignment(value)


def _smooth_realization(real_features: torch.Tensor, record: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    alignment = realization._feature_alignment(real_features, _tensor(record["smooth_features"], "Smooth CMOS features"), labels)
    return {"feature_alignment": _public_alignment(alignment), "smooth_metrics": dict(record["smooth"]), "paired_feature_distance": alignment["paired_target_cmos_distance"], "centroid_matched_nearest_fraction": float(alignment["centroid_matched_nearest_fraction"])}


def _restore_authoritative_optimizer(consumer: Any, step0: Mapping[str, Any], prior_report: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    return prior_alignment._restore_authoritative_optimizer(consumer, step0, prior_report)


def _run_preflight(
    consumer: Any,
    current_teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    confirmation: FrozenMNISTEvaluator,
    authorities: Mapping[str, Any],
    *,
    beta: float,
    gamma: float,
) -> dict[str, Any]:
    initial_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    initial_rng = _rng_snapshot()
    smooth_initial = _state_digest(smooth.state_dict())
    confirmation_initial = _state_digest(confirmation.state_dict())
    teacher_initial = _state_digest(current_teacher.state_dict())
    bad_optimizer = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        _validate_generator_optimizer(consumer, bad_optimizer)
    except (ExecutionError, successor.RecoveryExecutionError):
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise ExecutionError("negative control accepted a full-runtime optimizer")
    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(consumer.device)
    calibration = _calibrate_delta_can(consumer, smooth, fixed_batch, authorities["training"]["references"], authorities["historical_gradient_norm"])
    delta_can = float(calibration["delta_can"])
    composition0 = _diagnose_composition(consumer, smooth, fixed_batch, authorities["training"]["references"], beta=beta, gamma=gamma, delta_can=delta_can)
    optimizer, optimizer_mode = _restore_authoritative_optimizer(consumer, authorities["step0"], authorities["accepted_report"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], authorities["labels"])
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == authorities["calibration_indices"].tolist(), "preflight first TRAIN batch differs from authority")
    _require(torch.equal(authorities["labels"].index_select(0, first_indices), first_labels), "preflight TRAIN labels changed")
    sampler.load_state_dict(sampler_state)
    trace = []
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), "preflight TRAIN sampler binding changed")
        batch = authorities["images"].index_select(0, indices).to(consumer.device)
        trace.append(_apply_update(consumer, smooth, optimizer, batch, authorities["training"]["references"], beta=beta, gamma=gamma, delta_can=delta_can, step=step + 1, ledger=None))
    post_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    _require(torch.equal(initial_state["encoder.phase"], post_state["encoder.phase"]), "preflight changed frozen encoder.phase")
    _require(not torch.equal(initial_state["generator.phase"], post_state["generator.phase"]), "preflight Generator did not depart")
    _require(smooth_initial == _state_digest(smooth.state_dict()), "preflight changed Smooth state")
    _require(confirmation_initial == _state_digest(confirmation.state_dict()), "preflight changed confirmation state")
    _require(teacher_initial == _state_digest(current_teacher.state_dict()), "preflight changed secondary teacher state")
    for projection_name in ("rec", "op"):
        projection = _mapping(composition0["projection"][projection_name], f"preflight {projection_name} projection")
        _require(math.isfinite(float(projection["projection_fraction_norm"])) and math.isfinite(float(projection["removed_norm"])), f"preflight {projection_name} projection is not finite")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    _freeze_generator_only(consumer)
    successor._clear_grads(consumer)
    _restore_rng(initial_rng)
    raw_check = _raw_validity(consumer.runtime.forward_from_image(fixed_batch).raw_intensity.detach())
    _require(raw_check["finite"] and raw_check["nonconstant_fraction"] == 1.0 and raw_check["zero_image_count"] == 0, "step-0 optical validity failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "starting_authority": {"path": PRIOR_STEP_0.as_posix(), "sha256": EXPECTED_PRIOR_STEP_0_SHA256, "selected_step": 0},
        "independent_confirmation_resolved": True,
        "independent_confirmation_in_backward": False,
        "smooth_training_oracle_frozen": True,
        "delta_can_calibration": calibration,
        "gradient_rule": "g_total=g_can+project_nonconflicting(g_rec,g_can)+project_nonconflicting(g_op,g_can)",
        "eps_proj": EPS_PROJ,
        "step0_gradient_composition": composition0,
        "optimizer_mode": optimizer_mode,
        "three_step_real_update": {"updates": 3, "trace": trace, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "frozen_parameter_probe": {"encoder_phase_unchanged": True, "ols_unchanged": True, "w_ols_unchanged": True, "smooth_unchanged": True, "confirmation_unchanged": True, "secondary_teacher_unchanged": True},
        "no_dev_in_updates": True,
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "step0_optical_validity": raw_check,
        "scientific_300_step_executed": False,
    }


def _paired_bootstrap(labels: torch.Tensor, step0: torch.Tensor, step300: torch.Tensor) -> dict[str, Any]:
    result = successor._paired_bootstrap(labels, step0, step300)
    result["schema"] = BOOTSTRAP_SCHEMA
    return result


def _checkpoint_payload(consumer: Any, optimizer: torch.optim.Adam, sampler: DeterministicTrainRoleSampler, history: list[Mapping[str, Any]], composition_trajectory: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, optimizer_mode: str, preflight: Mapping[str, Any], delta_can: float) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "completed_updates": step,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "source_manifest_id": source_manifest.get("source_manifest_id"),
        "accepted_starting_checkpoint": {"path": PRIOR_STEP_0.as_posix(), "sha256": EXPECTED_PRIOR_STEP_0_SHA256, "selected_step": 0},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "SmoothMNISTEvaluator", "independent_confirmation_evaluator", "current_teacher_secondary"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_state(consumer)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": successor._cpu_state(sampler.state_dict()),
        "rng_state": _rng_state_payload(),
        "canonical_objective": {"delta_can": delta_can, "feature_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier", "normalization_epsilon": FEATURE_EPS},
        "gradient_composition": {"eps_proj": EPS_PROJ, "rule": "canonical-primary; remove only negative first-order components from reconstruction and optical-power gradients", "preflight": dict(preflight), "trajectory": dict(composition_trajectory)},
        "training_history": [dict(record) for record in history],
    }


def _recovery_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _checkpoint_payload(*args, **kwargs)
    payload["scientific_artifact"] = False
    payload["operational_artifact"] = True
    payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
    return payload


def _classification(smooth_summary: Mapping[str, Any], confirmation: Mapping[str, Any], reconstruction: Mapping[str, Any], optical: Mapping[str, Any], transfer: Mapping[str, Any]) -> tuple[str, dict[str, bool]]:
    canonical_improved = bool(smooth_summary["canonical_loss_delta"] < 0.0)
    smooth_moved = bool(smooth_summary["paired_feature_distance_delta"] < 0.0 or smooth_summary["centroid_matched_nearest_fraction_delta"] > 0.0)
    independent_supported = bool(confirmation["delta_macro_recall"] > 0.0 and confirmation["paired_bootstrap_ci"]["lower"] > 0.0)
    reconstruction_pass = bool(reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    optical_pass = bool(optical["all_finite"] and optical["all_nonconstant"] and optical["all_zero_image_counts_zero"])
    raw_valid = bool(transfer["successor_raw_validity"]["finite"] and transfer["successor_raw_validity"]["nonconstant_fraction"] == 1.0 and transfer["successor_raw_validity"]["zero_image_count"] == 0)
    checks = {"canonical_loss_improved": canonical_improved, "smooth_development_moved": smooth_moved, "independent_confirmation_supported": independent_supported, "reconstruction_pass": reconstruction_pass, "optical_validity_pass": optical_pass, "p2q2_raw_valid": raw_valid}
    if independent_supported and not reconstruction_pass:
        return TRADEOFF_CLASSIFICATION, checks
    if canonical_improved and smooth_moved and independent_supported and reconstruction_pass and optical_pass and raw_valid:
        return SUPPORTED_CLASSIFICATION, checks
    if canonical_improved and smooth_moved and not independent_supported:
        return SMOOTH_ONLY_CLASSIFICATION, checks
    if not canonical_improved or not smooth_moved:
        return LOCAL_FAILURE_CLASSIFICATION, checks
    return NOT_SUPPORTED_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any], weights: Mapping[str, float], delta_can: float) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"updates": SCIENTIFIC_UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "objective": "L_canonical=Smooth target-image feature alignment + auxiliary L_rec/L_op with conflict projection", "beta": weights["beta"], "gamma": weights["gamma"], "delta_can": delta_can, "data_source": dict(data_identity), "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256, "confirmation_checkpoint_sha256": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256},
        model_identity="LIGHTGEN_2D_CLEAN_2L_GENERATOR_CANONICAL_SEMANTIC_ALIGNMENT_WITH_INDEPENDENT_CONFIRMATION_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "prior_step0_optimizer_state": EXPECTED_PRIOR_STEP_0_SHA256},
        sampler_identity={"seed": 1234, "steps": SCIENTIFIC_UPDATES, "batch_size": LOGICAL_BATCH_SIZE, "class_balanced": True, "parent": "accepted clean conflict-aware step-0 sampler cursor reset to 0"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def run_successor(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "successor output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"successor output already exists: {target}")
    start_git = successor.git_identity(root)
    _require(start_git["branch"] == EXPECTED_BRANCH and start_git["head"] == EXPECTED_HEAD, "local branch/HEAD conflicts with accepted authorities")
    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal successor requires CUDA")
    device = torch.device("cuda:0")
    consumer, current_teacher, smooth, confirmation, evaluator_identity = _load_models(root, authorities, device)
    _freeze_generator_only(consumer)
    consumer.runtime.train()
    current_teacher.eval()
    smooth.eval()
    confirmation.eval()
    preflight = _run_preflight(consumer, current_teacher, smooth, confirmation, authorities, beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"])
    delta_can = float(preflight["delta_can_calibration"]["delta_can"])
    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(Path(__file__).resolve(), Path(prior_alignment.__file__).resolve(), Path(conflict.__file__).resolve(), Path(successor.__file__).resolve(), Path(realization.__file__).resolve(), root / "src/ldm_project/v2/application/research/lightgen_2d_chen_bogt_complex_ols_training.py", root / "src/ldm_project/v2/application/research/research_run_control.py", root / "src/ldm_project/v2/student/evaluator.py"),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "updates": SCIENTIFIC_UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "prior_hashes": authorities["authority_hashes"], "confirmation_checkpoint_sha256": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, authorities["data_identity"], authorities["weights"], delta_can)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(target / "preflight.json", {**preflight, "independent_confirmation": authorities["independent"], "data_firewall": {"protected": False, "test": False, "sealed": False}})
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(SCIENTIFIC_ATTEMPT_ID)
    _write_json(target / "execution_manifest.json", {"schema": "canonical_semantic_alignment_independent_confirmation_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization_id": AUTHORIZATION_ID, "authorization_token": AUTHORIZATION_TOKEN, "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", "CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_STEP0", "CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED", "SMOOTH_MNIST_EVALUATOR_V2_1", "INDEPENDENT_FROZEN_MNIST_EVALUATOR_20260724_114819"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "data_identity": authorities["data_identity"], "git_before": start_git, "cuda_runtime": cuda_runtime, "scope": {"bounded_updates": SCIENTIFIC_UPDATES, "trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "SmoothMNISTEvaluator", "independent confirmation evaluator"], "protected": False, "test": False, "sealed": False}})

    optimizer, optimizer_mode = _restore_authoritative_optimizer(consumer, authorities["step0"], authorities["accepted_report"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], authorities["labels"])
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == authorities["calibration_indices"].tolist(), "scientific first TRAIN batch differs from authority")
    _require(torch.equal(authorities["labels"].index_select(0, first_indices), first_labels), "scientific first TRAIN labels changed")
    sampler.load_state_dict(sampler_state)
    ledger = DataReadLedger()
    history: list[dict[str, Any]] = []
    dev_records: dict[str, dict[str, Any]] = {}
    smooth_records: dict[str, dict[str, Any]] = {}
    composition_trajectory: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)
    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(device)
    composition_trajectory["0"] = _diagnose_composition(consumer, smooth, fixed_batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_can=delta_can)
    dev0 = _evaluate_dev(consumer, smooth, confirmation, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=0, include_confirmation=True)
    _require(abs(float(dev0["smooth"]["macro_recall"]) - EXPECTED_SMOOTH_STEP0_MACRO) <= 5.0e-4, "step-0 Smooth baseline diverges from accepted 0.1030")
    dev_records["0"] = dev0
    smooth_records["0"] = _smooth_realization(dev0["smooth_target_features"], dev0, authorities["labels"].index_select(0, authorities["development_indices"]))
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev0.items() if not key.startswith("_") and not isinstance(value, torch.Tensor)}})
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=0, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_can=delta_can), kind="canonical semantic alignment scientific checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_can=delta_can)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "canonical Smooth target-image alignment primary", "delta_can": delta_can}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            trace = _apply_update(consumer, smooth, optimizer, batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_can=delta_can, step=step + 1, ledger=ledger)
            history.append(trace)
            now = time.time()
            if recovery_due(step=step + 1, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                last_recovery_path = recovery.save(recovery_factory(step + 1), step=step + 1)
                last_recovery_step = step + 1
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_canonical")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            interrupted = interrupt.handle_safe_boundary(step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_canonical")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": step + 1, "recovery_checkpoint": interrupted.name}
            completed = step + 1
            if completed in CHECKPOINT_STEPS[1:]:
                composition_trajectory[str(completed)] = _diagnose_composition(consumer, smooth, fixed_batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_can=delta_can)
                diagnostic = _evaluate_dev(consumer, smooth, confirmation, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed, include_confirmation=(completed == 300))
                dev_records[str(completed)] = diagnostic
                smooth_records[str(completed)] = _smooth_realization(dev0["smooth_target_features"], diagnostic, authorities["labels"].index_select(0, authorities["development_indices"]))
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in diagnostic.items() if not key.startswith("_") and not isinstance(value, torch.Tensor)}})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_can=delta_can), kind="canonical semantic alignment scientific checkpoint")

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=SCIENTIFIC_UPDATES, latest_losses={"completed_updates": SCIENTIFIC_UPDATES}, force=True)
    labels_dev = authorities["labels"].index_select(0, authorities["development_indices"])
    correctness_payload = {"schema": "canonical_semantic_alignment_independent_confirmation_correctness.v1", "development_indices_sha256": tensor_sha256(authorities["development_indices"]), "labels": labels_dev, "smooth_steps": {step: {"correct": dev_records[step]["_smooth_correct"]} for step in dev_records}, "confirmation_steps": {"0": {"correct": dev_records["0"]["_confirmation_correct"]}, "300": {"correct": dev_records["300"]["_confirmation_correct"]}}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="canonical alignment correctness vectors")
    bootstrap = _paired_bootstrap(labels_dev, correctness_payload["confirmation_steps"]["0"]["correct"], correctness_payload["confirmation_steps"]["300"]["correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="independent confirmation paired bootstrap")
    bootstrap_report = {key: value for key, value in bootstrap.items() if key not in {"macro_delta_tensor", "draws"}}
    independent0 = dev_records["0"]["confirmation"]
    independent300 = dev_records["300"]["confirmation"]
    confirmation_summary = {"evaluator": authorities["independent"], "step_0": independent0, "step_300": independent300, "delta_macro_recall": float(independent300["macro_recall"] - independent0["macro_recall"]), "per_class_delta": list(bootstrap_report["per_class_delta"]), "paired_bootstrap_ci": dict(bootstrap_report["ci95"]), "paired_bootstrap_artifact": _artifact_ref(root, target / "paired_bootstrap.pt", role="independent confirmation paired bootstrap"), "endpoint_steps": [0, 300], "step_100_200_confirmation_inspected": False}
    smooth_summary = {"evaluator": evaluator_identity["smooth"], "step_0_macro_recall": float(dev_records["0"]["smooth"]["macro_recall"]), "step_300_macro_recall": float(dev_records["300"]["smooth"]["macro_recall"]), "delta_macro_recall": float(dev_records["300"]["smooth"]["macro_recall"] - dev_records["0"]["smooth"]["macro_recall"]), "per_class_recall": {step: list(dev_records[step]["smooth"]["per_class_recall"]) for step in dev_records}, "canonical_loss_step_0": float(dev_records["0"]["canonical_loss"]), "canonical_loss_step_300": float(dev_records["300"]["canonical_loss"]), "canonical_loss_delta": float(dev_records["300"]["canonical_loss"] - dev_records["0"]["canonical_loss"]), "paired_feature_distance_step_0": float(smooth_records["0"]["feature_alignment"]["paired_target_cmos_distance"]["mean"]), "paired_feature_distance_step_300": float(smooth_records["300"]["feature_alignment"]["paired_target_cmos_distance"]["mean"]), "paired_feature_distance_delta": float(smooth_records["300"]["feature_alignment"]["paired_target_cmos_distance"]["mean"] - smooth_records["0"]["feature_alignment"]["paired_target_cmos_distance"]["mean"]), "centroid_matched_nearest_fraction_step_0": float(smooth_records["0"]["centroid_matched_nearest_fraction"]), "centroid_matched_nearest_fraction_step_300": float(smooth_records["300"]["centroid_matched_nearest_fraction"]), "centroid_matched_nearest_fraction_delta": float(smooth_records["300"]["centroid_matched_nearest_fraction"] - smooth_records["0"]["centroid_matched_nearest_fraction"]), "trajectory": {step: smooth_records[step] for step in smooth_records}, "role": "DEVELOPMENT_DIAGNOSTIC; not the scientific confirmation endpoint"}
    reconstruction = {"step_0": float(dev_records["0"]["reconstruction"]["L_rec"]), "step_100": float(dev_records["100"]["reconstruction"]["L_rec"]), "step_200": float(dev_records["200"]["reconstruction"]["L_rec"]), "step_300": float(dev_records["300"]["reconstruction"]["L_rec"]), "delta_step_300_minus_step_0": float(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"]), "tolerance": RECONSTRUCTION_TOLERANCE, "trajectory_pass": bool(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"] <= RECONSTRUCTION_TOLERANCE)}
    optical = {"checkpoints": {step: {"finite": bool(dev_records[step]["raw_cmos"]["finite"]), "nonconstant_fraction": float(dev_records[step]["raw_cmos"]["nonconstant_fraction"]), "zero_image_count": int(dev_records[step]["raw_cmos"]["zero_image_count"]), "L_op": float(dev_records[step]["optical"]["L_op"])} for step in dev_records}, "all_finite": all(bool(dev_records[step]["raw_cmos"]["finite"]) for step in dev_records), "all_nonconstant": all(float(dev_records[step]["raw_cmos"]["nonconstant_fraction"]) == 1.0 for step in dev_records), "all_zero_image_counts_zero": all(int(dev_records[step]["raw_cmos"]["zero_image_count"]) == 0 for step in dev_records)}
    _require(optical["all_finite"] and optical["all_nonconstant"] and optical["all_zero_image_counts_zero"], "optical validity failed")
    transfer = successor._p2q2_transfer(root, consumer, {"phase_latent": authorities["phase_latent"], "selected_generated_raw": authorities["selected_generated_raw"], "real_control_raw": authorities["real_control_raw"], "selected_metrics": authorities["selected_metrics"], "authority_hashes": {"phase_latent": authorities["authority_hashes"]["phase_latent"]}}, root / CURRENT_TEACHER_PATH, target)
    classification, checks = _classification(smooth_summary, confirmation_summary, reconstruction, optical, transfer)
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)
    report = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "result": "VALID_COMPLETED", "status": "VALID_COMPLETED", "outer_gate": OUTER_GATE, "outer_gate_passed": classification == SUPPORTED_CLASSIFICATION, "classification": classification, "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True}, "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)}, "starting_authority": {"accepted_clean_conflict_report": _artifact_ref(root, root / PRIOR_REPORT, role="accepted clean conflict-aware Generator-only report"), "accepted_clean_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="accepted clean 2L Generator step-0 state")}, "independent_confirmation": confirmation_summary, "training_oracle": {"identity": evaluator_identity["smooth"], "role": "frozen development/training oracle; no confirmation claim", "objective": "mean(||normalize(F_smooth(CMOS))-normalize(F_smooth(target))||_2^2)", "target_stop_gradient": True}, "secondary_teacher": {"checkpoint": _artifact_ref(root, root / CURRENT_TEACHER_PATH, role="frozen secondary P2/Q2 transfer evaluator"), "checkpoint_sha256": CURRENT_TEACHER_SHA256, "used_in_backward": False}, "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "SmoothMNISTEvaluator", "independent confirmation evaluator"]}, "loss_weights": {"beta": authorities["weights"]["beta"], "gamma": authorities["weights"]["gamma"], "delta_can": delta_can, "delta_can_calibration": preflight["delta_can_calibration"], "weight_sweep": False}, "gradient_composition": {"formula": "g_total=g_can+g_rec_safe+g_op_safe", "g_can": "grad_generator(delta_can*L_canonical)", "g_rec": "grad_generator(beta*L_rec)", "g_op": "grad_generator(gamma*L_op)", "projection": "project only auxiliary components with negative dot product against g_can", "eps_proj": EPS_PROJ, "preflight": preflight, "trajectory": composition_trajectory}, "preflight": preflight, "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "sampler_cursor": int(sampler.cursor)}, "smooth_development": smooth_summary, "reconstruction": reconstruction, "optical_validity": optical, "p2_q2_transfer": transfer, "counterfactual_integrity": {"same_start_step0": True, "same_data_order": True, "same_generator_architecture": True, "same_encoder_ols": True, "same_auxiliary_losses": True, "same_beta_gamma": True, "same_optimizer": True, "same_update_budget": True, "smooth_in_objective": True, "independent_confirmation_excluded_from_backward": True, "protected_test_sealed": False}, "outer_gate_checks": checks, "input_authorities": {**authorities["authority_hashes"], "independent_confirmation_checkpoint": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, "independent_confirmation_qualification": EXPECTED_INDEPENDENT_QUALIFICATION_SHA256, "independent_confirmation_split": EXPECTED_INDEPENDENT_SPLIT_SHA256}, "data_reads": {**ledger.to_dict(), "preflight_train_rows": 30}, "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0}, "claim_boundary": {"scientific_training_executed": True, "updates": SCIENTIFIC_UPDATES, "direct_canonical_alignment_transfer": classification == SUPPORTED_CLASSIFICATION, "global_architecture_inadequacy": False, "formal_promotion": False, "physical_hardware_validity": False, "architecture_change": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False}, "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1.py"]}, "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "checkpoints": [0, 100, 200, 300], "smooth_diagnostics": [0, 100, 200, 300], "independent_confirmation": [0, 300], "paired_bootstrap": True, "frozen_gradient_firewall": True, "no_protected_test_sealed": True}, "unresolved": [] if classification == SUPPORTED_CLASSIFICATION else ["The bounded run did not satisfy every independent canonical semantic alignment condition under the frozen clean 2L architecture."], "next_action": "NONE", "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "output_dir": _relative(root, target)}
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the bounded CUDA arm")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "updates": SCIENTIFIC_UPDATES, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_successor(output_dir=args.output_dir)
    except (ExecutionError, successor.RecoveryExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
