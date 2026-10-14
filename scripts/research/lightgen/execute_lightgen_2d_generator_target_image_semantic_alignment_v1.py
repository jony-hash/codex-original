"""Train the frozen clean 2L Generator toward paired target-image semantics.

The consumer keeps the accepted Encoder/OLS manifold, clean 2L Generator
parameterization, auxiliary reconstruction/optical losses, optimizer family,
data order, and independent Smooth DEV evaluator fixed.  Its only scientific
intervention is a TRAIN-only target-image feature-alignment loss from the
accepted FrozenMNISTEvaluator, with the alignment gradient protected from
locally conflicting auxiliary components.
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

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import ADAM_BETAS, ADAM_EPSILON, AMSGRAD, CHECKPOINT_STEPS, DataReadLedger, DeterministicTrainRoleSampler, SCIENTIFIC_UPDATES, TRAIN_ROLE, WEIGHT_DECAY, contracted_plane_powers, optical_power_loss_from_powers, _rng_state_payload
from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research.lightgen import execute_lightgen_2d_resolve_cmos_to_smooth_semantic_realization_v1 as realization


TASK_ID = "TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "generator-target-image-semantic-alignment-v1"
TRAJECTORY_ID = "generator-target-image-semantic-alignment-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

PRIOR_RUN = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01")
PRIOR_REPORT = PRIOR_RUN / "report.json"
PRIOR_STEP_0 = PRIOR_RUN / "scientific_step_000000.pt"
EXPECTED_PRIOR_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
EXPECTED_PRIOR_STEP_0_SHA256 = "a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34"

SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
TRAINING_SOURCE_SHA256 = successor.TRAINING_SOURCE_SHA256
EXPECTED_TRAIN_EVALUATOR_SHA256 = conflict.EXPECTED_TRAIN_EVALUATOR_SHA256
PHASE_REPORT = conflict.PHASE_REPORT
PHASE_SELECTION = conflict.PHASE_SELECTION
PHASE_LATENT = conflict.PHASE_LATENT
REAL_CONTROL = conflict.REAL_CONTROL
EXPECTED_PHASE_REPORT_SHA256 = conflict.EXPECTED_PHASE_REPORT_SHA256
EXPECTED_PHASE_SELECTION_SHA256 = conflict.EXPECTED_PHASE_SELECTION_SHA256
EXPECTED_PHASE_LATENT_SHA256 = conflict.EXPECTED_PHASE_LATENT_SHA256
EXPECTED_REAL_CONTROL_SHA256 = conflict.EXPECTED_REAL_CONTROL_SHA256

DEFAULT_OUTPUT_DIR = Path("outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01")
LOGICAL_BATCH_SIZE = successor.LOGICAL_BATCH_SIZE
BOOTSTRAP_REPLICATES = successor.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = successor.BOOTSTRAP_SEED
RECONSTRUCTION_TOLERANCE = successor.RECONSTRUCTION_TOLERANCE
RECOVERY_INTERVAL_STEPS = successor.RECOVERY_INTERVAL_STEPS
EPS_PROJ = 1.0e-12
FEATURE_EPS = 1.0e-12
EXPECTED_HISTORICAL_SEMANTIC_GRADIENT_NORM = 0.0019319

REPORT_SCHEMA = "generator_target_image_semantic_alignment_report.v1"
PREFLIGHT_SCHEMA = "generator_target_image_semantic_alignment_preflight.v1"
CHECKPOINT_SCHEMA = "generator_target_image_semantic_alignment_checkpoint.v1"
DEV_SCHEMA = "generator_target_image_semantic_alignment_dev_diagnostic.v1"
BOOTSTRAP_SCHEMA = "generator_target_image_semantic_alignment_paired_bootstrap.v1"
REALIZATION_SCHEMA = "generator_target_image_semantic_realization_diagnostic.v1"

SUPPORTED_CLASSIFICATION = "TARGET_IMAGE_SEMANTIC_ALIGNMENT_SUPPORTED"
TEACHER_ONLY_CLASSIFICATION = "TARGET_FEATURE_ALIGNMENT_WITHOUT_INDEPENDENT_SEMANTIC_TRANSFER"
TRADEOFF_CLASSIFICATION = "SEMANTIC_REALIZATION_RECONSTRUCTION_TRADEOFF"
NOT_SUPPORTED_CLASSIFICATION = "TARGET_IMAGE_SEMANTIC_ALIGNMENT_NOT_SUPPORTED"


ExecutionError = conflict.ConflictAwareExecutionError

_require = conflict._require
_mapping = conflict._mapping
_tensor = conflict._tensor
_finite = conflict._finite
_repo_root = conflict._repo_root
_relative = conflict._relative
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


def _restore_authoritative_optimizer(consumer: Any, step0: Mapping[str, Any], prior_report: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    """Restore the exact Generator moments recorded by the accepted clean step-0."""

    optimizer = successor._new_generator_optimizer(consumer)
    config = _mapping(step0.get("optimizer_configuration"), "step-0 optimizer configuration")
    expected_config = {
        "name": "Adam",
        "parameter_names": ["generator.phase"],
        "learning_rate": 1.0e-3,
        "betas": list(ADAM_BETAS),
        "eps": ADAM_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": AMSGRAD,
    }
    for name, value in expected_config.items():
        _require(config.get(name) == value, f"accepted clean step-0 optimizer configuration changed: {name}")
    _require(step0.get("optimizer_mode") == "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED", "accepted clean step-0 optimizer mode changed")
    old_state = _mapping(step0.get("optimizer_state"), "step-0 optimizer state")
    old_groups = old_state.get("param_groups")
    state_entries = old_state.get("state")
    _require(isinstance(old_groups, list) and len(old_groups) == 1, "step-0 optimizer groups changed")
    _require(isinstance(state_entries, Mapping), "step-0 optimizer state entries are missing")
    old_params = old_groups[0].get("params")
    _require(isinstance(old_params, list) and len(old_params) == 1, "step-0 optimizer parameter boundary changed")
    old_generator_state = state_entries.get(old_params[0])
    _require(isinstance(old_generator_state, Mapping), "step-0 Generator moments are missing")
    _require({"step", "exp_avg", "exp_avg_sq"}.issubset(old_generator_state), "step-0 Generator moments are incomplete")
    generator = _generator(consumer)
    for name in ("exp_avg", "exp_avg_sq"):
        value = old_generator_state[name]
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(generator.shape), f"step-0 optimizer moment shape changed: {name}")
        _finite(value, f"step-0 optimizer moment {name}")
    template = optimizer.state_dict()
    new_id = template["param_groups"][0]["params"][0]
    template["state"] = {new_id: copy.deepcopy(dict(old_generator_state))}
    group = dict(old_groups[0])
    group["params"] = [new_id]
    template["param_groups"] = [group]
    optimizer.load_state_dict(template)
    _validate_generator_optimizer(consumer, optimizer)
    return optimizer, "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED"


def _cmos_teacher_view(raw: torch.Tensor) -> torch.Tensor:
    """Apply the fixed raw-CMOS view used by the accepted TRAIN evaluator."""

    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "raw CMOS view shape changed")
    view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))
    _finite(view, "CMOS teacher view")
    return view


def _normalized_feature_alignment_loss(
    cmos_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed mean squared distance between normalized features."""

    _require(cmos_features.ndim == target_features.ndim == 2, "teacher feature rank changed")
    _require(cmos_features.shape == target_features.shape and cmos_features.shape[1] == 128, "teacher feature shape changed")
    target = target_features.detach()
    cmos_normalized = F.normalize(cmos_features, dim=1, eps=FEATURE_EPS)
    target_normalized = F.normalize(target, dim=1, eps=FEATURE_EPS)
    loss = (cmos_normalized - target_normalized).square().sum(dim=1).mean()
    _finite(loss, "L_align")
    return loss


def _alignment_components(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "target-alignment real path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/SLM target shape changed")
    _require(raw.dtype == torch.float32 and target_image.dtype == torch.float32, "raw/SLM target dtype changed")
    _finite(raw, "alignment raw CMOS")
    _finite(target_image, "alignment SLM target")
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    with torch.no_grad():
        target_features, _ = teacher(batch)
    cmos_features, _ = teacher(_cmos_teacher_view(raw))
    l_align = _normalized_feature_alignment_loss(cmos_features, target_features)
    for name, value in (("L_rec", l_rec), ("L_op", l_op), ("L_align", l_align)):
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target": target_image,
        "target_features": target_features.detach(),
        "cmos_features": cmos_features,
        "L_rec": l_rec,
        "L_op": l_op,
        "L_align": l_align,
    }


def _compose_gradients(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: Any,
    *,
    beta: float,
    gamma: float,
    delta_align: float,
) -> dict[str, Any]:
    generator = _generator(consumer)
    components = _alignment_components(consumer, teacher, batch, references)
    g_rec = torch.autograd.grad(beta * components["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
    g_op = torch.autograd.grad(gamma * components["L_op"], generator, retain_graph=True, allow_unused=False)[0]
    g_align = torch.autograd.grad(delta_align * components["L_align"], generator, retain_graph=False, allow_unused=False)[0]
    for name, value in (("g_rec", g_rec), ("g_op", g_op), ("g_align", g_align)):
        _finite(value, name)
    _norm(g_align, "g_align")
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec, g_align, "g_rec")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op, g_align, "g_op")
    g_total = g_align + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_align) > 0.0, "composed gradient lacks a positive alignment component")
    return {
        "components": components,
        "gradients": {
            "g_rec": g_rec,
            "g_op": g_op,
            "g_align": g_align,
            "g_rec_safe": g_rec_safe,
            "g_op_safe": g_op_safe,
            "g_total": g_total,
        },
        "gradient_norms": {
            name: _gradient_stats(value, name)
            for name, value in (
                ("g_rec", g_rec),
                ("g_op", g_op),
                ("g_align", g_align),
                ("g_rec_safe", g_rec_safe),
                ("g_op_safe", g_op_safe),
                ("g_total", g_total),
            )
        },
        "gradient_cosines": {
            "cos_g_rec_g_align": _cosine(g_rec, g_align),
            "cos_g_op_g_align": _cosine(g_op, g_align),
            "cos_g_rec_safe_g_align": _cosine(g_rec_safe, g_align),
            "cos_g_op_safe_g_align": _cosine(g_op_safe, g_align),
            "cos_g_total_g_align": _cosine(g_total, g_align),
        },
        "projection": {"rec": rec_projection, "op": op_projection},
        "old_tn_distribution_in_objective": False,
        "label_cross_entropy_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    components = _mapping(bundle["components"], "alignment components")
    return {
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "L_align": float(components["L_align"].detach().cpu()),
        "gradient_norms": bundle["gradient_norms"],
        "gradient_cosines": bundle["gradient_cosines"],
        "projection": bundle["projection"],
        "old_tn_distribution_in_objective": False,
        "label_cross_entropy_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _diagnose_composition(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: Any,
    *,
    beta: float,
    gamma: float,
    delta_align: float,
) -> dict[str, Any]:
    before = _state_digest(consumer.runtime.state_dict())
    bundle = _compose_gradients(consumer, teacher, batch, references, beta=beta, gamma=gamma, delta_align=delta_align)
    for parameter in consumer.runtime.parameters():
        _require(parameter.grad is None, "diagnostic composition populated a runtime gradient")
    successor._clear_grads(consumer)
    after = _state_digest(consumer.runtime.state_dict())
    _require(before == after, "diagnostic composition changed runtime state")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "diagnostic composition populated teacher gradients")
    return _composition_record(bundle)


def _apply_update(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    references: Any,
    *,
    beta: float,
    gamma: float,
    delta_align: float,
    step: int,
    ledger: DataReadLedger | None,
) -> dict[str, Any]:
    _validate_generator_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    successor._clear_grads(consumer)
    bundle = _compose_gradients(consumer, teacher, batch, references, beta=beta, gamma=gamma, delta_align=delta_align)
    for name, parameter in consumer.runtime.named_parameters():
        _require(parameter.grad is None, f"runtime gradient populated before assignment: {name}")
    generator.grad = bundle["gradients"]["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    after_frozen = _frozen_state(consumer)
    _require(_state_digest(before_frozen) == _state_digest(after_frozen), "frozen runtime state changed")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "teacher received an optimizer gradient")
    update_norm = float((generator.detach() - before_generator).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "generator.phase did not update")
    successor._clear_grads(consumer)
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    record = _composition_record(bundle)
    record.update(
        {
            "step": step,
            "generator_update_norm": update_norm,
            "encoder_unchanged": True,
            "frozen_runtime_unchanged": True,
            "trainable_parameter_names": ["generator.phase"],
        }
    )
    return record


def _calibrate_delta_align(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: Any,
    target_gradient_norm: float,
) -> dict[str, Any]:
    before = _state_digest(consumer.runtime.state_dict())
    components = _alignment_components(consumer, teacher, batch, references)
    raw_gradient = torch.autograd.grad(components["L_align"], _generator(consumer), retain_graph=False, allow_unused=False)[0]
    raw_norm = _norm(raw_gradient, "unweighted target-alignment gradient")
    _require(math.isfinite(target_gradient_norm) and target_gradient_norm > 0.0, "historical semantic gradient scale is invalid")
    delta_align = target_gradient_norm / raw_norm
    _require(math.isfinite(delta_align) and delta_align > 0.0, "delta_align calibration is invalid")
    weighted_norm = _norm(delta_align * raw_gradient, "calibrated target-alignment gradient")
    _require(math.isclose(weighted_norm, target_gradient_norm, rel_tol=2.0e-5, abs_tol=1.0e-12), "delta_align calibration did not reach the fixed target scale")
    successor._clear_grads(consumer)
    _require(before == _state_digest(consumer.runtime.state_dict()), "delta_align calibration changed runtime state")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "delta_align calibration populated teacher gradients")
    return {
        "method": "one TRAIN-only calibration batch at accepted clean step-0",
        "target_gradient_norm": target_gradient_norm,
        "unweighted_gradient_norm": raw_norm,
        "delta_align": delta_align,
        "weighted_gradient_norm": weighted_norm,
        "frozen_after_calibration": True,
        "sweep": False,
    }


def _teacher_positive_control(
    teacher: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    teacher.eval()
    selected_images = images.index_select(0, development_indices)
    selected_labels = labels.index_select(0, development_indices)
    correct_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(selected_images.shape[0]), LOGICAL_BATCH_SIZE):
            batch = selected_images[start : start + LOGICAL_BATCH_SIZE].to(device)
            _require(tuple(batch.shape[1:]) == (1, 28, 28), "canonical teacher positive-control shape changed")
            features, logits = teacher(batch)
            _require(features.shape[1] == 128 and logits.shape[1] == 10, "teacher positive-control geometry changed")
            _finite(features, "teacher positive-control features")
            _finite(logits, "teacher positive-control logits")
            correct_parts.append((logits.argmax(dim=1).cpu() == selected_labels[start : start + LOGICAL_BATCH_SIZE]).contiguous())
    correct = torch.cat(correct_parts)
    per_class = [float(correct[selected_labels == digit].to(torch.float64).mean()) for digit in range(10)]
    macro = float(sum(per_class) / 10.0)
    _require(macro >= 0.985, "TRAIN semantic teacher canonical positive control failed")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "teacher positive control populated gradients")
    return {
        "role": "canonical DEV images; diagnostic positive control only",
        "rows": int(selected_images.shape[0]),
        "macro_recall": macro,
        "per_class_recall": per_class,
        "feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier",
        "feature_dimension": 128,
        "selection_or_comparison": False,
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    docs = conflict._validate_current_docs(root)
    _require(OUTER_GATE not in "\n".join((root / "docs/CURRENT_STATUS.md").read_text(encoding="utf-8").splitlines()[:45]), "this target result is already recorded")

    attribution_path = root / conflict.ATTRIBUTION_REPORT
    _require(file_sha256(attribution_path) == conflict.EXPECTED_ATTRIBUTION_REPORT_SHA256, "accepted attribution report identity changed")
    attribution_report = json.loads(attribution_path.read_text(encoding="utf-8"))
    _require(attribution_report.get("result") == "VALID_COMPLETED", "accepted attribution result is not valid completed")
    _require(attribution_report.get("outer_gate") == conflict.attribution.OUTER_GATE, "accepted attribution gate changed")
    _require(attribution_report.get("primary_attribution") == "GENERATOR_SEMANTIC_CONTROLLABILITY_MIXED", "accepted attribution classification changed")
    _require(attribution_report.get("secondary_conflict") == "RECONSTRUCTION_SEMANTIC_GRADIENT_CONFLICT_SUPPORTED", "accepted conflict authority changed")

    accepted_report_path = root / PRIOR_REPORT
    _require(file_sha256(accepted_report_path) == EXPECTED_PRIOR_REPORT_SHA256, "accepted clean conflict-aware report identity changed")
    accepted_report = json.loads(accepted_report_path.read_text(encoding="utf-8"))
    _require(accepted_report.get("result") == "VALID_COMPLETED", "accepted clean conflict-aware result changed")
    _require(accepted_report.get("outer_gate") == conflict.OUTER_GATE, "accepted clean conflict-aware gate changed")
    _require(accepted_report.get("classification") == conflict.NOT_SUPPORTED_CLASSIFICATION, "accepted clean conflict-aware classification changed")
    _require(accepted_report.get("training_trajectory", {}).get("updates") == SCIENTIFIC_UPDATES, "accepted clean conflict-aware update count changed")

    step0 = safe_torch_load(root / PRIOR_STEP_0, expected_sha256=EXPECTED_PRIOR_STEP_0_SHA256, map_location="cpu")
    _require(isinstance(step0, Mapping), "accepted clean step-0 checkpoint is malformed")
    _require(step0.get("completed_updates") == 0, "accepted clean step-0 checkpoint is not step 0")
    _require(step0.get("scientific_artifact") is True and step0.get("operational_artifact") is False, "accepted clean step-0 role changed")
    _require(step0.get("outer_gate") == conflict.OUTER_GATE, "accepted clean step-0 gate changed")
    _require(step0.get("trainable_parameter_names") == ["generator.phase"], "accepted clean step-0 trainable boundary changed")
    _require("encoder.phase" in step0.get("frozen_parameter_names", []), "accepted clean step-0 encoder freeze is missing")
    _finite(_tensor(step0.get("generator_state"), "accepted clean step-0 generator"), "accepted clean step-0 generator")
    _finite(_tensor(step0.get("encoder_state"), "accepted clean step-0 encoder"), "accepted clean step-0 encoder")
    _require(step0.get("optimizer_mode") == "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED", "accepted clean step-0 optimizer mode changed")

    cpu_context = phase_repair._load_cpu_context(root)
    training = successor._load_successor_training_authority(root, {"training": cpu_context})
    images, labels, train_indices, development_indices, data_identity = conflict.producer._load_scientific_dataset(root)
    _require(data_identity.get("source_sha256") == TRAINING_SOURCE_SHA256, "TRAIN source identity changed")
    _require(data_identity == training["data_identity"], "accepted TRAIN/DEV data identity changed")
    calibration_identity = _mapping(training["weights"].calibration_batch_identity, "accepted calibration identity")
    calibration_indices = torch.tensor(calibration_identity.get("source_indices"), dtype=torch.long)
    _require(calibration_indices.numel() == LOGICAL_BATCH_SIZE, "accepted calibration batch size changed")
    calibration_labels = labels.index_select(0, calibration_indices)
    _require(calibration_labels.tolist() == list(range(10)), "accepted calibration labels changed")
    _require(bool(torch.all(torch.isin(calibration_indices, train_indices)).item()), "accepted calibration batch left TRAIN role")
    _require(math.isclose(float(training["weights"].beta), 2.4509953404, rel_tol=0.0, abs_tol=1.0e-10), "accepted beta authority changed")
    _require(math.isclose(float(training["weights"].gamma), 0.5330424285, rel_tol=0.0, abs_tol=1.0e-10), "accepted gamma authority changed")

    trajectory = _mapping(_mapping(accepted_report.get("gradient_composition"), "accepted gradient composition").get("trajectory"), "accepted gradient trajectory")
    step0_composition = _mapping(trajectory.get("0"), "accepted step-0 gradient composition")
    historical_gradient_norm = float(_mapping(_mapping(step0_composition.get("gradient_norms"), "accepted step-0 gradient norms").get("g_align", {}), "unused alignment gradient").get("norm", 0.0))
    if historical_gradient_norm <= 0.0:
        historical_gradient_norm = float(_mapping(_mapping(step0_composition.get("gradient_norms"), "accepted step-0 gradient norms").get("g_sem"), "accepted historical semantic gradient").get("norm"))
    _require(math.isfinite(historical_gradient_norm) and math.isclose(historical_gradient_norm, EXPECTED_HISTORICAL_SEMANTIC_GRADIENT_NORM, rel_tol=5.0e-4, abs_tol=1.0e-9), "accepted historical semantic gradient scale changed")

    phase = conflict._load_phase_transfer_authority(root, cpu_context)
    hashes = dict(conflict.attribution._load_authorities(root)["hashes"])
    hashes.update(phase["authority_hashes"])
    hashes.update({"accepted_conflict_report": EXPECTED_PRIOR_REPORT_SHA256, "accepted_clean_step0": EXPECTED_PRIOR_STEP_0_SHA256, "attribution_report": conflict.EXPECTED_ATTRIBUTION_REPORT_SHA256})
    return {
        "docs": docs,
        "attribution_report": attribution_report,
        "accepted_report": accepted_report,
        "step0": dict(step0),
        "training": training,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "data_identity": data_identity,
        "calibration_indices": calibration_indices,
        "calibration_labels": calibration_labels,
        "phase_latent": phase["phase_latent"],
        "selected_generated_raw": phase["selected_generated_raw"],
        "real_control_raw": phase["real_control_raw"],
        "selected_metrics": phase["selected_metrics"],
        "authority_hashes": hashes,
        "weights": {"beta": float(training["weights"].beta), "gamma": float(training["weights"].gamma)},
        "historical_gradient_norm": historical_gradient_norm,
    }


def _state_identity(teacher: FrozenMNISTEvaluator) -> str:
    return _state_digest(teacher.state_dict())


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[Any, FrozenMNISTEvaluator, SmoothMNISTEvaluator, dict[str, Any]]:
    consumer, train_teacher, smooth_evaluator, identity = conflict._load_models(root, authorities, device)
    _require(identity["train_architecture"] == "frozen_mnist_evaluator_v1", "TRAIN semantic teacher architecture changed")
    _require(identity["train_checkpoint_sha256"] == EXPECTED_TRAIN_EVALUATOR_SHA256, "TRAIN semantic teacher checkpoint changed")
    _require(identity["smooth"]["checkpoint_sha256"] == EXPECTED_SMOOTH_CHECKPOINT_SHA256, "Smooth evaluator checkpoint changed")
    _require(_state_identity(train_teacher) != _state_identity(smooth_evaluator), "TRAIN teacher is not distinct from Smooth evaluator")
    return consumer, train_teacher, smooth_evaluator, identity


def _run_preflight(
    consumer: Any,
    teacher: FrozenMNISTEvaluator,
    authorities: Mapping[str, Any],
    *,
    beta: float,
    gamma: float,
) -> dict[str, Any]:
    initial_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    initial_rng = _rng_snapshot()
    teacher_initial = _state_identity(teacher)
    bad_optimizer = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        _validate_generator_optimizer(consumer, bad_optimizer)
    except (ExecutionError, successor.RecoveryExecutionError):
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise ExecutionError("negative control accepted a full-runtime optimizer")

    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(consumer.device)
    calibration = _calibrate_delta_align(
        consumer,
        teacher,
        fixed_batch,
        authorities["training"]["references"],
        authorities["historical_gradient_norm"],
    )
    delta_align = float(calibration["delta_align"])
    composition0 = _diagnose_composition(
        consumer,
        teacher,
        fixed_batch,
        authorities["training"]["references"],
        beta=beta,
        gamma=gamma,
        delta_align=delta_align,
    )
    _require(composition0["old_tn_distribution_in_objective"] is False, "old TN term entered alignment objective")
    _require(composition0["label_cross_entropy_in_objective"] is False, "label CE entered alignment objective")
    optimizer, optimizer_mode = _restore_authoritative_optimizer(consumer, authorities["step0"], authorities["accepted_report"])
    sampler = DeterministicTrainRoleSampler(authorities["train_indices"], authorities["labels"])
    sampler_state = dict(authorities["step0"]["sampler_state"])
    sampler_state["cursor"] = 0
    sampler.load_state_dict(sampler_state)
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == authorities["calibration_indices"].tolist(), "preflight first TRAIN batch differs from authority")
    _require(torch.equal(authorities["labels"].index_select(0, first_indices), first_labels), "preflight TRAIN labels changed")
    sampler.load_state_dict(sampler_state)
    trace: list[dict[str, Any]] = []
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), "preflight TRAIN sampler binding changed")
        batch = authorities["images"].index_select(0, indices).to(consumer.device)
        trace.append(
            _apply_update(
                consumer,
                teacher,
                optimizer,
                batch,
                authorities["training"]["references"],
                beta=beta,
                gamma=gamma,
                delta_align=delta_align,
                step=step + 1,
                ledger=None,
            )
        )
    post_state = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    _require(torch.equal(initial_state["encoder.phase"], post_state["encoder.phase"]), "preflight changed frozen encoder.phase")
    _require(not torch.equal(initial_state["generator.phase"], post_state["generator.phase"]), "preflight Generator did not depart")
    _require(teacher_initial == _state_identity(teacher), "preflight changed TRAIN teacher state")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    _freeze_generator_only(consumer)
    successor._clear_grads(consumer)
    _restore_rng(initial_rng)
    runtime_restored = conflict._state_digest(initial_state) == conflict._state_digest(consumer.runtime.state_dict())
    teacher_unchanged = teacher_initial == _state_identity(teacher)
    _require(runtime_restored and teacher_unchanged, "preflight did not restore runtime or changed the frozen teacher")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "runtime_restored": runtime_restored,
        "frozen_evaluators_unchanged": teacher_unchanged,
        "starting_authority": {"path": PRIOR_STEP_0.as_posix(), "sha256": EXPECTED_PRIOR_STEP_0_SHA256, "selected_step": 0},
        "teacher_positive_control_required": True,
        "target_feature_contract": "stopgrad(FrozenMNISTEvaluator(batch)) paired with FrozenMNISTEvaluator(clamp(raw_cmos,0,1)->adaptive_avg_pool2d(28,28))",
        "feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier",
        "feature_dimension": 128,
        "normalization": {"kind": "L2", "epsilon": FEATURE_EPS},
        "delta_align_calibration": calibration,
        "gradient_rule": "g_total=g_align+project_nonconflicting(g_rec,g_align)+project_nonconflicting(g_op,g_align)",
        "eps_proj": EPS_PROJ,
        "step0_gradient_composition": composition0,
        "optimizer_mode": optimizer_mode,
        "three_step_real_update": {"updates": 3, "trace": trace, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "frozen_parameter_probe": {"encoder_phase_unchanged": True, "ols_unchanged": True, "w_ols_unchanged": True, "teacher_parameters_frozen": True},
        "old_tn_term": {"included_in_backward": False, "label_cross_entropy_in_backward": False, "scalar_weighted_sum_backward": False},
        "same_data_order": True,
        "same_optimizer_hyperparameters": True,
        "same_beta_gamma": True,
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "scientific_300_step_executed": False,
    }


def _public_alignment(alignment: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"same_reference_indices", "different_reference_indices", "paired_distances", "same_distances", "different_distances"}
    result: dict[str, Any] = {}
    for key, value in alignment.items():
        if key in excluded:
            continue
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


@torch.no_grad()
def _measure_smooth_realization(
    consumer: Any,
    smooth_evaluator: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    real_reference: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    smooth_evaluator.eval()
    cmos_features = torch.empty((development_indices.numel(), 128), dtype=torch.float32)
    cmos_logits = torch.empty((development_indices.numel(), 10), dtype=torch.float32)
    try:
        for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
            indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
            batch = images.index_select(0, indices).to(device)
            raw = consumer.runtime.forward_from_image(batch).raw_intensity
            _finite(raw, "Smooth diagnostic raw CMOS")
            features, logits = smooth_evaluator(realization._cmos_image_view(raw))
            _require(features.shape == (indices.numel(), 128) and logits.shape == (indices.numel(), 10), "Smooth diagnostic geometry changed")
            _finite(features, "Smooth diagnostic features")
            _finite(logits, "Smooth diagnostic logits")
            cmos_features[start : start + indices.numel()].copy_(features.cpu())
            cmos_logits[start : start + indices.numel()].copy_(logits.cpu())
    finally:
        consumer.runtime.train(was_training)
    labels_dev = labels.index_select(0, development_indices)
    alignment = realization._feature_alignment(real_reference["features"], cmos_features, labels_dev)
    metrics = realization._metrics(cmos_logits.argmax(dim=1), labels_dev)
    _require(all(parameter.grad is None for parameter in smooth_evaluator.parameters()), "Smooth diagnostic populated evaluator gradients")
    return {"alignment": alignment, "metrics": metrics, "features": cmos_features, "logits": cmos_logits, "labels": labels_dev}


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: DeterministicTrainRoleSampler,
    history: list[Mapping[str, Any]],
    composition_trajectory: Mapping[str, Any],
    *,
    step: int,
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
    preflight: Mapping[str, Any],
    delta_align: float,
) -> dict[str, Any]:
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
        "frozen_parameter_names": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_state(consumer)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": successor._cpu_state(sampler.state_dict()),
        "rng_state": _rng_state_payload(),
        "alignment": {"delta_align": delta_align, "feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier", "normalization_epsilon": FEATURE_EPS},
        "gradient_composition": {"eps_proj": EPS_PROJ, "rule": "alignment-primary; remove only negative first-order components from reconstruction and optical-power gradients", "preflight": dict(preflight), "trajectory": dict(composition_trajectory)},
        "training_history": [dict(record) for record in history],
    }


def _recovery_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _checkpoint_payload(*args, **kwargs)
    payload["scientific_artifact"] = False
    payload["operational_artifact"] = True
    payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
    return payload


def _classification(
    dev_semantic: Mapping[str, Any],
    dev_reconstruction: Mapping[str, Any],
    optical_validity: Mapping[str, Any],
    realization_summary: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    smooth_supported = bool(dev_semantic["delta_macro_recall"] > 0.0 and dev_semantic["paired_bootstrap_ci"]["lower"] > 0.0)
    reconstruction_pass = bool(dev_reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    optical_pass = bool(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"])
    centroid_moved = bool(realization_summary["final"]["centroid_matched_nearest_fraction"] > realization_summary["step_0"]["centroid_matched_nearest_fraction"])
    teacher_improved = bool(realization_summary["training_teacher"]["step_300_L_align"] < realization_summary["training_teacher"]["step_0_L_align"])
    raw_valid = bool(transfer["successor_raw_validity"]["finite"] and transfer["successor_raw_validity"]["nonconstant_fraction"] == 1.0 and transfer["successor_raw_validity"]["zero_image_count"] == 0)
    checks = {"dev_semantic_supported": smooth_supported, "dev_reconstruction_pass": reconstruction_pass, "optical_validity_pass": optical_pass, "smooth_centroid_alignment_moved": centroid_moved, "training_teacher_alignment_improved": teacher_improved, "sampled_raw_valid": raw_valid}
    if smooth_supported and not reconstruction_pass:
        return TRADEOFF_CLASSIFICATION, checks
    if smooth_supported and reconstruction_pass and optical_pass and centroid_moved:
        return SUPPORTED_CLASSIFICATION, checks
    if teacher_improved and not smooth_supported:
        return TEACHER_ONLY_CLASSIFICATION, checks
    return NOT_SUPPORTED_CLASSIFICATION, checks


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any], weights: Mapping[str, float], delta_align: float) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"updates": SCIENTIFIC_UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "objective": "L_align primary + auxiliary L_rec/L_op with conflict projection", "beta": weights["beta"], "gamma": weights["gamma"], "delta_align": delta_align, "data_source": dict(data_identity), "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256},
        model_identity="LIGHTGEN_2D_CLEAN_2L_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "prior_step0_optimizer_state": EXPECTED_PRIOR_STEP_0_SHA256},
        sampler_identity={"seed": 1234, "steps": SCIENTIFIC_UPDATES, "batch_size": LOGICAL_BATCH_SIZE, "class_balanced": True, "parent": "accepted clean conflict-aware step-0 sampler cursor reset to 0"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=None,
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
    consumer, teacher, smooth_evaluator, evaluator_identity = _load_models(root, authorities, device)
    _freeze_generator_only(consumer)
    consumer.runtime.train()
    teacher.eval()
    smooth_evaluator.eval()

    teacher_positive = _teacher_positive_control(teacher, authorities["images"], authorities["labels"], authorities["development_indices"], device)
    preflight = _run_preflight(consumer, teacher, authorities, beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"])
    delta_align = float(preflight["delta_align_calibration"]["delta_align"])

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(conflict.__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "updates": SCIENTIFIC_UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "prior_hashes": authorities["authority_hashes"]},
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, authorities["data_identity"], authorities["weights"], delta_align)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(target / "preflight.json", {**preflight, "teacher_positive_control": teacher_positive, "data_firewall": {"protected": False, "test": False, "sealed": False}})
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    preflight_updates = preflight.get("three_step_real_update")
    if not isinstance(preflight_updates, Mapping):
        preflight_updates = {}
    negative_control = preflight.get("negative_control")
    negative_control_passed = negative_control.get("passed") is True if isinstance(negative_control, Mapping) else False
    selected_engine = "target_image_alignment_cuda_generator_phase"
    receipt = control.make_ready_receipt(
        correctness={
            "passed": bool(preflight.get("status") == "READY" and teacher_positive.get("passed") is True and preflight.get("runtime_restored") is True and preflight.get("frozen_evaluators_unchanged") is True and preflight_updates.get("updates") == 3 and negative_control_passed),
            "three_step_updates": preflight_updates.get("updates"),
            "negative_control_rejected": negative_control_passed,
            "data_firewall": {"protected": False, "test": False, "sealed": False},
        },
        resource={"passed": int(cuda_total) > 0 and 0 <= int(cuda_free) <= int(cuda_total) and preflight.get("status") == "READY", "free_bytes": int(cuda_free), "total_bytes": int(cuda_total)},
        implementation={"passed": conflict._generator(consumer).requires_grad and not consumer.runtime.encoder.phase.requires_grad, "selected_engine": selected_engine},
        selected_engine=selected_engine,
        created_at=time.time(),
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, SCIENTIFIC_ATTEMPT_ID, receipt, resume=True)
    _write_json(target / "preflight_receipt.json", receipt.to_dict())
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "generator_target_image_semantic_alignment_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization_id": AUTHORIZATION_ID,
            "authorization_token": AUTHORIZATION_TOKEN,
            "input_authorities": ["M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION", "CONFLICT_AWARE_GENERATOR_SEMANTIC_RECOVERY_V1_STEP0", "GENERATOR_SEMANTIC_CONTROLLABILITY_ATTRIBUTION_RESOLVED", "CHEN_BOGT_PHASE_COMPLETION_REPAIR_V1_Q2", "SMOOTH_MNIST_EVALUATOR_V2_1"],
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "data_identity": authorities["data_identity"],
            "git_before": start_git,
            "cuda_runtime": cuda_runtime,
            "scope": {"bounded_updates": SCIENTIFIC_UPDATES, "trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"], "protected": False, "test": False, "sealed": False},
        },
    )

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
    composition_trajectory: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=SCIENTIFIC_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)

    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(device)
    composition_trajectory["0"] = _diagnose_composition(consumer, teacher, fixed_batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_align=delta_align)
    dev0 = successor._evaluate_dev(consumer, teacher, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=0)
    _require(abs(float(dev0["smooth_evaluator"]["macro_recall"]) - 0.1030) <= 5.0e-4, "step-0 CMOS-to-Smooth baseline diverges from accepted 0.1030")
    dev_records["0"] = dev0
    dev_images = authorities["images"].index_select(0, authorities["development_indices"])
    dev_labels = authorities["labels"].index_select(0, authorities["development_indices"])
    real_reference = realization._run_smooth(smooth_evaluator, dev_images, dev_labels, device=device, require_balanced=True)
    smooth_realization0 = _measure_smooth_realization(consumer, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], real_reference, device)
    _require(abs(float(smooth_realization0["metrics"]["macro_recall"]) - 0.1030) <= 5.0e-4, "step-0 Smooth diagnostic diverges from accepted 0.1030")
    _write_json(target / "dev_diagnostics_step_000000.json", {"schema": DEV_SCHEMA, "step": 0, "diagnostic": {key: value for key, value in dev0.items() if not key.startswith("_")}})
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=0, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_align=delta_align), kind="target-image semantic alignment scientific checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()
    last_recovery_path: Path | None = None

    def recovery_factory(step: int) -> Mapping[str, Any]:
        return _recovery_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_align=delta_align)

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "target-image alignment primary", "delta_align": delta_align}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_factory, phase="TRAINING", total_steps=SCIENTIFIC_UPDATES)
    with interrupt:
        for step in range(SCIENTIFIC_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            trace = _apply_update(consumer, teacher, optimizer, batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_align=delta_align, step=step + 1, ledger=ledger)
            history.append(trace)
            now = time.time()
            if recovery_due(step=step + 1, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                last_recovery_path = recovery.save(recovery_factory(step + 1), step=step + 1)
                last_recovery_step = step + 1
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_align")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            interrupted = interrupt.handle_safe_boundary(step=step + 1, latest_losses={key: trace[key] for key in ("L_rec", "L_op", "L_align")}, last_recovery_checkpoint=None if last_recovery_path is None else last_recovery_path.name)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": step + 1, "recovery_checkpoint": interrupted.name}
            completed = step + 1
            if completed in CHECKPOINT_STEPS[1:]:
                composition_trajectory[str(completed)] = _diagnose_composition(consumer, teacher, fixed_batch, authorities["training"]["references"], beta=authorities["weights"]["beta"], gamma=authorities["weights"]["gamma"], delta_align=delta_align)
                diagnostic = successor._evaluate_dev(consumer, teacher, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed)
                dev_records[str(completed)] = diagnostic
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {"schema": DEV_SCHEMA, "step": completed, "diagnostic": {key: value for key, value in diagnostic.items() if not key.startswith("_")}})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, preflight=preflight, delta_align=delta_align), kind="target-image semantic alignment scientific checkpoint")

    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=SCIENTIFIC_UPDATES, latest_losses={"completed_updates": SCIENTIFIC_UPDATES}, force=True)
    correctness_payload = {"schema": "generator_target_image_semantic_alignment_dev_correctness.v1", "development_indices_sha256": tensor_sha256(authorities["development_indices"]), "labels": authorities["labels"].index_select(0, authorities["development_indices"]), "steps": {step: {"train_correct": dev_records[step]["_train_correct"], "smooth_correct": dev_records[step]["_smooth_correct"]} for step in dev_records}}
    _write_torch(target / "dev_correctness.pt", correctness_payload, kind="target-image alignment DEV correctness vectors")
    bootstrap = successor._paired_bootstrap(correctness_payload["labels"], correctness_payload["steps"]["0"]["smooth_correct"], correctness_payload["steps"]["300"]["smooth_correct"])
    _write_torch(target / "paired_bootstrap.pt", bootstrap, kind="target-image alignment paired DEV Smooth bootstrap")
    bootstrap_report = dict(bootstrap)
    bootstrap_report.pop("macro_delta_tensor", None)
    bootstrap_report.pop("draws", None)
    dev_semantic = {
        "primary_evaluator": evaluator_identity["smooth"],
        "step_0_macro_recall": float(dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_300_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"]),
        "delta_macro_recall": float(dev_records["300"]["smooth_evaluator"]["macro_recall"] - dev_records["0"]["smooth_evaluator"]["macro_recall"]),
        "step_0_per_class_recall": list(dev_records["0"]["smooth_evaluator"]["per_class_recall"]),
        "step_300_per_class_recall": list(dev_records["300"]["smooth_evaluator"]["per_class_recall"]),
        "per_class_delta": list(bootstrap_report["per_class_delta"]),
        "paired_bootstrap_ci": dict(bootstrap_report["ci95"]),
        "paired_bootstrap_artifact": _artifact_ref(root, target / "paired_bootstrap.pt", role="paired DEV Smooth bootstrap"),
        "trajectory": {step: {"smooth_macro_recall": float(record["smooth_evaluator"]["macro_recall"]), "smooth_per_class_recall": list(record["smooth_evaluator"]["per_class_recall"])} for step, record in dev_records.items()},
        "evaluator_separation": {"train_loss_provider": "FrozenMNISTEvaluator v1 target-image feature teacher", "dev_primary": "SmoothMNISTEvaluator v2.1", "same_checkpoint": False, "smooth_in_objective": False},
    }
    dev_reconstruction = {
        "step_0": float(dev_records["0"]["reconstruction"]["L_rec"]),
        "step_100": float(dev_records["100"]["reconstruction"]["L_rec"]),
        "step_200": float(dev_records["200"]["reconstruction"]["L_rec"]),
        "step_300": float(dev_records["300"]["reconstruction"]["L_rec"]),
        "delta_step_300_minus_step_0": float(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"]),
        "tolerance": RECONSTRUCTION_TOLERANCE,
        "trajectory_pass": bool(dev_records["300"]["reconstruction"]["L_rec"] - dev_records["0"]["reconstruction"]["L_rec"] <= RECONSTRUCTION_TOLERANCE),
    }
    optical_validity = {
        "checkpoints": {step: {"finite": bool(record["raw_cmos"]["finite"]), "nonconstant": bool(record["raw_cmos"]["nonconstant"]), "zero_image_count": int(record["raw_cmos"]["zero_image_count"]), "L_op": float(record["optical"]["L_op"])} for step, record in dev_records.items()},
        "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in dev_records.values()),
        "all_nonconstant": all(bool(record["raw_cmos"]["nonconstant"]) for record in dev_records.values()),
        "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in dev_records.values()),
    }
    _require(optical_validity["all_finite"] and optical_validity["all_nonconstant"] and optical_validity["all_zero_image_counts_zero"], "optical validity failed")

    smooth_realization300 = _measure_smooth_realization(consumer, smooth_evaluator, authorities["images"], authorities["labels"], authorities["development_indices"], real_reference, device)
    realization_payload = {"schema": REALIZATION_SCHEMA, "labels": smooth_realization300["labels"], "real_features": real_reference["features"], "step_0_cmos_features": smooth_realization0["features"], "step_300_cmos_features": smooth_realization300["features"], "step_0_cmos_logits": smooth_realization0["logits"], "step_300_cmos_logits": smooth_realization300["logits"], "step_0_alignment": smooth_realization0["alignment"], "step_300_alignment": smooth_realization300["alignment"]}
    _write_torch(target / "smooth_semantic_realization.pt", realization_payload, kind="independent Smooth semantic realization diagnostics")
    realization_report = {
        "artifact": _artifact_ref(root, target / "smooth_semantic_realization.pt", role="independent Smooth feature/centroid realization diagnostics"),
        "step_0": _public_alignment(smooth_realization0["alignment"]),
        "final": _public_alignment(smooth_realization300["alignment"]),
        "step_0_smooth_metrics": smooth_realization0["metrics"],
        "step_300_smooth_metrics": smooth_realization300["metrics"],
        "training_teacher": {"step_0_L_align": float(composition_trajectory["0"]["L_align"]), "step_100_L_align": float(composition_trajectory["100"]["L_align"]), "step_200_L_align": float(composition_trajectory["200"]["L_align"]), "step_300_L_align": float(composition_trajectory["300"]["L_align"]), "delta_step_300_minus_step_0": float(composition_trajectory["300"]["L_align"] - composition_trajectory["0"]["L_align"])},
    }

    transfer_authorities = {"phase_latent": authorities["phase_latent"], "selected_generated_raw": authorities["selected_generated_raw"], "real_control_raw": authorities["real_control_raw"], "selected_metrics": authorities["selected_metrics"], "authority_hashes": {"phase_latent": EXPECTED_PHASE_LATENT_SHA256}}
    transfer = successor._p2q2_transfer(root, consumer, transfer_authorities, root / conflict.producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH, target)
    classification, checks = _classification(dev_semantic, dev_reconstruction, optical_validity, realization_report, transfer)
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=SCIENTIFIC_UPDATES, latest_losses={"classification": classification}, force=True)

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": classification == SUPPORTED_CLASSIFICATION,
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "git_end": successor.git_identity(root)},
        "starting_authority": {"accepted_clean_conflict_report": _artifact_ref(root, root / PRIOR_REPORT, role="accepted clean conflict-aware Generator-only report"), "accepted_clean_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="accepted clean 2L Generator step-0 state"), "rejected_as_start": ["failed weighted-sum step-300", "failed conflict-aware step-300", "failed 3L endpoint", "failed nonlinear endpoint"]},
        "training_teacher": {"identity": {"architecture": "FrozenMNISTEvaluator v1", "checkpoint": _artifact_ref(root, root / conflict.producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH, role="accepted frozen TRAIN semantic teacher"), "feature_layer": "embedding output immediately before classifier", "feature_dimension": 128, "checkpoint_validation_accuracy": float(evaluator_identity.get("train_validation_accuracy", 0.0)) if evaluator_identity.get("train_validation_accuracy") is not None else None}, "independent_from_smooth": True, "smooth_checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256, "positive_control": teacher_positive},
        "alignment_contract": {"target": "stopgrad(FrozenMNISTEvaluator(canonical TRAIN image))", "cmos": "FrozenMNISTEvaluator(adaptive_avg_pool2d(clamp(raw_cmos,0,1),(28,28)))", "feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier", "feature_dimension": 128, "normalization": "L2 normalization with epsilon 1e-12", "loss": "mean(||normalize(f_cmos)-normalize(f_target)||_2^2)", "delta_align": delta_align, "calibration": preflight["delta_align_calibration"]},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "scale", "T=I_100", "W_OLS", "latent normalization", "P2", "Q2", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator"]},
        "loss_weights": {"beta": authorities["weights"]["beta"], "gamma": authorities["weights"]["gamma"], "delta_align": delta_align, "beta_gamma_reused_exactly": True, "weight_sweep": False},
        "gradient_composition": {"formula": "g_total=g_align+g_rec_safe+g_op_safe", "g_align": "grad_generator(delta_align*L_align)", "g_rec": "grad_generator(beta*L_rec)", "g_op": "grad_generator(gamma*L_op)", "g_rec_safe": "project g_rec only when dot(g_rec,g_align)<0", "g_op_safe": "project g_op only when dot(g_op,g_align)<0", "eps_proj": EPS_PROJ, "preflight": preflight, "trajectory": composition_trajectory},
        "preflight": {**preflight, "teacher_positive_control": teacher_positive},
        "training_trajectory": {"updates": len(history), "checkpoint_steps": list(CHECKPOINT_STEPS), "trace": history, "sampler_cursor": int(sampler.cursor)},
        "alignment_loss_trajectory": realization_report["training_teacher"],
        "dev_smooth": dev_semantic,
        "smooth_semantic_realization": realization_report,
        "dev_reconstruction": dev_reconstruction,
        "optical_validity": optical_validity,
        "p2_q2_transfer": transfer,
        "counterfactual_integrity": {"same_start_step0": True, "same_data_order": True, "same_generator_architecture": True, "same_encoder_ols": True, "same_auxiliary_losses": True, "same_beta_gamma": True, "same_optimizer": True, "same_update_budget": True, "smooth_excluded_from_backward": True, "only_changed_variable": "paired target-image feature-alignment objective and alignment-primary gradient composition", "protected_test_sealed": False},
        "outer_gate_checks": checks,
        "input_authorities": authorities["authority_hashes"],
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "claim_boundary": {"scientific_training_executed": True, "updates": SCIENTIFIC_UPDATES, "generation_success": classification == SUPPORTED_CLASSIFICATION, "teacher_optimality": False, "global_architecture_inadequacy": False, "formal_promotion": False, "physical_hardware_validity": False, "architecture_change": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False},
        "files_changed": {"scientific_artifacts": sorted(path.name for path in target.iterdir()), "repository_source": ["scripts/research/lightgen/execute_lightgen_2d_generator_target_image_semantic_alignment_v1.py"]},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "checkpoints": [0, 100, 200, 300], "dev_evaluations": [0, 100, 200, 300], "paired_bootstrap": True, "smooth_feature_realization": True, "frozen_gradient_firewall": True, "alignment_gradient_calibrated_once": True, "no_protected_test_sealed": True},
        "unresolved": [] if classification == SUPPORTED_CLASSIFICATION else ["The bounded run did not satisfy every independent target-image semantic realization condition under the frozen clean 2L architecture."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
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
