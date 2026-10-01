"""Execute the preregistered evaluator-robust semantic training trajectory.

The consumer starts from the accepted clean 2L Generator step-0 authority and
updates only ``generator.phase``.  Smooth and independent evaluators define an
equal-weight normalized joint semantic direction; the newly established third
evaluator is confirmation-only and never enters backward.  Stage 1 runs 300
updates, and the already-authorized Stage 2 continues the same optimizer and
ordering to a fixed total of 3000 updates only when the frozen Stage-1 gate
passes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import ADAM_BETAS, ADAM_EPSILON, AMSGRAD, DataReadLedger, DeterministicTrainRoleSampler, FrozenBogtReferences, LightGen2DChenBOGTComplexOLSTrainingConsumer, PRODUCTION_PROPAGATION_GRID, TRAIN_ROLE, WEIGHT_DECAY, contracted_plane_powers, git_identity, optical_power_loss_from_powers, _rng_state_payload
from ldm_project.v2.application.research.lightgen import canonical_semantics
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
    stratified_mnist_indices,
)

from scripts.research.lightgen import execute_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1 as cross
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary
from scripts.research.lightgen import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research.lightgen import execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1 as canonical_training
from ldm_project.v2.application.research.lightgen import confirmation_evaluators as third_confirmation


TASK_ID = "EXECUTE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1"
CONDITIONAL_AUTHORIZATION_TOKEN = "AUTHORIZED_CONDITIONAL_CONTINUE_EVALUATOR_ROBUST_SEMANTIC_LONG_TRAINING_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
CONDITIONAL_AUTHORIZATION_ID = f"{TASK_ID}:authorization:02"
OUTER_GATE = "EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "evaluator-robust-semantic-training-v1"
TRAJECTORY_ID = "evaluator-robust-semantic-training-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_PRODUCER_BRANCH = EXPECTED_BRANCH
EXPECTED_PRODUCER_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"
# Historical import compatibility only.  This is producer provenance, not a
# current-consumer exact-source lock; all consumers use ``_bind_git_provenance``.
EXPECTED_HEAD = EXPECTED_PRODUCER_HEAD

CROSS_REPORT = cross.DEFAULT_OUTPUT_DIR / "report.json"
EXPECTED_CROSS_REPORT_SHA256 = "c4c390df298ba82563505bec5988b0383709b57e342dfdd623efef9e70425951"
CLEAN_REPORT = cross.PRIOR_REPORT
CLEAN_STEP_0 = cross.PRIOR_STEP_0
EXPECTED_CLEAN_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
EXPECTED_CLEAN_STEP_0_SHA256 = cross.EXPECTED_PRIOR_STEP_0_SHA256

EC_RUN = Path("outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03")
EC_REPORT = EC_RUN / "report.json"
EC_CHECKPOINT = EC_RUN / "evaluator_checkpoint.pt"
EC_QUALIFICATION = EC_RUN / "qualification_metrics.json"
EC_SPLIT = EC_RUN / "split_manifest.json"
EXPECTED_EC_REPORT_SHA256 = "48401db6b3958d7ad29cb587180f977bfe375cfe887ac1cd618db1f526c82020"
EXPECTED_EC_CHECKPOINT_SHA256 = "e5688c68a6fba506a22530770efd3f6a09d5c72097437c28c535ddec26a50f95"
EXPECTED_EC_QUALIFICATION_SHA256 = "0ec6e40637ced8d3de13485d2dec85672933b845a211ac23017f4dedcf40b6fe"
EXPECTED_EC_SPLIT_SHA256 = "66086d50dc536371b6acf052008ec41132029de0929c81f3abc33bb01647851a"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
INDEPENDENT_CHECKPOINT = canonical_training.INDEPENDENT_CHECKPOINT
EXPECTED_INDEPENDENT_CHECKPOINT_SHA256 = canonical_training.EXPECTED_INDEPENDENT_CHECKPOINT_SHA256
SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
TRAIN_EVALUATOR_CHECKPOINT = producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
EXPECTED_TRAIN_EVALUATOR_CHECKPOINT_SHA256 = producer.EVALUATOR_CHECKPOINT_SHA256

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_evaluator_robust_semantic_training_v1/runs/attempt_01"
)
STAGE_1_UPDATES = 300
TOTAL_UPDATES = 3000
CHECKPOINT_STEPS = (0, 100, 200, 300, 600, 1000, 1500, 2000, 2500, 3000)
LOGICAL_BATCH_SIZE = 10
SCHEDULE_SEED = producer.SCHEDULE_SEED
FEATURE_EPS = 1.0e-12
EPS_PROJ = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12
DELTA_TARGET_GRADIENT_NORM = 0.0019317993727
BETA = 2.4509953404
GAMMA = 0.5330424285
RECONSTRUCTION_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
PRELIGHT_EPSILON = 1.0e-4
RECOVERY_INTERVAL_STEPS = 25

REPORT_SCHEMA = "evaluator_robust_semantic_training_report.v1"
PREFLIGHT_SCHEMA = "evaluator_robust_semantic_training_preflight.v1"
CHECKPOINT_SCHEMA = "evaluator_robust_semantic_training_checkpoint.v1"
DEV_SCHEMA = "evaluator_robust_semantic_training_dev_diagnostic.v1"
BOOTSTRAP_SCHEMA = "evaluator_robust_semantic_training_paired_bootstrap.v1"
CMOS_SCHEMA = "evaluator_robust_semantic_training_p2_q2_cmos.v1"

CLASSIFICATION_SUPPORTED = "EVALUATOR_ROBUST_SEMANTIC_RECOVERY_SCALES_TO_LONG_HORIZON"
CLASSIFICATION_SHORT_HORIZON = "SHORT_HORIZON_ROBUST_RECOVERY_WITH_LONG_HORIZON_INSTABILITY"
CLASSIFICATION_DEGENERATED = "SHARED_SEMANTIC_DIRECTION_DEGENERATES_AT_LONG_HORIZON"
CLASSIFICATION_RECON_TRADEOFF = "LONG_HORIZON_SEMANTIC_RECONSTRUCTION_TRADEOFF"
CLASSIFICATION_STAGE1_NO_EC = "JOINT_DEVELOPMENT_SEMANTIC_GAIN_WITHOUT_INDEPENDENT_TRANSFER"
CLASSIFICATION_STAGE1_DEGENERATED = "SHARED_SEMANTIC_DIRECTION_NOT_GLOBALLY_STABLE"
CLASSIFICATION_STAGE1_RECON = "EVALUATOR_ROBUST_SEMANTIC_RECOVERY_RECONSTRUCTION_TRADEOFF"
CLASSIFICATION_OPTIMIZATION = "JOINT_SEMANTIC_TRAINING_OPTIMIZATION_NOT_SUPPORTED"


class RobustTrainingError(RuntimeError):
    """Raised when a frozen input or execution boundary cannot be consumed."""


class JointDirectionDegenerated(RobustTrainingError):
    """Raised only for the preregistered scientific joint-direction stop."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RobustTrainingError(message)


def _bind_git_provenance(
    consumer_git: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record producer and consumer Git facts without using them as locks."""

    historical = {} if contract is None else contract.get("git", contract)
    _require(isinstance(historical, Mapping), "historical Git provenance must be a mapping")
    producer_branch = historical.get("branch", EXPECTED_PRODUCER_BRANCH)
    producer_head = historical.get("head", EXPECTED_PRODUCER_HEAD)
    if producer_head is None:
        for field in ("producer_head", "source_head", "git_head"):
            candidate = historical.get(field, contract.get(field) if contract is not None else None)
            if candidate is not None:
                producer_head = candidate
                break
    _require(isinstance(producer_branch, str) and bool(producer_branch), "historical producer branch is unavailable")
    producer_head = str(producer_head or "")

    consumer_branch = consumer_git.get("branch")
    consumer_head = str(consumer_git.get("head") or "")

    return {
        "producer": {"branch": producer_branch, "head": producer_head},
        "consumer": {"branch": consumer_branch, "head": consumer_head},
        "head_equal": consumer_head == producer_head,
        "producer_head_is_provenance_only": True,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    try:
        canonical_semantics.finite(value, name)
    except ValueError as error:
        _require(False, str(error))


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_write_new_json(path, dict(payload), artifact_kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _runtime_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}


def _frozen_runtime_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.runtime.state_dict().items()
        if name != "generator.phase"
    }


def _generator(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
    names = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(names == ["generator.phase"], f"trainable firewall changed: {names}")


def _clear_runtime_grads(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for parameter in consumer.runtime.parameters():
        parameter.grad = None


def _clear_evaluator_grads(*evaluators: torch.nn.Module) -> None:
    for evaluator in evaluators:
        for parameter in evaluator.parameters():
            parameter.grad = None


def _norm(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    try:
        return canonical_semantics.norm(value, name, allow_zero=allow_zero)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return canonical_semantics.dot(left, right)


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    try:
        return canonical_semantics.cosine(left, right, name)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    try:
        values = canonical_semantics.gradient_stats(value, name)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error
    values.pop("rms")
    return values


def _project_auxiliary_gradient(
    auxiliary: torch.Tensor,
    semantic: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        return canonical_semantics.project_auxiliary_gradient(
            auxiliary,
            semantic,
            name,
            epsilon=EPS_PROJ,
            absolute_tolerance=1.0e-8,
            relative_tolerance=1.0e-5,
            norm_floor=1.0e-12,
        )
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    try:
        return canonical_semantics.cmos_view(value)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _normalized_feature_loss(cmos_features: torch.Tensor, target_features: torch.Tensor, name: str) -> torch.Tensor:
    try:
        return canonical_semantics.normalized_feature_loss(
            cmos_features, target_features, name, epsilon=FEATURE_EPS
        )
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    _finite(raw, "raw CMOS")
    flat = raw.detach().to(torch.float64).flatten(1)
    std = flat.std(dim=1, unbiased=False)
    return {
        "finite": True,
        "nonconstant_fraction": float((std > 0.0).to(torch.float64).mean().cpu()),
        "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
    }


def _loss_components(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "real optical path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/target geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "raw/target dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_image, "target image")
    smooth_cmos, _ = smooth(_cmos_view(raw))
    independent_cmos, _ = independent(_cmos_view(raw))
    with torch.no_grad():
        smooth_target, _ = smooth(_cmos_view(batch))
        independent_target, _ = independent(_cmos_view(batch))
    l_s = _normalized_feature_loss(smooth_cmos, smooth_target, "L_S")
    l_i = _normalized_feature_loss(independent_cmos, independent_target, "L_I")
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    for name, value in (("L_S", l_s), ("L_I", l_i), ("L_rec", l_rec), ("L_op", l_op)):
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target_image": target_image,
        "L_S": l_s,
        "L_I": l_i,
        "L_rec": l_rec,
        "L_op": l_op,
    }


def _joint_direction(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
) -> dict[str, Any]:
    generator = _generator(consumer)
    components = _loss_components(consumer, smooth, independent, batch, references)
    g_s = torch.autograd.grad(components["L_S"], generator, retain_graph=True, allow_unused=False)[0]
    g_i = torch.autograd.grad(components["L_I"], generator, retain_graph=True, allow_unused=False)[0]
    for name, value in (("g_S", g_s), ("g_I", g_i)):
        _finite(value, name)
    norm_s = _norm(g_s, "g_S")
    norm_i = _norm(g_i, "g_I")
    u_s = g_s / norm_s
    u_i = g_i / norm_i
    v_joint = u_s + u_i
    v_norm = _norm(v_joint, "v_joint", allow_zero=True)
    if not math.isfinite(v_norm) or v_norm <= JOINT_DEGENERACY_TOLERANCE:
        raise JointDirectionDegenerated("joint direction norm is nonfinite or <= 1e-12")
    try:
        q_joint, v_rms = canonical_semantics.rms_normalize(v_joint, "v_joint")
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error
    d_joint = -q_joint
    _finite(q_joint, "q_joint")
    _require(torch.equal((-d_joint).detach(), q_joint.detach()), "joint sign convention changed")
    return {
        "components": components,
        "g_S": g_s,
        "g_I": g_i,
        "u_S": u_s,
        "u_I": u_i,
        "v_joint": v_joint,
        "q_joint": q_joint,
        "d_joint": d_joint,
        "g_S_norm": norm_s,
        "g_I_norm": norm_i,
        "v_joint_norm": v_norm,
        "v_joint_rms": v_rms,
    }


def _compose_gradients(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
    *,
    delta_joint: float,
) -> dict[str, Any]:
    direction = _joint_direction(consumer, smooth, independent, batch, references)
    components = direction["components"]
    generator = _generator(consumer)
    g_rec_unweighted = torch.autograd.grad(components["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
    g_op_unweighted = torch.autograd.grad(components["L_op"], generator, retain_graph=False, allow_unused=False)[0]
    g_rec_w = BETA * g_rec_unweighted
    g_op_w = GAMMA * g_op_unweighted
    g_sem = delta_joint * direction["q_joint"]
    for name, value in (("g_rec_w", g_rec_w), ("g_op_w", g_op_w), ("g_sem", g_sem)):
        _finite(value, name)
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec_w, g_sem, "g_rec_w")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op_w, g_sem, "g_op_w")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_sem) > 0.0, "combined gradient has no positive semantic component")
    return {
        **direction,
        "gradients": {
            "g_rec_unweighted": g_rec_unweighted,
            "g_op_unweighted": g_op_unweighted,
            "g_rec_w": g_rec_w,
            "g_op_w": g_op_w,
            "g_sem": g_sem,
            "g_rec_safe": g_rec_safe,
            "g_op_safe": g_op_safe,
            "g_total": g_total,
        },
        "projection": {"g_rec": rec_projection, "g_op": op_projection},
        "gradient_cosines": {
            "cos_g_S_g_I": _cosine(direction["g_S"], direction["g_I"], "cos_g_S_g_I"),
            "cos_g_total_g_sem": _cosine(g_total, g_sem, "cos_g_total_g_sem"),
        },
        "delta_joint": delta_joint,
        "evaluator_parameters_in_backward": {"E_S": True, "E_I": True, "E_C": False},
        "target_features_stop_gradient": True,
        "label_cross_entropy_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any], *, step: int, phase_displacement: float, phase_update_norm: float) -> dict[str, Any]:
    components = _mapping(bundle["components"], "joint components")
    gradients = _mapping(bundle["gradients"], "joint gradients")
    return {
        "step": step,
        "L_S": float(components["L_S"].detach().cpu()),
        "L_I": float(components["L_I"].detach().cpu()),
        "L_rec": float(components["L_rec"].detach().cpu()),
        "L_op": float(components["L_op"].detach().cpu()),
        "gradient_norms": {
            name: _gradient_stats(value, name)
            for name, value in (
                ("g_S", bundle["g_S"]),
                ("g_I", bundle["g_I"]),
                ("v_joint", bundle["v_joint"]),
                ("g_sem", gradients["g_sem"]),
                ("g_total", gradients["g_total"]),
            )
        },
        "cos_g_S_g_I": float(bundle["gradient_cosines"]["cos_g_S_g_I"]),
        "cos_g_total_g_sem": float(bundle["gradient_cosines"]["cos_g_total_g_sem"]),
        "v_joint_norm": float(bundle["v_joint_norm"]),
        "v_joint_rms": float(bundle["v_joint_rms"]),
        "delta_joint": float(bundle["delta_joint"]),
        "phase_displacement_from_clean_step0": phase_displacement,
        "phase_update_norm": phase_update_norm,
        "projection": bundle["projection"],
        "optical_validity": _raw_validity(components["raw"].detach()),
        "trainable_parameter_names": ["generator.phase"],
        "frozen_evaluator_parameters": True,
        "E_C_in_backward": False,
    }


def _calibrate_delta_joint(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
) -> dict[str, Any]:
    before = _state_digest(_runtime_state(consumer))
    direction = _joint_direction(consumer, smooth, independent, batch, references)
    q_norm = _norm(direction["q_joint"], "q_joint")
    delta_joint = DELTA_TARGET_GRADIENT_NORM / q_norm
    applied_norm = _norm(delta_joint * direction["q_joint"], "calibrated g_sem")
    _require(math.isclose(applied_norm, DELTA_TARGET_GRADIENT_NORM, rel_tol=2.0e-5, abs_tol=1.0e-12), "delta_joint calibration missed target norm")
    _clear_runtime_grads(consumer)
    _clear_evaluator_grads(smooth, independent)
    _require(before == _state_digest(_runtime_state(consumer)), "delta_joint calibration changed runtime state")
    return {
        "method": "exactly one fixed clean step-0 TRAIN calibration batch",
        "target_gradient_norm": DELTA_TARGET_GRADIENT_NORM,
        "unweighted_joint_direction_norm": q_norm,
        "delta_joint": delta_joint,
        "applied_joint_gradient_norm": applied_norm,
        "sweep": False,
        "schedule": False,
        "g_S_norm": direction["g_S_norm"],
        "g_I_norm": direction["g_I_norm"],
        "v_joint_norm": direction["v_joint_norm"],
        "v_joint_rms": direction["v_joint_rms"],
        "cos_g_S_g_I": _cosine(direction["g_S"], direction["g_I"], "calibration cos_g_S_g_I"),
        "sign_convention": "d_joint=-v_joint/RMS(v_joint); q_joint=-d_joint; optimizer.grad uses +q_joint",
    }


def _new_optimizer(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        [_generator(consumer)],
        lr=1.0e-3,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    _validate_optimizer(consumer, optimizer)
    return optimizer


def _validate_optimizer(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, optimizer: torch.optim.Optimizer) -> None:
    _require(isinstance(optimizer, torch.optim.Adam), "optimizer is not Adam")
    _require(len(optimizer.param_groups) == 1, "optimizer has unexpected parameter groups")
    group = optimizer.param_groups[0]
    parameters = list(group["params"])
    _require(len(parameters) == 1 and parameters[0] is _generator(consumer), "optimizer contains a non-generator parameter")
    _require(group["lr"] == 1.0e-3 and tuple(group["betas"]) == ADAM_BETAS, "optimizer lr/betas changed")
    _require(group["eps"] == ADAM_EPSILON and group["weight_decay"] == WEIGHT_DECAY and group["amsgrad"] is AMSGRAD, "optimizer options changed")


def _restore_clean_optimizer(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    step0: Mapping[str, Any],
    clean_report: Mapping[str, Any],
) -> tuple[torch.optim.Adam, str]:
    optimizer, mode = canonical_training._restore_authoritative_optimizer(consumer, step0, clean_report)
    _validate_optimizer(consumer, optimizer)
    _require(mode == "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED", "clean optimizer moments were not restored")
    return optimizer, mode


class LongHorizonSampler:
    """Reuse the clean sampler permutations without a post-300 reshuffle."""

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor) -> None:
        reference = DeterministicTrainRoleSampler(train_indices, labels)
        reference_state = reference.state_dict()
        self.seed = int(reference_state["seed"])
        self.steps = TOTAL_UPDATES
        self.cursor = 0
        self.permutations = [value.cpu().clone() for value in reference_state["permutations"]]

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.cursor < self.steps, "long-horizon sampler exhausted")
        indices = torch.stack([permutation[self.cursor] for permutation in self.permutations])
        self.cursor += 1
        return indices, torch.arange(10, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "evaluator_robust_long_horizon_sampler.v1",
            "seed": self.seed,
            "steps": self.steps,
            "cursor": self.cursor,
            "permutations": [value.clone() for value in self.permutations],
        }


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    torch.random.set_rng_state(snapshot[0])
    torch.cuda.set_rng_state_all(snapshot[1])


def _loss_values_no_grad(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: FrozenBogtReferences,
) -> dict[str, float]:
    with torch.no_grad():
        components = _loss_components(consumer, smooth, independent, batch, references)
        return {name: float(components[name].cpu()) for name in ("L_S", "L_I", "L_rec", "L_op")}


def _run_preflight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    authorities: Mapping[str, Any],
    *,
    delta_joint: float,
) -> dict[str, Any]:
    initial_state = _runtime_state(consumer)
    initial_rng = _rng_snapshot()
    initial_smooth = _state_digest(smooth.state_dict())
    initial_independent = _state_digest(independent.state_dict())
    initial_ec = _state_digest(ec.state_dict())
    bad_optimizer = torch.optim.Adam(list(consumer.runtime.parameters()), lr=1.0e-3)
    try:
        _validate_optimizer(consumer, bad_optimizer)
    except RobustTrainingError:
        negative_control = "FULL_RUNTIME_OPTIMIZER_REJECTED"
    else:
        raise RobustTrainingError("negative control accepted an optimizer outside generator.phase")

    batch_indices = authorities["calibration_indices"]
    batch = authorities["images"].index_select(0, batch_indices).to(consumer.device)
    direction = _joint_direction(consumer, smooth, independent, batch, authorities["training"]["references"])
    _require(_dot(direction["g_S"], direction["q_joint"]) > 0.0, "q_joint is not descent-compatible for L_S")
    _require(_dot(direction["g_I"], direction["q_joint"]) > 0.0, "q_joint is not descent-compatible for L_I")
    before_values = _loss_values_no_grad(consumer, smooth, independent, batch, authorities["training"]["references"])
    generator = _generator(consumer)
    generator_before = generator.detach().clone()
    with torch.no_grad():
        generator.add_(-PRELIGHT_EPSILON * direction["q_joint"])
    after_values = _loss_values_no_grad(consumer, smooth, independent, batch, authorities["training"]["references"])
    with torch.no_grad():
        generator.copy_(generator_before)
    _clear_runtime_grads(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "joint sign reproduction did not restore runtime")
    _require(after_values["L_S"] < before_values["L_S"] and after_values["L_I"] < before_values["L_I"], "joint direction is not first-order beneficial for both evaluators")

    optimizer, optimizer_mode = _restore_clean_optimizer(consumer, authorities["step0"], authorities["clean_report"])
    sampler = LongHorizonSampler(authorities["train_indices"], authorities["labels"])
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == batch_indices.tolist(), "long-horizon first TRAIN batch differs from clean authority")
    _require(torch.equal(authorities["labels"].index_select(0, first_indices), first_labels), "first TRAIN labels changed")
    sampler.cursor = 0
    traces: list[dict[str, Any]] = []
    for step in range(3):
        indices, batch_labels = sampler.next_batch()
        batch_step = authorities["images"].index_select(0, indices).to(consumer.device)
        traces.append(_apply_update(consumer, smooth, independent, optimizer, batch_step, batch_labels.to(consumer.device), authorities["training"]["references"], delta_joint=delta_joint, step=step + 1, ledger=None, clean_generator_state=initial_state["generator.phase"]))
    post_state = _runtime_state(consumer)
    _require(torch.equal(initial_state["encoder.phase"], post_state["encoder.phase"]), "preflight changed frozen encoder")
    _require(not torch.equal(initial_state["generator.phase"], post_state["generator.phase"]), "preflight Generator did not depart")
    _require(initial_smooth == _state_digest(smooth.state_dict()), "preflight changed E_S")
    _require(initial_independent == _state_digest(independent.state_dict()), "preflight changed E_I")
    _require(initial_ec == _state_digest(ec.state_dict()), "preflight changed E_C")
    _require(all(parameter.grad is None for parameter in ec.parameters()), "E_C received a backward gradient")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    _freeze_generator_only(consumer)
    _clear_runtime_grads(consumer)
    _clear_evaluator_grads(smooth, independent, ec)
    _restore_rng(initial_rng)
    step0_raw = consumer.runtime.forward_from_image(batch).raw_intensity.detach()
    raw_validity = _raw_validity(step0_raw)
    _require(raw_validity["finite"] and raw_validity["nonconstant_fraction"] == 1.0 and raw_validity["zero_image_count"] == 0, "step-0 optical validity failed")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "starting_authority": {"path": CLEAN_STEP_0.as_posix(), "sha256": EXPECTED_CLEAN_STEP_0_SHA256, "selected_step": 0},
        "authority_checks": {
            "cross_evaluator_report": True,
            "clean_step0": True,
            "E_S": True,
            "E_I": True,
            "E_C": True,
            "E_C_qualification": True,
        },
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "P2", "Q2", "E_S", "E_I", "E_C"],
        "optimizer_mode": optimizer_mode,
        "delta_joint_calibration": authorities["calibration"],
        "joint_sign_convention": {
            "d_joint": "-v_joint/RMS(v_joint)",
            "q_joint": "-d_joint",
            "optimizer_gradient": "+g_sem where g_sem=delta_joint*q_joint",
            "dot_g_S_q_joint": _dot(direction["g_S"], direction["q_joint"]),
            "dot_g_I_q_joint": _dot(direction["g_I"], direction["q_joint"]),
            "epsilon": PRELIGHT_EPSILON,
            "before": before_values,
            "after": after_values,
            "runtime_restored": True,
        },
        "three_step_real_update": {"updates": 3, "trace": traces, "generator_departed": True, "encoder_unchanged": True},
        "negative_control": {"name": negative_control, "passed": True},
        "E_C_absent_from_backward": True,
        "optical_validity": raw_validity,
        "protected_test_sealed": {"protected": False, "test": False, "sealed": False},
        "scientific_300_update_stage_executed": False,
    }


def _apply_update(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    references: FrozenBogtReferences,
    *,
    delta_joint: float,
    step: int,
    ledger: DataReadLedger | None,
    clean_generator_state: torch.Tensor,
) -> dict[str, Any]:
    _validate_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_runtime_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    _clear_runtime_grads(consumer)
    bundle = _compose_gradients(consumer, smooth, independent, batch, references, delta_joint=delta_joint)
    generator.grad = bundle["gradients"]["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    for name, parameter in consumer.runtime.named_parameters():
        if name != "generator.phase":
            _require(parameter.grad is None, f"frozen parameter received a gradient: {name}")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    after_frozen = _frozen_runtime_state(consumer)
    _require(_state_digest(before_frozen) == _state_digest(after_frozen), "frozen runtime state changed")
    update_norm = float((generator.detach() - before_generator).to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "generator.phase did not update")
    _require(all(parameter.grad is None for parameter in independent.parameters()), "E_I received an optimizer gradient")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "E_S received an optimizer gradient")
    if ledger is not None:
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    displacement = float((generator.detach() - clean_generator_state.to(generator.device)).to(torch.float64).square().sum().sqrt().cpu())
    return _composition_record(bundle, step=step, phase_displacement=displacement, phase_update_norm=update_norm)


def _class_metrics(correct: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(correct.shape == labels.shape == (5000,), "DEV correctness shape changed")
    recalls: list[float] = []
    for digit in range(10):
        mask = labels == digit
        _require(int(mask.sum()) == 500, f"DEV class count changed for {digit}")
        recalls.append(float(correct[mask].to(torch.float64).mean()))
    return {
        "accuracy": float(correct.to(torch.float64).mean()),
        "macro_recall": float(sum(recalls) / 10.0),
        "per_class_recall": recalls,
        "rows": int(correct.numel()),
    }


def _evaluate_dev(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    development_indices: torch.Tensor,
    references: FrozenBogtReferences,
    ledger: DataReadLedger,
    *,
    step: int,
    include_ec: bool,
) -> dict[str, Any]:
    labels_dev = labels.index_select(0, development_indices)
    smooth_correct: list[torch.Tensor] = []
    independent_correct: list[torch.Tensor] = []
    ec_correct: list[torch.Tensor] = []
    smooth_l = 0.0
    independent_l = 0.0
    rec_sum = 0.0
    op_sum = 0.0
    total_elements = 0
    raw_finite = True
    raw_nonconstant = 0
    raw_zero = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    try:
        with torch.no_grad():
            for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
                indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
                batch = images.index_select(0, indices).to(consumer.device)
                batch_labels = labels.index_select(0, indices).to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                _require(result.image_input is not None and result.ols_input_field is not None, "DEV optical path is incomplete")
                raw = result.raw_intensity
                target = result.image_input.slm_intensity
                _finite(raw, "DEV raw CMOS")
                _finite(target, "DEV target image")
                smooth_cmos, smooth_logits = smooth(_cmos_view(raw))
                independent_cmos, independent_logits = independent(_cmos_view(raw))
                smooth_target, _ = smooth(_cmos_view(batch))
                independent_target, _ = independent(_cmos_view(batch))
                smooth_l += float(_normalized_feature_loss(smooth_cmos, smooth_target, "DEV L_S").cpu()) * int(indices.numel())
                independent_l += float(_normalized_feature_loss(independent_cmos, independent_target, "DEV L_I").cpu()) * int(indices.numel())
                smooth_correct.append((smooth_logits.argmax(dim=1) == batch_labels).cpu())
                independent_correct.append((independent_logits.argmax(dim=1) == batch_labels).cpu())
                if include_ec:
                    _, ec_logits = ec(_cmos_view(raw))
                    ec_correct.append((ec_logits.argmax(dim=1) == batch_labels).cpu())
                rec_sum += float((raw - target).square().sum().cpu())
                powers = contracted_plane_powers(result)
                ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
                op_sum += float(torch.log(ratios).square().sum().cpu())
                flat = raw.to(torch.float64).flatten(1)
                per_sample_std = flat.std(dim=1, unbiased=False)
                raw_finite = raw_finite and bool(torch.isfinite(raw).all().item())
                raw_nonconstant += int((per_sample_std > 0.0).sum().item())
                raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum().item())
                raw_min = min(raw_min, float(raw.min().cpu()))
                raw_max = max(raw_max, float(raw.max().cpu()))
                total_elements += int(raw.numel())
                ledger.record("dev", int(indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    smooth_correct_all = torch.cat(smooth_correct)
    independent_correct_all = torch.cat(independent_correct)
    raw_count = int(development_indices.numel())
    result: dict[str, Any] = {
        "schema": DEV_SCHEMA,
        "step": step,
        "rows": raw_count,
        "role": "DEV diagnostics; E_C is confirmation-only and endpoint-limited",
        "E_S": {"architecture": "SmoothMNISTEvaluator v2.1", **_class_metrics(smooth_correct_all, labels_dev)},
        "E_I": {"architecture": "FrozenMNISTEvaluator v1", **_class_metrics(independent_correct_all, labels_dev)},
        "L_S": smooth_l / raw_count,
        "L_I": independent_l / raw_count,
        "L_rec": rec_sum / float(total_elements),
        "L_op": op_sum / float(raw_count * 5),
        "raw_cmos": {
            "finite": raw_finite,
            "nonconstant_fraction": float(raw_nonconstant / raw_count),
            "zero_image_count": raw_zero,
            "min": raw_min,
            "max": raw_max,
        },
        "E_C_in_backward": False,
        "evaluator_parameter_gradients_none": bool(
            all(parameter.grad is None for parameter in smooth.parameters())
            and all(parameter.grad is None for parameter in independent.parameters())
            and all(parameter.grad is None for parameter in ec.parameters())
        ),
        "_E_S_correct": smooth_correct_all,
        "_E_I_correct": independent_correct_all,
    }
    if include_ec:
        ec_correct_all = torch.cat(ec_correct)
        result["E_C"] = {"architecture": third_confirmation.ThirdConfirmationEvaluator.architecture_kind, **_class_metrics(ec_correct_all, labels_dev)}
        result["_E_C_correct"] = ec_correct_all
    _require(result["raw_cmos"]["finite"] and result["raw_cmos"]["nonconstant_fraction"] == 1.0 and result["raw_cmos"]["zero_image_count"] == 0, f"DEV optical validity failed at step {step}")
    _require(result["evaluator_parameter_gradients_none"], "DEV evaluator received a gradient")
    return result


def _paired_bootstrap(labels: torch.Tensor, step0: torch.Tensor, step300: torch.Tensor) -> dict[str, Any]:
    _require(labels.shape == step0.shape == step300.shape == (5000,), "E_C bootstrap vectors have wrong shape")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    deltas: list[torch.Tensor] = []
    per_class: list[float] = []
    draws: list[torch.Tensor] = []
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 500, f"E_C bootstrap class {digit} count changed")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        draws.append(draw)
        base = step0.index_select(0, matches)[draw]
        candidate = step300.index_select(0, matches)[draw]
        deltas.append((candidate.to(torch.float64) - base.to(torch.float64)).mean(dim=1))
        per_class.append(float(step300.index_select(0, matches).to(torch.float64).mean() - step0.index_select(0, matches).to(torch.float64).mean()))
    macro_delta = torch.stack(deltas).mean(dim=0)
    quantiles = torch.quantile(macro_delta, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "class_stratified_paired": True,
        "delta_macro_recall": float(macro_delta.mean()),
        "ci95": {"lower": float(quantiles[0]), "upper": float(quantiles[1])},
        "per_class_delta": per_class,
        "macro_delta_tensor": macro_delta,
        "draws": draws,
    }


def _p2q2_transfer(
    root: Path,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    phase_authority: Mapping[str, Any],
    target: Path,
    *,
    step: int,
) -> dict[str, Any]:
    latent = _mapping(phase_authority["phase_latent"], "Q2 latent")
    z_sample = _tensor(latent.get("z_sample"), "Q2 z")[:200].contiguous()
    baseline_raw = _tensor(phase_authority["selected_generated_raw"], "accepted Q2 generated raw")
    real_raw = _tensor(phase_authority["real_control_raw"], "accepted real control raw")
    outputs: list[torch.Tensor] = []
    consumer.runtime.eval()
    for start in range(0, 200, LOGICAL_BATCH_SIZE):
        result = consumer.generated_forward(z_sample[start : start + LOGICAL_BATCH_SIZE].to(consumer.device))
        raw = result.raw_intensity.detach().cpu()
        _finite(raw, "P2/Q2 generated raw")
        outputs.append(raw)
    candidate_raw = torch.cat(outputs)
    _require(candidate_raw.shape == (200, 1, 200, 200), "P2/Q2 raw shape changed")
    raw_path = target / f"p2_q2_generated_step_{step:06d}.pt"
    _write_torch(raw_path, {"schema": CMOS_SCHEMA, "candidate_id": "Q2", "selected_step": step, "input_z": z_sample, "raw_cmos_intensity": candidate_raw}, kind=f"P2/Q2 generated CMOS step {step}")
    evaluator_report, _ = canary._evaluate_frozen_semantics(checkpoint=root / TRAIN_EVALUATOR_CHECKPOINT, generated_raw=candidate_raw, real_raw=real_raw)
    baseline_report, _ = canary._evaluate_frozen_semantics(checkpoint=root / TRAIN_EVALUATOR_CHECKPOINT, generated_raw=baseline_raw, real_raw=real_raw)
    real_view = real_raw.to(torch.float64).flatten(1)
    raw_mean = real_view.mean(dim=0)
    raw_std = real_view.std(dim=0, unbiased=False)
    candidate_view = canary._standardize(candidate_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    baseline_view = canary._standardize(baseline_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    real_view = canary._standardize(real_view, raw_mean, raw_std)
    raw_self = canary._energy_distance(real_view[:100], real_view[100:])
    candidate_raw_ed = canary._energy_distance(candidate_view, real_view)
    baseline_raw_ed = canary._energy_distance(baseline_view, real_view)
    expected = _mapping(phase_authority["selected_metrics"], "accepted Q2 metrics")
    baseline = {key: float(expected[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")}
    recomputed_baseline_raw = baseline_raw_ed / max(raw_self, 1.0e-12)
    _require(abs(recomputed_baseline_raw - baseline["R_raw"]) < 1.0e-9, "accepted Q2 R_raw baseline changed")
    _require(abs(float(baseline_report["comparison"]["R_feature"]) - baseline["R_feature"]) < 1.0e-9, "accepted Q2 R_feature baseline changed")
    candidate = {
        "R_u": baseline["R_u"],
        "R_phi": baseline["R_phi"],
        "R_z": baseline["R_z"],
        "R_raw": candidate_raw_ed / max(raw_self, 1.0e-12),
        "R_feature": float(evaluator_report["comparison"]["R_feature"]),
    }
    return {
        "step": step,
        "candidate_id": "Q2",
        "sample_count": 200,
        "latent_bank_sha256": phase_authority["authority_hashes"]["phase_latent"],
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {key: candidate[key] - baseline[key] for key in candidate},
        "R_u": baseline["R_u"],
        "R_z": baseline["R_z"],
        "R_raw": candidate["R_raw"],
        "R_feature": candidate["R_feature"],
        "evaluator": {"candidate": evaluator_report, "baseline": baseline_report},
        "histogram": evaluator_report["generated"]["predicted_class_histogram"],
        "dominant_fraction": evaluator_report["generated"]["predicted_class_dominant_fraction"],
        "entropy_nats": evaluator_report["generated"]["predicted_class_entropy_nats"],
        "raw_validity": canary._raw_cmos_statistics(candidate_raw, name=f"P2/Q2 step {step}"),
        "generated_cmos_artifact": _artifact_ref(root, raw_path, role=f"P2/Q2 generated CMOS at step {step}"),
    }


def _load_canonical_train(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    source = root / TRAINING_SOURCE
    _require(source.is_file(), f"canonical TRAIN source is missing: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise RobustTrainingError("canonical TRAIN source cannot be read") from error
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "canonical TRAIN source schema changed")
    images, labels = payload
    _require(isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor), "canonical TRAIN tensors missing")
    _require(images.shape == (60000, 28, 28) and images.dtype == torch.uint8, "canonical TRAIN image schema changed")
    _require(labels.shape == (60000,) and labels.dtype == torch.long, "canonical TRAIN label schema changed")
    _require(int(images.min()) >= 0 and int(images.max()) <= 255, "canonical TRAIN range changed")
    _require(torch.equal(torch.unique(labels), torch.arange(10, dtype=torch.long)), "canonical TRAIN class set changed")
    return images.unsqueeze(1).to(torch.float32).div_(255.0), labels


def _load_authorities(root: Path) -> dict[str, Any]:
    git_provenance = _bind_git_provenance(git_identity(root))
    status = root / "docs/CURRENT_STATUS.md"
    handoff = root / "docs/status/current_handoff.md"
    status_top = "\n".join(status.read_text(encoding="utf-8").splitlines()[:45]) if status.is_file() else ""
    handoff_top = "\n".join(handoff.read_text(encoding="utf-8").splitlines()[:45]) if handoff.is_file() else ""

    cross_path = root / CROSS_REPORT
    cross_report = json.loads(cross_path.read_text(encoding="utf-8"))
    _require(cross_report.get("result") == "VALID_COMPLETED", "cross-evaluator authority is not valid completed")
    _require(cross_report.get("outer_gate") == cross.OUTER_GATE, "cross-evaluator gate changed")
    _require(cross_report.get("classification") == cross.CLASSIFICATION_SHARED, "cross-evaluator classification changed")

    clean_report_path = root / CLEAN_REPORT
    clean_step0_path = root / CLEAN_STEP_0
    clean_report = json.loads(clean_report_path.read_text(encoding="utf-8"))
    clean_step0 = safe_torch_load(clean_step0_path, map_location="cpu")
    _require(clean_report.get("result") == "VALID_COMPLETED", "clean step-0 report is not valid completed")
    _require(clean_report.get("outer_gate") == conflict.OUTER_GATE, "clean step-0 gate changed")
    _require(clean_report.get("classification") == conflict.NOT_SUPPORTED_CLASSIFICATION, "clean step-0 classification changed")
    _require(isinstance(clean_step0, Mapping) and clean_step0.get("completed_updates") == 0, "clean step-0 checkpoint changed")
    _require(clean_step0.get("trainable_parameter_names") == ["generator.phase"], "clean trainable boundary changed")
    _finite(_tensor(clean_step0.get("generator_state"), "clean generator state"), "clean generator state")
    _finite(_tensor(clean_step0.get("encoder_state"), "clean encoder state"), "clean encoder state")

    cpu_context = phase_repair._load_cpu_context(root)
    training = successor._load_successor_training_authority(root, {"training": cpu_context})
    images, labels = _load_canonical_train(root)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SCHEDULE_SEED, validation_per_label=500)
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    expected_data = _mapping(training["data_identity"], "accepted data identity")
    _require(train_indices.numel() == 55000 and development_indices.numel() == 5000, "TRAIN/DEV counts changed")
    _require(torch.bincount(labels.index_select(0, development_indices), minlength=10).tolist() == [500] * 10, "DEV role is not class-balanced")

    calibration_identity = _mapping(training["weights"].calibration_batch_identity, "calibration identity")
    calibration_indices = torch.tensor(calibration_identity["source_indices"], dtype=torch.long)
    _require(calibration_indices.tolist() == [41252, 57579, 31150, 36488, 36208, 15778, 46903, 14076, 39350, 23349], "calibration batch identity changed")
    _require(labels.index_select(0, calibration_indices).tolist() == list(range(10)), "calibration labels changed")
    _require(bool(torch.all(torch.isin(calibration_indices, train_indices)).item()), "calibration left TRAIN role")

    phase = conflict._load_phase_transfer_authority(root, cpu_context)

    ec_report_path = root / EC_REPORT
    ec_qualification_path = root / EC_QUALIFICATION
    ec_report = json.loads(ec_report_path.read_text(encoding="utf-8"))
    ec_qualification = json.loads(ec_qualification_path.read_text(encoding="utf-8"))
    _require(ec_report.get("result") == "VALID_COMPLETED" and ec_report.get("classification") == third_confirmation.CLASSIFICATION_SUPPORTED, "E_C result changed")
    _require(ec_report.get("outer_gate") == third_confirmation.OUTER_GATE, "E_C gate changed")
    _require(ec_qualification.get("passed") is True and ec_qualification.get("evaluation_calls") == 1, "E_C qualification changed")
    _require(ec_qualification.get("count") == 5500 and ec_qualification.get("split_role") == "E_C_QUALIFICATION", "E_C qualification role changed")
    _require(float(ec_qualification.get("accuracy")) >= 0.985 and float(ec_qualification.get("macro_recall")) >= 0.985 and min(ec_qualification.get("per_class_recall")) >= 0.95, "E_C qualification thresholds changed")

    return {
        "git_provenance": git_provenance,
        "local_state": dict(git_provenance["consumer"]),
        "docs": {
            "CURRENT_STATUS_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(),
            "current_handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest(),
        },
        "cross_report": cross_report,
        "clean_report": clean_report,
        "step0": dict(clean_step0),
        "training": training,
        "cpu_context": cpu_context,
        "phase": phase,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "data_identity": dict(expected_data),
        "calibration_indices": calibration_indices,
        "ec_report": ec_report,
        "ec_qualification": ec_qualification,
        "authority_hashes": {
            "cross_report": EXPECTED_CROSS_REPORT_SHA256,
            "clean_report": EXPECTED_CLEAN_REPORT_SHA256,
            "clean_step0": EXPECTED_CLEAN_STEP_0_SHA256,
            "E_C_report": EXPECTED_EC_REPORT_SHA256,
            "E_C_checkpoint": EXPECTED_EC_CHECKPOINT_SHA256,
            "E_C_qualification": EXPECTED_EC_QUALIFICATION_SHA256,
            "E_C_split": EXPECTED_EC_SPLIT_SHA256,
            "training_source": TRAINING_SOURCE_SHA256,
        },
        "calibration": None,
    }


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, SmoothMNISTEvaluator, FrozenMNISTEvaluator, third_confirmation.ThirdConfirmationEvaluator]:
    training = _mapping(authorities["training"], "training authority")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    bundle = _mapping(training["bundle"], "generation bundle")
    _require(consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"), "runtime identity differs from clean authority")
    consumer.load_trainable_state_dict({"encoder.phase": authorities["step0"]["encoder_state"], "generator.phase": authorities["step0"]["generator_state"]})
    consumer.bind_references(training["references"], for_scientific=True)
    consumer.bind_weights(training["weights"], for_scientific=True)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()

    smooth, _ = successor._load_smooth_evaluator(root, device)
    independent, independent_payload = load_frozen_evaluator(root / INDEPENDENT_CHECKPOINT, device=device, min_validation_accuracy=0.985)
    _require(independent_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "E_I architecture changed")
    _require(all(not parameter.requires_grad for parameter in smooth.parameters()), "E_S is not frozen")
    _require(all(not parameter.requires_grad for parameter in independent.parameters()), "E_I is not frozen")

    payload = safe_torch_load(root / EC_CHECKPOINT, map_location="cpu")
    payload_map = _mapping(payload, "E_C checkpoint")
    _require(payload_map.get("semantic_authority_id") == third_confirmation.SEMANTIC_AUTHORITY_ID, "E_C semantic authority changed")
    _require(payload_map.get("architecture_kind") == third_confirmation.ThirdConfirmationEvaluator.architecture_kind, "E_C architecture changed")
    ec = third_confirmation.ThirdConfirmationEvaluator().to(device)
    ec.load_state_dict(_mapping(payload_map.get("state_dict"), "E_C state"), strict=True)
    ec.freeze()
    _require(all(not parameter.requires_grad for parameter in ec.parameters()), "E_C is not frozen")
    return consumer, smooth, independent, ec


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"stage1_updates": STAGE_1_UPDATES, "total_updates": TOTAL_UPDATES, "checkpoint_steps": list(CHECKPOINT_STEPS), "beta": BETA, "gamma": GAMMA, "delta_target_gradient_norm": DELTA_TARGET_GRADIENT_NORM, "joint_rule": "equal-weight normalized E_S/E_I; q_joint=-d_joint"},
        model_identity="LIGHTGEN_2D_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1",
        optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "state": "clean step-0 generator moments restored; no Stage-2 reset"},
        sampler_identity={"seed": SCHEDULE_SEED, "batch_size": LOGICAL_BATCH_SIZE, "total_updates": TOTAL_UPDATES, "stage2": "continue same permutations without reshuffle"},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def _checkpoint_payload(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Adam,
    sampler: LongHorizonSampler,
    history: list[Mapping[str, Any]],
    composition_trajectory: Mapping[str, Any],
    *,
    step: int,
    delta_calibration: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    run_identity: ResearchRunIdentity,
    optimizer_mode: str,
    stage1_condition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_optimizer(consumer, optimizer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_ids": [AUTHORIZATION_ID, CONDITIONAL_AUTHORIZATION_ID],
        "completed_updates": step,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "checkpoint_selection": "none; every preregistered scientific checkpoint persisted",
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "starting_authority": {"path": CLEAN_STEP_0.as_posix(), "sha256": EXPECTED_CLEAN_STEP_0_SHA256, "selected_step": 0},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "P2", "Q2", "E_S", "E_I", "E_C"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_runtime_state(consumer)),
        "optimizer_mode": optimizer_mode,
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "delta_joint_calibration": dict(delta_calibration),
        "stage1_continuation_condition": None if stage1_condition is None else dict(stage1_condition),
        "gradient_contract": {"g_sem": "delta_joint*q_joint", "g_rec_w": "beta*g_rec", "g_op_w": "gamma*g_op", "g_total": "g_sem+g_rec_safe+g_op_safe", "projection": "remove only negative dot-product components relative to g_sem", "eps_proj": EPS_PROJ},
        "training_history": [dict(record) for record in history],
        "composition_trajectory": dict(composition_trajectory),
    }


def _stage1_condition(
    dev0: Mapping[str, Any],
    dev300: Mapping[str, Any],
    e_c_bootstrap: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    optical: Mapping[str, Any],
    *,
    joint_degenerate: bool = False,
) -> tuple[bool, str, dict[str, bool]]:
    train_s = bool(float(dev300["L_S"]) < float(dev0["L_S"]))
    train_i = bool(float(dev300["L_I"]) < float(dev0["L_I"]))
    ec_delta = float(e_c_bootstrap["delta_macro_recall"])
    ec_ci = float(e_c_bootstrap["ci95"]["lower"])
    ec_supported = bool(ec_delta > 0.0 and ec_ci > 0.0)
    rec_pass = bool(reconstruction["delta_step_300_minus_step_0"] <= RECONSTRUCTION_TOLERANCE)
    optical_pass = bool(optical["all_finite"] and optical["all_nonconstant"] and optical["all_zero_image_counts_zero"] and optical["all_power_checks_finite"])
    checks = {"joint_direction_never_degenerate": not joint_degenerate, "L_S_improved": train_s, "L_I_improved": train_i, "E_C_delta_positive": ec_delta > 0.0, "E_C_CI_lower_positive": ec_ci > 0.0, "reconstruction_constraint": rec_pass, "optical_validity": optical_pass, "no_protected_test_sealed_violation": True, "no_authority_counterfactual_violation": True}
    if all(checks.values()):
        return True, CLASSIFICATION_SUPPORTED, checks
    if joint_degenerate:
        return False, CLASSIFICATION_STAGE1_DEGENERATED, checks
    if train_s and train_i and not ec_supported:
        return False, CLASSIFICATION_STAGE1_NO_EC, checks
    if ec_supported and not rec_pass:
        return False, CLASSIFICATION_STAGE1_RECON, checks
    return False, CLASSIFICATION_OPTIMIZATION, checks


def _long_classification(
    stage1_passed: bool,
    stage1_dev: Mapping[str, Any],
    final_dev: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    joint_degenerate: bool,
) -> str:
    if joint_degenerate:
        return CLASSIFICATION_DEGENERATED
    if float(final_dev["L_S"]) >= float(stage1_dev["L_S"]) or float(final_dev["L_I"]) >= float(stage1_dev["L_I"]):
        return CLASSIFICATION_SHORT_HORIZON
    if float(reconstruction["delta_step_3000_minus_step_0"]) > RECONSTRUCTION_TOLERANCE:
        return CLASSIFICATION_RECON_TRADEOFF
    return CLASSIFICATION_SUPPORTED if stage1_passed else CLASSIFICATION_OPTIMIZATION


def _prepare_cuda_runtime() -> dict[str, Any]:
    return successor._prepare_cuda_runtime_dll_search_path()


def run_training(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")
    current_git = git_identity(root)
    git_provenance = _bind_git_provenance(current_git)
    start_git = current_git

    authorities = _load_authorities(root)
    authorities["git_provenance"] = git_provenance
    cuda_runtime = _prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal evaluator-robust training requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, ec = _load_models(root, authorities, device)
    calibration_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(device)
    delta_calibration = _calibrate_delta_joint(consumer, smooth, independent, calibration_batch, authorities["training"]["references"])
    authorities["calibration"] = delta_calibration
    preflight = _run_preflight(consumer, smooth, independent, ec, authorities, delta_joint=float(delta_calibration["delta_joint"]))
    _require(preflight["status"] == "READY", "preflight did not reach READY")

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(cross.__file__).resolve(),
            Path(conflict.__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(canonical_training.__file__).resolve(),
            Path(third_confirmation.__file__).resolve(),
            Path(producer.__file__).resolve(),
            Path(canary.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
            root / TRAINING_SOURCE,
            root / EC_CHECKPOINT,
        ),
        config_identity={
            "authorization_token": AUTHORIZATION_TOKEN,
            "conditional_authorization_token": CONDITIONAL_AUTHORIZATION_TOKEN,
            "outer_gate": OUTER_GATE,
            "stage1_updates": STAGE_1_UPDATES,
            "total_updates": TOTAL_UPDATES,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "beta": BETA,
            "gamma": GAMMA,
            "delta_target_gradient_norm": DELTA_TARGET_GRADIENT_NORM,
            "delta_joint": delta_calibration["delta_joint"],
            "authority_hashes": authorities["authority_hashes"],
        },
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, authorities["data_identity"])
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(SCIENTIFIC_ATTEMPT_ID)
    _write_json(target / "source_manifest.json", source_manifest, kind="evaluator-robust source manifest")
    _write_json(target / "preflight.json", preflight, kind="evaluator-robust preflight")
    _write_json(target / "execution_manifest.json", {"schema": "evaluator_robust_semantic_training_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorizations": [{"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN}, {"id": CONDITIONAL_AUTHORIZATION_ID, "token": CONDITIONAL_AUTHORIZATION_TOKEN}], "input_authorities": authorities["authority_hashes"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "local_state": start_git, "cuda_runtime": cuda_runtime, "scope": {"stage1_updates": STAGE_1_UPDATES, "total_updates": TOTAL_UPDATES, "trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "P2", "Q2", "E_S", "E_I", "E_C"], "protected": False, "test": False, "sealed": False}}, kind="evaluator-robust execution manifest")

    optimizer, optimizer_mode = _restore_clean_optimizer(consumer, authorities["step0"], authorities["clean_report"])
    sampler = LongHorizonSampler(authorities["train_indices"], authorities["labels"])
    first_indices, first_labels = sampler.next_batch()
    _require(first_indices.tolist() == authorities["calibration_indices"].tolist(), "scientific first TRAIN batch differs from authority")
    _require(torch.equal(authorities["labels"].index_select(0, first_indices), first_labels), "scientific first TRAIN labels changed")
    sampler.cursor = 0
    ledger = DataReadLedger()
    history: list[dict[str, Any]] = []
    composition_trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    p2q2_records: dict[str, dict[str, Any]] = {}
    clean_generator_state = authorities["step0"]["generator_state"].to(device)

    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)
    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(device)
    composition0 = _compose_gradients(consumer, smooth, independent, fixed_batch, authorities["training"]["references"], delta_joint=float(delta_calibration["delta_joint"]))
    composition_trajectory["0"] = _composition_record(composition0, step=0, phase_displacement=0.0, phase_update_norm=0.0)
    _clear_runtime_grads(consumer)
    dev0 = _evaluate_dev(consumer, smooth, independent, ec, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=0, include_ec=True)
    dev_records["0"] = dev0
    _write_json(target / "dev_diagnostics_step_000000.json", {key: value for key, value in dev0.items() if not key.startswith("_")}, kind="step-0 DEV diagnostics")
    _write_torch(target / "scientific_step_000000.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=0, delta_calibration=delta_calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, stage1_condition=None), kind="evaluator-robust step-0 checkpoint")

    last_recovery_step = 0
    last_recovery_time = time.time()

    def recovery_payload(step: int) -> Mapping[str, Any]:
        payload = _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, delta_calibration=delta_calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, stage1_condition=None)
        payload["scientific_artifact"] = False
        payload["operational_artifact"] = True
        payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        return payload

    heartbeat.update(state=RUNNING, phase="TRAINING", step=0, latest_losses={"rule": "E_S/E_I equal-weight normalized joint semantic direction", "delta_joint": delta_calibration["delta_joint"]}, force=True)
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=TOTAL_UPDATES)
    stage1_degenerate = False
    with interrupt:
        for step in range(TOTAL_UPDATES):
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            try:
                trace = _apply_update(consumer, smooth, independent, optimizer, batch, batch_labels.to(device), authorities["training"]["references"], delta_joint=float(delta_calibration["delta_joint"]), step=step + 1, ledger=ledger, clean_generator_state=clean_generator_state)
            except JointDirectionDegenerated:
                stage1_degenerate = True
                raise
            history.append(trace)
            completed = step + 1
            now = time.time()
            if recovery_due(step=completed, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                recovery.save(recovery_payload(completed), step=completed)
                last_recovery_step = completed
                last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=completed, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_rec", "L_op")})
            interrupted = interrupt.handle_safe_boundary(step=completed, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_rec", "L_op")}, last_recovery_checkpoint=None)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": completed, "recovery_checkpoint": interrupted.name}
            if completed in CHECKPOINT_STEPS[1:]:
                composition = _compose_gradients(consumer, smooth, independent, fixed_batch, authorities["training"]["references"], delta_joint=float(delta_calibration["delta_joint"]))
                displacement = float((_generator(consumer).detach() - clean_generator_state).to(torch.float64).square().sum().sqrt().cpu())
                composition_trajectory[str(completed)] = _composition_record(composition, step=completed, phase_displacement=displacement, phase_update_norm=float(history[-1]["phase_update_norm"]))
                _clear_runtime_grads(consumer)
                if completed in (100, 200, 300):
                    if completed == 300:
                        diagnostic = _evaluate_dev(consumer, smooth, independent, ec, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed, include_ec=True)
                        dev_records[str(completed)] = diagnostic
                    _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {key: value for key, value in (dev_records.get(str(completed)) or {"step": completed, "composition": composition_trajectory[str(completed)]}).items() if not key.startswith("_")}, kind=f"step-{completed} diagnostics")
                if completed == 300:
                    _write_torch(target / "scientific_step_000300.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=300, delta_calibration=delta_calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, stage1_condition=None), kind="evaluator-robust stage-1 checkpoint")
                    labels_dev = authorities["labels"].index_select(0, authorities["development_indices"])
                    e_c_bootstrap = _paired_bootstrap(labels_dev, dev_records["0"]["_E_C_correct"], dev_records["300"]["_E_C_correct"])
                    _write_torch(target / "paired_bootstrap_stage1.pt", e_c_bootstrap, kind="E_C Stage-1 paired bootstrap")
                    reconstruction = {"step_0": dev_records["0"]["L_rec"], "step_300": dev_records["300"]["L_rec"], "delta_step_300_minus_step_0": dev_records["300"]["L_rec"] - dev_records["0"]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records["300"]["L_rec"] - dev_records["0"]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
                    optical = {"checkpoints": {step_name: dev_records[step_name]["raw_cmos"] for step_name in ("0", "300")}, "all_finite": all(dev_records[step_name]["raw_cmos"]["finite"] for step_name in ("0", "300")), "all_nonconstant": all(dev_records[step_name]["raw_cmos"]["nonconstant_fraction"] == 1.0 for step_name in ("0", "300")), "all_zero_image_counts_zero": all(dev_records[step_name]["raw_cmos"]["zero_image_count"] == 0 for step_name in ("0", "300")), "all_power_checks_finite": all(math.isfinite(float(dev_records[step_name]["L_op"])) for step_name in ("0", "300"))}
                    stage1_passed, stage1_classification, continuation_checks = _stage1_condition(dev_records["0"], dev_records["300"], e_c_bootstrap, reconstruction, optical, joint_degenerate=stage1_degenerate)
                    heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=300, latest_losses={"stage1_passed": stage1_passed}, force=True)
                    p2q2_records["300"] = _p2q2_transfer(root, consumer, authorities["phase"], target, step=300)
                    if not stage1_passed:
                        break
                elif completed in (600, 1000, 1500, 2000, 2500, 3000):
                    include_ec = completed == 3000
                    diagnostic = _evaluate_dev(consumer, smooth, independent, ec, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed, include_ec=include_ec)
                    dev_records[str(completed)] = diagnostic
                    _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", {key: value for key, value in diagnostic.items() if not key.startswith("_")}, kind=f"step-{completed} DEV diagnostics")
                    _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=completed, delta_calibration=delta_calibration, source_manifest=source_manifest, run_identity=effective_identity, optimizer_mode=optimizer_mode, stage1_condition=continuation_checks), kind=f"evaluator-robust step-{completed} checkpoint")
                    if completed == 3000:
                        p2q2_records["3000"] = _p2q2_transfer(root, consumer, authorities["phase"], target, step=3000)

    _require("0" in dev_records and "300" in dev_records, "Stage-1 endpoints are incomplete")
    labels_dev = authorities["labels"].index_select(0, authorities["development_indices"])
    e_c_bootstrap = _paired_bootstrap(labels_dev, dev_records["0"]["_E_C_correct"], dev_records["300"]["_E_C_correct"])
    reconstruction = {"step_0": dev_records["0"]["L_rec"], "step_300": dev_records["300"]["L_rec"], "delta_step_300_minus_step_0": dev_records["300"]["L_rec"] - dev_records["0"]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records["300"]["L_rec"] - dev_records["0"]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    optical = {"checkpoints": {step_name: dev_records[step_name]["raw_cmos"] for step_name in ("0", "300")}, "all_finite": all(dev_records[step_name]["raw_cmos"]["finite"] for step_name in ("0", "300")), "all_nonconstant": all(dev_records[step_name]["raw_cmos"]["nonconstant_fraction"] == 1.0 for step_name in ("0", "300")), "all_zero_image_counts_zero": all(dev_records[step_name]["raw_cmos"]["zero_image_count"] == 0 for step_name in ("0", "300")), "all_power_checks_finite": all(math.isfinite(float(dev_records[step_name]["L_op"])) for step_name in ("0", "300"))}
    stage1_passed, stage1_classification, continuation_checks = _stage1_condition(dev_records["0"], dev_records["300"], e_c_bootstrap, reconstruction, optical, joint_degenerate=stage1_degenerate)
    stage2_executed = stage1_passed and len(history) >= TOTAL_UPDATES
    final_dev = dev_records.get("3000")
    long_reconstruction = None
    final_classification = stage1_classification
    long_checks: dict[str, Any] = {}
    if stage2_executed and final_dev is not None:
        long_reconstruction = {"step_0": dev_records["0"]["L_rec"], "step_300": dev_records["300"]["L_rec"], "step_3000": final_dev["L_rec"], "delta_step_3000_minus_step_0": final_dev["L_rec"] - dev_records["0"]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": final_dev["L_rec"] - dev_records["0"]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
        final_classification = _long_classification(stage1_passed, dev_records["300"], final_dev, long_reconstruction, joint_degenerate=stage1_degenerate)
        long_checks = {"joint_direction_stable": not stage1_degenerate, "L_S_stage300_to_3000_preserved": final_dev["L_S"] < dev_records["300"]["L_S"], "L_I_stage300_to_3000_preserved": final_dev["L_I"] < dev_records["300"]["L_I"], "reconstruction_constraint": long_reconstruction["passed"], "optical_validity": final_dev["raw_cmos"]["finite"] and final_dev["raw_cmos"]["nonconstant_fraction"] == 1.0 and final_dev["raw_cmos"]["zero_image_count"] == 0, "no_protected_test_sealed_violation": True, "no_authority_counterfactual_violation": True}
    elif not stage1_passed:
        final_classification = stage1_classification

    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=len(history), latest_losses={"classification": final_classification}, force=True)
    public_dev = {step: {key: value for key, value in value.items() if not key.startswith("_")} for step, value in dev_records.items()}
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(stage1_passed),
        "classification": final_classification,
        "authorizations_consumed": [{"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "Stage 1 300-update experiment"}, {"id": CONDITIONAL_AUTHORIZATION_ID, "token": CONDITIONAL_AUTHORIZATION_TOKEN, "scope": "preauthorized conditional continuation to total 3000"}],
        "local_state": {"branch": start_git["branch"], "head": start_git["head"], "dirty_before": start_git["worktree_dirty"], "worktree_preserved": True},
        "starting_authority": {"cross_evaluator_report": _artifact_ref(root, root / CROSS_REPORT, role="accepted cross-evaluator semantic divergence authority"), "clean_report": _artifact_ref(root, root / CLEAN_REPORT, role="accepted clean 2L Generator report"), "clean_step0": _artifact_ref(root, root / CLEAN_STEP_0, role="accepted clean 2L Generator step-0"), "not_started_from": ["teacher-alignment step-300", "Smooth-guided step-300", "conflict-aware step-300", "3L step-300", "nonlinear step-300"]},
        "evaluators": {"E_S": {"role": "frozen development semantic objective", "architecture": "SmoothMNISTEvaluator v2.1", "checkpoint": SMOOTH_CHECKPOINT.as_posix(), "checkpoint_sha256": EXPECTED_SMOOTH_CHECKPOINT_SHA256, "feature_layer": "embedding output immediately before classifier", "preprocessing": cross.SMOOTH_PREPROCESSING}, "E_I": {"role": "frozen development semantic objective", "architecture": "FrozenMNISTEvaluator v1", "checkpoint": INDEPENDENT_CHECKPOINT.as_posix(), "checkpoint_sha256": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, "feature_layer": "embedding output immediately before classifier", "preprocessing": cross.INDEPENDENT_PREPROCESSING}, "E_C": {"role": "primary Stage-1 independent confirmation; descriptive only in Stage 2", "architecture": third_confirmation.ThirdConfirmationEvaluator.architecture_kind, "checkpoint": EC_CHECKPOINT.as_posix(), "checkpoint_sha256": EXPECTED_EC_CHECKPOINT_SHA256, "qualification": _artifact_ref(root, root / EC_QUALIFICATION, role="accepted E_C qualification"), "qualification_accuracy": authorities["ec_qualification"]["accuracy"], "qualification_macro_recall": authorities["ec_qualification"]["macro_recall"], "qualification_min_class_recall": min(authorities["ec_qualification"]["per_class_recall"]) }},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "P2", "Q2", "E_S", "E_I", "E_C"], "evaluator_parameter_updates": 0},
        "data_identity": authorities["data_identity"],
        "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "joint_gradient_contract": {"g_S": "grad_generator(L_S)", "g_I": "grad_generator(L_I)", "u_S": "g_S/||g_S||_2", "u_I": "g_I/||g_I||_2", "v_joint": "u_S+u_I", "d_joint": "-v_joint/RMS(v_joint)", "q_joint": "-d_joint", "g_sem": "delta_joint*q_joint", "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "target_features_stop_gradient": True, "E_C_in_backward": False},
        "delta_joint": delta_calibration,
        "auxiliary_gradient_contract": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "g_rec_w": "beta*g_rec", "g_op_w": "gamma*g_op", "g_total": "g_sem+g_rec_safe+g_op_safe", "projection": "remove only component opposing protected g_sem"},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "clean_step0_state_restored": True, "stage2_state_reset": False, "scheduler_change": False},
        "preflight": preflight,
        "stage1": {"updates": min(len(history), STAGE_1_UPDATES), "checkpoints": [0, 100, 200, 300], "trajectory": {step: composition_trajectory[step] for step in ("0", "100", "200", "300") if step in composition_trajectory}, "development_semantics": {step: public_dev[step] for step in ("0", "300") if step in public_dev}, "independent_confirmation": {"E_C_step0": public_dev["0"]["E_C"], "E_C_step300": public_dev["300"]["E_C"], "delta_macro_recall": e_c_bootstrap["delta_macro_recall"], "paired_bootstrap_ci": e_c_bootstrap["ci95"], "per_class_delta": e_c_bootstrap["per_class_delta"], "artifact": _artifact_ref(root, target / "paired_bootstrap_stage1.pt", role="E_C Stage-1 paired bootstrap")}, "reconstruction": reconstruction, "optical_validity": optical, "P2_Q2_transfer": p2q2_records.get("300"), "continuation_condition": {"passed": stage1_passed, "checks": continuation_checks, "classification_if_stop": stage1_classification}},
        "stage2": {"executed": stage2_executed, "additional_updates": max(0, len(history) - STAGE_1_UPDATES), "total_updates": len(history), "checkpoints": [300, 600, 1000, 1500, 2000, 2500, 3000], "trajectory": {step: composition_trajectory[step] for step in ("300", "600", "1000", "1500", "2000", "2500", "3000") if step in composition_trajectory}, "development_semantics": {step: public_dev[step] for step in ("600", "1000", "1500", "2000", "2500", "3000") if step in public_dev}, "E_C_descriptive_transfer": None if final_dev is None else {"role": "LONG_HORIZON_DESCRIPTIVE_TRANSFER", "step300_macro_recall": public_dev["300"]["E_C"]["macro_recall"], "step3000_macro_recall": public_dev["3000"]["E_C"]["macro_recall"], "delta_step3000_minus_step300": public_dev["3000"]["E_C"]["macro_recall"] - public_dev["300"]["E_C"]["macro_recall"], "delta_step3000_minus_step0": public_dev["3000"]["E_C"]["macro_recall"] - public_dev["0"]["E_C"]["macro_recall"], "per_class_recall_step3000": public_dev["3000"]["E_C"]["per_class_recall"]}, "reconstruction": long_reconstruction, "P2_Q2_transfer": p2q2_records.get("3000"), "continuation_checks": long_checks},
        "outer_gate_checks": {"stage1_independent_confirmation": stage1_passed, "conditional_long_horizon_executed_only_after_stage1": (not stage2_executed) or stage1_passed, "protected_test_sealed_zero": True},
        "training_trajectory": {"updates": len(history), "checkpoint_steps_persisted": [int(step) for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()], "trace": history},
        "p2_q2_transfer": p2q2_records,
        "claim_boundary": {"stage1_independent_confirmation": stage1_passed, "stage2_is_scaling_study": stage2_executed, "final_semantic_recovery_claim": False, "new_confirmation_required_before_promotion": True, "architecture_acceptance": False, "promotion": False, "physical_hardware_validity": False},
        "counterfactual_integrity": {"same_clean_step0": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_data_order_through_stage2": True, "same_optimizer": True, "same_delta_joint": True, "same_beta_gamma": True, "same_evaluators": True, "E_C_in_backward": False, "no_hyperparameter_search": True, "no_protected_test_sealed": True},
        "files_changed": {"repository_source": ["scripts/research/lightgen/execute_lightgen_2d_evaluator_robust_semantic_training_v1.py"], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "stage1_checkpoints": [0, 100, 200, 300], "stage2_checkpoints": [300, 600, 1000, 1500, 2000, 2500, 3000], "E_C_stage1_endpoints": [0, 300], "E_C_stage2_endpoint": None if not stage2_executed else 3000, "frozen_gradient_firewall": True, "no_protected_test_sealed": True},
        "unresolved": ["Stage-2 final results require a new independent or sealed confirmation authority before any formal promotion or final semantic-recovery claim."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="evaluator-robust semantic training report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "conditional_authorization": CONDITIONAL_AUTHORIZATION_TOKEN, "stage1_updates": STAGE_1_UPDATES, "total_updates": TOTAL_UPDATES, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": "authorization token mismatch"}, ensure_ascii=False, indent=2))
        return 2
    try:
        report = run_training(output_dir=args.output_dir)
    except JointDirectionDegenerated as error:
        print(json.dumps({"status": "VALID_FAILED", "task_id": TASK_ID, "classification": CLASSIFICATION_STAGE1_DEGENERATED, "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    except (RobustTrainingError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "outer_gate_passed": report["outer_gate_passed"], "stage2_executed": report["stage2"]["executed"], "updates": report["training_trajectory"]["updates"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
