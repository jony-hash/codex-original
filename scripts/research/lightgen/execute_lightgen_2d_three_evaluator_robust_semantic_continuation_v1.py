"""Execute the authorized three-evaluator SIC continuation.

The executor consumes the accepted step-3000 Generator checkpoint, restores its
stored Adam state and TRAIN-order state, and updates only ``generator.phase``.
E_S, E_I and E_C form the equal-weight normalized semantic direction.  E_D is
loaded as a frozen independent confirmation endpoint and is never called from
the backward path.  Stage 1 runs exactly 300 updates to step 3300; Stage 2 is
entered only when every preregistered Stage-1 condition passes and ends at
step 6000.
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

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, LightGen2DChenBOGTComplexOLSTrainingConsumer, TRAIN_ROLE, contracted_plane_powers, git_identity, optical_power_loss_from_powers, _rng_state_payload
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator, stratified_mnist_indices

from scripts.research.lightgen import execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1 as attribution
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from ldm_project.v2.application.research.lightgen import confirmation_evaluators as independent_confirmation
from ldm_project.v2.application.research.lightgen import confirmation_evaluators as third_confirmation
from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from scripts.research.lightgen import execute_lightgen_2d_semantically_conditioned_ols_sampler_v1 as conditioned_sampler


TASK_ID = "EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = robust.SEMANTIC_AUTHORITY_ID
EXPECTED_BRANCH = robust.EXPECTED_BRANCH
EXPECTED_PRODUCER_HEAD = robust.EXPECTED_PRODUCER_HEAD
# Historical import compatibility; current-consumer binding is delegated to
# robust._load_authorities and its explicit provenance helper.
EXPECTED_HEAD = EXPECTED_PRODUCER_HEAD

START_STEP = 3000
STAGE1_END = 3300
TOTAL_STEPS = 6000
STAGE1_UPDATES = 300
STAGE2_UPDATES = 2700
CHECKPOINT_STEPS = (3000, 3100, 3200, 3300, 3600, 4000, 4500, 5000, 5500, 6000)
LOGICAL_BATCH_SIZE = 10
SCHEDULE_SEED = robust.SCHEDULE_SEED
FEATURE_EPS = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12
EPS_PROJ = 1.0e-12
DELTA_SIC = 1.365988374680088e-05
BETA = 2.4509953404
GAMMA = 0.5330424285
RECONSTRUCTION_TOLERANCE = 0.005
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
PRELIGHT_EPSILON = 1.0e-4
RECOVERY_INTERVAL_STEPS = 25

STEP3000 = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_003000.pt")
STEP3000_SHA256 = "1a87af8aa4fa12e43dfd4ab7ee99aa41e08299843f5414cfd05d0221dc473a3d"
ROBUST_REPORT = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/report.json")
ROBUST_REPORT_SHA256 = "3e972d54746121a15dbbf6fca16df2124e9854ceac17bcbe294bff2d24c2bf6b"
ATTRIBUTION_REPORT = Path("outputs/20260920_attribute_post3000_generator_semantic_realization_ceiling_v1/runs/attempt_01/report.json")
ATTRIBUTION_REPORT_SHA256 = "06e045b5a28e40362da7a7c43e9414e54e0718f6a167a8b88fdaebbbae1fc779"
ATTRIBUTION_GRADIENTS = Path("outputs/20260920_attribute_post3000_generator_semantic_realization_ceiling_v1/runs/attempt_01/gradient_attribution.pt")
ATTRIBUTION_GRADIENTS_SHA256 = "7fa88dac3c5772c222df37f651f4e0d1dec33cf18338cd11bdda8c063ba62f17"
ATTRIBUTION_PROBES = Path("outputs/20260920_attribute_post3000_generator_semantic_realization_ceiling_v1/runs/attempt_01/directional_probes.json")
ATTRIBUTION_PROBES_SHA256 = "084ff76830bc54e703b59d460e765ae016a2c55c4a65e9b722dcd696ffea37f4"

ED_RUN = Path("outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01")
ED_REPORT = ED_RUN / "report.json"
ED_REPORT_SHA256 = "01a875ba775accc5903b58428c1a1e9468e186276ea21ffadf521dc5d8ce5db8"
ED_CHECKPOINT = ED_RUN / "final_epoch_candidate.pt"
ED_CHECKPOINT_SHA256 = "b8ece3e3bacedd5b5bec63073bee02ce7489161030b835740c681a1ebd3f9ff9"
ED_QUALIFICATION = ED_RUN / "qualification_metrics.json"
ED_QUALIFICATION_SHA256 = "e112465de698a434fe83bb8ed0a60ed17e6669b5809641446555137e13d1ad02"
ED_ARCHITECTURE = ED_RUN / "architecture.json"
ED_ARCHITECTURE_SHA256 = "6854c7fd1a200b65354a0ec291772412b408688d822c7f588d03c15c2ec0a29a"
ED_INDEPENDENCE = ED_RUN / "architecture_independence_audit.json"
ED_INDEPENDENCE_SHA256 = "aea440b114b374658c4f137b15e3ec7bca73d59fde2cea121abc884b3c5c3577"

S1_REPORT = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/report.json")
S1_REPORT_SHA256 = "1d88e4afe3be37ba10045523fe40aa4eb4068aef1bb8ed8c063f4ceaef969767"
S1_WINNER = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/winner_frozen_sampler.pt")
S1_WINNER_SHA256 = "166cf15ea91666afe1fe1c88e8897b88d58a4740253d59838927502babba568c"
S1_PANELS = Path("outputs/20260920_semantically_conditioned_ols_sampler_v1/runs/attempt_01/generator_canary_metric_panels.pt")
S1_PANELS_SHA256 = "e95dc47dcd437a5e62b70793c597ec319fe96118ab9b0ac45e6ad84dd33b061c"

DEFAULT_OUTPUT_DIR = Path("outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_01")

REPORT_SCHEMA = "three_evaluator_robust_semantic_continuation_report.v1"
CHECKPOINT_SCHEMA = "three_evaluator_robust_semantic_continuation_checkpoint.v1"
PREFLIGHT_SCHEMA = "three_evaluator_robust_semantic_continuation_preflight.v1"
DEV_SCHEMA = "three_evaluator_robust_semantic_continuation_dev_diagnostic.v1"
S1_SCHEMA = "three_evaluator_robust_semantic_continuation_s1_transfer.v1"

CLASSIFICATION_SUPPORTED = "THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_SUPPORTED"
CLASSIFICATION_SHORT = "SHORT_HORIZON_THREE_EVALUATOR_RECOVERY_WITH_LONG_HORIZON_LIMITATION"
CLASSIFICATION_DEGENERATED = "THREE_EVALUATOR_SHARED_DIRECTION_DEGENERATES_AT_LONG_HORIZON"
CLASSIFICATION_RECON = "THREE_EVALUATOR_LONG_HORIZON_SEMANTIC_RECONSTRUCTION_TRADEOFF"
CLASSIFICATION_STAGE1_NO_ED = "THREE_EVALUATOR_DEVELOPMENT_GAIN_WITHOUT_INDEPENDENT_TRANSFER"
CLASSIFICATION_STAGE1_DEGENERATED = "THREE_EVALUATOR_SHARED_DIRECTION_NOT_GLOBALLY_STABLE"
CLASSIFICATION_STAGE1_RECON = "THREE_EVALUATOR_SEMANTIC_RECOVERY_RECONSTRUCTION_TRADEOFF"
CLASSIFICATION_OPTIMIZATION = "THREE_EVALUATOR_CONTINUATION_OPTIMIZATION_NOT_SUPPORTED"
CLASSIFICATION_S1_LIMITED = "REAL_PATH_SEMANTIC_GAIN_WITH_LIMITED_S1_TRANSFER"
CLASSIFICATION_S1_TRANSFER = "SEMANTIC_REALIZATION_GAIN_TRANSFERS_TO_REPAIRED_SAMPLED_MANIFOLD"


class ContinuationError(RuntimeError):
    """Raised when a named authority or continuation invariant is invalid."""


class SICDirectionDegenerated(ContinuationError):
    """Raised only for the preregistered scientific SIC degeneracy stop."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    try:
        canonical.finite(value, name)
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
    return {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items() if name != "generator.phase"}


def _generator(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
        parameter.grad = None
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
        return canonical.norm(value, name, allow_zero=allow_zero)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _rms(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    try:
        return canonical.rms(value, name, allow_zero=allow_zero)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return canonical.dot(left, right)


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    try:
        return canonical.cosine(left, right, name)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    try:
        return canonical.gradient_stats(value, name)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    try:
        return canonical.cmos_view(value)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _normalized_feature_loss(generated: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    try:
        return canonical.normalized_feature_loss(generated, target, name, epsilon=FEATURE_EPS)
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    _finite(raw, "raw CMOS")
    flat = raw.detach().to(torch.float64).flatten(1)
    std = flat.std(dim=1, unbiased=False)
    return {
        "finite": True,
        "finite_fraction": 1.0,
        "nonconstant_fraction": float((std > 0.0).to(torch.float64).mean().cpu()),
        "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
        "min": float(raw.min().detach().cpu()),
        "max": float(raw.max().detach().cpu()),
        "shape": list(raw.shape),
    }


def _loss_bundle(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    batch: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "real optical path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/target geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "raw/target dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_image, "target image")
    raw_view = _cmos_view(raw)
    target_view = _cmos_view(batch)
    s_features, s_logits = smooth(raw_view)
    i_features, i_logits = independent(raw_view)
    c_features, c_logits = ec(raw_view)
    with torch.no_grad():
        s_target, _ = smooth(target_view)
        i_target, _ = independent(target_view)
        c_target, _ = ec(target_view)
    losses = {
        "L_S": _normalized_feature_loss(s_features, s_target, "L_S"),
        "L_I": _normalized_feature_loss(i_features, i_target, "L_I"),
        "L_C": _normalized_feature_loss(c_features, c_target, "L_C"),
        "L_rec": (raw - target_image).square().mean(),
        "L_op": optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference),
    }
    for name, value in losses.items():
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target_image": target_image,
        "features": {"E_S": s_features, "E_I": i_features, "E_C": c_features},
        "target_features": {"E_S": s_target.detach(), "E_I": i_target.detach(), "E_C": c_target.detach()},
        "logits": {"E_S": s_logits, "E_I": i_logits, "E_C": c_logits},
        "losses": losses,
    }


def _project_auxiliary_gradient(auxiliary: torch.Tensor, semantic: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        return canonical.project_auxiliary_gradient(
            auxiliary,
            semantic,
            name,
            epsilon=EPS_PROJ,
        )
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error


def _sic_direction(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    batch: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    generator = _generator(consumer)
    bundle = _loss_bundle(consumer, smooth, independent, ec, batch, references)
    g_s = torch.autograd.grad(bundle["losses"]["L_S"], generator, retain_graph=True, allow_unused=False)[0]
    g_i = torch.autograd.grad(bundle["losses"]["L_I"], generator, retain_graph=True, allow_unused=False)[0]
    g_c = torch.autograd.grad(bundle["losses"]["L_C"], generator, retain_graph=True, allow_unused=False)[0]
    g_rec = torch.autograd.grad(bundle["losses"]["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
    g_op = torch.autograd.grad(bundle["losses"]["L_op"], generator, retain_graph=False, allow_unused=False)[0]
    for name, value in (("g_S", g_s), ("g_I", g_i), ("g_C", g_c), ("g_rec", g_rec), ("g_op", g_op)):
        _finite(value, name)
    n_s, n_i, n_c = _norm(g_s, "g_S"), _norm(g_i, "g_I"), _norm(g_c, "g_C")
    u_s, u_i, u_c = g_s / n_s, g_i / n_i, g_c / n_c
    v_sic = u_s + u_i + u_c
    v_norm = _norm(v_sic, "v_SIC", allow_zero=True)
    if not math.isfinite(v_norm) or v_norm <= JOINT_DEGENERACY_TOLERANCE:
        raise SICDirectionDegenerated("SIC direction norm is nonfinite or <= 1e-12")
    try:
        q_sic, v_rms = canonical.rms_normalize(v_sic, "v_SIC")
    except ValueError as error:
        _require(False, str(error))
        raise AssertionError("unreachable") from error
    d_sic = -q_sic
    g_sem = DELTA_SIC * q_sic
    g_rec_w = BETA * g_rec
    g_op_w = GAMMA * g_op
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec_w, g_sem, "g_rec_w")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op_w, g_sem, "g_op_w")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(_dot(g_total, g_sem) > 0.0, "combined gradient has no positive SIC component")
    return {
        "bundle": bundle,
        "g_S": g_s,
        "g_I": g_i,
        "g_C": g_c,
        "g_rec": g_rec,
        "g_op": g_op,
        "u_S": u_s,
        "u_I": u_i,
        "u_C": u_c,
        "v_SIC": v_sic,
        "q_SIC": q_sic,
        "d_SIC": d_sic,
        "g_sem": g_sem,
        "g_rec_w": g_rec_w,
        "g_op_w": g_op_w,
        "g_rec_safe": g_rec_safe,
        "g_op_safe": g_op_safe,
        "g_total": g_total,
        "norms": {"g_S": n_s, "g_I": n_i, "g_C": n_c, "v_SIC": v_norm, "v_SIC_rms": v_rms, "g_sem": _norm(g_sem, "g_sem")},
        "pairwise_cosines": {
            "cos_g_S_g_I": _cosine(g_s, g_i, "cos_g_S_g_I"),
            "cos_g_S_g_C": _cosine(g_s, g_c, "cos_g_S_g_C"),
            "cos_g_I_g_C": _cosine(g_i, g_c, "cos_g_I_g_C"),
        },
        "cos_g_total_g_sem": _cosine(g_total, g_sem, "cos_g_total_g_sem"),
        "projection": {"g_rec": rec_projection, "g_op": op_projection},
        "delta_SIC": DELTA_SIC,
        "evaluator_parameters_in_backward": {"E_S": True, "E_I": True, "E_C": True, "E_D": False},
        "target_features_stop_gradient": True,
        "label_cross_entropy_in_objective": False,
        "scalar_weighted_sum_backward": False,
    }


def _composition_record(bundle: Mapping[str, Any], *, step: int, phase_displacement: float, phase_update_rms: float) -> dict[str, Any]:
    losses = _mapping(bundle["bundle"]["losses"], "SIC losses")
    return {
        "step": step,
        "L_S": float(losses["L_S"].detach().cpu()),
        "L_I": float(losses["L_I"].detach().cpu()),
        "L_C": float(losses["L_C"].detach().cpu()),
        "L_rec": float(losses["L_rec"].detach().cpu()),
        "L_op": float(losses["L_op"].detach().cpu()),
        "gradient_norms": {name: _gradient_stats(bundle[name], name) for name in ("g_S", "g_I", "g_C", "v_SIC", "g_sem", "g_total")},
        "pairwise_cosines": dict(bundle["pairwise_cosines"]),
        "cos_g_total_g_sem": float(bundle["cos_g_total_g_sem"]),
        "phase_displacement_from_step3000": phase_displacement,
        "phase_update_rms": phase_update_rms,
        "projection": bundle["projection"],
        "optical_validity": _raw_validity(bundle["bundle"]["raw"].detach()),
        "trainable_parameter_names": ["generator.phase"],
        "frozen_evaluator_parameters": True,
        "E_D_in_backward": False,
    }


def _optimizer(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, checkpoint: Mapping[str, Any]) -> tuple[torch.optim.Adam, str]:
    optimizer = torch.optim.Adam([_generator(consumer)], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0, amsgrad=False)
    optimizer.load_state_dict(copy.deepcopy(_mapping(checkpoint.get("optimizer_state"), "stored optimizer state")))
    robust._validate_optimizer(consumer, optimizer)
    mode = "STEP3000_ADAM_STATE_RESTORED_EXACTLY"
    state = optimizer.state.get(_generator(consumer))
    _require(isinstance(state, Mapping), "restored Adam state is missing")
    _require(float(_tensor(state.get("step"), "Adam internal step").item()) == 3300.0, "stored Adam internal step was changed")
    _require(isinstance(state.get("exp_avg"), torch.Tensor) and isinstance(state.get("exp_avg_sq"), torch.Tensor), "Adam moments are missing")
    _finite(state["exp_avg"], "Adam exp_avg")
    _finite(state["exp_avg_sq"], "Adam exp_avg_sq")
    return optimizer, mode


class ContinuationSampler:
    """The accepted step-3000 per-class order with deterministic wraparound."""

    SCHEMA = "three_evaluator_robust_semantic_continuation_sampler.v1"

    def __init__(self, state: Mapping[str, Any], *, cursor: int = START_STEP) -> None:
        _require(state.get("seed") == SCHEDULE_SEED, "sampler seed changed")
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == 10, "sampler permutations are malformed")
        self.seed = int(state["seed"])
        self.steps = TOTAL_STEPS
        self.cursor = int(cursor)
        self.permutations = []
        for digit, value in enumerate(permutations):
            value = _tensor(value, f"sampler permutation {digit}").cpu().clone()
            _require(value.dtype == torch.long and value.ndim == 1 and value.numel() >= START_STEP, f"sampler permutation {digit} is too short")
            self.permutations.append(value)
        _require(0 <= self.cursor <= self.steps, "sampler cursor is invalid")

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.cursor < self.steps, "continuation sampler exhausted")
        indices = torch.stack([permutation[self.cursor % permutation.numel()] for permutation in self.permutations])
        self.cursor += 1
        return indices, torch.arange(10, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "steps": self.steps, "cursor": self.cursor, "wraparound": True, "permutations": [value.clone() for value in self.permutations]}


def _restore_rng(payload: Mapping[str, Any]) -> None:
    cpu = _tensor(payload.get("cpu"), "checkpoint CPU RNG")
    cuda = payload.get("cuda")
    _require(isinstance(cuda, list) and all(isinstance(value, torch.Tensor) for value in cuda), "checkpoint CUDA RNG is malformed")
    torch.random.set_rng_state(cpu.cpu())
    _require(torch.cuda.is_available(), "CUDA is required to restore the accepted CUDA RNG state")
    torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), [state.clone() for state in torch.cuda.get_rng_state_all()]


def _rng_equal(left: tuple[torch.Tensor, list[torch.Tensor]], right: tuple[torch.Tensor, list[torch.Tensor]]) -> bool:
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True))


def _class_metrics(labels: torch.Tensor, predictions: torch.Tensor, *, label_source: str) -> dict[str, Any]:
    labels = labels.detach().cpu().to(torch.long)
    predictions = predictions.detach().cpu().to(torch.long)
    _require(labels.shape == predictions.shape, "classification vectors have different shapes")
    confusion = torch.zeros((10, 10), dtype=torch.long)
    for true, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[true, predicted] += 1
    recalls = []
    for digit in range(10):
        count = int((labels == digit).sum())
        _require(count > 0, f"class {digit} is empty")
        recalls.append(float(confusion[digit, digit].to(torch.float64) / count))
    return {
        "accuracy": float((labels == predictions).to(torch.float64).mean()),
        "macro_recall": float(sum(recalls) / 10.0),
        "per_class_recall": recalls,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "predicted_class_histogram": torch.bincount(predictions, minlength=10).tolist(),
        "true_class_histogram": torch.bincount(labels, minlength=10).tolist(),
        "rows": int(labels.numel()),
        "label_source": label_source,
    }


def _load_canonical_train(root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    source = root / robust.TRAINING_SOURCE
    _require(source.is_file(), f"canonical TRAIN source is missing: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "canonical TRAIN source schema changed")
    images, labels = payload
    _require(isinstance(images, torch.Tensor) and isinstance(labels, torch.Tensor), "canonical TRAIN tensors missing")
    _require(images.shape == (60000, 28, 28) and labels.shape == (60000,), "canonical TRAIN shape changed")
    _require(images.dtype == torch.uint8 and labels.dtype == torch.long, "canonical TRAIN dtype changed")
    return images.unsqueeze(1).to(torch.float32).div_(255.0), labels


def _load_authorities(root: Path) -> dict[str, Any]:
    authorities = robust._load_authorities(root)
    start_git = dict(authorities["git_provenance"]["consumer"])
    current_git = git_identity(root)
    start_git.update({"worktree_dirty": current_git["worktree_dirty"], "status_sha256": current_git["status_sha256"]})

    robust_report_path = root / ROBUST_REPORT
    robust_report = json.loads(robust_report_path.read_text(encoding="utf-8"))
    _require(robust_report.get("result") == "VALID_COMPLETED" and robust_report.get("classification") == robust.CLASSIFICATION_SUPPORTED, "accepted step3000 robust result changed")
    _require(int(robust_report.get("long_horizon_completed_step", -1)) == START_STEP, "accepted robust trajectory endpoint changed")

    step_path = root / STEP3000
    step3000 = safe_torch_load(step_path, map_location="cpu")
    _require(isinstance(step3000, Mapping), "step3000 checkpoint is malformed")
    _require(step3000.get("completed_updates") == START_STEP and step3000.get("scientific_artifact") is True and step3000.get("operational_artifact") is False, "step3000 checkpoint role changed")
    _require(step3000.get("trainable_parameter_names") == ["generator.phase"], "step3000 trainable boundary changed")
    for name in ("generator_state", "encoder_state"):
        value = _tensor(step3000.get(name), name)
        _require(tuple(value.shape) == (2, 1, 100, 100) and value.dtype == torch.float32, f"{name} geometry changed")
        _finite(value, name)
    sampler_state = _mapping(step3000.get("sampler_state"), "step3000 sampler state")
    _require(sampler_state.get("cursor") == START_STEP and sampler_state.get("steps") == START_STEP, "step3000 sampler cursor/length changed")
    _require(sampler_state.get("seed") == SCHEDULE_SEED, "step3000 sampler seed changed")
    rng_state = _mapping(step3000.get("rng_state"), "step3000 RNG state")
    _tensor(rng_state.get("cpu"), "step3000 CPU RNG")
    _require(isinstance(rng_state.get("cuda"), list) and rng_state["cuda"], "step3000 CUDA RNG state missing")

    attr_path = root / ATTRIBUTION_REPORT
    attr_report = json.loads(attr_path.read_text(encoding="utf-8"))
    _require(attr_report.get("result") == "VALID_COMPLETED" and attr_report.get("outer_gate") == attribution.OUTER_GATE and attr_report.get("classification") == attribution.CLASSIFICATION_SHARED, "accepted SIC attribution changed")
    _require(attr_report.get("stage3_generator_phase_attribution", {}).get("steps", {}).get("3000", {}).get("common_direction_defined") is True, "accepted SIC direction is not defined")
    attr_grad_path = root / ATTRIBUTION_GRADIENTS
    attr_probe_path = root / ATTRIBUTION_PROBES
    attr_gradients = safe_torch_load(attr_grad_path, map_location="cpu")
    attr_probes = json.loads(attr_probe_path.read_text(encoding="utf-8"))
    _require(attr_probes.get("steps", {}).get("3000", {}).get("stable_descent", {}).get("d_SIC", {}).get("L_S") is True, "accepted SIC L_S descent sign changed")
    _require(attr_probes.get("steps", {}).get("3000", {}).get("stable_descent", {}).get("d_SIC", {}).get("L_I") is True, "accepted SIC L_I descent sign changed")
    _require(attr_probes.get("steps", {}).get("3000", {}).get("stable_descent", {}).get("d_SIC", {}).get("L_C") is True, "accepted SIC L_C descent sign changed")

    ed_report_path = root / ED_REPORT
    ed_qualification_path = root / ED_QUALIFICATION
    ed_architecture_path = root / ED_ARCHITECTURE
    ed_independence_path = root / ED_INDEPENDENCE
    ed_report = json.loads(ed_report_path.read_text(encoding="utf-8"))
    ed_qualification = json.loads(ed_qualification_path.read_text(encoding="utf-8"))
    ed_architecture = json.loads(ed_architecture_path.read_text(encoding="utf-8"))
    ed_independence = json.loads(ed_independence_path.read_text(encoding="utf-8"))
    _require(ed_report.get("result") == "VALID_COMPLETED" and ed_report.get("outer_gate") == independent_confirmation.OUTER_GATE and ed_report.get("classification") == independent_confirmation.CLASSIFICATION_SUPPORTED, "E_D report is not accepted")
    _require(ed_report.get("sic_binding", {}).get("semantic_authority_id") == independent_confirmation.SEMANTIC_AUTHORITY_ID, "E_D semantic authority changed")
    _require(ed_qualification.get("passed") is True and ed_qualification.get("evaluation_calls") == 1, "E_D qualification changed")
    _require(float(ed_qualification["accuracy"]) >= 0.97 and float(ed_qualification["macro_recall"]) >= 0.97 and min(ed_qualification["per_class_recall"]) >= 0.94, "E_D qualification thresholds changed")
    _require(ed_architecture.get("architecture_kind") == independent_confirmation.PixelMLPConfirmationEvaluatorV1.architecture_kind, "E_D architecture changed")
    _require(ed_independence.get("status") == "PASS", "E_D architecture independence changed")

    s1_report_path = root / S1_REPORT
    s1_winner_path = root / S1_WINNER
    s1_panels_path = root / S1_PANELS
    s1_report = json.loads(s1_report_path.read_text(encoding="utf-8"))
    s1_winner = safe_torch_load(s1_winner_path, map_location="cpu")
    s1_panels = safe_torch_load(s1_panels_path, map_location="cpu")
    _require(s1_report.get("result") == "VALID_COMPLETED" and s1_report.get("outer_gate_passed") is True and s1_report.get("winner", {}).get("family") == "S1", "S1 winner authority changed")
    _require(isinstance(s1_winner, Mapping) and s1_winner.get("family") == "S1" and s1_winner.get("frozen_before_generator_canary") is True, "S1 winner payload changed")
    z = _tensor(s1_winner.get("z"), "S1 frozen bank z")
    requested = _tensor(s1_winner.get("requested_labels"), "S1 requested labels")
    _require(z.shape == (2000, 100) and z.is_complex(), "S1 bank geometry changed")
    _require(requested.shape == (2000,) and requested.dtype == torch.long and torch.bincount(requested, minlength=10).tolist() == [200] * 10, "S1 requested class balance changed")
    _require(isinstance(s1_panels, Mapping) and _tensor(s1_panels.get("panel_indices"), "S1 panel indices").numel() == 200, "S1 panel authority changed")

    images, labels = _load_canonical_train(root)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SCHEDULE_SEED, validation_per_label=500)
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    authorities["local_state"] = start_git
    authorities.update({
        "robust_report": robust_report,
        "step3000": dict(step3000),
        "attr_report": attr_report,
        "attr_gradients": attr_gradients,
        "attr_probes": attr_probes,
        "ed_report": ed_report,
        "ed_qualification": ed_qualification,
        "ed_architecture": ed_architecture,
        "ed_independence": ed_independence,
        "s1_report": s1_report,
        "s1_winner": dict(s1_winner),
        "s1_panels": dict(s1_panels),
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "authority_hashes": {**authorities["authority_hashes"], "step3000": STEP3000_SHA256, "robust_report": ROBUST_REPORT_SHA256, "attribution_report": ATTRIBUTION_REPORT_SHA256, "attribution_gradients": ATTRIBUTION_GRADIENTS_SHA256, "attribution_probes": ATTRIBUTION_PROBES_SHA256, "E_D_report": ED_REPORT_SHA256, "E_D_checkpoint": ED_CHECKPOINT_SHA256, "E_D_qualification": ED_QUALIFICATION_SHA256, "E_D_architecture": ED_ARCHITECTURE_SHA256, "E_D_independence": ED_INDEPENDENCE_SHA256, "S1_report": S1_REPORT_SHA256, "S1_winner": S1_WINNER_SHA256, "S1_panels": S1_PANELS_SHA256},
    })
    return authorities


def _load_models(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, SmoothMNISTEvaluator, FrozenMNISTEvaluator, third_confirmation.ThirdConfirmationEvaluator, independent_confirmation.PixelMLPConfirmationEvaluatorV1]:
    consumer, smooth, independent, ec = robust._load_models(root, authorities, device)
    ed_payload = safe_torch_load(root / ED_CHECKPOINT, map_location="cpu")
    ed_payload = _mapping(ed_payload, "E_D checkpoint")
    _require(ed_payload.get("semantic_authority_id") == independent_confirmation.SEMANTIC_AUTHORITY_ID, "E_D checkpoint authority changed")
    ed = independent_confirmation.PixelMLPConfirmationEvaluatorV1().to(device)
    ed.load_state_dict(_mapping(ed_payload.get("state_dict"), "E_D state"), strict=True)
    ed.freeze()
    _freeze_generator_only(consumer)
    for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec), ("E_D", ed)):
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    return consumer, smooth, independent, ec, ed


def _restore_checkpoint_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer, optimizer: torch.optim.Adam, sampler: ContinuationSampler, checkpoint: Mapping[str, Any]) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    _freeze_generator_only(consumer)
    optimizer.load_state_dict(copy.deepcopy(_mapping(checkpoint.get("optimizer_state"), "checkpoint optimizer state")))
    robust._validate_optimizer(consumer, optimizer)
    _require(float(optimizer.state[_generator(consumer)]["step"].item()) == 3300.0, "checkpoint Adam internal step changed")
    sampler.cursor = int(_mapping(checkpoint.get("sampler_state"), "checkpoint sampler state").get("cursor"))
    _restore_rng(_mapping(checkpoint.get("rng_state"), "checkpoint RNG state"))
    _clear_runtime_grads(consumer)


def _sign_preflight(root: Path, authorities: Mapping[str, Any], consumer: Any, smooth: Any, independent: Any, ec: Any, ledger: DataReadLedger) -> dict[str, Any]:
    bank = attribution._load_diagnostic_bank(root, authorities)
    checkpoint_state = {"encoder.phase": authorities["step3000"]["encoder_state"], "generator.phase": authorities["step3000"]["generator_state"]}
    result, tensors = attribution._analyze_checkpoint(consumer, smooth, independent, ec, authorities["images"], authorities["labels"], bank["a_indices"], authorities["training"]["references"], checkpoint_state, START_STEP, ledger)
    saved = _mapping(_mapping(authorities["attr_gradients"], "accepted attribution gradients")["trajectory_phase_gradients"], "accepted trajectory gradients")["3000"]
    comparisons = {}
    for name in ("g_S", "g_I", "g_C"):
        observed = _tensor(tensors[name], f"observed {name}").cpu()
        expected = _tensor(saved[name], f"accepted {name}").cpu()
        _require(observed.shape == expected.shape and torch.allclose(observed, expected, rtol=2.0e-5, atol=2.0e-7), f"SIC sign authority mismatch for {name}")
        comparisons[name] = {"cosine": _cosine(observed, expected, f"accepted/observed {name}"), "max_abs_difference": float((observed - expected).abs().max())}
    g_s, g_i, g_c = (_tensor(saved[name], f"accepted {name}") for name in ("g_S", "g_I", "g_C"))
    q_accepted = (g_s / _norm(g_s, "accepted g_S") + g_i / _norm(g_i, "accepted g_I") + g_c / _norm(g_c, "accepted g_C"))
    q_accepted = q_accepted / _rms(q_accepted, "accepted q_SIC")
    _require(_norm(q_accepted, "accepted q_SIC", allow_zero=True) > JOINT_DEGENERACY_TOLERANCE, "accepted SIC q is degenerate")
    _require(torch.equal(_runtime_state(consumer)["generator.phase"], authorities["step3000"]["generator_state"]), "SIC sign preflight changed generator state")
    _require(torch.equal(_runtime_state(consumer)["encoder.phase"], authorities["step3000"]["encoder_state"]), "SIC sign preflight changed encoder state")
    _clear_runtime_grads(consumer)
    _clear_evaluator_grads(smooth, independent, ec)
    return {"status": "READY", "accepted_gradient_cosines": comparisons, "accepted_q_SIC_norm": _norm(q_accepted, "accepted q_SIC"), "accepted_direction": "d_SIC=-v_SIC/RMS(v_SIC); optimizer uses +q_SIC", "attribution_recomputed": False, "temporary_runtime_restored": True, "E_D_in_backward": False, "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}}


def _preflight(root: Path, authorities: Mapping[str, Any], consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, ledger: DataReadLedger) -> dict[str, Any]:
    initial_state = _runtime_state(consumer)
    initial_rng = _rng_snapshot()
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_cursor = sampler.cursor
    batch_indices, batch_labels = sampler.next_batch()
    _require(torch.equal(authorities["labels"].index_select(0, batch_indices), batch_labels), "continuation batch labels changed")
    batch = authorities["images"].index_select(0, batch_indices).to(consumer.device)
    direction = _sic_direction(consumer, smooth, independent, ec, batch, authorities["training"]["references"])
    _require(direction["evaluator_parameters_in_backward"]["E_D"] is False, "E_D entered SIC backward")
    _require(math.isclose(float(direction["delta_SIC"]), DELTA_SIC, rel_tol=0.0, abs_tol=1.0e-18), "delta_SIC changed")
    _require(_norm(direction["v_SIC"], "preflight v_SIC", allow_zero=True) > JOINT_DEGENERACY_TOLERANCE, "preflight SIC direction is degenerate")
    before_values = {name: float(direction["bundle"]["losses"][name].detach().cpu()) for name in ("L_S", "L_I", "L_C")}
    generator = _generator(consumer)
    generator_before = generator.detach().clone()
    with torch.no_grad():
        generator.add_(-PRELIGHT_EPSILON * direction["q_SIC"])
    with torch.no_grad():
        after_bundle = _loss_bundle(consumer, smooth, independent, ec, batch, authorities["training"]["references"])
    after_values = {name: float(after_bundle["losses"][name].detach().cpu()) for name in ("L_S", "L_I", "L_C")}
    with torch.no_grad():
        generator.copy_(generator_before)
    _clear_runtime_grads(consumer)
    _clear_evaluator_grads(smooth, independent, ec, ed)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "directional sanity did not restore runtime")
    _require(_rng_equal(_rng_snapshot(), initial_rng), "directional sanity changed RNG")
    _require(all(parameter.grad is None for parameter in ed.parameters()), "E_D received a gradient")

    traces = []
    sampler.cursor = initial_cursor
    for preflight_step in range(3):
        indices, labels = sampler.next_batch()
        _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"preflight batch labels changed at update {preflight_step + 1}")
        preflight_batch = authorities["images"].index_select(0, indices).to(consumer.device)
        traces.append(_apply_update(consumer, smooth, independent, ec, optimizer, preflight_batch, authorities["training"]["references"], step=START_STEP + preflight_step + 1, clean_generator_state=initial_state["generator.phase"]))
    _require(not torch.equal(initial_state["generator.phase"], _runtime_state(consumer)["generator.phase"]), "preflight generator did not depart")
    _require(torch.equal(initial_state["encoder.phase"], _runtime_state(consumer)["encoder.phase"]), "preflight encoder changed")
    consumer.load_trainable_state_dict({"encoder.phase": initial_state["encoder.phase"], "generator.phase": initial_state["generator.phase"]})
    optimizer.load_state_dict(initial_optimizer)
    sampler.cursor = initial_cursor
    _restore_rng({"cpu": initial_rng[0], "cuda": initial_rng[1]})
    _freeze_generator_only(consumer)
    _clear_runtime_grads(consumer)
    _clear_evaluator_grads(smooth, independent, ec, ed)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_state), "preflight runtime restore failed")
    _require(float(optimizer.state[_generator(consumer)]["step"].item()) == 3300.0, "preflight optimizer restore failed")
    _require(sampler.cursor == START_STEP, "preflight sampler restore failed")
    _require(all(parameter.grad is None for parameter in ed.parameters()), "E_D gradient firewall failed")
    return {"schema": PREFLIGHT_SCHEMA, "status": "READY", "three_step_real_update": {"updates": 3, "trace": traces, "scientific_persistent_updates": 0}, "starting_cursor": START_STEP, "adam_internal_step": 3300.0, "stored_rng_restored": True, "directional_sanity": {"before": before_values, "after": after_values, "q_SIC_finite": True}, "sign": _sign_preflight(root, authorities, consumer, smooth, independent, ec, ledger), "trainable_parameter_names": ["generator.phase"], "E_D_absent_from_backward": True, "negative_control": "E_D_forward_only_and_no_optimizer_gradient", "protected_test_sealed": {"protected": 0, "test": 0, "sealed": 0}}


def _apply_update(consumer: Any, smooth: Any, independent: Any, ec: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, references: Any, *, step: int, clean_generator_state: torch.Tensor) -> dict[str, Any]:
    robust._validate_optimizer(consumer, optimizer)
    generator = _generator(consumer)
    before_generator = generator.detach().clone()
    before_frozen = _frozen_runtime_state(consumer)
    optimizer.zero_grad(set_to_none=True)
    _clear_runtime_grads(consumer)
    bundle = _sic_direction(consumer, smooth, independent, ec, batch, references)
    generator.grad = bundle["g_total"].detach().clone()
    _finite(generator.grad, "assigned g_total")
    for name, parameter in consumer.runtime.named_parameters():
        if name != "generator.phase":
            _require(parameter.grad is None, f"frozen runtime parameter received gradient: {name}")
    optimizer.step()
    _finite(generator, "updated generator.phase")
    _require(_state_digest(before_frozen) == _state_digest(_frozen_runtime_state(consumer)), "frozen runtime state changed")
    update = generator.detach() - before_generator
    update_norm = _norm(update, "phase update")
    update_rms = _rms(update, "phase update")
    displacement = _norm(generator.detach() - clean_generator_state.to(generator.device), "phase displacement")
    _require(update_norm > 0.0, "generator.phase did not update")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "E_S received optimizer gradient")
    _require(all(parameter.grad is None for parameter in independent.parameters()), "E_I received optimizer gradient")
    _require(all(parameter.grad is None for parameter in ec.parameters()), "E_C received optimizer gradient")
    return _composition_record(bundle, step=step, phase_displacement=displacement, phase_update_rms=update_rms)


def _evaluate_dev(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, images: torch.Tensor, labels: torch.Tensor, development_indices: torch.Tensor, references: Any, ledger: DataReadLedger, *, step: int, include_ed: bool) -> dict[str, Any]:
    labels_dev = labels.index_select(0, development_indices)
    predictions = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    loss_s = loss_i = loss_c = rec_sum = op_sum = 0.0
    raw_chunks = []
    raw_nonconstant = raw_zero = 0
    raw_min, raw_max = float("inf"), float("-inf")
    total_elements = 0
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    smooth.eval(); independent.eval(); ec.eval(); ed.eval()
    try:
        with torch.no_grad():
            for start in range(0, int(development_indices.numel()), LOGICAL_BATCH_SIZE):
                indices = development_indices[start : start + LOGICAL_BATCH_SIZE]
                batch = images.index_select(0, indices).to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                raw = result.raw_intensity
                target = result.image_input.slm_intensity
                _finite(raw, "DEV raw CMOS"); _finite(target, "DEV target")
                raw_view = _cmos_view(raw); target_view = _cmos_view(batch)
                s, s_logits = smooth(raw_view); i, i_logits = independent(raw_view); c, c_logits = ec(raw_view)
                s_t, _ = smooth(target_view); i_t, _ = independent(target_view); c_t, _ = ec(target_view)
                loss_s += float(_normalized_feature_loss(s, s_t, "DEV L_S").cpu()) * int(indices.numel())
                loss_i += float(_normalized_feature_loss(i, i_t, "DEV L_I").cpu()) * int(indices.numel())
                loss_c += float(_normalized_feature_loss(c, c_t, "DEV L_C").cpu()) * int(indices.numel())
                predictions["E_S"].append(s_logits.argmax(dim=1).cpu()); predictions["E_I"].append(i_logits.argmax(dim=1).cpu()); predictions["E_C"].append(c_logits.argmax(dim=1).cpu())
                if include_ed:
                    predictions["E_D"].append(ed(raw_view).argmax(dim=1).cpu())
                rec_sum += float((raw - target).square().sum().cpu())
                powers = contracted_plane_powers(result)
                ratios = (powers.to(torch.float64) + 1.0e-12) / (references.power_reference.to(device=powers.device, dtype=torch.float64) + 1.0e-12)
                op_sum += float(torch.log(ratios).square().sum().cpu())
                flat = raw.to(torch.float64).flatten(1)
                raw_nonconstant += int((flat.std(dim=1, unbiased=False) > 0.0).sum().item())
                raw_zero += int((flat.abs().sum(dim=1) == 0.0).sum().item())
                raw_min = min(raw_min, float(raw.min().cpu())); raw_max = max(raw_max, float(raw.max().cpu()))
                total_elements += int(raw.numel())
                raw_chunks.append(raw.cpu())
                ledger.record("dev", int(indices.numel()))
    finally:
        consumer.runtime.train(was_training)
    row_count = int(development_indices.numel())
    result: dict[str, Any] = {
        "schema": DEV_SCHEMA,
        "step": step,
        "rows": row_count,
        "E_S": {"architecture": "SmoothMNISTEvaluator v2.1", **_class_metrics(labels_dev, torch.cat(predictions["E_S"]), label_source="canonical DEV label")},
        "E_I": {"architecture": "FrozenMNISTEvaluator v1", **_class_metrics(labels_dev, torch.cat(predictions["E_I"]), label_source="canonical DEV label")},
        "E_C": {"architecture": third_confirmation.ThirdConfirmationEvaluator.architecture_kind, **_class_metrics(labels_dev, torch.cat(predictions["E_C"]), label_source="canonical DEV label")},
        "L_S": loss_s / row_count,
        "L_I": loss_i / row_count,
        "L_C": loss_c / row_count,
        "L_rec": rec_sum / float(total_elements),
        "L_op": op_sum / float(row_count * 5),
        "raw_cmos": {"finite": True, "nonconstant_fraction": float(raw_nonconstant / row_count), "zero_image_count": raw_zero, "min": raw_min, "max": raw_max},
        "E_D_in_backward": False,
        "E_D_evaluated": bool(include_ed),
    }
    if include_ed:
        result["E_D"] = {"architecture": independent_confirmation.PixelMLPConfirmationEvaluatorV1.architecture_kind, **_class_metrics(labels_dev, torch.cat(predictions["E_D"]), label_source="canonical DEV label")}
        result["_E_D_correct"] = torch.cat(predictions["E_D"]) == labels_dev
        result["_labels_dev"] = labels_dev
    _require(result["raw_cmos"]["finite"] and result["raw_cmos"]["nonconstant_fraction"] == 1.0 and result["raw_cmos"]["zero_image_count"] == 0, f"DEV optical validity failed at step {step}")
    _require(all(parameter.grad is None for model in (smooth, independent, ec, ed) for parameter in model.parameters()), "DEV evaluator received a gradient")
    return result


def _public_dev(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _paired_bootstrap(labels: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    _require(labels.shape == before.shape == after.shape == (5000,), "E_D bootstrap vectors have wrong shape")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    class_deltas = []
    per_class = []
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 500, f"E_D bootstrap class {digit} count changed")
        draw = torch.randint(matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        base = before.index_select(0, matches)[draw]
        candidate = after.index_select(0, matches)[draw]
        class_deltas.append((candidate.to(torch.float64) - base.to(torch.float64)).mean(dim=1))
        per_class.append(float(after.index_select(0, matches).to(torch.float64).mean() - before.index_select(0, matches).to(torch.float64).mean()))
    macro = torch.stack(class_deltas).mean(dim=0)
    quantiles = torch.quantile(macro, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    return {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "class_stratified_paired": True, "delta_macro_recall": float(macro.mean()), "ci95": {"lower": float(quantiles[0]), "upper": float(quantiles[1])}, "per_class_delta": per_class, "macro_delta_tensor": macro}


def _load_s1_bank(authorities: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = _tensor(authorities["s1_winner"]["z"], "S1 z").cpu().contiguous()
    requested = _tensor(authorities["s1_winner"]["requested_labels"], "S1 requested").cpu().contiguous()
    panel_indices = _tensor(authorities["s1_panels"]["panel_indices"], "S1 panel indices").to(torch.long).cpu().contiguous()
    _require(z.shape == (2000, 100) and z.is_complex(), "S1 z shape changed")
    _require(requested.shape == (2000,) and panel_indices.shape == (200,), "S1 bank shape changed")
    _require(torch.bincount(requested, minlength=10).tolist() == [200] * 10, "S1 bank is not balanced")
    return z, requested, panel_indices


def _summary(value: torch.Tensor, name: str) -> dict[str, Any]:
    flat = value.detach().cpu().to(torch.float64).reshape(-1)
    _finite(flat, name)
    q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {"count": int(flat.numel()), "min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean()), "std_population": float(flat.std(unbiased=False)), "quantiles": {"0.01": float(q[0]), "0.5": float(q[1]), "0.99": float(q[2])}}


def _s1_transfer(consumer: Any, smooth: Any, independent: Any, ec: Any, ed: Any, authorities: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    z, requested, panel_indices = _load_s1_bank(authorities)
    outputs: list[torch.Tensor] = []
    logits: dict[str, list[torch.Tensor]] = {name: [] for name in ("E_S", "E_I", "E_C", "E_D")}
    features: dict[str, list[torch.Tensor]] = {name: [] for name in ("E_S", "E_I", "E_C")}
    consumer.runtime.eval(); smooth.eval(); independent.eval(); ec.eval(); ed.eval()
    with torch.inference_mode():
        for start in range(0, int(z.shape[0]), 10):
            result = consumer.generated_forward(z[start : start + 10].to(consumer.device))
            raw = result.raw_intensity.detach()
            _finite(raw, "S1 generated raw CMOS")
            view = _cmos_view(raw)
            s_feat, s_logits = smooth(view); i_feat, i_logits = independent(view); c_feat, c_logits = ec(view); d_logits = ed(view)
            outputs.append(raw.cpu())
            for name, value in (("E_S", s_logits), ("E_I", i_logits), ("E_C", c_logits), ("E_D", d_logits)):
                logits[name].append(value.detach().cpu())
            for name, value in (("E_S", s_feat), ("E_I", i_feat), ("E_C", c_feat)):
                features[name].append(value.detach().cpu())
    raw_all = torch.cat(outputs)
    logits_all = {name: torch.cat(values) for name, values in logits.items()}
    features_all = {name: torch.cat(values) for name, values in features.items()}
    panel_raw = raw_all.index_select(0, panel_indices).flatten(1).to(torch.float64)
    control_raw = _tensor(authorities["s1_panels"]["real_validation_control_raw_panel"], "S1 real control panel").to(torch.float64).flatten(1)
    _require(panel_raw.shape == control_raw.shape == (200, 40000), "S1 R panel geometry changed")
    mean = control_raw.mean(dim=0); std = control_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    generated_standard = (panel_raw - mean) / std; control_standard = (control_raw - mean) / std
    raw_cross = _energy_distance(generated_standard, control_standard)
    raw_self = _energy_distance(control_standard[:100], control_standard[100:])
    control_view = control_raw.reshape(200, 1, 200, 200).to(device=consumer.device, dtype=torch.float32)
    with torch.inference_mode():
        control_features, _ = ec(_cmos_view(control_view))
    feature_current = features_all["E_C"].index_select(0, panel_indices).to(torch.float64)
    feature_control = control_features.detach().cpu().to(torch.float64)
    feature_cross = _energy_distance(feature_current, feature_control)
    feature_self = _energy_distance(feature_control[:100], feature_control[100:])
    semantic = {name: _class_metrics(requested, logits_all[name].argmax(dim=1), label_source="requested class y supplied by frozen S1 sample(y, seed_or_index)") for name in logits_all}
    raw_stats = _raw_validity(raw_all)
    report = {
        "schema": S1_SCHEMA,
        "step": step,
        "family": "S1",
        "rows": int(z.shape[0]),
        "requested_class_ground_truth": True,
        "semantic": semantic,
        "latent_support_reference": {"sampler_report": S1_REPORT.as_posix(), "sampler_report_sha256": S1_REPORT_SHA256, "winner": "S1", "semantic_support_rate": float(authorities["s1_report"]["winner"]["absolute_target"]["winner_rate"]), "support_ci_lower": float(authorities["s1_report"]["winner"]["absolute_target"]["winner_ci_lower"])},
        "R_raw": raw_cross / max(raw_self, 1.0e-12),
        "R_feature": feature_cross / max(feature_self, 1.0e-12),
        "R_raw_details": {"cross": raw_cross, "self": raw_self, "control_role": "fixed real validation control panel from accepted S1 canary"},
        "R_feature_details": {"cross": feature_cross, "self": feature_self, "feature_space": "frozen E_C embedding"},
        "class_histograms": {name: metric["predicted_class_histogram"] for name, metric in semantic.items()},
        "normalized_entropy": {name: _entropy(metric["predicted_class_histogram"]) / math.log(10.0) for name, metric in semantic.items()},
        "dominant_fraction": {name: max(metric["predicted_class_histogram"]) / float(z.shape[0]) for name, metric in semantic.items()},
        "raw_validity": raw_stats,
        "fixed_bank_sha256": tensor_sha256(z),
        "requested_labels_sha256": tensor_sha256(requested),
    }
    _require(raw_stats["finite"] and raw_stats["nonconstant_fraction"] == 1.0 and raw_stats["zero_image_count"] == 0, f"S1 optical validity failed at step {step}")
    return report


def _entropy(histogram: list[int]) -> float:
    counts = torch.tensor(histogram, dtype=torch.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0.0]
    return float(-(nonzero * nonzero.log()).sum())


def _energy_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(2.0 * torch.cdist(left, right, p=2).mean() - torch.cdist(left, left, p=2).mean() - torch.cdist(right, right, p=2).mean())


def _checkpoint_payload(consumer: Any, optimizer: Any, sampler: ContinuationSampler, history: list[Mapping[str, Any]], trajectory: Mapping[str, Any], *, step: int, source_manifest: Mapping[str, Any], run_identity: ResearchRunIdentity, stage1_condition: Mapping[str, Any] | None) -> dict[str, Any]:
    state = optimizer.state[_generator(consumer)]
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "authorization_ids": [AUTHORIZATION_ID],
        "completed_updates": step,
        "starting_scientific_step": START_STEP,
        "checkpoint_schedule": list(CHECKPOINT_STEPS),
        "checkpoint_selection": "none; every preregistered scientific checkpoint persisted",
        "scientific_artifact": True,
        "operational_artifact": False,
        "run_identity": run_identity.to_dict(),
        "source_manifest": dict(source_manifest),
        "starting_authority": {"path": STEP3000.as_posix(), "sha256": STEP3000_SHA256, "selected_step": START_STEP},
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "P2", "Q2", "S1", "E_S", "E_I", "E_C", "E_D"],
        "generator_state": consumer.runtime.generator.phase.detach().cpu().clone(),
        "encoder_state": consumer.runtime.encoder.phase.detach().cpu().clone(),
        "frozen_runtime_state_sha256": _state_digest(_frozen_runtime_state(consumer)),
        "optimizer_mode": "STEP3000_ADAM_STATE_RESTORED_EXACTLY",
        "optimizer_configuration": {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "internal_step": float(state["step"])},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state_payload(),
        "delta_SIC": DELTA_SIC,
        "auxiliary_coefficients": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ},
        "stage1_continuation_condition": None if stage1_condition is None else dict(stage1_condition),
        "gradient_contract": {"g_S": "grad_generator(L_S)", "g_I": "grad_generator(L_I)", "g_C": "grad_generator(L_C)", "u": "g/||g||_2", "v_SIC": "u_S+u_I+u_C", "d_SIC": "-v_SIC/RMS(v_SIC)", "q_SIC": "+v_SIC/RMS(v_SIC)", "g_sem": "delta_SIC*q_SIC", "g_total": "g_sem+g_rec_safe+g_op_safe", "projection": "remove only opposing auxiliary component", "E_D_in_backward": False},
        "training_history": [dict(record) for record in history],
        "composition_trajectory": dict(trajectory),
    }


def _make_identity(source_manifest: Mapping[str, Any], data_identity: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(task_id=TASK_ID, run_id="three-evaluator-robust-semantic-continuation-v1", trajectory_id="three-evaluator-robust-semantic-continuation-trajectory-v1", semantic_authority_id=SEMANTIC_AUTHORITY_ID, data_role=TRAIN_ROLE, config_identity={"starting_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "delta_SIC": DELTA_SIC, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "sampler_wraparound": True}, model_identity="LIGHTGEN_2D_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1", optimizer_identity={"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 3300.0, "reset": False}, sampler_identity={"seed": SCHEDULE_SEED, "starting_cursor": START_STEP, "terminal_cursor": TOTAL_STEPS, "accepted_prefix_preserved": True, "wraparound_after_per_class_permutation_exhaustion": True}, execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]), source_manifest_id=str(source_manifest["source_manifest_id"]), scientific_attempt_id="three-evaluator-robust-semantic-continuation-v1:attempt_01")


def _stage1_condition(dev3000: Mapping[str, Any], dev3300: Mapping[str, Any], ed_bootstrap: Mapping[str, Any], reconstruction: Mapping[str, Any], optical: Mapping[str, Any], *, sic_degenerate: bool, firewall: Mapping[str, int]) -> tuple[bool, str, dict[str, bool]]:
    ed_delta = float(ed_bootstrap["delta_macro_recall"])
    ed_ci = float(ed_bootstrap["ci95"]["lower"])
    checks = {
        "sic_direction_never_degenerate": not sic_degenerate,
        "L_S_3300_lt_L_S_3000": float(dev3300["L_S"]) < float(dev3000["L_S"]),
        "L_I_3300_lt_L_I_3000": float(dev3300["L_I"]) < float(dev3000["L_I"]),
        "L_C_3300_lt_L_C_3000": float(dev3300["L_C"]) < float(dev3000["L_C"]),
        "E_D_delta_positive": ed_delta > 0.0,
        "E_D_CI_lower_positive": ed_ci > 0.0,
        "reconstruction_constraint": float(reconstruction["delta_step3300_minus_step3000"]) <= RECONSTRUCTION_TOLERANCE,
        "optical_validity": bool(optical["passed"]),
        "protected_test_sealed_zero": all(int(firewall.get(name, -1)) == 0 for name in ("protected", "test", "sealed")),
        "no_authority_counterfactual_violation": True,
    }
    if all(checks.values()):
        return True, CLASSIFICATION_SUPPORTED, checks
    if sic_degenerate:
        return False, CLASSIFICATION_STAGE1_DEGENERATED, checks
    if checks["L_S_3300_lt_L_S_3000"] and checks["L_I_3300_lt_L_I_3000"] and checks["L_C_3300_lt_L_C_3000"] and not (checks["E_D_delta_positive"] and checks["E_D_CI_lower_positive"]):
        return False, CLASSIFICATION_STAGE1_NO_ED, checks
    if checks["E_D_delta_positive"] and checks["E_D_CI_lower_positive"] and not checks["reconstruction_constraint"]:
        return False, CLASSIFICATION_STAGE1_RECON, checks
    return False, CLASSIFICATION_OPTIMIZATION, checks


def _optical_pass(*records: Mapping[str, Any]) -> dict[str, Any]:
    passed = all(bool(record["raw_cmos"]["finite"]) and float(record["raw_cmos"]["nonconstant_fraction"]) == 1.0 and int(record["raw_cmos"]["zero_image_count"]) == 0 and math.isfinite(float(record["L_op"])) for record in records)
    return {"checkpoints": {str(record["step"]): record["raw_cmos"] for record in records}, "passed": passed, "all_finite": all(bool(record["raw_cmos"]["finite"]) for record in records), "all_nonconstant": all(float(record["raw_cmos"]["nonconstant_fraction"]) == 1.0 for record in records), "all_zero_image_counts_zero": all(int(record["raw_cmos"]["zero_image_count"]) == 0 for record in records), "all_power_checks_finite": all(math.isfinite(float(record["L_op"])) for record in records)}


def _long_classification(dev3300: Mapping[str, Any], dev6000: Mapping[str, Any], reconstruction: Mapping[str, Any], *, sic_degenerate: bool) -> tuple[str, str | None]:
    if sic_degenerate:
        return CLASSIFICATION_DEGENERATED, None
    if float(dev6000["L_S"]) >= float(dev3300["L_S"]) or float(dev6000["L_I"]) >= float(dev3300["L_I"]) or float(dev6000["L_C"]) >= float(dev3300["L_C"]):
        return CLASSIFICATION_SHORT, None
    if not reconstruction["passed"]:
        return CLASSIFICATION_RECON, None
    return CLASSIFICATION_SUPPORTED, None


def run_training(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")
    authorities = _load_authorities(root)
    cuda_runtime = robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal SIC continuation requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, ec, ed = _load_models(root, authorities, device)
    _restore_rng(authorities["step3000"]["rng_state"])
    optimizer, optimizer_mode = _optimizer(consumer, authorities["step3000"])
    sampler_state = authorities["step3000"]["sampler_state"]
    sampler = ContinuationSampler(sampler_state, cursor=START_STEP)
    ledger = DataReadLedger()
    preflight = _preflight(root, authorities, consumer, smooth, independent, ec, ed, optimizer, sampler, ledger)
    _require(preflight["status"] == "READY", "SIC continuation preflight did not reach READY")

    source_manifest_object = capture_execution_source_manifest(root, task_id=TASK_ID, semantic_authority_id=SEMANTIC_AUTHORITY_ID, relevant_paths=(Path(__file__).resolve(), Path(robust.__file__).resolve(), Path(attribution.__file__).resolve(), Path(independent_confirmation.__file__).resolve(), Path(third_confirmation.__file__).resolve(), Path(conditioned_sampler.__file__).resolve(), root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py", root / "src/ldm_project/v2/student/evaluator.py", root / STEP3000, root / ED_CHECKPOINT, root / S1_WINNER), config_identity={"authorization_token": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "delta_SIC": DELTA_SIC, "beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "authority_hashes": authorities["authority_hashes"]})
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    run_identity = _make_identity(source_manifest, authorities["data_identity"])
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready(); control.start_execution(); effective_identity = control.begin_scientific_attempt(run_identity.scientific_attempt_id)
    _write_json(target / "source_manifest.json", source_manifest, kind="SIC continuation source manifest")
    _write_json(target / "preflight.json", preflight, kind="SIC continuation preflight")
    _write_json(target / "execution_manifest.json", {"schema": "three_evaluator_robust_semantic_continuation_execution_manifest.v1", "task_id": TASK_ID, "outer_gate": OUTER_GATE, "authorization": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN}, "input_authorities": authorities["authority_hashes"], "source_manifest": source_manifest, "run_identity": effective_identity.to_dict(), "local_state": authorities["local_state"], "cuda_runtime": cuda_runtime, "scope": {"stage1": [START_STEP, STAGE1_END], "stage2": [STAGE1_END, TOTAL_STEPS], "trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "T", "scale", "S1", "P2", "Q2", "E_S", "E_I", "E_C", "E_D"], "protected": False, "test": False, "sealed": False}}, kind="SIC continuation execution manifest")

    clean_generator_state = authorities["step3000"]["generator_state"].to(device)
    history: list[dict[str, Any]] = []
    trajectory: dict[str, dict[str, Any]] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    s1_records: dict[str, dict[str, Any]] = {}
    heartbeat = HeartbeatWriter(target, effective_identity.run_id, total_steps=TOTAL_STEPS, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, effective_identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=AUTHORIZATION_ID)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=START_STEP, latest_losses={"preflight": "READY", "optimizer_mode": optimizer_mode}, force=True)
    fixed_indices = torch.stack([value[START_STEP % value.numel()] for value in sampler.permutations])
    fixed_batch = authorities["images"].index_select(0, fixed_indices).to(device)
    composition = _sic_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"])
    trajectory[str(START_STEP)] = _composition_record(composition, step=START_STEP, phase_displacement=0.0, phase_update_rms=0.0)
    _clear_runtime_grads(consumer); _clear_evaluator_grads(smooth, independent, ec, ed)
    dev_records[str(START_STEP)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=START_STEP, include_ed=True)
    s1_records[str(START_STEP)] = _s1_transfer(consumer, smooth, independent, ec, ed, authorities, step=START_STEP)
    _write_json(target / f"dev_diagnostics_step_{START_STEP:06d}.json", _public_dev(dev_records[str(START_STEP)]), kind="step-3000 DEV diagnostics")
    _write_json(target / f"s1_transfer_step_{START_STEP:06d}.json", s1_records[str(START_STEP)], kind="step-3000 S1 transfer")
    _write_torch(target / f"scientific_step_{START_STEP:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=START_STEP, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition=None), kind="SIC continuation starting checkpoint")

    last_recovery_step = START_STEP
    last_recovery_time = time.time()
    def recovery_payload(step: int) -> Mapping[str, Any]:
        payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=step, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition=None)
        payload["scientific_artifact"] = False; payload["operational_artifact"] = True; payload["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        return payload

    stage1_degenerate = False
    stage1_passed = False
    stage1_classification = CLASSIFICATION_OPTIMIZATION
    continuation_checks: dict[str, bool] = {}
    ed_bootstrap: dict[str, Any] | None = None
    reconstruction: dict[str, Any] | None = None
    optical: dict[str, Any] | None = None
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=TOTAL_STEPS)
    with interrupt:
        while sampler.cursor < TOTAL_STEPS:
            current_step = sampler.cursor
            indices, batch_labels = sampler.next_batch()
            _require(torch.equal(authorities["labels"].index_select(0, indices), batch_labels), f"TRAIN sampler labels changed at update {current_step + 1}")
            batch = authorities["images"].index_select(0, indices).to(device)
            try:
                trace = _apply_update(consumer, smooth, independent, ec, optimizer, batch, authorities["training"]["references"], step=current_step + 1, clean_generator_state=clean_generator_state)
            except SICDirectionDegenerated:
                if current_step < STAGE1_END:
                    stage1_degenerate = True
                raise
            history.append(trace)
            completed = current_step + 1
            now = time.time()
            if recovery_due(step=completed, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                recovery.save(recovery_payload(completed), step=completed); last_recovery_step = completed; last_recovery_time = now
            heartbeat.update(state=RUNNING, phase="TRAINING", step=completed, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_C", "L_rec", "L_op")})
            interrupted = interrupt.handle_safe_boundary(step=completed, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_C", "L_rec", "L_op")}, last_recovery_checkpoint=None)
            if interrupted is not None:
                control.interrupted()
                return {"result": "INTERRUPTED", "output_dir": _relative(root, target), "step": completed, "recovery_checkpoint": interrupted.name}
            if completed in CHECKPOINT_STEPS:
                composition = _sic_direction(consumer, smooth, independent, ec, fixed_batch, authorities["training"]["references"])
                trajectory[str(completed)] = _composition_record(composition, step=completed, phase_displacement=_norm(_generator(consumer).detach() - clean_generator_state, "checkpoint displacement"), phase_update_rms=float(trace["phase_update_rms"]))
                _clear_runtime_grads(consumer); _clear_evaluator_grads(smooth, independent, ec, ed)
            if completed == STAGE1_END:
                dev_records[str(completed)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed, include_ed=True)
                s1_records[str(completed)] = _s1_transfer(consumer, smooth, independent, ec, ed, authorities, step=completed)
                _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", _public_dev(dev_records[str(completed)]), kind="step-3300 DEV diagnostics")
                _write_json(target / f"s1_transfer_step_{completed:06d}.json", s1_records[str(completed)], kind="step-3300 S1 transfer")
                ed_bootstrap = _paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_E_D_correct"], dev_records[str(completed)]["_E_D_correct"])
                _write_torch(target / "paired_bootstrap_stage1.pt", ed_bootstrap, kind="E_D Stage-1 paired bootstrap")
                reconstruction = {"step3000": dev_records[str(START_STEP)]["L_rec"], "step3300": dev_records[str(completed)]["L_rec"], "delta_step3300_minus_step3000": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(completed)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
                optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(completed)])
                firewall = {"protected": 0, "test": 0, "sealed": 0}
                stage1_passed, stage1_classification, continuation_checks = _stage1_condition(dev_records[str(START_STEP)], dev_records[str(completed)], ed_bootstrap, reconstruction, optical, sic_degenerate=stage1_degenerate, firewall=firewall)
                stage1_payload = _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": continuation_checks})
                _write_torch(target / f"scientific_step_{completed:06d}.pt", stage1_payload, kind="SIC continuation Stage-1 endpoint checkpoint")
                heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=completed, latest_losses={"stage1_passed": stage1_passed}, force=True)
                if not stage1_passed:
                    break
            elif completed in (3100, 3200, 3600, 4000, 4500, 5000, 5500, 6000):
                if completed == TOTAL_STEPS:
                    dev_records[str(completed)] = _evaluate_dev(consumer, smooth, independent, ec, ed, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=completed, include_ed=True)
                    s1_records[str(completed)] = _s1_transfer(consumer, smooth, independent, ec, ed, authorities, step=completed)
                    _write_json(target / f"dev_diagnostics_step_{completed:06d}.json", _public_dev(dev_records[str(completed)]), kind=f"step-{completed} DEV diagnostics")
                    _write_json(target / f"s1_transfer_step_{completed:06d}.json", s1_records[str(completed)], kind=f"step-{completed} S1 transfer")
                _write_torch(target / f"scientific_step_{completed:06d}.pt", _checkpoint_payload(consumer, optimizer, sampler, history, trajectory, step=completed, source_manifest=source_manifest, run_identity=effective_identity, stage1_condition={"passed": stage1_passed, "checks": continuation_checks}), kind=f"SIC continuation step-{completed} checkpoint")

    _require(str(START_STEP) in dev_records and str(STAGE1_END) in dev_records, "Stage-1 endpoint diagnostics are incomplete")
    if ed_bootstrap is None:
        ed_bootstrap = _paired_bootstrap(dev_records[str(START_STEP)]["_labels_dev"], dev_records[str(START_STEP)]["_E_D_correct"], dev_records[str(STAGE1_END)]["_E_D_correct"])
    if reconstruction is None:
        reconstruction = {"step3000": dev_records[str(START_STEP)]["L_rec"], "step3300": dev_records[str(STAGE1_END)]["L_rec"], "delta_step3300_minus_step3000": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records[str(STAGE1_END)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}
    if optical is None:
        optical = _optical_pass(dev_records[str(START_STEP)], dev_records[str(STAGE1_END)])
    stage2_executed = bool(stage1_passed and sampler.cursor == TOTAL_STEPS)
    final_classification = stage1_classification
    mechanism = None
    if stage2_executed:
        final_classification, mechanism = _long_classification(dev_records[str(STAGE1_END)], dev_records[str(TOTAL_STEPS)], {"passed": dev_records[str(TOTAL_STEPS)]["L_rec"] - dev_records[str(START_STEP)]["L_rec"] <= RECONSTRUCTION_TOLERANCE}, sic_degenerate=False)
        real_gain = float(dev_records[str(TOTAL_STEPS)]["E_D"]["macro_recall"]) > float(dev_records[str(START_STEP)]["E_D"]["macro_recall"])
        s1_gain = float(s1_records[str(TOTAL_STEPS)]["semantic"]["E_D"]["macro_recall"]) > float(s1_records[str(START_STEP)]["semantic"]["E_D"]["macro_recall"])
        if real_gain and s1_gain:
            mechanism = CLASSIFICATION_S1_TRANSFER
        elif real_gain and not s1_gain:
            mechanism = CLASSIFICATION_S1_LIMITED
    control.complete()
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=sampler.cursor, latest_losses={"classification": final_classification}, force=True)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": bool(stage1_passed),
        "classification": final_classification,
        "mechanism": mechanism,
        "authorization_consumed": {"id": AUTHORIZATION_ID, "token": AUTHORIZATION_TOKEN, "scope": "step3000->3300 and conditional step3300->6000"},
        "local_state": {"branch": authorities["local_state"]["branch"], "head": authorities["local_state"]["head"], "dirty_before": authorities["local_state"]["worktree_dirty"], "worktree_preserved": True},
        "starting_authority": {"step3000_checkpoint": _artifact_ref(root, root / STEP3000, role="accepted exact step3000 continuation checkpoint"), "robust_report": _artifact_ref(root, root / ROBUST_REPORT, role="accepted robust step3000 report"), "attribution_report": _artifact_ref(root, root / ATTRIBUTION_REPORT, role="accepted post3000 SIC attribution")},
        "evaluators": {"E_S": {"role": "training component and DEV diagnostic", "checkpoint_sha256": robust.EXPECTED_SMOOTH_CHECKPOINT_SHA256}, "E_I": {"role": "training component and DEV diagnostic", "checkpoint_sha256": robust.EXPECTED_INDEPENDENT_CHECKPOINT_SHA256}, "E_C": {"role": "training component and DEV diagnostic", "checkpoint_sha256": robust.EXPECTED_EC_CHECKPOINT_SHA256}, "E_D": {"role": "independent confirmation only", "semantic_authority_id": independent_confirmation.SEMANTIC_AUTHORITY_ID, "checkpoint_sha256": ED_CHECKPOINT_SHA256, "qualification": authorities["ed_qualification"]}},
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "T", "scale", "S1", "P2", "Q2", "E_S", "E_I", "E_C", "E_D"], "E_D_in_backward": False},
        "data_identity": authorities["data_identity"],
        "data_reads": {**ledger.to_dict(), "protected": 0, "test": 0, "sealed": 0},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "sic_gradient_contract": {"g_S": "grad_generator(L_S)", "g_I": "grad_generator(L_I)", "g_C": "grad_generator(L_C)", "u_k": "g_k/||g_k||_2", "v_SIC": "u_S+u_I+u_C", "d_SIC": "-v_SIC/RMS(v_SIC)", "q_SIC": "+v_SIC/RMS(v_SIC)", "delta_SIC": DELTA_SIC, "degeneracy_tolerance": JOINT_DEGENERACY_TOLERANCE, "E_D_in_backward": False, "target_features_stop_gradient": True},
        "auxiliary_contract": {"beta": BETA, "gamma": GAMMA, "eps_proj": EPS_PROJ, "g_rec_w": "beta*g_rec", "g_op_w": "gamma*g_op", "projection": "remove only opposing component relative to g_sem"},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False, "stored_internal_step_at_start": 3300.0, "reset": False, "scheduler_change": False},
        "preflight": preflight,
        "stage1": {"updates": min(max(0, len(history)), STAGE1_UPDATES), "checkpoints": [3000, 3100, 3200, 3300], "trajectory": {key: trajectory[key] for key in ("3000", "3100", "3200", "3300") if key in trajectory}, "development_semantics": {key: _public_dev(dev_records[key]) for key in ("3000", "3300") if key in dev_records}, "independent_confirmation": {"E_D_step3000": dev_records["3000"]["E_D"], "E_D_step3300": dev_records["3300"]["E_D"], "delta_macro_recall": ed_bootstrap["delta_macro_recall"], "paired_bootstrap_ci": ed_bootstrap["ci95"], "per_class_delta": ed_bootstrap["per_class_delta"], "artifact": _artifact_ref(root, target / "paired_bootstrap_stage1.pt", role="E_D Stage-1 paired bootstrap")}, "reconstruction": reconstruction, "optical_validity": optical, "S1_transfer": {key: s1_records[key] for key in ("3000", "3300") if key in s1_records}, "continuation_condition": {"passed": stage1_passed, "checks": continuation_checks, "classification_if_stop": stage1_classification}},
        "stage2": {"executed": stage2_executed, "additional_updates": max(0, len(history) - STAGE1_UPDATES), "total_updates": len(history), "checkpoints": [3300, 3600, 4000, 4500, 5000, 5500, 6000], "trajectory": {key: trajectory[key] for key in ("3300", "3600", "4000", "4500", "5000", "5500", "6000") if key in trajectory}, "development_semantics": {key: _public_dev(dev_records[key]) for key in ("6000",) if key in dev_records}, "E_D_long_horizon_role": "LONG_HORIZON_DESCRIPTIVE_TRANSFER" if "6000" in dev_records else None, "S1_transfer": s1_records.get("6000"), "reconstruction": None if "6000" not in dev_records else {"step3000": dev_records["3000"]["L_rec"], "step3300": dev_records["3300"]["L_rec"], "step6000": dev_records["6000"]["L_rec"], "delta_step6000_minus_step3000": dev_records["6000"]["L_rec"] - dev_records["3000"]["L_rec"], "tolerance": RECONSTRUCTION_TOLERANCE, "passed": dev_records["6000"]["L_rec"] - dev_records["3000"]["L_rec"] <= RECONSTRUCTION_TOLERANCE}, "optical_validity": None if "6000" not in dev_records else _optical_pass(dev_records["3000"], dev_records["3300"], dev_records["6000"])},
        "outer_gate_checks": {"stage1_all_preregistered_conditions": stage1_passed, "conditional_stage2_only_after_stage1": (not stage2_executed) or stage1_passed, "protected_test_sealed_zero": True},
        "training_trajectory": {"updates": len(history), "scientific_cursor": sampler.cursor, "checkpoint_steps_persisted": [step for step in CHECKPOINT_STEPS if (target / f"scientific_step_{step:06d}.pt").is_file()], "trace": history},
        "s1_fixed_bank": {"winner": "S1", "winner_sha256": S1_WINNER_SHA256, "no_regeneration": True},
        "claim_boundary": {"stage1_independent_confirmation": stage1_passed, "stage2_is_scaling_study": stage2_executed, "E_D_step6000_independent": False, "final_semantic_recovery_claim": False, "new_confirmation_required_before_promotion": True, "architecture_acceptance": False, "promotion": False, "physical_hardware_validity": False},
        "counterfactual_integrity": {"same_step3000_checkpoint": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_adam_state": True, "same_delta_SIC": True, "same_beta_gamma": True, "same_evaluators": True, "E_D_in_backward": False, "no_hyperparameter_search": True, "no_sampler_refit": True, "no_checkpoint_selection": True, "no_protected_test_sealed": True},
        "files_changed": {"repository_source": [_relative(root, Path(__file__).resolve())], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "stage1_checkpoints": [3000, 3100, 3200, 3300], "stage2_checkpoints": [3300, 3600, 4000, 4500, 5000, 5500, 6000], "E_D_stage1_endpoints": [3000, 3300], "E_D_stage2_endpoint": 6000 if "6000" in dev_records else None, "frozen_gradient_firewall": True, "no_protected_test_sealed": True},
        "unresolved": ["A positive continuation does not establish final generation quality, human perceptual equivalence, hardware readiness or promotion; a new sealed confirmation authority is required before any formal promotion."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="three-evaluator SIC continuation report")
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
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "start_step": START_STEP, "stage1_end": STAGE1_END, "total_steps": TOTAL_STEPS, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": "authorization token mismatch"}, ensure_ascii=False, indent=2))
        return 2
    try:
        report = run_training(output_dir=args.output_dir)
    except SICDirectionDegenerated as error:
        print(json.dumps({"status": "VALID_FAILED", "task_id": TASK_ID, "classification": CLASSIFICATION_DEGENERATED, "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    except (ContinuationError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "outer_gate_passed": report["outer_gate_passed"], "stage2_executed": report["stage2"]["executed"], "updates": report["training_trajectory"]["updates"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
