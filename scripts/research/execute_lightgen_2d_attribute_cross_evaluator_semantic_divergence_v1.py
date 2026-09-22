"""Attribute Smooth/independent-evaluator semantic divergence on frozen 2L states.

This is a TRAIN-only, no-update consumer.  It compares the accepted clean
step-0 Generator with the accepted Smooth-guided step-300 Generator using one
fixed class-balanced A/B diagnostic bank.  A supplies output- and
parameter-space gradients; disjoint B supplies symmetric non-persistent
directional probes.  No evaluator is used as a teacher and no optimizer is
created.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen_2d_chen_bogt_complex_ols_training import (
    DataReadLedger,
    LightGen2DChenBOGTComplexOLSTrainingConsumer,
    PRODUCTION_PROPAGATION_GRID,
    TRAIN_ROLE,
    contracted_plane_powers,
    git_identity,
    optical_power_loss_from_powers,
)
from ldm_project.v2.application.research.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator, load_frozen_evaluator

from scripts.research import execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1 as prior_attribution
from scripts.research import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research import execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1 as canonical_training


TASK_ID = "ATTRIBUTE_CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ATTRIBUTE_CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "cross-evaluator-semantic-divergence-v1"

EXPECTED_BRANCH = canonical_training.EXPECTED_BRANCH
EXPECTED_HEAD = canonical_training.EXPECTED_HEAD

PRIOR_RUN = conflict.DEFAULT_OUTPUT_DIR
PRIOR_REPORT = conflict.DEFAULT_OUTPUT_DIR / "report.json"
PRIOR_STEP_0 = conflict.DEFAULT_OUTPUT_DIR / "scientific_step_000000.pt"
EXPECTED_PRIOR_REPORT_SHA256 = "dc05742fa2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
EXPECTED_PRIOR_STEP_0_SHA256 = "a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34"

ALIGNMENT_RUN = canonical_training.DEFAULT_OUTPUT_DIR
ALIGNMENT_REPORT = ALIGNMENT_RUN / "report.json"
ALIGNMENT_STEP_300 = ALIGNMENT_RUN / "scientific_step_000300.pt"
EXPECTED_ALIGNMENT_REPORT_SHA256 = "440230967def4f758a9d72a7999aac34d6127bce487fb078a2320a9b02ab4423"
EXPECTED_ALIGNMENT_STEP_300_SHA256 = "5efdcfeef9c10a403223c43ae76c464591279261fc593e8e3c1c42016efc1843"

SMOOTH_REALIZATION_REPORT = prior_attribution.SMOOTH_REALIZATION_REPORT
EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256 = prior_attribution.EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256

INDEPENDENT_CHECKPOINT = canonical_training.INDEPENDENT_CHECKPOINT
EXPECTED_INDEPENDENT_CHECKPOINT_SHA256 = canonical_training.EXPECTED_INDEPENDENT_CHECKPOINT_SHA256
EXPECTED_INDEPENDENT_QUALIFICATION_SHA256 = canonical_training.EXPECTED_INDEPENDENT_QUALIFICATION_SHA256
EXPECTED_INDEPENDENT_SPLIT_SHA256 = canonical_training.EXPECTED_INDEPENDENT_SPLIT_SHA256

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03"
)
BANK_SEED = 2026091901
ROWS_PER_CLASS_PER_HALF = 2
MICRO_BATCH_SIZE = 10
PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
INTERPOLATION_LAMBDAS = (0.25, 0.50, 0.75, 1.00)
FEATURE_EPS = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12

SMOOTH_PREPROCESSING = [
    "clamp(x,0,1)",
    "adaptive_avg_pool2d(x,(28,28))",
    "preserve singleton grayscale channel",
    "no mean/std or exposure normalization",
]
INDEPENDENT_PREPROCESSING = list(SMOOTH_PREPROCESSING)
FEATURE_LAYER = "embedding output immediately before classifier"

REPORT_SCHEMA = "cross_evaluator_semantic_divergence_attribution_report.v1"
GRADIENT_SCHEMA = "cross_evaluator_semantic_divergence_gradients.v1"
PROBE_SCHEMA = "cross_evaluator_semantic_divergence_probes.v1"
INTERPOLATION_SCHEMA = "cross_evaluator_semantic_divergence_interpolation.v1"

CLASSIFICATION_OFF_MANIFOLD = "EVALUATOR_SPECIFIC_OFF_MANIFOLD_SEMANTIC_SHORTCUT_SUPPORTED"
CLASSIFICATION_SHARED = "SHARED_CROSS_EVALUATOR_LOCAL_SEMANTIC_DIRECTION_SUPPORTED"
CLASSIFICATION_JACOBIAN = "OPTICAL_REALIZATION_JACOBIAN_CROSS_EVALUATOR_RESTRICTION_SUPPORTED"
CLASSIFICATION_TRAJECTORY = "SMOOTH_GUIDED_TRAJECTORY_INCREASES_SEMANTIC_OFF_MANIFOLD_DIVERGENCE"
CLASSIFICATION_MIXED = "CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_MIXED"
CLASSIFICATION_UNRESOLVED = "CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_MECHANISM_UNRESOLVED"


class AttributionExecutionError(RuntimeError):
    """Raised when an authority or fail-closed diagnostic boundary fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AttributionExecutionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, dict(payload))


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return prior_attribution._state_digest(state)


def _runtime_state(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> dict[str, torch.Tensor]:
    return prior_attribution._runtime_state(consumer)


def _generator(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    return prior_attribution._generator(consumer)


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    prior_attribution._freeze_generator_only(consumer)


def _checkpoint_state(checkpoint: Mapping[str, Any], name: str, expected_step: int) -> dict[str, torch.Tensor]:
    return prior_attribution._checkpoint_state(checkpoint, name, expected_step)


def _select_diagnostic_bank(train_indices: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    return prior_attribution._select_diagnostic_bank(train_indices, labels)


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    return prior_attribution._cmos_view(value)


def _normalized_feature_loss(cmos: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    return prior_attribution._normalized_feature_loss(cmos, target, name)


def _norm(value: torch.Tensor, name: str) -> float:
    return prior_attribution._norm(value, name)


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    return prior_attribution._cosine(left, right, name)


def _gradient_stats(value: torch.Tensor, name: str, *, phase: bool = False) -> dict[str, Any]:
    return prior_attribution._gradient_stats(value, name, phase=phase)


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    return prior_attribution._raw_validity(raw)


def _preprocessing_support(raw: torch.Tensor, gradient: torch.Tensor, evaluator_name: str) -> dict[str, Any]:
    active = (raw.detach() > 0.0) & (raw.detach() < 1.0)
    blocked = ~active
    return {
        "active_unclamped_fraction": float(active.to(torch.float64).mean().cpu()),
        "saturation_or_clamp_blocked_fraction": float(blocked.to(torch.float64).mean().cpu()),
        "gradient_nonzero_fraction": float((gradient != 0.0).to(torch.float64).mean().cpu()),
        "gradient_nonzero_inside_active_fraction": float(((gradient != 0.0) & active).to(torch.float64).mean().cpu()),
        "evaluator": evaluator_name,
        "definition": "raw CMOS values strictly inside (0,1) are active; values at or outside clamp bounds are blocked/descriptive",
    }


def _load_smooth_contract(root: Path) -> dict[str, Any]:
    report_path = root / SMOOTH_REALIZATION_REPORT
    _require(file_sha256(report_path) == EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256, "accepted Smooth realization report identity changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    positive = _mapping(report.get("real_image_positive_control"), "Smooth real-image positive control")
    _require(positive.get("assessment") == "strong", "Smooth canonical-real-image positive control is not strong")
    metrics = _mapping(positive.get("metrics"), "Smooth positive-control metrics")
    _require(int(metrics.get("rows")) == 5000, "Smooth positive-control row count changed")
    _require(math.isclose(float(metrics.get("macro_recall")), 0.988, rel_tol=0.0, abs_tol=1.0e-12), "Smooth positive-control macro recall changed")
    contract = _mapping(report.get("smooth_contract"), "Smooth contract")
    current = _mapping(contract.get("current_contract"), "Smooth current contract")
    evaluator = _mapping(current.get("evaluator"), "Smooth evaluator contract")
    _require(evaluator.get("feature_layer") == FEATURE_LAYER, "Smooth feature layer changed")
    _require(evaluator.get("feature_dimension") == 128, "Smooth feature dimension changed")
    _require(current.get("preprocessing_sequence") == SMOOTH_PREPROCESSING, "Smooth preprocessing contract changed")
    _require(_mapping(current.get("qualification"), "Smooth qualification").get("status") == "passed", "Smooth qualification did not pass")
    checkpoint = _mapping(evaluator.get("checkpoint"), "Smooth checkpoint contract")
    _require(checkpoint.get("sha256") == successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256, "Smooth checkpoint contract changed")
    return {
        "report": _artifact_ref(root, report_path, role="accepted Smooth semantic realization contract"),
        "positive_control": dict(positive),
        "contract": dict(current),
        "role": "DEVELOPMENT_DIAGNOSTIC",
    }


def _load_alignment_authority(root: Path) -> dict[str, Any]:
    authorities = canonical_training._load_authorities(root)
    report_path = root / ALIGNMENT_REPORT
    _require(file_sha256(report_path) == EXPECTED_ALIGNMENT_REPORT_SHA256, "accepted Smooth alignment report identity changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("result") == "VALID_COMPLETED", "accepted Smooth alignment result is not valid completed")
    _require(report.get("outer_gate") == canonical_training.OUTER_GATE, "accepted Smooth alignment outer gate changed")
    _require(report.get("classification") == canonical_training.SMOOTH_ONLY_CLASSIFICATION, "accepted Smooth alignment classification changed")
    _require(report.get("outer_gate_passed") is False, "accepted Smooth alignment unexpectedly passed its gate")
    _require(report.get("training_trajectory", {}).get("updates") == 300, "accepted Smooth alignment update count changed")
    _require(report.get("validation", {}).get("independent_confirmation") == [0, 300], "accepted confirmation endpoint changed")
    _require(report.get("independent_confirmation", {}).get("step_100_200_confirmation_inspected") is False, "accepted confirmation inspection scope changed")
    _require(report.get("trainable_frozen", {}).get("trainable") == ["generator.phase"], "accepted Generator trainable boundary changed")
    _require(report.get("trainable_frozen", {}).get("frozen")[-1] == "independent confirmation evaluator", "accepted independent evaluator freeze boundary changed")
    starting = _mapping(report.get("starting_authority"), "accepted alignment starting authority")
    _require(starting.get("accepted_clean_step0", {}).get("sha256") == EXPECTED_PRIOR_STEP_0_SHA256, "accepted alignment clean step-0 identity changed")
    clean_step0_payload = _mapping(authorities.get("step0"), "accepted clean step-0 checkpoint")
    clean_step0 = _checkpoint_state(clean_step0_payload, "clean step-0", 0)
    step300_payload = safe_torch_load(root / ALIGNMENT_STEP_300, expected_sha256=EXPECTED_ALIGNMENT_STEP_300_SHA256, map_location="cpu")
    _require(isinstance(step300_payload, Mapping), "accepted Smooth alignment step-300 checkpoint is malformed")
    step300 = _checkpoint_state(step300_payload, "Smooth-guided step-300", 300)
    _require(torch.equal(clean_step0["encoder.phase"], step300["encoder.phase"]), "accepted Encoder changed between step 0 and Smooth step 300")
    _require(not torch.equal(clean_step0["generator.phase"], step300["generator.phase"]), "accepted Smooth-guided Generator did not change")
    smooth_contract = _load_smooth_contract(root)
    independent = _mapping(authorities.get("independent"), "independent evaluator authority")
    _require(independent.get("architecture") == "FrozenMNISTEvaluator v1", "independent evaluator architecture changed")
    _require(independent.get("feature_layer") == FEATURE_LAYER and independent.get("feature_dimension") == 128, "independent feature contract changed")
    authorities["bank"] = _select_diagnostic_bank(authorities["train_indices"], authorities["labels"])
    return {
        **authorities,
        "step0": clean_step0,
        "alignment_report": report,
        "alignment_step300": step300,
        "smooth_contract": smooth_contract,
        "authority_hashes": {
            **authorities["authority_hashes"],
            "accepted_alignment_report": EXPECTED_ALIGNMENT_REPORT_SHA256,
            "accepted_alignment_step300": EXPECTED_ALIGNMENT_STEP_300_SHA256,
            "accepted_smooth_realization_report": EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256,
            "independent_confirmation_checkpoint": EXPECTED_INDEPENDENT_CHECKPOINT_SHA256,
            "independent_confirmation_qualification": EXPECTED_INDEPENDENT_QUALIFICATION_SHA256,
            "independent_confirmation_split": EXPECTED_INDEPENDENT_SPLIT_SHA256,
        },
    }


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, SmoothMNISTEvaluator, FrozenMNISTEvaluator, dict[str, Any]]:
    training = _mapping(authorities.get("training"), "training authority")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    bundle = _mapping(training.get("bundle"), "accepted generation bundle")
    _require(consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"), "runtime identity differs from accepted bundle")
    consumer.load_trainable_state_dict(authorities["step0"])
    consumer.bind_references(training["references"], for_scientific=True)
    consumer.bind_weights(training["weights"], for_scientific=True)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()

    smooth, smooth_identity = successor._load_smooth_evaluator(root, device)
    independent, independent_payload = load_frozen_evaluator(root / INDEPENDENT_CHECKPOINT, device=device, min_validation_accuracy=0.985)
    _require(independent_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "independent evaluator architecture changed")
    _require(all(not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator is not frozen")
    _require(all(not parameter.requires_grad for parameter in independent.parameters()), "independent evaluator is not frozen")
    _require(file_sha256(root / INDEPENDENT_CHECKPOINT) == EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, "independent evaluator identity changed")
    _require(prior_attribution._state_digest(smooth.state_dict()) != prior_attribution._state_digest(independent.state_dict()), "Smooth and independent evaluator states are not distinct")
    return consumer, smooth, independent, {
        "smooth": smooth_identity,
        "smooth_feature_layer": FEATURE_LAYER,
        "smooth_preprocessing": SMOOTH_PREPROCESSING,
        "independent": authorities["independent"],
        "independent_feature_layer": FEATURE_LAYER,
        "independent_preprocessing": INDEPENDENT_PREPROCESSING,
        "independent_in_backward": False,
        "smooth_in_backward": False,
    }


def _feature_bundle(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    batch: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    result = consumer.runtime.forward_from_image(batch)
    _require(result.image_input is not None and result.ols_input_field is not None, "real optical path is incomplete")
    raw = result.raw_intensity
    target_image = result.image_input.slm_intensity
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/target optical geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "raw/target optical dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_image, "SLM target image")
    target_view = _cmos_view(batch)
    smooth_cmos, _ = smooth(_cmos_view(raw))
    independent_cmos, _ = independent(_cmos_view(raw))
    with torch.no_grad():
        smooth_target, _ = smooth(target_view)
        independent_target, _ = independent(target_view)
    l_smooth = _normalized_feature_loss(smooth_cmos, smooth_target, "L_S")
    l_independent = _normalized_feature_loss(independent_cmos, independent_target, "L_I")
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    for name, value in (("L_S", l_smooth), ("L_I", l_independent), ("L_rec", l_rec), ("L_op", l_op)):
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target_image": target_image,
        "target_view": target_view.detach(),
        "smooth_target": smooth_target.detach(),
        "independent_target": independent_target.detach(),
        "L_S": l_smooth,
        "L_I": l_independent,
        "L_rec": l_rec,
        "L_op": l_op,
    }


def _analyze_step(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    images: torch.Tensor,
    indices: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    checkpoint_name: str,
    ledger: DataReadLedger,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    generator = _generator(consumer)
    n = int(indices.numel())
    _require(n > 0 and n % MICRO_BATCH_SIZE == 0, "TRAIN A bank is not divisible by the fixed micro-batch")
    expected_step = {"step-0": 0, "step-300": 300}.get(checkpoint_name)
    _require(expected_step is not None, f"unknown accepted checkpoint name: {checkpoint_name}")
    accumulator = {name: torch.zeros_like(generator) for name in ("g_S", "g_I")}
    h_s_chunks: list[torch.Tensor] = []
    h_i_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    target_view_chunks: list[torch.Tensor] = []
    smooth_target_chunks: list[torch.Tensor] = []
    independent_target_chunks: list[torch.Tensor] = []
    scalar_totals = {name: 0.0 for name in ("L_S", "L_I", "L_rec", "L_op")}
    for start in range(0, n, MICRO_BATCH_SIZE):
        batch_indices = indices[start : start + MICRO_BATCH_SIZE]
        batch = images.index_select(0, batch_indices).to(consumer.device)
        bundle = _feature_bundle(consumer, smooth, independent, batch, references)
        h_s, g_s = torch.autograd.grad(bundle["L_S"], (bundle["raw"], generator), retain_graph=True, allow_unused=False)
        h_i, g_i = torch.autograd.grad(bundle["L_I"], (bundle["raw"], generator), retain_graph=False, allow_unused=False)
        weight = batch_indices.numel() / n
        accumulator["g_S"] += weight * g_s.detach()
        accumulator["g_I"] += weight * g_i.detach()
        h_s_chunks.append((weight * h_s.detach()).cpu())
        h_i_chunks.append((weight * h_i.detach()).cpu())
        raw_chunks.append(bundle["raw"].detach().cpu())
        target_view_chunks.append(bundle["target_view"].cpu())
        smooth_target_chunks.append(bundle["smooth_target"].cpu())
        independent_target_chunks.append(bundle["independent_target"].cpu())
        for name in scalar_totals:
            scalar_totals[name] += weight * float(bundle[name].detach().cpu())
        ledger.record(TRAIN_ROLE, int(batch_indices.numel()))
        _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "diagnostic autograd populated runtime .grad")
    h_s_all = torch.cat(h_s_chunks, dim=0)
    h_i_all = torch.cat(h_i_chunks, dim=0)
    raw_all = torch.cat(raw_chunks, dim=0)
    target_views = torch.cat(target_view_chunks, dim=0)
    smooth_targets = torch.cat(smooth_target_chunks, dim=0)
    independent_targets = torch.cat(independent_target_chunks, dim=0)
    _finite(h_s_all, f"{checkpoint_name} h_S")
    _finite(h_i_all, f"{checkpoint_name} h_I")
    _finite(raw_all, f"{checkpoint_name} raw")
    gradients = {"h_S": h_s_all, "h_I": h_i_all, "g_S": accumulator["g_S"], "g_I": accumulator["g_I"]}
    result = {
        "checkpoint": checkpoint_name,
        "completed_updates": expected_step,
        "rows": n,
        "losses": scalar_totals,
        "output_space": {
            "h_S": _gradient_stats(h_s_all, f"{checkpoint_name} h_S"),
            "h_I": _gradient_stats(h_i_all, f"{checkpoint_name} h_I"),
            "cos_h_S_h_I": _cosine(h_s_all, h_i_all, f"{checkpoint_name} h_S/h_I"),
            "preprocessing_support": {
                "Smooth": _preprocessing_support(raw_all, h_s_all, "Smooth"),
                "independent": _preprocessing_support(raw_all, h_i_all, "independent"),
            },
        },
        "parameter_space": {
            "g_S": _gradient_stats(accumulator["g_S"], f"{checkpoint_name} g_S", phase=True),
            "g_I": _gradient_stats(accumulator["g_I"], f"{checkpoint_name} g_I", phase=True),
            "cos_g_S_g_I": _cosine(accumulator["g_S"], accumulator["g_I"], f"{checkpoint_name} g_S/g_I"),
        },
        "raw_validity": _raw_validity(raw_all),
    }
    intermediates = {
        **gradients,
        "raw": raw_all,
        "target_views": target_views,
        "smooth_targets": smooth_targets,
        "independent_targets": independent_targets,
    }
    return result, intermediates


def _normalize_rms(value: torch.Tensor, name: str) -> torch.Tensor:
    _finite(value, name)
    rms = float(value.detach().to(torch.float64).square().mean().sqrt().cpu())
    _require(math.isfinite(rms) and rms > 0.0, f"{name} RMS is zero or invalid")
    direction = -value.detach() / rms
    _require(math.isclose(float(direction.to(torch.float64).square().mean().sqrt().cpu()), 1.0, rel_tol=1.0e-5), f"{name} RMS normalization failed")
    return direction


def _direction_bundle(gradients: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    g_s = gradients["g_S"]
    g_i = gradients["g_I"]
    s_norm = _norm(g_s, "g_S")
    i_norm = _norm(g_i, "g_I")
    u_s = g_s.detach() / s_norm
    u_i = g_i.detach() / i_norm
    v_joint = u_s + u_i
    joint_norm = float(v_joint.to(torch.float64).norm().cpu())
    _require(math.isfinite(joint_norm), "joint bisector norm is nonfinite")
    direction = {
        "d_S": _normalize_rms(g_s, "d_S"),
        "d_I": _normalize_rms(g_i, "d_I"),
    }
    if joint_norm > JOINT_DEGENERACY_TOLERANCE:
        direction["d_joint"] = _normalize_rms(v_joint, "d_joint")
        joint = {"status": "DEFINED", "v_joint_l2_norm": joint_norm, "u_S_l2_norm": s_norm, "u_I_l2_norm": i_norm}
    else:
        joint = {"status": "JOINT_BISECTOR_DEGENERATE", "v_joint_l2_norm": joint_norm, "u_S_l2_norm": s_norm, "u_I_l2_norm": i_norm}
    return direction, joint


@torch.no_grad()
def _evaluate_b(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    images: torch.Tensor,
    references: Any,
    ledger: DataReadLedger,
) -> dict[str, Any]:
    total = {name: 0.0 for name in ("L_S", "L_I", "L_rec", "L_op")}
    raw_parts: list[torch.Tensor] = []
    count = int(images.shape[0])
    for start in range(0, count, MICRO_BATCH_SIZE):
        batch = images[start : start + MICRO_BATCH_SIZE]
        bundle = _feature_bundle(consumer, smooth, independent, batch, references)
        weight = batch.shape[0] / count
        for name in total:
            total[name] += weight * float(bundle[name].detach().cpu())
        raw_parts.append(bundle["raw"].detach().cpu())
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    raw_all = torch.cat(raw_parts, dim=0)
    return {"losses": total, "raw_validity": _raw_validity(raw_all)}


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(candidate["losses"][name] - base["losses"][name]) for name in ("L_S", "L_I", "L_rec", "L_op")}


def _run_probes(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    images: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    gradients: Mapping[str, torch.Tensor],
    checkpoint_name: str,
    ledger: DataReadLedger,
) -> dict[str, Any]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    base_state = _runtime_state(consumer)
    directions, joint = _direction_bundle(gradients)
    b_images = images.to(consumer.device)
    base = _evaluate_b(consumer, smooth, independent, b_images, references, ledger)
    records: dict[str, Any] = {}
    try:
        for direction_name, direction in directions.items():
            epsilon_records: list[dict[str, Any]] = []
            for epsilon in PERTURBATION_EPSILONS:
                signed: dict[str, Any] = {}
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    with torch.no_grad():
                        _generator(consumer).add_(sign * epsilon * direction.to(consumer.device))
                    candidate = _evaluate_b(consumer, smooth, independent, b_images, references, ledger)
                    signed[sign_name] = {
                        "deltas": _metric_deltas(base, candidate),
                        "losses": candidate["losses"],
                        "raw_validity": candidate["raw_validity"],
                    }
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{checkpoint_name} {direction_name} probe was not restored")
                plus = signed["plus"]["deltas"]
                minus = signed["minus"]["deltas"]
                signed["central_slope_per_epsilon"] = {metric: float((plus[metric] - minus[metric]) / (2.0 * epsilon)) for metric in ("L_S", "L_I", "L_rec", "L_op")}
                epsilon_records.append({"epsilon": epsilon, **signed})
            records[direction_name] = epsilon_records
    finally:
        consumer.load_trainable_state_dict(checkpoint)
        _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{checkpoint_name} probes did not restore the checkpoint")
    return {
        "schema": PROBE_SCHEMA,
        "checkpoint": checkpoint_name,
        "epsilon_definition": "epsilon radians RMS phase added to generator.phase",
        "epsilons": list(PERTURBATION_EPSILONS),
        "base": base,
        "directions": records,
        "joint_bisector": joint,
        "checkpoint_restored_after_every_probe": True,
    }


def _stable_descent(probes: Mapping[str, Any], direction: str, metric: str) -> bool:
    records = _mapping(probes.get("directions"), "probe directions").get(direction)
    if not isinstance(records, list) or not records:
        return False
    return all(
        float(_mapping(record["plus"], "plus probe")["deltas"][metric]) < 0.0
        and float(_mapping(record["minus"], "minus probe")["deltas"][metric]) > 0.0
        for record in records
    )


def _sign_summary(probes: Mapping[str, Any], direction: str, metric: str) -> dict[str, Any]:
    records = _mapping(probes.get("directions"), "probe directions").get(direction)
    _require(isinstance(records, list), f"missing probe direction {direction}")
    plus = [float(record["plus"]["deltas"][metric]) for record in records]
    minus = [float(record["minus"]["deltas"][metric]) for record in records]
    optical_validity = all(
        bool(record[side]["raw_validity"]["finite"])
        and float(record[side]["raw_validity"]["nonconstant_fraction"]) == 1.0
        and int(record[side]["raw_validity"]["zero_image_count"]) == 0
        for record in records
        for side in ("plus", "minus")
    )
    return {
        "plus_deltas": plus,
        "minus_deltas": minus,
        "stable_descent": all(p < 0.0 and m > 0.0 for p, m in zip(plus, minus)),
        "stable_ascent": all(p > 0.0 and m < 0.0 for p, m in zip(plus, minus)),
        "optical_validity": optical_validity,
    }


def _direction_score(probes: Mapping[str, Any], direction: str) -> int:
    return int(_stable_descent(probes, direction, "L_S")) + int(_stable_descent(probes, direction, "L_I"))


def _classify(
    step0: Mapping[str, Any],
    step300: Mapping[str, Any],
    probes0: Mapping[str, Any],
    probes300: Mapping[str, Any],
    interpolation0: Mapping[str, Any],
    interpolation300: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str, list[str]]:
    joint0 = _mapping(probes0.get("joint_bisector"), "step-0 joint bisector")
    joint300 = _mapping(probes300.get("joint_bisector"), "step-300 joint bisector")
    joint0_defined = joint0.get("status") == "DEFINED"
    joint300_defined = joint300.get("status") == "DEFINED"
    joint0_both = joint0_defined and _stable_descent(probes0, "d_joint", "L_S") and _stable_descent(probes0, "d_joint", "L_I")
    joint300_both = joint300_defined and _stable_descent(probes300, "d_joint", "L_S") and _stable_descent(probes300, "d_joint", "L_I")
    cross0 = _stable_descent(probes0, "d_S", "L_S") and not _stable_descent(probes0, "d_S", "L_I") and _stable_descent(probes0, "d_I", "L_I") and not _stable_descent(probes0, "d_I", "L_S")
    cross300 = _stable_descent(probes300, "d_S", "L_S") and not _stable_descent(probes300, "d_S", "L_I") and _stable_descent(probes300, "d_I", "L_I") and not _stable_descent(probes300, "d_I", "L_S")
    interpolation0 = _mapping(interpolation0.get("summary"), "step-0 interpolation summary")
    interpolation300 = _mapping(interpolation300.get("summary"), "step-300 interpolation summary")
    off_manifold0 = float(interpolation0.get("cos_at_lambda_0.25")) > float(interpolation0.get("cos_at_lambda_1.0"))
    off_manifold300 = float(interpolation300.get("cos_at_lambda_0.25")) > float(interpolation300.get("cos_at_lambda_1.0"))
    output_parameter_gap0 = float(step0["output_space"]["cos_h_S_h_I"]) > float(step0["parameter_space"]["cos_g_S_g_I"])
    output_parameter_gap300 = float(step300["output_space"]["cos_h_S_h_I"]) > float(step300["parameter_space"]["cos_g_S_g_I"])
    trajectory = joint0_both and not joint300_both
    jacobian = (output_parameter_gap0 or output_parameter_gap300) and not joint0_both and not joint300_both
    off_manifold = (cross0 or cross300) and not joint0_both and not joint300_both and (off_manifold0 or off_manifold300)
    mechanisms: list[str] = []
    if off_manifold:
        mechanisms.append(CLASSIFICATION_OFF_MANIFOLD)
    if joint0_both or joint300_both:
        mechanisms.append(CLASSIFICATION_SHARED)
    if jacobian:
        mechanisms.append(CLASSIFICATION_JACOBIAN)
    if trajectory:
        mechanisms.append(CLASSIFICATION_TRAJECTORY)
    if not mechanisms:
        classification = CLASSIFICATION_UNRESOLVED
    elif len(mechanisms) == 1:
        classification = mechanisms[0]
    else:
        classification = CLASSIFICATION_MIXED
    if classification == CLASSIFICATION_OFF_MANIFOLD:
        recommendation = "DEFINE_EVALUATOR_AGNOSTIC_IMAGE_MANIFOLD_ALIGNMENT_V1"
    elif classification == CLASSIFICATION_SHARED:
        recommendation = "DEFINE_EVALUATOR_ROBUST_SEMANTIC_TRAINING_V1"
    elif classification == CLASSIFICATION_JACOBIAN:
        recommendation = "DEFINE_OPTICAL_REALIZATION_JACOBIAN_CROSS_EVALUATOR_STUDY_V1"
    elif classification == CLASSIFICATION_TRAJECTORY:
        recommendation = "USE_STEP0_SHARED_DIRECTION_AS_INPUT_TO_A_FOLLOWUP_GATE"
    elif classification == CLASSIFICATION_MIXED:
        recommendation = "SELECT_ONE_SUCCESSOR_ONLY_AFTER_REVIEWING_SUPPORTED_MECHANISMS"
    else:
        recommendation = "NONE"
    details = {
        "flags": {
            "step0_joint_direction_improves_both": joint0_both,
            "step300_joint_direction_improves_both": joint300_both,
            "step0_evaluator_specific_cross_effects": cross0,
            "step300_evaluator_specific_cross_effects": cross300,
            "step0_interpolation_agreement_decreases_toward_cmos": off_manifold0,
            "step300_interpolation_agreement_decreases_toward_cmos": off_manifold300,
            "step0_output_space_more_aligned_than_parameter_space": output_parameter_gap0,
            "step300_output_space_more_aligned_than_parameter_space": output_parameter_gap300,
        },
        "mechanisms_supported": mechanisms,
        "step0_direction_scores": {name: _direction_score(probes0, name) for name in ("d_S", "d_I")},
        "step300_direction_scores": {name: _direction_score(probes300, name) for name in ("d_S", "d_I")},
        "no_universal_cosine_cutoff": True,
    }
    return classification, details, recommendation, mechanisms


def _interpolation_step(
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    target_views: torch.Tensor,
    cmos_views: torch.Tensor,
    smooth_targets: torch.Tensor,
    independent_targets: torch.Tensor,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for value in INTERPOLATION_LAMBDAS:
        interpolated = ((1.0 - value) * target_views + value * cmos_views).detach().clone().requires_grad_(True)
        smooth_features, _ = smooth(_cmos_view(interpolated))
        independent_features, _ = independent(_cmos_view(interpolated))
        l_smooth = _normalized_feature_loss(smooth_features, smooth_targets, "interpolation L_S")
        l_independent = _normalized_feature_loss(independent_features, independent_targets, "interpolation L_I")
        h_s = torch.autograd.grad(l_smooth, interpolated, retain_graph=True, allow_unused=False)[0]
        h_i = torch.autograd.grad(l_independent, interpolated, retain_graph=False, allow_unused=False)[0]
        records.append({
            "lambda": value,
            "L_S": float(l_smooth.detach().cpu()),
            "L_I": float(l_independent.detach().cpu()),
            "h_S": _gradient_stats(h_s, f"interpolation lambda {value} h_S"),
            "h_I": _gradient_stats(h_i, f"interpolation lambda {value} h_I"),
            "cos_h_S_h_I": _cosine(h_s, h_i, f"interpolation lambda {value} h_S/h_I"),
        })
    summary = {f"cos_at_lambda_{record['lambda']}": record["cos_h_S_h_I"] for record in records}
    return {"schema": INTERPOLATION_SCHEMA, "gradient_space": "shared 28x28 evaluator image input after exact raw-CMOS clamp/pool preprocessing", "lambdas": records, "summary": summary}


def _run_preflight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    images: torch.Tensor,
    indices: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    ledger: DataReadLedger,
) -> dict[str, Any]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    before = _state_digest(_runtime_state(consumer))
    batch_indices = indices[:MICRO_BATCH_SIZE]
    batch = images.index_select(0, batch_indices).to(consumer.device)
    bundle = _feature_bundle(consumer, smooth, independent, batch, references)
    h_s, g_s = torch.autograd.grad(bundle["L_S"], (bundle["raw"], _generator(consumer)), retain_graph=True, allow_unused=False)
    h_i, g_i = torch.autograd.grad(bundle["L_I"], (bundle["raw"], _generator(consumer)), retain_graph=False, allow_unused=False)
    ledger.record(TRAIN_ROLE, int(batch_indices.numel()))
    _finite(h_s, "preflight h_S")
    _finite(h_i, "preflight h_I")
    _finite(g_s, "preflight g_S")
    _finite(g_i, "preflight g_I")
    _require(before == _state_digest(_runtime_state(consumer)), "preflight changed runtime state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "preflight left runtime gradients")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "preflight left Smooth gradients")
    _require(all(parameter.grad is None for parameter in independent.parameters()), "preflight left independent gradients")
    return {
        "status": "READY",
        "optimizer_created": False,
        "parameter_updates": 0,
        "representative_rows": int(batch_indices.numel()),
        "output_gradient": True,
        "parameter_gradient": True,
        "runtime_state_restored": True,
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "SmoothMNISTEvaluator", "FrozenMNISTEvaluator", "P2", "Q2"],
        "protected_test_sealed_zero": True,
    }


def _make_identity(source_manifest: Mapping[str, Any], bank: Mapping[str, torch.Tensor], attempt_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=f"{RUN_ID}:trajectory",
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={"no_update": True, "bank_seed": BANK_SEED, "rows_per_class_per_half": ROWS_PER_CLASS_PER_HALF, "epsilons": list(PERTURBATION_EPSILONS), "interpolation_lambdas": list(INTERPOLATION_LAMBDAS), "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()}},
        model_identity="LIGHTGEN_2D_CROSS_EVALUATOR_SEMANTIC_DIVERGENCE_ATTRIBUTION_V1",
        optimizer_identity={"created": False, "parameter_updates": 0},
        sampler_identity={"kind": "deterministic class-balanced TRAIN-only A/B bank", "seed": BANK_SEED},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=attempt_id,
    )


def run_analysis(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")
    local_state = git_identity(root)
    _require(local_state["branch"] == EXPECTED_BRANCH and local_state["head"] == EXPECTED_HEAD, "local branch/HEAD conflicts with accepted authority")
    authorities = _load_alignment_authority(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "cross-evaluator attribution requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, evaluator_identity = _load_models(root, authorities, device)
    bank = authorities["bank"]
    ledger = DataReadLedger()
    preflight = _run_preflight(consumer, smooth, independent, authorities["images"], bank["a_indices"], authorities["training"]["references"], authorities["step0"], ledger)
    initial_runtime_state = _runtime_state(consumer)

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(prior_attribution.__file__).resolve(),
            Path(canonical_training.__file__).resolve(),
            Path(conflict.__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            Path(producer.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={
            "authorization_mode": "IMPLEMENT_PREPARED",
            "authorization_token": AUTHORIZATION_TOKEN,
            "outer_gate": OUTER_GATE,
            "no_update": True,
            "bank_seed": BANK_SEED,
            "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()},
            "epsilons": list(PERTURBATION_EPSILONS),
            "interpolation_lambdas": list(INTERPOLATION_LAMBDAS),
            "input_hashes": authorities["authority_hashes"],
        },
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    attempt_id = f"{RUN_ID}:{target.name}"
    run_identity = _make_identity(source_manifest, bank, attempt_id)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    control.mark_preflight_ready()
    control.start_execution()
    effective_identity = control.begin_scientific_attempt(attempt_id)

    _write_torch(
        target / "diagnostic_bank.pt",
        {
            "schema": "cross_evaluator_semantic_divergence_bank.v1",
            "task_id": TASK_ID,
            "seed": BANK_SEED,
            "data_role": TRAIN_ROLE,
            "a_indices": bank["a_indices"],
            "a_labels": bank["a_labels"],
            "b_indices": bank["b_indices"],
            "b_labels": bank["b_labels"],
            "rows_per_class_per_half": ROWS_PER_CLASS_PER_HALF,
            "development_rows_consumed": 0,
        },
        kind="TRAIN-only cross-evaluator attribution bank",
    )
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "cross_evaluator_semantic_divergence_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
            "input_authorities": authorities["authority_hashes"],
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "local_state": local_state,
            "cuda_runtime": cuda_runtime,
            "scope": {"optimizer_created": False, "parameter_updates": 0, "train_rows_unique": int(torch.cat((bank["a_indices"], bank["b_indices"])).numel()), "development_rows_consumed": 0, "protected": False, "test": False, "sealed": False},
        },
    )
    a_indices = bank["a_indices"]
    b_indices = bank["b_indices"]
    references = authorities["training"]["references"]
    step0_result, step0_gradients = _analyze_step(consumer, smooth, independent, authorities["images"], a_indices, references, authorities["step0"], "step-0", ledger)
    step0_interpolation = _interpolation_step(smooth, independent, step0_gradients["target_views"].to(device), _cmos_view(step0_gradients["raw"].to(device)), step0_gradients["smooth_targets"].to(device), step0_gradients["independent_targets"].to(device))
    step0_probes = _run_probes(consumer, smooth, independent, authorities["images"].index_select(0, b_indices), references, authorities["step0"], step0_gradients, "step-0", ledger)
    step300_result, step300_gradients = _analyze_step(consumer, smooth, independent, authorities["images"], a_indices, references, authorities["alignment_step300"], "step-300", ledger)
    step300_interpolation = _interpolation_step(smooth, independent, step300_gradients["target_views"].to(device), _cmos_view(step300_gradients["raw"].to(device)), step300_gradients["smooth_targets"].to(device), step300_gradients["independent_targets"].to(device))
    step300_probes = _run_probes(consumer, smooth, independent, authorities["images"].index_select(0, b_indices), references, authorities["alignment_step300"], step300_gradients, "step-300", ledger)

    consumer.load_trainable_state_dict(authorities["step0"])
    _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_runtime_state), "final runtime restoration failed")
    _require(ledger.counts.get("dev", 0) == 0 and ledger.counts.get("protected", 0) == 0 and ledger.counts.get("test", 0) == 0 and ledger.counts.get("sealed", 0) == 0, "forbidden data role was read")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "runtime gradients remained after attribution")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "Smooth gradients remained after attribution")
    _require(all(parameter.grad is None for parameter in independent.parameters()), "independent gradients remained after attribution")

    _write_torch(
        target / "gradient_vectors.pt",
        {
            "schema": GRADIENT_SCHEMA,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "checkpoint_hashes": {"step0": EXPECTED_PRIOR_STEP_0_SHA256, "step300": EXPECTED_ALIGNMENT_STEP_300_SHA256},
            "bank_indices": {"a": a_indices, "b": b_indices},
            "step0": {name: value.detach().cpu() for name, value in step0_gradients.items() if isinstance(value, torch.Tensor)},
            "step300": {name: value.detach().cpu() for name, value in step300_gradients.items() if isinstance(value, torch.Tensor)},
        },
        kind="cross-evaluator output and parameter gradient vectors",
    )
    _write_json(target / "directional_probes.json", {"schema": PROBE_SCHEMA, "step0": step0_probes, "step300": step300_probes})
    _write_json(target / "interpolation.json", {"schema": INTERPOLATION_SCHEMA, "step0": step0_interpolation, "step300": step300_interpolation})

    classification, classification_details, recommendation, mechanisms = _classify(step0_result, step300_result, step0_probes, step300_probes, step0_interpolation, step300_interpolation)
    control.complete()
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "classification": classification,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": local_state,
        "input_authorities": {
            "accepted_clean_conflict_report": _artifact_ref(root, root / PRIOR_REPORT, role="accepted clean 2L Generator authority"),
            "accepted_clean_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="accepted clean step-0 Generator"),
            "accepted_smooth_alignment_report": _artifact_ref(root, root / ALIGNMENT_REPORT, role="accepted Smooth-guided alignment result"),
            "accepted_smooth_alignment_step300": _artifact_ref(root, root / ALIGNMENT_STEP_300, role="accepted Smooth-guided step-300 Generator"),
            "accepted_smooth_realization_report": _artifact_ref(root, root / SMOOTH_REALIZATION_REPORT, role="accepted Smooth contract and positive control"),
            "authority_hashes": authorities["authority_hashes"],
            "train_data_identity": authorities["data_identity"],
        },
        "evaluator_qualification": {
            "Smooth": {"role": "DEVELOPMENT_DIAGNOSTIC", "feature_layer": FEATURE_LAYER, "preprocessing": SMOOTH_PREPROCESSING, "positive_control": authorities["smooth_contract"]["positive_control"], "contract": authorities["smooth_contract"]["contract"]},
            "independent": {**dict(authorities["independent"]), "role": "DEVELOPMENT_DIAGNOSTIC", "preprocessing": INDEPENDENT_PREPROCESSING},
        },
        "frozen_contracts": {"encoder": True, "OLS": True, "W_OLS": True, "generator_architecture": True, "P2": True, "Q2": True, "Smooth_parameters": True, "independent_parameters": True, "no_evaluator_teacher": True},
        "diagnostic_bank": {"artifact": _artifact_ref(root, target / "diagnostic_bank.pt", role="frozen TRAIN-only A/B bank"), "seed": BANK_SEED, "a_rows": int(a_indices.numel()), "b_rows": int(b_indices.numel()), "class_balanced": True, "a_indices_sha256": tensor_sha256(a_indices), "b_indices_sha256": tensor_sha256(b_indices), "development_rows_consumed": 0},
        "objective_definitions": {"L_S": "mean(||normalize(F_S(preprocess_S(CMOS)))-normalize(F_S(preprocess_S(target)))||_2^2), target features stop-gradient", "L_I": "mean(||normalize(F_I(preprocess_I(CMOS)))-normalize(F_I(preprocess_I(target)))||_2^2), target features stop-gradient", "feature_layer": FEATURE_LAYER, "normalization": "L2 normalization with epsilon 1e-12", "target_image_contract": "canonical TRAIN image paired with the same exact evaluator preprocessing used for raw CMOS; physical interpolation is diagnostic-only"},
        "preflight": preflight,
        "step0": step0_result,
        "step300": step300_result,
        "gradient_attribution": {"artifact": _artifact_ref(root, target / "gradient_vectors.pt", role="A-bank output/parameter-space gradient vectors"), "raw_cmos_gradient_space": "common raw CMOS tensor before evaluator preprocessing", "cosines": {"step0_h_S_h_I": step0_result["output_space"]["cos_h_S_h_I"], "step300_h_S_h_I": step300_result["output_space"]["cos_h_S_h_I"], "step0_g_S_g_I": step0_result["parameter_space"]["cos_g_S_g_I"], "step300_g_S_g_I": step300_result["parameter_space"]["cos_g_S_g_I"]}, "output_minus_parameter": {"step0": float(step0_result["output_space"]["cos_h_S_h_I"] - step0_result["parameter_space"]["cos_g_S_g_I"]), "step300": float(step300_result["output_space"]["cos_h_S_h_I"] - step300_result["parameter_space"]["cos_g_S_g_I"])}, "no_universal_cosine_cutoff": True},
        "held_out_evaluator_specific_probes": {"artifact": _artifact_ref(root, target / "directional_probes.json", role="symmetric held-out B probes"), "step0": step0_probes, "step300": step300_probes, "primary_metrics": ["delta_L_S", "delta_L_I", "delta_L_rec", "raw_validity"], "d_S_effects": {"step0": {"Smooth": _sign_summary(step0_probes, "d_S", "L_S"), "independent": _sign_summary(step0_probes, "d_S", "L_I")}, "step300": {"Smooth": _sign_summary(step300_probes, "d_S", "L_S"), "independent": _sign_summary(step300_probes, "d_S", "L_I")}}, "d_I_effects": {"step0": {"Smooth": _sign_summary(step0_probes, "d_I", "L_S"), "independent": _sign_summary(step0_probes, "d_I", "L_I")}, "step300": {"Smooth": _sign_summary(step300_probes, "d_I", "L_S"), "independent": _sign_summary(step300_probes, "d_I", "L_I")}}, "checkpoint_restored_after_every_probe": True},
        "shared_direction_canary": {"step0": {"joint": step0_probes["joint_bisector"], "effects": None if step0_probes["joint_bisector"]["status"] != "DEFINED" else {"Smooth": _sign_summary(step0_probes, "d_joint", "L_S"), "independent": _sign_summary(step0_probes, "d_joint", "L_I"), "reconstruction": _sign_summary(step0_probes, "d_joint", "L_rec")}}, "step300": {"joint": step300_probes["joint_bisector"], "effects": None if step300_probes["joint_bisector"]["status"] != "DEFINED" else {"Smooth": _sign_summary(step300_probes, "d_joint", "L_S"), "independent": _sign_summary(step300_probes, "d_joint", "L_I"), "reconstruction": _sign_summary(step300_probes, "d_joint", "L_rec")}}, "construction": "equal-weight normalized bisector; no coefficient sweep"},
        "off_manifold_interpolation": {"artifact": _artifact_ref(root, target / "interpolation.json", role="fixed off-manifold interpolation diagnostic"), "step0": step0_interpolation, "step300": step300_interpolation, "physical_optical_path": False},
        "step0_to_step300_change": {"output_cosine_delta": float(step300_result["output_space"]["cos_h_S_h_I"] - step0_result["output_space"]["cos_h_S_h_I"]), "parameter_cosine_delta": float(step300_result["parameter_space"]["cos_g_S_g_I"] - step0_result["parameter_space"]["cos_g_S_g_I"]), "joint_status_step0": step0_probes["joint_bisector"]["status"], "joint_status_step300": step300_probes["joint_bisector"]["status"], "shared_direction_score_step0": _direction_score(step0_probes, "d_S") + _direction_score(step0_probes, "d_I"), "shared_direction_score_step300": _direction_score(step300_probes, "d_S") + _direction_score(step300_probes, "d_I")},
        "classification_evidence": classification_details,
        "mechanisms": mechanisms,
        "recommended_single_successor": recommendation,
        "evaluator_role_update": {"Smooth": "DEVELOPMENT_DIAGNOSTIC", "independent": "DEVELOPMENT_DIAGNOSTIC"},
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"dev": ledger.counts.get("dev", 0), "protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "counterfactual_integrity": {"optimizer_created": False, "parameter_updates": 0, "persistent_probe_changes": False, "checkpoint_restored_after_every_probe": True, "runtime_restored_to_step0": True, "Smooth_in_backward": False, "independent_in_backward": False, "evaluator_search": False, "protected_test_sealed": False},
        "claim_boundary": {"local_cross_evaluator_attribution_only": True, "shared_direction_if_supported_is_local": True, "global_architecture_inadequacy": False, "generation_acceptance": False, "physical_hardware_validity": False, "training_executed": False, "parameter_persisted": False},
        "files_changed": {"repository_source": ["scripts/research/execute_lightgen_2d_attribute_cross_evaluator_semantic_divergence_v1.py"], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "step0_and_step300_complete": True, "symmetric_epsilons": list(PERTURBATION_EPSILONS), "interpolation_lambdas": list(INTERPOLATION_LAMBDAS), "train_only": True, "no_dev": True, "protected_test_sealed_zero": True},
        "unresolved": ["The attribution is local to the frozen clean 2L runtime and deterministic TRAIN A/B bank; it is not a global semantic or hardware conclusion."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and run the no-update attribution")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "no_update": True, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_analysis(output_dir=args.output_dir)
    except (AttributionExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
