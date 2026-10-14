"""Attribute the failed canonical semantic realization under a frozen 2L runtime.

This is a TRAIN-only, no-update consumer.  It compares the accepted teacher
target-image objective with the accepted Smooth target-image diagnostic at the
clean step-0 and failed step-300 Generator states.  Directions are built from
the deterministic A bank and tested symmetrically on a disjoint B bank.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, LightGen2DChenBOGTComplexOLSTrainingConsumer, PRODUCTION_PROPAGATION_GRID, TRAIN_ROLE, contracted_plane_powers, git_identity, optical_power_loss_from_powers
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_frozen_evaluator,
)

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_complex_ols_training_v1 as producer
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 as conflict
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as successor
from scripts.research.lightgen import execute_lightgen_2d_generator_target_image_semantic_alignment_v1 as target_alignment


TASK_ID = "ATTRIBUTE_CANONICAL_SEMANTIC_REALIZATION_FAILURE_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ATTRIBUTE_CANONICAL_SEMANTIC_REALIZATION_FAILURE_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "canonical-semantic-realization-failure-attribution-v1"
TRAJECTORY_ID = "canonical-semantic-realization-failure-attribution-trajectory-v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

PRIOR_RUN = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01")
PRIOR_REPORT = PRIOR_RUN / "report.json"
PRIOR_STEP_0 = PRIOR_RUN / "scientific_step_000000.pt"
EXPECTED_PRIOR_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"
EXPECTED_PRIOR_STEP_0_SHA256 = "a61dd84728aa220aa567ab6cd5f9daa788b21e76be3c506054c2b7c0a23dcd34"

TARGET_RUN = Path("outputs/20260919_generator_target_image_semantic_alignment_v1/runs/attempt_01")
TARGET_REPORT = TARGET_RUN / "report.json"
TARGET_STEP_300 = TARGET_RUN / "scientific_step_000300.pt"
EXPECTED_TARGET_REPORT_SHA256 = "500289c9e17e28500e5b103a5d9bfcd7b92b1ace5bd16fdccd9dea683e570f68"
EXPECTED_TARGET_STEP_300_SHA256 = "7b51f01e0965295f059e0b7f52de1b7c68cdb5cb9ec438c4bca69d4052adcacf"

SMOOTH_REALIZATION_RUN = Path("outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_04")
SMOOTH_REALIZATION_REPORT = SMOOTH_REALIZATION_RUN / "report.json"
EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256 = "46fcfd60e5d240a92a9c4ca472f8bb88a90c641474ba11eea9db9da7236dba65"

SMOOTH_CHECKPOINT = successor.SMOOTH_CHECKPOINT
EXPECTED_SMOOTH_CHECKPOINT_SHA256 = successor.EXPECTED_SMOOTH_CHECKPOINT_SHA256
EXPECTED_TRAIN_EVALUATOR_SHA256 = producer.EVALUATOR_CHECKPOINT_SHA256
TRAINING_SOURCE_SHA256 = successor.TRAINING_SOURCE_SHA256

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260919_canonical_semantic_realization_failure_attribution_v1/runs/attempt_03"
)
BANK_SEED = 2026091901
ROWS_PER_CLASS_PER_HALF = 2
MICRO_BATCH_SIZE = 10
PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
FEATURE_EPS = 1.0e-12
EPS_PROJ = 1.0e-12

REPORT_SCHEMA = "canonical_semantic_realization_failure_attribution_report.v1"
GRADIENT_SCHEMA = "canonical_semantic_realization_failure_attribution_gradients.v1"
PROBE_SCHEMA = "canonical_semantic_realization_failure_attribution_probes.v1"

CLASSIFICATION_TEACHER_MISMATCH = "TEACHER_TO_CANONICAL_OBJECTIVE_TRANSFER_MISMATCH_SUPPORTED"
CLASSIFICATION_CAPACITY_LIMITATION = "CLEAN_2L_CANONICAL_SEMANTIC_LOCAL_CAPACITY_NOT_SUPPORTED"
CLASSIFICATION_TRAJECTORY_FAILURE = "LOCAL_CANONICAL_CONTROLLABILITY_WITH_GLOBAL_TRAINING_TRANSFER_FAILURE"
CLASSIFICATION_JACOBIAN = "OPTICAL_REALIZATION_JACOBIAN_RESTRICTION_SUPPORTED"
CLASSIFICATION_MIXED = "CANONICAL_SEMANTIC_REALIZATION_FAILURE_MIXED"


class AttributionExecutionError(RuntimeError):
    """Raised when a named authority or fail-closed measurement boundary fails."""


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
    return Path(__file__).resolve().parents[3]


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
    return {
        name: value.detach().cpu().clone()
        for name, value in consumer.runtime.state_dict().items()
    }


def _generator(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> torch.nn.Parameter:
    parameters = dict(consumer.runtime.named_parameters())
    _require(set(parameters) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    return parameters["generator.phase"]


def _freeze_generator_only(consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer) -> None:
    for name, parameter in consumer.runtime.named_parameters():
        parameter.requires_grad_(name == "generator.phase")
        parameter.grad = None
    trainable = [name for name, parameter in consumer.runtime.named_parameters() if parameter.requires_grad]
    _require(trainable == ["generator.phase"], f"attribution trainable firewall changed: {trainable}")


def _checkpoint_state(checkpoint: Mapping[str, Any], name: str, expected_step: int) -> dict[str, torch.Tensor]:
    _require(checkpoint.get("completed_updates") == expected_step, f"accepted {name} checkpoint step changed")
    _require(
        checkpoint.get("scientific_artifact") is True
        and checkpoint.get("operational_artifact") is False,
        f"accepted {name} checkpoint role changed",
    )
    _require(
        checkpoint.get("trainable_parameter_names") == ["generator.phase"],
        f"accepted {name} trainable boundary changed",
    )
    _require("encoder.phase" in checkpoint.get("frozen_parameter_names", []), f"accepted {name} encoder freeze is missing")
    encoder = _tensor(checkpoint.get("encoder_state"), f"accepted {name} encoder state")
    generator = _tensor(checkpoint.get("generator_state"), f"accepted {name} generator state")
    _finite(encoder, f"accepted {name} encoder state")
    _finite(generator, f"accepted {name} generator state")
    _require(tuple(generator.shape) == (2, 1, 100, 100), f"accepted {name} generator geometry changed")
    return {"encoder.phase": encoder.cpu(), "generator.phase": generator.cpu()}


def _select_diagnostic_bank(
    train_indices: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _require(train_indices.dtype == torch.long and labels.dtype == torch.long, "diagnostic bank roles require long tensors")
    _require(train_indices.numel() == 55_000 and labels.numel() == 60_000, "diagnostic bank roles changed")
    generator = torch.Generator(device="cpu").manual_seed(BANK_SEED)
    a_parts: list[torch.Tensor] = []
    b_parts: list[torch.Tensor] = []
    for digit in range(10):
        candidates = train_indices[labels.index_select(0, train_indices) == digit]
        _require(candidates.numel() >= 2 * ROWS_PER_CLASS_PER_HALF, f"TRAIN class {digit} cannot form A/B bank")
        shuffled = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator))
        a_parts.append(shuffled[:ROWS_PER_CLASS_PER_HALF])
        b_parts.append(shuffled[ROWS_PER_CLASS_PER_HALF : 2 * ROWS_PER_CLASS_PER_HALF])
    a = torch.cat(a_parts).contiguous()
    b = torch.cat(b_parts).contiguous()
    bank = torch.cat((a, b))
    _require(torch.unique(bank).numel() == bank.numel(), "diagnostic A/B bank overlaps")
    _require(bool(torch.all(torch.isin(bank, train_indices)).item()), "diagnostic bank left TRAIN role")
    a_labels = labels.index_select(0, a)
    b_labels = labels.index_select(0, b)
    _require(torch.bincount(a_labels, minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10, "A bank is not class-balanced")
    _require(torch.bincount(b_labels, minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10, "B bank is not class-balanced")
    return {"a_indices": a, "a_labels": a_labels, "b_indices": b, "b_labels": b_labels}


def _load_authorities(root: Path) -> dict[str, Any]:
    local_state = git_identity(root)
    _require(local_state["branch"] == EXPECTED_BRANCH, "local branch conflicts with accepted authority")
    _require(local_state["head"] == EXPECTED_HEAD, "local HEAD conflicts with accepted authority")

    target_report_path = root / TARGET_REPORT
    _require(file_sha256(target_report_path) == EXPECTED_TARGET_REPORT_SHA256, "accepted target-alignment report identity changed")
    target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
    _require(target_report.get("result") == "VALID_COMPLETED", "accepted target-alignment result is not valid completed")
    _require(target_report.get("outer_gate") == target_alignment.OUTER_GATE, "accepted target-alignment outer gate changed")
    _require(
        target_report.get("classification") == target_alignment.TEACHER_ONLY_CLASSIFICATION,
        "accepted target-alignment classification changed",
    )
    _require(target_report.get("training_trajectory", {}).get("updates") == 300, "accepted target-alignment update count changed")

    prior_report_path = root / PRIOR_REPORT
    _require(file_sha256(prior_report_path) == EXPECTED_PRIOR_REPORT_SHA256, "accepted clean conflict report identity changed")
    prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    _require(prior_report.get("result") == "VALID_COMPLETED", "accepted clean conflict result is not valid completed")
    _require(prior_report.get("outer_gate") == conflict.OUTER_GATE, "accepted clean conflict outer gate changed")

    step0_payload = safe_torch_load(root / PRIOR_STEP_0, expected_sha256=EXPECTED_PRIOR_STEP_0_SHA256, map_location="cpu")
    step300_payload = safe_torch_load(root / TARGET_STEP_300, expected_sha256=EXPECTED_TARGET_STEP_300_SHA256, map_location="cpu")
    _require(isinstance(step0_payload, Mapping), "accepted clean step-0 checkpoint is malformed")
    _require(isinstance(step300_payload, Mapping), "accepted target step-300 checkpoint is malformed")
    step0 = _checkpoint_state(step0_payload, "clean step-0", 0)
    step300 = _checkpoint_state(step300_payload, "target-alignment step-300", 300)
    _require(torch.equal(step0["encoder.phase"], step300["encoder.phase"]), "accepted encoder changed between step 0 and step 300")
    _require(not torch.equal(step0["generator.phase"], step300["generator.phase"]), "accepted generator did not change between step 0 and step 300")

    smooth_report_path = root / SMOOTH_REALIZATION_REPORT
    _require(
        file_sha256(smooth_report_path) == EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256,
        "accepted CMOS-to-Smooth report identity changed",
    )
    smooth_report = json.loads(smooth_report_path.read_text(encoding="utf-8"))
    smooth_contract = _mapping(smooth_report.get("smooth_contract"), "accepted Smooth contract")
    current_contract = _mapping(smooth_contract.get("current_contract"), "accepted current Smooth contract")
    evaluator_contract = _mapping(current_contract.get("evaluator"), "accepted Smooth evaluator contract")
    _require(
        evaluator_contract.get("feature_layer") == "embedding output immediately before classifier",
        "accepted Smooth feature layer changed",
    )
    _require(evaluator_contract.get("feature_dimension") == 128, "accepted Smooth feature dimension changed")
    _require(
        current_contract.get("preprocessing_sequence")
        == ["clamp(x,0,1)", "adaptive_avg_pool2d(x,(28,28))", "preserve singleton grayscale channel", "no mean/std or exposure normalization"],
        "accepted Smooth preprocessing contract changed",
    )

    cpu_context = phase_repair._load_cpu_context(root)
    training = successor._load_successor_training_authority(root, {"training": cpu_context})
    images, labels, train_indices, _development_indices, data_identity = producer._load_scientific_dataset(root)
    _require(data_identity.get("source_sha256") == TRAINING_SOURCE_SHA256, "TRAIN source identity changed")
    _require(data_identity == training["data_identity"], "accepted TRAIN source identity changed")
    _require(images.shape == (60_000, 1, 28, 28), "TRAIN source image geometry changed")
    _require(labels.shape == (60_000,), "TRAIN source label geometry changed")
    bank = _select_diagnostic_bank(train_indices, labels)

    target_objective = _mapping(target_report.get("alignment_contract"), "accepted target alignment contract")
    loss_weights = _mapping(target_report.get("loss_weights"), "accepted target alignment weights")
    beta = float(loss_weights.get("beta"))
    gamma = float(loss_weights.get("gamma"))
    delta_align = float(loss_weights.get("delta_align"))
    _require(math.isclose(beta, 2.4509953404250195, rel_tol=0.0, abs_tol=1.0e-12), "accepted beta changed")
    _require(math.isclose(gamma, 0.5330424284576345, rel_tol=0.0, abs_tol=1.0e-12), "accepted gamma changed")
    _require(math.isclose(delta_align, 0.024553859594563188, rel_tol=0.0, abs_tol=1.0e-15), "accepted delta_align changed")
    _require(target_objective.get("feature_layer") == "FrozenMNISTEvaluator.embedding output immediately before classifier", "teacher layer changed")
    _require(target_objective.get("feature_dimension") == 128, "teacher feature dimension changed")
    _require(target_objective.get("normalization") == "L2 normalization with epsilon 1e-12", "teacher normalization contract changed")

    return {
        "local_state": local_state,
        "target_report": target_report,
        "prior_report": prior_report,
        "smooth_report": smooth_report,
        "step0_payload": dict(step0_payload),
        "step300_payload": dict(step300_payload),
        "step0": step0,
        "step300": step300,
        "training": training,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "data_identity": data_identity,
        "bank": bank,
        "weights": {"beta": beta, "gamma": gamma, "delta_align": delta_align},
        "authority_hashes": {
            "prior_report": EXPECTED_PRIOR_REPORT_SHA256,
            "prior_step0": EXPECTED_PRIOR_STEP_0_SHA256,
            "target_report": EXPECTED_TARGET_REPORT_SHA256,
            "target_step300": EXPECTED_TARGET_STEP_300_SHA256,
            "smooth_realization_report": EXPECTED_SMOOTH_REALIZATION_REPORT_SHA256,
            "smooth_checkpoint": EXPECTED_SMOOTH_CHECKPOINT_SHA256,
            "training_report": successor.EXPECTED_SOURCE_REPORT_SHA256,
            "training_checkpoint": successor.EXPECTED_SOURCE_CHECKPOINT_SHA256,
            "generation_bundle": successor.EXPECTED_SOURCE_BUNDLE_SHA256,
            "frozen_references": successor.EXPECTED_SOURCE_REFERENCES_SHA256,
            "frozen_weights": successor.EXPECTED_SOURCE_WEIGHTS_SHA256,
        },
    }


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, FrozenMNISTEvaluator, SmoothMNISTEvaluator, dict[str, Any]]:
    training = _mapping(authorities.get("training"), "training authority")
    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    bundle = _mapping(training.get("bundle"), "accepted generation bundle")
    _require(
        consumer.runtime_identity() == _mapping(bundle.get("ols"), "bundle OLS").get("interface"),
        "runtime identity differs from accepted bundle",
    )
    consumer.load_trainable_state_dict(authorities["step0"])
    consumer.bind_references(training["references"], for_scientific=True)
    consumer.bind_weights(training["weights"], for_scientific=True)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()

    train_checkpoint = root / producer.EVALUATOR_CHECKPOINT_RELATIVE_PATH
    _require(file_sha256(train_checkpoint) == EXPECTED_TRAIN_EVALUATOR_SHA256, "TRAIN teacher checkpoint identity changed")
    teacher, teacher_payload = load_frozen_evaluator(train_checkpoint, device=device)
    _require(all(not parameter.requires_grad for parameter in teacher.parameters()), "TRAIN teacher is not frozen")

    smooth, smooth_identity = successor._load_smooth_evaluator(root, device)
    _require(all(not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator is not frozen")
    return consumer, teacher, smooth, {
        "train_checkpoint": _relative(root, train_checkpoint),
        "train_checkpoint_sha256": EXPECTED_TRAIN_EVALUATOR_SHA256,
        "train_architecture": teacher_payload.get("architecture_kind"),
        "train_feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier",
        "smooth": smooth_identity,
        "smooth_feature_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier",
    }


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    _require(value.ndim == 4 and value.shape[1] == 1 and tuple(value.shape[-2:]) in {(28, 28), (200, 200)}, "Smooth view geometry changed")
    _finite(value, "Smooth view input")
    return F.adaptive_avg_pool2d(value.clamp(0.0, 1.0), (28, 28))


def _normalized_feature_loss(cmos: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    _require(cmos.ndim == target.ndim == 2 and cmos.shape == target.shape and cmos.shape[1] == 128, f"{name} feature shape changed")
    loss = (
        F.normalize(cmos, dim=1, eps=FEATURE_EPS)
        - F.normalize(target.detach(), dim=1, eps=FEATURE_EPS)
    ).square().sum(dim=1).mean()
    _finite(loss, name)
    return loss


def _loss_bundle(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
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
    _finite(target_image, "target SLM intensity")

    with torch.no_grad():
        teacher_target, _ = teacher(batch)
        smooth_target, _ = smooth(_cmos_view(batch))
    teacher_cmos, _ = teacher(target_alignment._cmos_teacher_view(raw))
    smooth_cmos, _ = smooth(_cmos_view(raw))
    l_teacher = target_alignment._normalized_feature_alignment_loss(teacher_cmos, teacher_target)
    l_smooth = _normalized_feature_loss(smooth_cmos, smooth_target, "L_S")
    l_rec = (raw - target_image).square().mean()
    l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
    for name, value in (("L_T", l_teacher), ("L_S", l_smooth), ("L_rec", l_rec), ("L_op", l_op)):
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target_image": target_image,
        "L_T": l_teacher,
        "L_S": l_smooth,
        "L_rec": l_rec,
        "L_op": l_op,
    }


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result) and result > 0.0, f"{name} is zero or invalid")
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    left_norm = _norm(left, f"{name} left")
    right_norm = _norm(right, f"{name} right")
    value = float(
        torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu()
        / (left_norm * right_norm)
    )
    _require(math.isfinite(value), f"{name} cosine is nonfinite")
    return value


def _gradient_stats(value: torch.Tensor, name: str, *, phase: bool = False) -> dict[str, Any]:
    _finite(value, name)
    flat = value.detach()
    norm = _norm(value, name)
    result: dict[str, Any] = {
        "norm": norm,
        "finite_fraction": float(torch.isfinite(flat).to(torch.float64).mean().cpu()),
        "zero_fraction": float((flat == 0.0).to(torch.float64).mean().cpu()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if phase:
        result["per_phase_plane_norms"] = {
            f"generator.phase[{index}]": float(flat[index].to(torch.float64).square().sum().sqrt().cpu())
            for index in range(int(flat.shape[0]))
        }
    return result


def _compose_gradient(
    g_t: torch.Tensor,
    g_rec_raw: torch.Tensor,
    g_op_raw: torch.Tensor,
    *,
    beta: float,
    gamma: float,
    delta_align: float,
) -> dict[str, Any]:
    g_align = delta_align * g_t
    g_rec = beta * g_rec_raw
    g_op = gamma * g_op_raw
    g_rec_safe, rec_projection = conflict._project_auxiliary_gradient(g_rec, g_align, "g_rec")
    g_op_safe, op_projection = conflict._project_auxiliary_gradient(g_op, g_align, "g_op")
    g_total = g_align + g_rec_safe + g_op_safe
    _finite(g_total, "g_total")
    _require(
        float(torch.sum(g_total.detach().to(torch.float64) * g_align.detach().to(torch.float64)).cpu()) > 0.0,
        "g_total has no positive accepted alignment component",
    )
    return {
        "g_align": g_align,
        "g_rec": g_rec,
        "g_op": g_op,
        "g_rec_safe": g_rec_safe,
        "g_op_safe": g_op_safe,
        "g_total": g_total,
        "projection": {"rec": rec_projection, "op": op_projection},
    }


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    flat = raw.detach().to(torch.float64).flatten(1)
    std = flat.std(dim=1, unbiased=False)
    return {
        "finite": bool(torch.isfinite(raw).all().item()),
        "nonconstant_fraction": float((std > 0.0).to(torch.float64).mean().cpu()),
        "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
    }


def _preprocessing_support(raw: torch.Tensor, h_t: torch.Tensor, h_s: torch.Tensor) -> dict[str, Any]:
    active = (raw.detach() > 0.0) & (raw.detach() < 1.0)
    blocked = ~active
    return {
        "active_differentiable_range_fraction": float(active.to(torch.float64).mean().cpu()),
        "clamp_or_saturation_blocked_fraction": float(blocked.to(torch.float64).mean().cpu()),
        "teacher_gradient_nonzero_inside_active_fraction": float(((h_t != 0.0) & active).to(torch.float64).mean().cpu()),
        "smooth_gradient_nonzero_inside_active_fraction": float(((h_s != 0.0) & active).to(torch.float64).mean().cpu()),
        "definition": "raw CMOS values strictly inside (0,1) are active; values at or outside clamp bounds are blocked/descriptive",
    }


def _analyze_step(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    images: torch.Tensor,
    indices: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    checkpoint_name: str,
    weights: Mapping[str, float],
    ledger: DataReadLedger | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    generator = _generator(consumer)
    n = int(indices.numel())
    _require(n > 0 and n % MICRO_BATCH_SIZE == 0, "diagnostic A bank is not divisible by the fixed micro-batch")

    accumulator = {
        name: torch.zeros_like(generator)
        for name in ("g_T", "g_S", "g_rec_raw", "g_op_raw")
    }
    h_t_chunks: list[torch.Tensor] = []
    h_s_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    scalar_totals = {name: 0.0 for name in ("L_T", "L_S", "L_rec", "L_op")}

    for start in range(0, n, MICRO_BATCH_SIZE):
        batch_indices = indices[start : start + MICRO_BATCH_SIZE]
        batch = images.index_select(0, batch_indices).to(consumer.device)
        losses = _loss_bundle(consumer, teacher, smooth, batch, references)
        raw = losses["raw"]
        h_t, g_t = torch.autograd.grad(losses["L_T"], (raw, generator), retain_graph=True, allow_unused=False)
        h_s, g_s = torch.autograd.grad(losses["L_S"], (raw, generator), retain_graph=True, allow_unused=False)
        g_rec = torch.autograd.grad(losses["L_rec"], generator, retain_graph=True, allow_unused=False)[0]
        g_op = torch.autograd.grad(losses["L_op"], generator, retain_graph=False, allow_unused=False)[0]
        weight = batch_indices.numel() / n
        accumulator["g_T"] += weight * g_t.detach()
        accumulator["g_S"] += weight * g_s.detach()
        accumulator["g_rec_raw"] += weight * g_rec.detach()
        accumulator["g_op_raw"] += weight * g_op.detach()
        h_t_chunks.append((weight * h_t.detach()).cpu())
        h_s_chunks.append((weight * h_s.detach()).cpu())
        raw_chunks.append(raw.detach().cpu())
        for name in scalar_totals:
            scalar_totals[name] += weight * float(losses[name].detach().cpu())
        if ledger is not None:
            ledger.record(TRAIN_ROLE, int(batch_indices.numel()))
        _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "autograd.grad populated runtime .grad")

    h_t_all = torch.cat(h_t_chunks, dim=0)
    h_s_all = torch.cat(h_s_chunks, dim=0)
    raw_all = torch.cat(raw_chunks, dim=0)
    _finite(h_t_all, f"{checkpoint_name} h_T")
    _finite(h_s_all, f"{checkpoint_name} h_S")
    _finite(raw_all, f"{checkpoint_name} raw")
    composed = _compose_gradient(
        accumulator["g_T"],
        accumulator["g_rec_raw"],
        accumulator["g_op_raw"],
        beta=float(weights["beta"]),
        gamma=float(weights["gamma"]),
        delta_align=float(weights["delta_align"]),
    )
    gradients = {
        **accumulator,
        **composed,
        "h_T": h_t_all,
        "h_S": h_s_all,
    }
    result = {
        "checkpoint": checkpoint_name,
        "rows": n,
        "losses": scalar_totals,
        "output_space": {
            "h_T": _gradient_stats(h_t_all, f"{checkpoint_name} h_T"),
            "h_S": _gradient_stats(h_s_all, f"{checkpoint_name} h_S"),
            "cos_h_T_h_S": _cosine(h_t_all, h_s_all, f"{checkpoint_name} h_T/h_S"),
            "preprocessing_support": _preprocessing_support(raw_all, h_t_all, h_s_all),
        },
        "parameter_space": {
            "g_T": _gradient_stats(accumulator["g_T"], f"{checkpoint_name} g_T", phase=True),
            "g_S": _gradient_stats(accumulator["g_S"], f"{checkpoint_name} g_S", phase=True),
            "g_total": _gradient_stats(composed["g_total"], f"{checkpoint_name} g_total", phase=True),
            "cos_g_T_g_S": _cosine(accumulator["g_T"], accumulator["g_S"], f"{checkpoint_name} g_T/g_S"),
            "cos_g_total_g_S": _cosine(composed["g_total"], accumulator["g_S"], f"{checkpoint_name} g_total/g_S"),
            "cos_g_total_g_T": _cosine(composed["g_total"], accumulator["g_T"], f"{checkpoint_name} g_total/g_T"),
            "g_align": _gradient_stats(composed["g_align"], f"{checkpoint_name} g_align", phase=True),
            "g_rec_raw": _gradient_stats(accumulator["g_rec_raw"], f"{checkpoint_name} g_rec_raw", phase=True),
            "g_op_raw": _gradient_stats(accumulator["g_op_raw"], f"{checkpoint_name} g_op_raw", phase=True),
            "g_rec_weighted": _gradient_stats(composed["g_rec"], f"{checkpoint_name} beta*g_rec", phase=True),
            "g_op_weighted": _gradient_stats(composed["g_op"], f"{checkpoint_name} gamma*g_op", phase=True),
            "projection": composed["projection"],
        },
        "raw_validity": _raw_validity(raw_all),
    }
    return result, gradients


def _normalize_rms(value: torch.Tensor, name: str) -> torch.Tensor:
    _finite(value, name)
    rms = float(value.detach().to(torch.float64).square().mean().sqrt().cpu())
    _require(math.isfinite(rms) and rms > 0.0, f"{name} RMS is zero or invalid")
    direction = -value.detach() / rms
    _require(math.isclose(float(direction.to(torch.float64).square().mean().sqrt().cpu()), 1.0, rel_tol=1.0e-5), f"{name} RMS normalization failed")
    return direction


@torch.no_grad()
def _target_features(
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_features, _ = teacher(images)
    smooth_features, _ = smooth(_cmos_view(images))
    _finite(teacher_features, "B teacher target features")
    _finite(smooth_features, "B Smooth target features")
    return teacher_features.detach(), smooth_features.detach()


@torch.no_grad()
def _evaluate_b(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
    target_teacher: torch.Tensor,
    target_smooth: torch.Tensor,
    ledger: DataReadLedger | None,
) -> dict[str, Any]:
    total = {name: 0.0 for name in ("L_T", "L_S", "L_rec", "L_op")}
    teacher_predictions: list[torch.Tensor] = []
    smooth_predictions: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    count = int(images.shape[0])
    for start in range(0, count, MICRO_BATCH_SIZE):
        stop = min(start + MICRO_BATCH_SIZE, count)
        batch = images[start:stop]
        result = consumer.runtime.forward_from_image(batch)
        _require(result.image_input is not None, "B real path is incomplete")
        raw = result.raw_intensity
        target_image = result.image_input.slm_intensity
        teacher_features, teacher_logits = teacher(target_alignment._cmos_teacher_view(raw))
        smooth_features, smooth_logits = smooth(_cmos_view(raw))
        teacher_dist = (
            F.normalize(teacher_features, dim=1, eps=FEATURE_EPS)
            - F.normalize(target_teacher[start:stop], dim=1, eps=FEATURE_EPS)
        ).square().sum(dim=1)
        smooth_dist = (
            F.normalize(smooth_features, dim=1, eps=FEATURE_EPS)
            - F.normalize(target_smooth[start:stop], dim=1, eps=FEATURE_EPS)
        ).square().sum(dim=1)
        rec_dist = (raw - target_image).square().flatten(1).mean(dim=1)
        l_op = optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference)
        for name, value in (
            ("L_T", teacher_dist),
            ("L_S", smooth_dist),
            ("L_rec", rec_dist),
            ("L_op", l_op),
        ):
            _finite(value if isinstance(value, torch.Tensor) else torch.as_tensor(value), f"B {name}")
        weight = (stop - start) / count
        total["L_T"] += weight * float(teacher_dist.mean().cpu())
        total["L_S"] += weight * float(smooth_dist.mean().cpu())
        total["L_rec"] += weight * float(rec_dist.mean().cpu())
        total["L_op"] += weight * float(l_op.cpu())
        teacher_predictions.append(teacher_logits.argmax(dim=1).cpu())
        smooth_predictions.append(smooth_logits.argmax(dim=1).cpu())
        raw_parts.append(raw.cpu())
        if ledger is not None:
            ledger.record(TRAIN_ROLE, stop - start)
    raw_all = torch.cat(raw_parts, dim=0)
    teacher_pred = torch.cat(teacher_predictions)
    smooth_pred = torch.cat(smooth_predictions)
    _require(teacher_pred.shape == labels.shape and smooth_pred.shape == labels.shape, "B classifier output shape changed")
    teacher_correct = teacher_pred == labels.cpu()
    smooth_correct = smooth_pred == labels.cpu()
    teacher_per_class = [float(teacher_correct[labels.cpu() == digit].to(torch.float64).mean()) for digit in range(10)]
    smooth_per_class = [float(smooth_correct[labels.cpu() == digit].to(torch.float64).mean()) for digit in range(10)]
    return {
        "losses": total,
        "raw_validity": _raw_validity(raw_all),
        "teacher_classifier": {
            "macro_recall": float(sum(teacher_per_class) / 10.0),
            "per_class_recall": teacher_per_class,
        },
        "smooth_classifier": {
            "macro_recall": float(sum(smooth_per_class) / 10.0),
            "per_class_recall": smooth_per_class,
        },
    }


def _delta_metrics(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(candidate["losses"][name] - base["losses"][name])
        for name in ("L_T", "L_S", "L_rec", "L_op")
    }


def _run_probes(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
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
    b_images = images.to(consumer.device)
    b_labels = labels.cpu()
    target_teacher, target_smooth = _target_features(teacher, smooth, b_images)
    base = _evaluate_b(consumer, teacher, smooth, b_images, b_labels, references, target_teacher, target_smooth, ledger)
    directions = {
        "d_T": _normalize_rms(gradients["g_T"], f"{checkpoint_name} d_T").to(consumer.device),
        "d_S": _normalize_rms(gradients["g_S"], f"{checkpoint_name} d_S").to(consumer.device),
        "d_total": _normalize_rms(gradients["g_total"], f"{checkpoint_name} d_total").to(consumer.device),
    }
    records: dict[str, Any] = {}
    for direction_name, direction in directions.items():
        epsilon_records: list[dict[str, Any]] = []
        for epsilon in PERTURBATION_EPSILONS:
            signed: dict[str, Any] = {}
            for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                consumer.load_trainable_state_dict(checkpoint)
                _freeze_generator_only(consumer)
                with torch.no_grad():
                    _generator(consumer).add_(sign * epsilon * direction)
                candidate = _evaluate_b(
                    consumer,
                    teacher,
                    smooth,
                    b_images,
                    b_labels,
                    references,
                    target_teacher,
                    target_smooth,
                    ledger,
                )
                signed[sign_name] = {
                    "epsilon": epsilon,
                    "deltas": _delta_metrics(base, candidate),
                    "losses": candidate["losses"],
                    "raw_validity": candidate["raw_validity"],
                    "teacher_classifier": candidate["teacher_classifier"],
                    "smooth_classifier": candidate["smooth_classifier"],
                }
                consumer.load_trainable_state_dict(checkpoint)
                _freeze_generator_only(consumer)
                _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{checkpoint_name} {direction_name} probe was not restored")
            epsilon_records.append({"epsilon": epsilon, **signed})
        records[direction_name] = epsilon_records
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"{checkpoint_name} final probe restoration failed")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "teacher received a probe gradient")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "Smooth received a probe gradient")
    return {
        "schema": PROBE_SCHEMA,
        "checkpoint": checkpoint_name,
        "rows": int(images.shape[0]),
        "base": base,
        "directions": records,
        "epsilons": list(PERTURBATION_EPSILONS),
        "restored_after_every_perturbation": True,
        "direction_definition": "d=-gradient/RMS(gradient), giving unit RMS phase perturbation before epsilon",
        "probe_parameter_updates": 0,
    }


def _signal(probes: Mapping[str, Any], direction: str, metric: str) -> dict[str, Any]:
    records = probes["directions"][direction]
    plus = [float(record["plus"]["deltas"][metric]) for record in records]
    minus = [float(record["minus"]["deltas"][metric]) for record in records]
    plus_mean = sum(plus) / len(plus)
    minus_mean = sum(minus) / len(minus)
    return {
        "plus_deltas": plus,
        "minus_deltas": minus,
        "plus_mean": plus_mean,
        "minus_mean": minus_mean,
        "plus_decreases": sum(value < 0.0 for value in plus),
        "minus_increases": sum(value > 0.0 for value in minus),
        "symmetric_descent_sign": bool(plus_mean < 0.0 and minus_mean > 0.0),
    }


def _probe_interpretation(probes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        direction: {
            metric: _signal(probes, direction, metric)
            for metric in ("L_T", "L_S", "L_rec")
        }
        for direction in ("d_T", "d_S", "d_total")
    }


def _classify(
    step0: Mapping[str, Any],
    step300: Mapping[str, Any],
    probes0: Mapping[str, Any],
    probes300: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    signal0 = _probe_interpretation(probes0)
    signal300 = _probe_interpretation(probes300)
    d_t_teacher = signal0["d_T"]["L_T"]["symmetric_descent_sign"] and signal300["d_T"]["L_T"]["symmetric_descent_sign"]
    d_t_smooth = signal0["d_T"]["L_S"]["symmetric_descent_sign"] and signal300["d_T"]["L_S"]["symmetric_descent_sign"]
    d_s_smooth0 = signal0["d_S"]["L_S"]["symmetric_descent_sign"]
    d_s_smooth300 = signal300["d_S"]["L_S"]["symmetric_descent_sign"]
    step0_to_step300_rotation = d_s_smooth0 and not d_s_smooth300
    cos_h0 = float(step0["output_space"]["cos_h_T_h_S"])
    cos_h300 = float(step300["output_space"]["cos_h_T_h_S"])
    cos_g0 = float(step0["parameter_space"]["cos_g_T_g_S"])
    cos_g300 = float(step300["parameter_space"]["cos_g_T_g_S"])
    jacobian_pattern = (cos_h0 > cos_g0 and cos_h300 > cos_g300) and not (d_s_smooth0 or d_s_smooth300)

    if d_t_teacher and not d_t_smooth and (d_s_smooth0 or d_s_smooth300):
        classification = CLASSIFICATION_TEACHER_MISMATCH
        recommendation = "DEFINE_CANONICAL_ALIGNMENT_TRAINING_WITH_INDEPENDENT_CONFIRMATION_V1"
    elif step0_to_step300_rotation:
        classification = CLASSIFICATION_TRAJECTORY_FAILURE
        recommendation = "DEFINE_CANONICAL_TRAJECTORY_CONTROL_TRAINING_V1"
    elif not d_s_smooth0 and not d_s_smooth300:
        classification = CLASSIFICATION_JACOBIAN if jacobian_pattern else CLASSIFICATION_CAPACITY_LIMITATION
        recommendation = "COMPARE_CANONICAL_CONTROLLABILITY_FUNCTION_CLASSES_V1" if classification == CLASSIFICATION_CAPACITY_LIMITATION else None
    else:
        classification = CLASSIFICATION_MIXED
        recommendation = None
    details = {
        "step0": signal0,
        "step300": signal300,
        "d_T_improves_teacher_B": d_t_teacher,
        "d_T_improves_smooth_B": d_t_smooth,
        "d_S_improves_smooth_B_step0": d_s_smooth0,
        "d_S_improves_smooth_B_step300": d_s_smooth300,
        "step0_to_step300_canonical_direction_rotation": step0_to_step300_rotation,
        "jacobian_relative_pattern": {
            "step0_output_cosine": cos_h0,
            "step0_parameter_cosine": cos_g0,
            "step300_output_cosine": cos_h300,
            "step300_parameter_cosine": cos_g300,
            "descriptive_support": jacobian_pattern,
            "no_universal_cosine_cutoff": True,
        },
    }
    return classification, details, recommendation


def _make_identity(
    source_manifest: Mapping[str, Any],
    bank: Mapping[str, torch.Tensor],
    scientific_attempt_id: str,
) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={
            "no_update": True,
            "bank_seed": BANK_SEED,
            "rows_per_class_per_half": ROWS_PER_CLASS_PER_HALF,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "epsilons": list(PERTURBATION_EPSILONS),
            "checkpoints": [0, 300],
            "teacher_loss": "mean squared distance between L2-normalized FrozenMNISTEvaluator 128D features",
            "smooth_loss": "mean squared distance between L2-normalized SmoothMNISTEvaluator 128D features",
            "bank_indices_sha256": {
                name: tensor_sha256(value)
                for name, value in bank.items()
            },
        },
        model_identity="LIGHTGEN_2D_CANONICAL_SEMANTIC_REALIZATION_FAILURE_ATTRIBUTION_V1",
        optimizer_identity={"created": False, "parameter_updates": 0},
        sampler_identity={"kind": "deterministic class-balanced TRAIN-only A/B bank", "seed": BANK_SEED},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=scientific_attempt_id,
    )


def _run_preflight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    teacher: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    authorities: Mapping[str, Any],
) -> dict[str, Any]:
    initial_state = _runtime_state(consumer)
    bank = _mapping(authorities["bank"], "diagnostic bank")
    a_indices = _tensor(bank["a_indices"], "A indices")
    _require(a_indices.numel() >= MICRO_BATCH_SIZE, "A bank cannot form representative preflight")
    ledger = DataReadLedger()
    step_result, gradients = _analyze_step(
        consumer,
        teacher,
        smooth,
        authorities["images"],
        a_indices[:MICRO_BATCH_SIZE],
        authorities["training"]["references"],
        authorities["step0"],
        "preflight-step-0",
        authorities["weights"],
        ledger,
    )
    for name in ("g_T", "g_S", "g_total"):
        _gradient_stats(gradients[name], f"preflight {name}", phase=True)
    _require(step_result["raw_validity"]["finite"], "preflight raw CMOS is nonfinite")
    _require(_state_digest(initial_state) == _state_digest(_runtime_state(consumer)), "preflight changed runtime state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "preflight left a runtime gradient")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "preflight left a teacher gradient")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "preflight left a Smooth gradient")
    _require(ledger.counts.get("dev", 0) == 0, "preflight touched DEV")
    return {
        "status": "READY",
        "optimizer_created": False,
        "parameter_updates": 0,
        "representative_rows": MICRO_BATCH_SIZE,
        "representative_step0_gradient": True,
        "runtime_state_restored": True,
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "FrozenMNISTEvaluator", "SmoothMNISTEvaluator", "P2", "Q2"],
        "data_reads": ledger.to_dict(),
        "protected_test_sealed_zero": True,
    }


def run_analysis(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")

    authorities = _load_authorities(root)
    cuda_runtime = successor._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "canonical attribution requires CUDA")
    device = torch.device("cuda:0")
    consumer, teacher, smooth, evaluator_identity = _load_models(root, authorities, device)
    preflight = _run_preflight(consumer, teacher, smooth, authorities)
    initial_runtime_state = _runtime_state(consumer)

    bank = authorities["bank"]
    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(target_alignment.__file__).resolve(),
            Path(conflict.__file__).resolve(),
            Path(successor.__file__).resolve(),
            Path(phase_repair.__file__).resolve(),
            Path(producer.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={
            "authorization_mode": "IMPLEMENT_PREPARED",
            "authorization_token": AUTHORIZATION_TOKEN,
            "outer_gate": OUTER_GATE,
            "no_update": True,
            "bank_seed": BANK_SEED,
            "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()},
            "input_hashes": authorities["authority_hashes"],
        },
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    scientific_attempt_id = f"{RUN_ID}:{target.name}"
    run_identity = _make_identity(source_manifest, bank, scientific_attempt_id)
    control = ResearchRunControl(run_identity, AUTHORIZATION_ID)
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    receipt = control.make_ready_receipt(
        correctness={
            "passed": bool(preflight.get("status") == "READY" and preflight.get("runtime_state_restored") is True and preflight.get("parameter_updates") == 0 and preflight.get("representative_step0_gradient") is True and preflight.get("protected_test_sealed_zero") is True),
            "representative_gradient_checked": bool(preflight.get("representative_step0_gradient")),
            "runtime_state_restored": bool(preflight.get("runtime_state_restored")),
            "protected_test_sealed_zero": bool(preflight.get("protected_test_sealed_zero")),
        },
        resource={"passed": int(cuda_total) > 0 and 0 <= int(cuda_free) <= int(cuda_total), "free_bytes": int(cuda_free), "total_bytes": int(cuda_total), "updates": 0},
        implementation={"passed": preflight.get("optimizer_created") is False and preflight.get("parameter_updates") == 0, "selected_engine": "read_only_cuda_attribution"},
        selected_engine="read_only_cuda_attribution",
        created_at=time.time(),
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, scientific_attempt_id, receipt, resume=True)
    _write_json(target / "preflight_receipt.json", receipt.to_dict())

    bank_payload = {
        "schema": "canonical_semantic_realization_failure_attribution_bank.v1",
        "task_id": TASK_ID,
        "seed": BANK_SEED,
        "data_role": TRAIN_ROLE,
        "a_indices": bank["a_indices"],
        "a_labels": bank["a_labels"],
        "b_indices": bank["b_indices"],
        "b_labels": bank["b_labels"],
        "rows_per_class_per_half": ROWS_PER_CLASS_PER_HALF,
        "development_rows_consumed": 0,
    }
    _write_torch(target / "diagnostic_bank.pt", bank_payload, kind="TRAIN-only canonical attribution bank")
    _write_json(target / "source_manifest.json", source_manifest)
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "canonical_semantic_realization_failure_attribution_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
            "input_authorities": {
                "accepted_clean_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="clean accepted 2L Generator step-0"),
                "accepted_target_alignment": _artifact_ref(root, root / TARGET_REPORT, role="accepted failed target-image alignment result"),
                "failed_target_alignment_step300": _artifact_ref(root, root / TARGET_STEP_300, role="failed target-alignment step-300 checkpoint"),
                "accepted_smooth_realization": _artifact_ref(root, root / SMOOTH_REALIZATION_REPORT, role="accepted CMOS-to-Smooth semantic realization contract"),
            },
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "local_state": authorities["local_state"],
            "cuda_runtime": cuda_runtime,
            "scope": {
                "optimizer_created": False,
                "parameter_updates": 0,
                "train_rows_unique": int(torch.cat((bank["a_indices"], bank["b_indices"])).numel()),
                "development_rows_consumed": 0,
                "protected": False,
                "test": False,
                "sealed": False,
            },
        },
    )

    ledger = DataReadLedger()
    a_indices = bank["a_indices"]
    b_indices = bank["b_indices"]
    step0_result, step0_gradients = _analyze_step(
        consumer,
        teacher,
        smooth,
        authorities["images"],
        a_indices,
        authorities["training"]["references"],
        authorities["step0"],
        "step-0",
        authorities["weights"],
        ledger,
    )
    step0_probes = _run_probes(
        consumer,
        teacher,
        smooth,
        authorities["images"].index_select(0, b_indices),
        authorities["labels"].index_select(0, b_indices),
        authorities["training"]["references"],
        authorities["step0"],
        step0_gradients,
        "step-0",
        ledger,
    )
    step300_result, step300_gradients = _analyze_step(
        consumer,
        teacher,
        smooth,
        authorities["images"],
        a_indices,
        authorities["training"]["references"],
        authorities["step300"],
        "step-300",
        authorities["weights"],
        ledger,
    )
    step300_probes = _run_probes(
        consumer,
        teacher,
        smooth,
        authorities["images"].index_select(0, b_indices),
        authorities["labels"].index_select(0, b_indices),
        authorities["training"]["references"],
        authorities["step300"],
        step300_gradients,
        "step-300",
        ledger,
    )

    consumer.load_trainable_state_dict(authorities["step0"])
    _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_runtime_state), "final checkpoint restoration failed")
    _require(ledger.counts.get("dev", 0) == 0, "scientific attribution touched DEV")
    _require(ledger.counts.get("protected", 0) == 0 and ledger.counts.get("test", 0) == 0 and ledger.counts.get("sealed", 0) == 0, "forbidden data role was read")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "scientific attribution left a runtime gradient")
    _require(all(parameter.grad is None for parameter in teacher.parameters()), "scientific attribution left a teacher gradient")
    _require(all(parameter.grad is None for parameter in smooth.parameters()), "scientific attribution left a Smooth gradient")

    gradient_payload = {
        "schema": GRADIENT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "checkpoint_hashes": {"step0": EXPECTED_PRIOR_STEP_0_SHA256, "step300": EXPECTED_TARGET_STEP_300_SHA256},
        "bank_indices": {"a": a_indices, "b": b_indices},
        "step0": {
            name: value.detach().cpu()
            for name, value in step0_gradients.items()
            if isinstance(value, torch.Tensor)
        },
        "step300": {
            name: value.detach().cpu()
            for name, value in step300_gradients.items()
            if isinstance(value, torch.Tensor)
        },
    }
    _write_torch(target / "gradient_vectors.pt", gradient_payload, kind="canonical attribution gradient vectors")
    _write_json(target / "directional_probes.json", {"schema": PROBE_SCHEMA, "step0": step0_probes, "step300": step300_probes})

    classification, classification_details, recommendation = _classify(
        step0_result,
        step300_result,
        step0_probes,
        step300_probes,
    )
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
        "local_state": authorities["local_state"],
        "input_authorities": {
            "accepted_clean_step0": _artifact_ref(root, root / PRIOR_STEP_0, role="clean accepted 2L Generator step-0"),
            "accepted_clean_conflict_report": _artifact_ref(root, root / PRIOR_REPORT, role="accepted clean conflict-aware report"),
            "accepted_target_alignment_report": _artifact_ref(root, root / TARGET_REPORT, role="accepted target-image alignment report"),
            "failed_target_alignment_step300": _artifact_ref(root, root / TARGET_STEP_300, role="failed target-alignment step-300 checkpoint"),
            "accepted_cmos_to_smooth_report": _artifact_ref(root, root / SMOOTH_REALIZATION_REPORT, role="accepted canonical Smooth realization diagnosis"),
            "authority_hashes": authorities["authority_hashes"],
            "train_data_identity": authorities["data_identity"],
        },
        "frozen_contracts": {
            "encoder": True,
            "OLS": True,
            "W_OLS": True,
            "generator_architecture": True,
            "train_teacher": evaluator_identity,
            "smooth_evaluator": evaluator_identity["smooth"],
            "teacher_feature_layer": "FrozenMNISTEvaluator.embedding output immediately before classifier",
            "smooth_feature_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier",
            "smooth_preprocessing": ["clamp(x,0,1)", "adaptive_avg_pool2d(x,(28,28))", "preserve singleton grayscale channel", "no mean/std or exposure normalization"],
        },
        "diagnostic_bank": {
            "artifact": _artifact_ref(root, target / "diagnostic_bank.pt", role="frozen TRAIN-only A/B bank"),
            "seed": BANK_SEED,
            "a_rows": int(a_indices.numel()),
            "b_rows": int(b_indices.numel()),
            "class_balanced": True,
            "a_indices_sha256": tensor_sha256(a_indices),
            "b_indices_sha256": tensor_sha256(b_indices),
            "development_rows_consumed": 0,
        },
        "objective_definitions": {
            "L_T": "mean(||normalize(F_teacher(CMOS))-normalize(F_teacher(target))||_2^2), target features stop-gradient, exact accepted TRAIN target-image contract",
            "L_S": "mean(||normalize(F_smooth(CMOS))-normalize(F_smooth(target))||_2^2), Smooth frozen and target features stop-gradient; diagnostic oracle only",
            "normalization": "L2 normalization with epsilon 1e-12",
            "g_total": "g_align + project_nonconflicting(beta*g_rec, g_align) + project_nonconflicting(gamma*g_op, g_align)",
            "weights": dict(authorities["weights"]),
            "projection_epsilon": EPS_PROJ,
        },
        "step0": step0_result,
        "step300": step300_result,
        "gradient_attribution": {
            "artifact": _artifact_ref(root, target / "gradient_vectors.pt", role="A-bank output/parameter-space gradient vectors"),
            "raw_unweighted_primary": True,
            "cosines": {
                "step0_h_T_h_S": step0_result["output_space"]["cos_h_T_h_S"],
                "step300_h_T_h_S": step300_result["output_space"]["cos_h_T_h_S"],
                "step0_g_T_g_S": step0_result["parameter_space"]["cos_g_T_g_S"],
                "step300_g_T_g_S": step300_result["parameter_space"]["cos_g_T_g_S"],
                "step0_g_total_g_S": step0_result["parameter_space"]["cos_g_total_g_S"],
                "step300_g_total_g_S": step300_result["parameter_space"]["cos_g_total_g_S"],
                "step0_g_total_g_T": step0_result["parameter_space"]["cos_g_total_g_T"],
                "step300_g_total_g_T": step300_result["parameter_space"]["cos_g_total_g_T"],
            },
            "jacobian_transfer": {
                "step0_output_minus_parameter": float(step0_result["output_space"]["cos_h_T_h_S"] - step0_result["parameter_space"]["cos_g_T_g_S"]),
                "step300_output_minus_parameter": float(step300_result["output_space"]["cos_h_T_h_S"] - step300_result["parameter_space"]["cos_g_T_g_S"]),
                "interpretation": "descriptive relative pattern only; no universal cosine cutoff",
            },
        },
        "held_out_directional_probes": {
            "artifact": _artifact_ref(root, target / "directional_probes.json", role="symmetric held-out B probes"),
            "step0": step0_probes,
            "step300": step300_probes,
            "checkpoint_restored_after_every_probe": True,
            "primary_quantities": ["delta_L_T", "delta_L_S", "delta_L_rec", "raw_validity"],
            "secondary_quantities": ["Smooth TRAIN diagnostic macro recall", "teacher classifier recall"],
        },
        "trajectory_comparison": {
            "step0_to_step300_generator_state_cosine": float(
                torch.sum(
                    authorities["step0"]["generator.phase"].to(torch.float64)
                    * authorities["step300"]["generator.phase"].to(torch.float64)
                ).item()
                / (
                    authorities["step0"]["generator.phase"].to(torch.float64).norm()
                    * authorities["step300"]["generator.phase"].to(torch.float64).norm()
                ).item()
            ),
            "step0_output_cosine": step0_result["output_space"]["cos_h_T_h_S"],
            "step300_output_cosine": step300_result["output_space"]["cos_h_T_h_S"],
            "step0_parameter_cosine": step0_result["parameter_space"]["cos_g_T_g_S"],
            "step300_parameter_cosine": step300_result["parameter_space"]["cos_g_T_g_S"],
            "step0_g_total_smooth_cosine": step0_result["parameter_space"]["cos_g_total_g_S"],
            "step300_g_total_smooth_cosine": step300_result["parameter_space"]["cos_g_total_g_S"],
        },
        "classification_evidence": classification_details,
        "recommended_single_successor": recommendation,
        "smooth_role_after_attribution": "DEVELOPMENT_DIAGNOSTIC",
        "preflight": preflight,
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {
            "dev": ledger.counts.get("dev", 0),
            "protected": ledger.counts.get("protected", 0),
            "test": ledger.counts.get("test", 0),
            "sealed": ledger.counts.get("sealed", 0),
        },
        "no_update_integrity": {
            "optimizer_created": False,
            "parameter_updates": 0,
            "persistent_probe_changes": False,
            "checkpoint_restored_after_every_probe": True,
            "runtime_restored_to_step0": True,
            "teacher_parameters_frozen": True,
            "smooth_parameters_frozen": True,
        },
        "claim_boundary": {
            "local_canonical_direction_transfer_only": True,
            "global_architecture_inadequacy": False,
            "global_optimum": False,
            "generation_acceptance": False,
            "physical_hardware_validity": False,
            "training_executed": False,
            "parameter_persisted": False,
            "smooth_prior_measurements_invalidated": False,
        },
        "files_changed": {
            "repository_source": ["scripts/research/lightgen/execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1.py"],
            "scientific_artifacts": sorted(path.name for path in target.iterdir()),
        },
        "validation": {
            "cuda": True,
            "cuda_runtime": cuda_runtime,
            "outer_gate": OUTER_GATE,
            "preflight": "READY",
            "step0_and_step300_complete": True,
            "symmetric_epsilons": list(PERTURBATION_EPSILONS),
            "train_only": True,
            "no_dev": True,
            "protected_test_sealed_zero": True,
        },
        "unresolved": [
            "The classification is local to the frozen clean 2L runtime and the deterministic TRAIN A/B bank; it is not a global capacity or final quality claim."
        ],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and run the no-update attribution")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "task_id": TASK_ID,
                    "authorization_required": AUTHORIZATION_TOKEN,
                    "outer_gate": OUTER_GATE,
                    "no_update": True,
                    "output_dir": str(args.output_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    try:
        report = run_analysis(output_dir=args.output_dir)
    except (AttributionExecutionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "outer_gate": report["outer_gate"],
                "output_dir": report["output_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
