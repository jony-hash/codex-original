"""Attribute the post-3000 Generator semantic-realization ceiling.

This is a TRAIN-only, no-update diagnostic.  It consumes the accepted clean,
step-300, and step-3000 Generator authorities and the accepted class-balanced
A/B bank.  The A half supplies frozen evaluator, reconstruction, and optical
gradients.  The disjoint B half supplies symmetric temporary phase probes.
No optimizer, persistent parameter update, sampler change, DEV read, or new
evaluator is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import DataReadLedger, LightGen2DChenBOGTComplexOLSTrainingConsumer, TRAIN_ROLE, contracted_plane_powers, git_identity, optical_power_loss_from_powers
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator

from ldm_project.v2.application.research.lightgen import confirmation_evaluators as third_confirmation
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 as frozen_latent


TASK_ID = "ATTRIBUTE_POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_ATTRIBUTE_POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_ATTRIBUTED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "attribute-post3000-generator-semantic-realization-ceiling-v1"
TRAJECTORY_ID = f"{RUN_ID}:trajectory"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

REPLAY_REPORT = Path(
    "outputs/20260920_real_ols_replay_conditioning_identity_gap_attributed_v1/runs/attempt_01/report.json"
)
REPLAY_BANK_PAYLOAD = Path(
    "outputs/20260919_cross_evaluator_semantic_divergence_v1/runs/attempt_03/diagnostic_bank.pt"
)
EXPECTED_REPLAY_REPORT_SHA256 = "7cf6033b2da4fe1258e78e51bfa3972a637852345ab3ea4594ebefb4383df523"
EXPECTED_REPLAY_BANK_SHA256 = "5213c40dc0f2141a92b771166e34e8b9e207e84e2b3b51f62a1d758f199ecb3d"

POST3000_REPORT = Path(
    "outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_05/report.json"
)
EXPECTED_POST3000_REPORT_SHA256 = "c745133868a0058576239c489f1daaead87214ca0c8bc2ba1dc73c786ee37647"

ROBUST_REPORT = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/report.json")
EXPECTED_ROBUST_REPORT_SHA256 = "3e972d54746121a15dbbf6fca16df2124e9854ceac17bcbe294bff2d24c2bf6b"

STEP_0 = robust.CLEAN_STEP_0
STEP_0_SHA256 = robust.EXPECTED_CLEAN_STEP_0_SHA256
STEP_300 = Path("outputs/20260919_evaluator_robust_semantic_training_v1/runs/attempt_02/scientific_step_000300.pt")
STEP_300_SHA256 = "3920c63636b374e16904d373142c15c8a95b20a3360de5ea52fa0f60825858cc"
STEP_3000 = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_003000.pt")
STEP_3000_SHA256 = "1a87af8aa4fa12e43dfd4ab7ee99aa41e08299843f5414cfd05d0221dc473a3d"
CHECKPOINTS = {
    0: (STEP_0, STEP_0_SHA256),
    300: (STEP_300, STEP_300_SHA256),
    600: (Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_000600.pt"), "cddefb57554c9982351807ea56635bb1268f36f8c490a45cb68f3c86275a6d14"),
    1000: (Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_001000.pt"), "94198d0fe518116305d3591bbecb8086b2f79d0bfad11370210fb34c3017138b"),
    1500: (Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_001500.pt"), "8005b63fc5427de0763a154b3ae8b6f79045be9eb0675dd28a88edb50632e093"),
    2000: (Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_002000.pt"), "7e876d15bd03cb480703cc45bce64d890086d63a33a9be4a84d18fadd75c14c8"),
    2500: (Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_002500.pt"), "1ccdc783ddc52df673f6bffcf4774d3fd98540f544315ca1068523388b588372"),
    3000: (STEP_3000, STEP_3000_SHA256),
}
KEY_STEPS = (0, 300, 3000)
TRAJECTORY_STEPS = (0, 300, 600, 1000, 1500, 2000, 2500, 3000)

BANK_SEED = 2026091901
MICRO_BATCH_SIZE = 10
PERTURBATION_EPSILONS = (1.0e-4, 3.0e-4, 1.0e-3)
FEATURE_EPS = 1.0e-12
JOINT_DEGENERACY_TOLERANCE = 1.0e-12
EPS_PROJ = 1.0e-12

# These are consumed as historical references only.  They are not changed or
# optimized by this diagnostic.
HISTORICAL_BETA = 2.4509953404
HISTORICAL_GAMMA = 0.5330424285
HISTORICAL_DELTA_JOINT = 1.365988374680088e-05
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8

FEATURE_LAYER = "embedding output immediately before classifier"
PREPROCESSING = [
    "clamp(x,0,1)",
    "adaptive_avg_pool2d(x,(28,28))",
    "preserve singleton grayscale channel",
    "no mean/std or exposure normalization",
]

REPORT_SCHEMA = "post3000_generator_semantic_realization_ceiling_attribution_report.v1"
GRADIENT_SCHEMA = "post3000_generator_semantic_realization_ceiling_gradients.v1"
PROBE_SCHEMA = "post3000_generator_semantic_realization_ceiling_probes.v1"

CLASSIFICATION_SHARED = "POST3000_SHARED_SEMANTIC_DIRECTION_REMAINS_AVAILABLE"
CLASSIFICATION_JACOBIAN = "OPTICAL_GENERATOR_JACOBIAN_SEMANTIC_REALIZATION_CEILING_SUPPORTED"
CLASSIFICATION_CONFLICT = "POST3000_OUTPUT_SPACE_CROSS_EVALUATOR_SEMANTIC_CONFLICT_SUPPORTED"
CLASSIFICATION_TWO_EVALUATOR = "TWO_EVALUATOR_TRAINING_OBJECTIVE_CEILING_SUPPORTED"
CLASSIFICATION_AUXILIARY = "AUXILIARY_CONSTRAINT_LIMITED_SEMANTIC_REALIZATION_SUPPORTED"
CLASSIFICATION_MIXED = "POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_MIXED"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260920_attribute_post3000_generator_semantic_realization_ceiling_v1/runs/attempt_01"
)


class CeilingAttributionError(RuntimeError):
    """Raised when a named authority or diagnostic boundary fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CeilingAttributionError(message)


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


def _clear_evaluator_grads(*evaluators: torch.nn.Module) -> None:
    for evaluator in evaluators:
        for parameter in evaluator.parameters():
            parameter.grad = None


def _checkpoint_state(payload: Mapping[str, Any], name: str, expected_step: int) -> dict[str, torch.Tensor]:
    _require(payload.get("completed_updates") == expected_step, f"accepted {name} checkpoint step changed")
    _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"accepted {name} checkpoint role changed")
    _require(payload.get("trainable_parameter_names") == ["generator.phase"], f"accepted {name} trainable boundary changed")
    _require("encoder.phase" in payload.get("frozen_parameter_names", []), f"accepted {name} encoder freeze is missing")
    encoder = _tensor(payload.get("encoder_state"), f"accepted {name} encoder state")
    generator = _tensor(payload.get("generator_state"), f"accepted {name} generator state")
    _finite(encoder, f"accepted {name} encoder state")
    _finite(generator, f"accepted {name} generator state")
    _require(tuple(encoder.shape) == (2, 1, 100, 100), f"accepted {name} encoder geometry changed")
    _require(tuple(generator.shape) == (2, 1, 100, 100), f"accepted {name} generator geometry changed")
    return {"encoder.phase": encoder.cpu(), "generator.phase": generator.cpu()}


def _load_checkpoint(root: Path, step: int, descriptor: tuple[Path, str]) -> tuple[dict[str, torch.Tensor], Mapping[str, Any], str]:
    path, expected_sha256 = descriptor
    target = root / path
    _require(target.is_file(), f"checkpoint step {step} is missing: {target}")
    observed = file_sha256(target)
    if not expected_sha256:
        expected_sha256 = observed
    _require(observed == expected_sha256, f"checkpoint step {step} identity changed")
    payload = safe_torch_load(target, expected_sha256=expected_sha256, map_location="cpu")
    mapping = _mapping(payload, f"checkpoint step {step}")
    state = _checkpoint_state(mapping, f"step {step}", step)
    return state, mapping, expected_sha256


def _load_diagnostic_bank(root: Path, authorities: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    path = root / REPLAY_BANK_PAYLOAD
    payload = safe_torch_load(path, expected_sha256=EXPECTED_REPLAY_BANK_SHA256, map_location="cpu")
    payload = _mapping(payload, "accepted A/B diagnostic bank")
    _require(payload.get("schema") == "cross_evaluator_semantic_divergence_bank.v1", "accepted A/B bank schema changed")
    _require(payload.get("seed") == BANK_SEED and payload.get("data_role") == TRAIN_ROLE, "accepted A/B bank seed/role changed")
    labels = _tensor(authorities["labels"], "canonical labels")
    train_indices = _tensor(authorities["train_indices"], "TRAIN indices")
    a = _tensor(payload.get("a_indices"), "A indices").to(dtype=torch.long)
    b = _tensor(payload.get("b_indices"), "B indices").to(dtype=torch.long)
    a_labels = _tensor(payload.get("a_labels"), "A labels").to(dtype=torch.long)
    b_labels = _tensor(payload.get("b_labels"), "B labels").to(dtype=torch.long)
    _require(a.numel() == b.numel() == 20, "accepted A/B bank size changed")
    _require(torch.unique(torch.cat((a, b))).numel() == 40, "accepted A/B bank overlaps")
    _require(bool(torch.all(torch.isin(torch.cat((a, b)), train_indices)).item()), "accepted A/B bank left TRAIN")
    _require(torch.equal(a_labels, labels.index_select(0, a)), "accepted A labels changed")
    _require(torch.equal(b_labels, labels.index_select(0, b)), "accepted B labels changed")
    _require(torch.bincount(a_labels, minlength=10).tolist() == [2] * 10, "A bank is not class-balanced")
    _require(torch.bincount(b_labels, minlength=10).tolist() == [2] * 10, "B bank is not class-balanced")
    return {"a_indices": a.contiguous(), "a_labels": a_labels.contiguous(), "b_indices": b.contiguous(), "b_labels": b_labels.contiguous()}


def _load_authorities(root: Path) -> dict[str, Any]:
    local_state = git_identity(root)
    _require(local_state["branch"] == EXPECTED_BRANCH and local_state["head"] == EXPECTED_HEAD, "branch/HEAD conflicts with accepted authorities")
    authorities = robust._load_authorities(root)

    replay_path = root / REPLAY_REPORT
    _require(file_sha256(replay_path) == EXPECTED_REPLAY_REPORT_SHA256, "accepted replay attribution identity changed")
    replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
    _require(replay_report.get("result") == "VALID_COMPLETED", "accepted replay attribution is not valid completed")
    _require(replay_report.get("outer_gate") == "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_ATTRIBUTED", "accepted replay attribution gate changed")
    _require(replay_report.get("classification") == "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_MIXED", "accepted replay attribution classification changed")

    post_path = root / POST3000_REPORT
    _require(file_sha256(post_path) == EXPECTED_POST3000_REPORT_SHA256, "accepted post3000 report identity changed")
    post_report = json.loads(post_path.read_text(encoding="utf-8"))
    _require(post_report.get("result") == "VALID_COMPLETED", "accepted post3000 report is not valid completed")
    _require(post_report.get("outer_gate") == "POST3000_GENERATION_READINESS_AND_SCALING_RESOLVED", "accepted post3000 gate changed")
    _require(post_report.get("classification") == "REAL_OLS_SEMANTIC_RECOVERY_WITHOUT_SAMPLED_GENERATION_TRANSFER", "accepted post3000 classification changed")

    robust_path = root / ROBUST_REPORT
    _require(file_sha256(robust_path) == EXPECTED_ROBUST_REPORT_SHA256, "accepted robust training report identity changed")
    robust_report = json.loads(robust_path.read_text(encoding="utf-8"))
    _require(robust_report.get("result") == "VALID_COMPLETED", "accepted robust training report is not valid completed")
    _require(robust_report.get("classification") == robust.CLASSIFICATION_SUPPORTED, "accepted robust training classification changed")
    _require(int(robust_report.get("long_horizon_completed_step")) == 3000, "accepted robust training endpoint changed")

    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    checkpoint_payloads: dict[int, Mapping[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for step, descriptor in CHECKPOINTS.items():
        state, payload, observed = _load_checkpoint(root, step, descriptor)
        checkpoints[step] = state
        checkpoint_payloads[step] = payload
        checkpoint_hashes[str(step)] = observed
    encoder0 = checkpoints[0]["encoder.phase"]
    for step in TRAJECTORY_STEPS[1:]:
        _require(torch.equal(encoder0, checkpoints[step]["encoder.phase"]), f"encoder changed at step {step}")
        _require(not torch.equal(checkpoints[0]["generator.phase"], checkpoints[step]["generator.phase"]), f"generator did not change at step {step}")

    bank = _load_diagnostic_bank(root, authorities)
    authorities.update(
        {
            "local_state": local_state,
            "replay_report": replay_report,
            "post3000_report": post_report,
            "robust_report": robust_report,
            "checkpoints": checkpoints,
            "checkpoint_payloads": checkpoint_payloads,
            "checkpoint_hashes": checkpoint_hashes,
            "bank": bank,
        }
    )
    return authorities


def _load_models(
    root: Path,
    authorities: Mapping[str, Any],
    device: torch.device,
) -> tuple[LightGen2DChenBOGTComplexOLSTrainingConsumer, SmoothMNISTEvaluator, FrozenMNISTEvaluator, third_confirmation.ThirdConfirmationEvaluator]:
    robust._prepare_cuda_runtime()
    consumer, smooth, independent, ec = robust._load_models(root, authorities, device)
    consumer.runtime.eval()
    smooth.eval()
    independent.eval()
    ec.eval()
    for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec)):
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} is not frozen")
    return consumer, smooth, independent, ec


def _cmos_view(value: torch.Tensor) -> torch.Tensor:
    _require(value.ndim == 4 and value.shape[1] == 1 and tuple(value.shape[-2:]) in {(28, 28), (200, 200)}, "CMOS view shape changed")
    _finite(value, "CMOS view input")
    view = F.adaptive_avg_pool2d(value.clamp(0.0, 1.0), (28, 28))
    _finite(view, "CMOS view")
    return view


def _normalized_feature_loss(cmos_features: torch.Tensor, target_features: torch.Tensor, name: str) -> torch.Tensor:
    _require(cmos_features.shape == target_features.shape and cmos_features.ndim == 2 and cmos_features.shape[1] == 128, f"{name} feature shape changed")
    value = (
        F.normalize(cmos_features, dim=1, eps=FEATURE_EPS)
        - F.normalize(target_features.detach(), dim=1, eps=FEATURE_EPS)
    ).square().sum(dim=1).mean()
    _finite(value, name)
    return value


def _raw_validity(raw: torch.Tensor) -> dict[str, Any]:
    _finite(raw, "raw CMOS")
    flat = raw.detach().to(torch.float64).flatten(1)
    std = flat.std(dim=1, unbiased=False)
    return {
        "finite": True,
        "finite_fraction": 1.0,
        "nonconstant_fraction": float((std > 0.0).to(torch.float64).mean().cpu()),
        "zero_image_count": int((flat.abs().sum(dim=1) == 0.0).sum().item()),
        "shape": list(raw.shape),
        "min": float(raw.min().detach().cpu()),
        "max": float(raw.max().detach().cpu()),
    }


def _norm(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result) and (allow_zero or result > 0.0), f"{name} is zero or invalid")
    return result


def _rms(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().mean().sqrt().cpu())
    _require(math.isfinite(result) and (allow_zero or result > 0.0), f"{name} RMS is zero or invalid")
    return result


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    value = _dot(left, right) / (_norm(left, f"{name} left") * _norm(right, f"{name} right"))
    _require(math.isfinite(value), f"{name} is nonfinite")
    return value


def _summary(value: torch.Tensor, name: str) -> dict[str, Any]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _finite(flat, name)
    _require(flat.numel() > 0, f"{name} is empty")
    quantiles = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float64))
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std_population": float(flat.std(unbiased=False)),
        "quantiles": {"0.01": float(quantiles[0]), "0.5": float(quantiles[1]), "0.99": float(quantiles[2])},
    }


def _gradient_stats(value: torch.Tensor, name: str, *, phase: bool = False) -> dict[str, Any]:
    _finite(value, name)
    result: dict[str, Any] = {
        "norm": _norm(value, name),
        "rms": _rms(value, name),
        "shape": list(value.shape),
        "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean().cpu()),
        "zero_fraction": float((value == 0.0).to(torch.float64).mean().cpu()),
    }
    if phase:
        result["per_phase_plane_norms"] = {
            f"generator.phase[{index}]": _norm(value[index], f"{name}[{index}]")
            for index in range(int(value.shape[0]))
        }
    return result


def _normalize_rms(value: torch.Tensor, name: str) -> torch.Tensor:
    scale = _rms(value, name)
    direction = -value.detach() / scale
    _require(math.isclose(_rms(direction, f"{name} normalized"), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-7), f"{name} RMS normalization failed")
    return direction


def _class_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().cpu().to(torch.long)
    logits = logits.detach().cpu().to(torch.float32)
    predictions = logits.argmax(dim=1)
    probabilities = torch.softmax(logits, dim=1)
    top2 = torch.topk(logits, k=2, dim=1).values
    recalls = []
    confusion = torch.zeros((10, 10), dtype=torch.long)
    for true_label, predicted_label in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[true_label, predicted_label] += 1
    for digit in range(10):
        count = int((labels == digit).sum())
        _require(count > 0, f"diagnostic class {digit} is empty")
        recalls.append(float(confusion[digit, digit].to(torch.float64) / count))
    confidence = probabilities.max(dim=1).values
    return {
        "accuracy": float((predictions == labels).to(torch.float64).mean()),
        "macro_recall": float(sum(recalls) / 10.0),
        "per_class_recall": recalls,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "predicted_class_histogram": torch.bincount(predictions, minlength=10).tolist(),
        "true_class_histogram": torch.bincount(labels, minlength=10).tolist(),
        "top1_confidence": _summary(confidence, "top1 confidence"),
        "top1_top2_logit_margin": _summary(top2[:, 0] - top2[:, 1], "top1-top2 margin"),
        "rows": int(labels.numel()),
        "label_source": "canonical TRAIN source label",
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
    _require(raw.shape == target_image.shape and tuple(raw.shape[-2:]) == (200, 200), "raw/target optical geometry changed")
    _require(raw.dtype == target_image.dtype == torch.float32, "raw/target dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_image, "target SLM image")
    raw_view = _cmos_view(raw)
    target_view = _cmos_view(batch)
    smooth_features, smooth_logits = smooth(raw_view)
    independent_features, independent_logits = independent(raw_view)
    ec_features, ec_logits = ec(raw_view)
    with torch.no_grad():
        smooth_target, _ = smooth(target_view)
        independent_target, _ = independent(target_view)
        ec_target, _ = ec(target_view)
    losses = {
        "L_S": _normalized_feature_loss(smooth_features, smooth_target, "L_S"),
        "L_I": _normalized_feature_loss(independent_features, independent_target, "L_I"),
        "L_C": _normalized_feature_loss(ec_features, ec_target, "L_C"),
        "L_rec": (raw - target_image).square().mean(),
        "L_op": optical_power_loss_from_powers(contracted_plane_powers(result), references.power_reference),
    }
    for name, value in losses.items():
        _finite(value, name)
    return {
        "result": result,
        "raw": raw,
        "target_image": target_image,
        "target_view": target_view.detach(),
        "features": {"E_S": smooth_features, "E_I": independent_features, "E_C": ec_features},
        "target_features": {"E_S": smooth_target.detach(), "E_I": independent_target.detach(), "E_C": ec_target.detach()},
        "logits": {"E_S": smooth_logits, "E_I": independent_logits, "E_C": ec_logits},
        "losses": losses,
    }


def _three_way_geometry(values: Mapping[str, torch.Tensor], name: str) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    _require(set(values) == {"E_S", "E_I", "E_C"}, f"{name} evaluator set changed")
    norms = {key: _norm(value, f"{name} {key}") for key, value in values.items()}
    units = {key: value.detach() / norms[key] for key, value in values.items()}
    pairwise = {
        "E_S_E_I": _cosine(units["E_S"], units["E_I"], f"{name} E_S/E_I"),
        "E_S_E_C": _cosine(units["E_S"], units["E_C"], f"{name} E_S/E_C"),
        "E_I_E_C": _cosine(units["E_I"], units["E_C"], f"{name} E_I/E_C"),
    }
    resultant = units["E_S"] + units["E_I"] + units["E_C"]
    resultant_norm = _norm(resultant, f"{name} three-way resultant", allow_zero=True)
    _require(resultant_norm > JOINT_DEGENERACY_TOLERANCE, f"{name} three-way resultant is degenerate")
    resultant_rms = _rms(resultant, f"{name} three-way resultant")
    result = {
        "gradient_norms": norms,
        "pairwise_cosines": pairwise,
        "three_way_resultant_l2_norm": resultant_norm,
        "three_way_resultant_rms": resultant_rms,
        "common_alignment_index": resultant_norm / 3.0,
        "cosine_to_common_resultant": {key: _cosine(value, resultant, f"{name} {key}/common") for key, value in units.items()},
        "common_direction_defined": True,
    }
    return result, resultant, units


def _two_way_vector(values: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    left = values["E_S"]
    right = values["E_I"]
    vector = left.detach() / _norm(left, f"{name} E_S") + right.detach() / _norm(right, f"{name} E_I")
    _require(_norm(vector, f"{name} resultant", allow_zero=True) > JOINT_DEGENERACY_TOLERANCE, f"{name} resultant is degenerate")
    return vector


def _cross_space_common_direction_comparison(output: Mapping[str, Any], phase: Mapping[str, Any]) -> dict[str, Any]:
    output_values = [float(output["cosine_to_common_resultant"][name]) for name in ("E_S", "E_I", "E_C")]
    phase_values = [float(phase["cosine_to_common_resultant"][name]) for name in ("E_S", "E_I", "E_C")]
    gaps = [left - right for left, right in zip(output_values, phase_values, strict=True)]
    return {
        "output_cosine_to_output_common": output_values,
        "phase_cosine_to_phase_common": phase_values,
        "output_minus_phase_common_cosine_gaps": gaps,
        "mean_absolute_gap": float(sum(abs(value) for value in gaps) / len(gaps)),
        "definition": "compare evaluator-to-common-direction alignment patterns; output and phase tensors are not directly cosine-comparable",
    }


def _feature_residual_metrics(
    labels: torch.Tensor,
    generated_features: Mapping[str, torch.Tensor],
    target_features: Mapping[str, torch.Tensor],
    logits: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    labels_cpu = labels.detach().cpu().to(torch.long)
    result: dict[str, Any] = {}
    for name in ("E_S", "E_I", "E_C"):
        generated = F.normalize(generated_features[name].detach().cpu().to(torch.float64), dim=1, eps=FEATURE_EPS)
        target = F.normalize(target_features[name].detach().cpu().to(torch.float64), dim=1, eps=FEATURE_EPS)
        centroids = torch.stack([target[labels_cpu == digit].mean(dim=0) for digit in range(10)])
        centroids = F.normalize(centroids, dim=1, eps=FEATURE_EPS)
        distances = torch.cdist(generated, centroids, p=2)
        own = distances.gather(1, labels_cpu[:, None]).squeeze(1)
        masked = distances.clone()
        masked[torch.arange(labels_cpu.numel()), labels_cpu] = float("inf")
        nearest_other = masked.min(dim=1).values
        result[name] = {
            "class_metrics": _class_metrics(labels_cpu, logits[name]),
            "target_vs_generated_feature_distance": _summary((generated - target).square().sum(dim=1), f"{name} target/generated feature distance"),
            "generated_vs_own_class_centroid_distance": _summary(own, f"{name} own centroid distance"),
            "generated_vs_nearest_other_class_centroid_margin": _summary(nearest_other - own, f"{name} nearest-other margin"),
            "centroid_definition": "L2-normalized target features from the same class-balanced TRAIN A bank; Euclidean distance",
        }
    return result


def _analyze_checkpoint(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    step: int,
    ledger: DataReadLedger,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    generator = _generator(consumer)
    _require(indices.numel() > 0 and indices.numel() % MICRO_BATCH_SIZE == 0, "A bank is not divisible by fixed micro-batch")
    accum = {name: torch.zeros_like(generator) for name in ("g_S", "g_I", "g_C", "g_rec", "g_op")}
    h_chunks = {name: [] for name in ("h_S", "h_I", "h_C")}
    generated_features = {name: [] for name in ("E_S", "E_I", "E_C")}
    target_features = {name: [] for name in ("E_S", "E_I", "E_C")}
    logits = {name: [] for name in ("E_S", "E_I", "E_C")}
    raw_chunks: list[torch.Tensor] = []
    target_image_chunks: list[torch.Tensor] = []
    scalar_totals = {name: 0.0 for name in ("L_S", "L_I", "L_C", "L_rec", "L_op")}
    row_count = int(indices.numel())
    for start in range(0, row_count, MICRO_BATCH_SIZE):
        batch_indices = indices[start : start + MICRO_BATCH_SIZE]
        batch = images.index_select(0, batch_indices).to(consumer.device)
        bundle = _loss_bundle(consumer, smooth, independent, ec, batch, references)
        grads: dict[str, tuple[torch.Tensor, torch.Tensor] | tuple[None, torch.Tensor]] = {}
        for loss_name, output_name in (("L_S", "S"), ("L_I", "I"), ("L_C", "C")):
            h, g = torch.autograd.grad(bundle["losses"][loss_name], (bundle["raw"], generator), retain_graph=True, allow_unused=False)
            grads[output_name] = (h.detach(), g.detach())
        g_rec = torch.autograd.grad(bundle["losses"]["L_rec"], generator, retain_graph=True, allow_unused=False)[0].detach()
        g_op = torch.autograd.grad(bundle["losses"]["L_op"], generator, retain_graph=False, allow_unused=False)[0].detach()
        weight = batch_indices.numel() / row_count
        for output_name in ("S", "I", "C"):
            h, g = grads[output_name]
            h_chunks[f"h_{output_name}"].append((weight * h).cpu())
            accum[f"g_{output_name}"] += weight * g
        accum["g_rec"] += weight * g_rec
        accum["g_op"] += weight * g_op
        for name, value in bundle["losses"].items():
            scalar_totals[name] += weight * float(value.detach().cpu())
        for name in generated_features:
            generated_features[name].append(bundle["features"][name].detach().cpu())
            target_features[name].append(bundle["target_features"][name].detach().cpu())
            logits[name].append(bundle["logits"][name].detach().cpu())
        raw_chunks.append(bundle["raw"].detach().cpu())
        target_image_chunks.append(bundle["target_image"].detach().cpu())
        ledger.record(TRAIN_ROLE, int(batch_indices.numel()))
        _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), f"step {step} populated runtime .grad")
        _clear_evaluator_grads(smooth, independent, ec)

    h_all = {name: torch.cat(values, dim=0) for name, values in h_chunks.items()}
    generated_all = {name: torch.cat(values, dim=0) for name, values in generated_features.items()}
    target_all = {name: torch.cat(values, dim=0) for name, values in target_features.items()}
    logits_all = {name: torch.cat(values, dim=0) for name, values in logits.items()}
    raw_all = torch.cat(raw_chunks, dim=0)
    target_image_all = torch.cat(target_image_chunks, dim=0)
    for name, value in {**h_all, **accum, "raw": raw_all}.items():
        _finite(value, f"step {step} {name}")
    output_geometry, v_output_3, _ = _three_way_geometry({"E_S": h_all["h_S"], "E_I": h_all["h_I"], "E_C": h_all["h_C"]}, f"step {step} output")
    phase_geometry, v_phase_3, _ = _three_way_geometry({"E_S": accum["g_S"], "E_I": accum["g_I"], "E_C": accum["g_C"]}, f"step {step} phase")
    v_si = _two_way_vector({"E_S": accum["g_S"], "E_I": accum["g_I"]}, f"step {step} v_SI")
    residual = _feature_residual_metrics(labels.index_select(0, indices), generated_all, target_all, logits_all)
    result = {
        "step": step,
        "rows": row_count,
        "losses": scalar_totals,
        "output_space": output_geometry,
        "phase_space": phase_geometry,
        "cross_space_common_direction": _cross_space_common_direction_comparison(output_geometry, phase_geometry),
        "historical_SI": {"cos_g_C_vs_v_SI": _cosine(accum["g_C"], v_si, f"step {step} g_C/v_SI"), "v_SI_l2_norm": _norm(v_si, f"step {step} v_SI")},
        "residual_decomposition": residual,
        "raw_validity": _raw_validity(raw_all),
        "reconstruction": {"L_rec": scalar_totals["L_rec"], "target_generated_rms": _rms(raw_all - target_image_all, f"step {step} raw/target")},
    }
    tensors = {
        **h_all,
        **accum,
        "raw": raw_all,
        "target_image": target_image_all,
        "generated_E_S": generated_all["E_S"],
        "generated_E_I": generated_all["E_I"],
        "generated_E_C": generated_all["E_C"],
        "target_E_S": target_all["E_S"],
        "target_E_I": target_all["E_I"],
        "target_E_C": target_all["E_C"],
        "logits_E_S": logits_all["E_S"],
        "logits_E_I": logits_all["E_I"],
        "logits_E_C": logits_all["E_C"],
        "v_output_3": v_output_3,
        "v_phase_3": v_phase_3,
        "v_SI": v_si,
    }
    return result, tensors


@torch.no_grad()
def _evaluate_bank(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    images: torch.Tensor,
    references: Any,
    ledger: DataReadLedger,
) -> dict[str, Any]:
    count = int(images.shape[0])
    totals = {name: 0.0 for name in ("L_S", "L_I", "L_C", "L_rec", "L_op")}
    raw_chunks: list[torch.Tensor] = []
    for start in range(0, count, MICRO_BATCH_SIZE):
        batch = images[start : start + MICRO_BATCH_SIZE]
        bundle = _loss_bundle(consumer, smooth, independent, ec, batch, references)
        weight = batch.shape[0] / count
        for name in totals:
            totals[name] += weight * float(bundle["losses"][name].detach().cpu())
        raw_chunks.append(bundle["raw"].detach().cpu())
        ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    raw = torch.cat(raw_chunks, dim=0)
    return {"losses": totals, "raw_validity": _raw_validity(raw)}


def _metric_deltas(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(candidate["losses"][name] - base["losses"][name]) for name in ("L_S", "L_I", "L_C", "L_rec", "L_op")}


def _stable_descent(probe: Mapping[str, Any], direction: str, metric: str) -> bool:
    records = probe["directions"].get(direction)
    if not isinstance(records, list) or not records:
        return False
    return all(
        float(record["plus"]["deltas"][metric]) < 0.0 and float(record["minus"]["deltas"][metric]) > 0.0
        for record in records
    )


def _run_probes(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    images: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    step: int,
    phase_gradients: Mapping[str, torch.Tensor],
    ledger: DataReadLedger,
) -> dict[str, Any]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    consumer.runtime.eval()
    base_state = _runtime_state(consumer)
    g_s, g_i, g_c = phase_gradients["g_S"], phase_gradients["g_I"], phase_gradients["g_C"]
    u_s, u_i, u_c = g_s / _norm(g_s, "probe g_S"), g_i / _norm(g_i, "probe g_I"), g_c / _norm(g_c, "probe g_C")
    v_si = u_s + u_i
    v_sic = v_si + u_c
    _require(_norm(v_si, "probe v_SI", allow_zero=True) > JOINT_DEGENERACY_TOLERANCE, "probe v_SI is degenerate")
    _require(_norm(v_sic, "probe v_SIC", allow_zero=True) > JOINT_DEGENERACY_TOLERANCE, "probe v_SIC is degenerate")
    directions = {"d_SI": _normalize_rms(v_si, "d_SI"), "d_C": _normalize_rms(u_c, "d_C"), "d_SIC": _normalize_rms(v_sic, "d_SIC")}
    b_images = images.to(consumer.device)
    base = _evaluate_bank(consumer, smooth, independent, ec, b_images, references, ledger)
    records: dict[str, list[dict[str, Any]]] = {}
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
                    candidate = _evaluate_bank(consumer, smooth, independent, ec, b_images, references, ledger)
                    signed[sign_name] = {
                        "deltas": _metric_deltas(base, candidate),
                        "losses": candidate["losses"],
                        "raw_validity": candidate["raw_validity"],
                    }
                    consumer.load_trainable_state_dict(checkpoint)
                    _freeze_generator_only(consumer)
                    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"step {step} {direction_name} probe was not restored")
                signed["central_slope_per_epsilon"] = {
                    metric: float((signed["plus"]["deltas"][metric] - signed["minus"]["deltas"][metric]) / (2.0 * epsilon))
                    for metric in ("L_S", "L_I", "L_C", "L_rec", "L_op")
                }
                epsilon_records.append({"epsilon": epsilon, **signed})
            records[direction_name] = epsilon_records
    finally:
        consumer.load_trainable_state_dict(checkpoint)
        _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(base_state), f"step {step} probes did not restore checkpoint")
    return {
        "schema": PROBE_SCHEMA,
        "step": step,
        "epsilon_definition": "epsilon radians RMS phase added to generator.phase",
        "epsilons": list(PERTURBATION_EPSILONS),
        "base": base,
        "directions": records,
        "stable_descent": {
            direction: {metric: _stable_descent({"directions": records}, direction, metric) for metric in ("L_S", "L_I", "L_C")}
            for direction in records
        },
        "checkpoint_restored_after_every_probe": True,
    }


def _project_auxiliary_gradient(auxiliary: torch.Tensor, semantic: torch.Tensor, name: str) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = _norm(auxiliary, f"{name} weighted")
    semantic_norm = _norm(semantic, "historical semantic")
    dot_before = _dot(auxiliary, semantic)
    projected = dot_before < 0.0
    removed = torch.zeros_like(auxiliary)
    if projected:
        coefficient = dot_before / (float(torch.sum(semantic.detach().to(torch.float64).square()).cpu()) + EPS_PROJ)
        removed = semantic * coefficient
    safe = auxiliary - removed
    _finite(safe, f"{name} safe")
    dot_after = _dot(safe, semantic)
    removed_norm = _norm(removed, f"{name} removed", allow_zero=True)
    return safe, {
        "dot_before": dot_before,
        "dot_after": dot_after,
        "projected": projected,
        "removed_norm": removed_norm,
        "projection_fraction_norm": removed_norm / auxiliary_norm,
        "semantic_norm": semantic_norm,
        "eps_proj": EPS_PROJ,
    }


def _auxiliary_record(gradients: Mapping[str, torch.Tensor], step: int, robust_report: Mapping[str, Any]) -> dict[str, Any]:
    g_s, g_i, g_c = gradients["g_S"], gradients["g_I"], gradients["g_C"]
    u_s, u_i = g_s / _norm(g_s, "aux g_S"), g_i / _norm(g_i, "aux g_I")
    u_c = g_c / _norm(g_c, "aux g_C")
    v_si = u_s + u_i
    v_sic = v_si + u_c
    q_si = v_si / _rms(v_si, "aux q_SI")
    delta_joint = float(robust_report["recovery_authority"]["delta_joint"]["delta_joint"])
    _require(math.isclose(delta_joint, HISTORICAL_DELTA_JOINT, rel_tol=0.0, abs_tol=1.0e-15), "historical delta_joint changed")
    g_sem = delta_joint * q_si
    g_rec_w = HISTORICAL_BETA * gradients["g_rec"]
    g_op_w = HISTORICAL_GAMMA * gradients["g_op"]
    g_rec_safe, rec_projection = _project_auxiliary_gradient(g_rec_w, g_sem, "g_rec")
    g_op_safe, op_projection = _project_auxiliary_gradient(g_op_w, g_sem, "g_op")
    g_total = g_sem + g_rec_safe + g_op_safe
    _finite(g_total, "historical g_total")
    return {
        "step": step,
        "historical_coefficients": {"beta": HISTORICAL_BETA, "gamma": HISTORICAL_GAMMA, "delta_joint": delta_joint, "coefficient_sweep": False},
        "gradient_norms": {name: _gradient_stats(value, f"step {step} {name}", phase=True) for name, value in {"g_rec": gradients["g_rec"], "g_op": gradients["g_op"], "g_rec_w": g_rec_w, "g_op_w": g_op_w, "g_sem": g_sem, "g_rec_safe": g_rec_safe, "g_op_safe": g_op_safe, "g_total": g_total}.items()},
        "cosines": {
            "g_rec_vs_g_S": _cosine(gradients["g_rec"], g_s, "g_rec/g_S"),
            "g_rec_vs_g_I": _cosine(gradients["g_rec"], g_i, "g_rec/g_I"),
            "g_rec_vs_g_C": _cosine(gradients["g_rec"], g_c, "g_rec/g_C"),
            "g_rec_vs_v_SI": _cosine(gradients["g_rec"], v_si, "g_rec/v_SI"),
            "g_rec_vs_v_SIC": _cosine(gradients["g_rec"], v_sic, "g_rec/v_SIC"),
            "g_op_vs_g_S": _cosine(gradients["g_op"], g_s, "g_op/g_S"),
            "g_op_vs_g_I": _cosine(gradients["g_op"], g_i, "g_op/g_I"),
            "g_op_vs_g_C": _cosine(gradients["g_op"], g_c, "g_op/g_C"),
            "g_op_vs_v_SI": _cosine(gradients["g_op"], v_si, "g_op/v_SI"),
            "g_op_vs_v_SIC": _cosine(gradients["g_op"], v_sic, "g_op/v_SIC"),
            "g_total_vs_g_sem": _cosine(g_total, g_sem, "g_total/g_sem"),
        },
        "projection": {"g_rec": rec_projection, "g_op": op_projection},
        "semantic_component_positive": _dot(g_total, g_sem) > 0.0,
        "no_coefficient_change": True,
    }


def _adam_effective_update_rms(payload: Mapping[str, Any]) -> float | None:
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping):
        return None
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list) or not groups:
        return None
    slot = state.get(0)
    if not isinstance(slot, Mapping):
        return None
    exp_avg = slot.get("exp_avg")
    exp_avg_sq = slot.get("exp_avg_sq")
    step_value = slot.get("step")
    group = groups[0]
    if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor) or not isinstance(step_value, torch.Tensor):
        return None
    if not isinstance(group, Mapping):
        return None
    step = float(step_value.item())
    lr = float(group.get("lr", 0.0))
    betas = tuple(float(value) for value in group.get("betas", ADAM_BETAS))
    eps = float(group.get("eps", ADAM_EPSILON))
    if step <= 0.0 or lr <= 0.0 or len(betas) != 2:
        return None
    bias_correction1 = 1.0 - betas[0] ** step
    bias_correction2 = 1.0 - betas[1] ** step
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    update = step_size * exp_avg.to(torch.float64) / (exp_avg_sq.to(torch.float64).sqrt() + eps)
    _finite(update, "Adam effective update")
    return _rms(update, "Adam effective update")


def _preflight(
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    smooth: SmoothMNISTEvaluator,
    independent: FrozenMNISTEvaluator,
    ec: third_confirmation.ThirdConfirmationEvaluator,
    images: torch.Tensor,
    indices: torch.Tensor,
    references: Any,
    checkpoint: Mapping[str, torch.Tensor],
    ledger: DataReadLedger,
) -> dict[str, Any]:
    consumer.load_trainable_state_dict(checkpoint)
    _freeze_generator_only(consumer)
    before = _state_digest(_runtime_state(consumer))
    batch = images.index_select(0, indices[:MICRO_BATCH_SIZE]).to(consumer.device)
    bundle = _loss_bundle(consumer, smooth, independent, ec, batch, references)
    generator = _generator(consumer)
    for loss_name in ("L_S", "L_I", "L_C", "L_rec", "L_op"):
        gradient = torch.autograd.grad(bundle["losses"][loss_name], generator, retain_graph=loss_name != "L_op", allow_unused=False)[0]
        _finite(gradient, f"preflight {loss_name} generator gradient")
    ledger.record(TRAIN_ROLE, int(batch.shape[0]))
    _require(before == _state_digest(_runtime_state(consumer)), "preflight changed runtime state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "preflight left runtime gradients")
    _clear_evaluator_grads(smooth, independent, ec)
    return {
        "status": "READY",
        "optimizer_created": False,
        "parameter_updates": 0,
        "representative_rows": int(batch.shape[0]),
        "output_gradients": ["E_S", "E_I", "E_C"],
        "phase_gradients": ["E_S", "E_I", "E_C", "L_rec", "L_op"],
        "runtime_state_restored": True,
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase", "OLS", "W_OLS", "E_S", "E_I", "E_C", "P2", "Q2"],
        "protected_test_sealed_zero": True,
    }


def _make_identity(source_manifest: Mapping[str, Any], bank: Mapping[str, torch.Tensor], attempt_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role=TRAIN_ROLE,
        config_identity={
            "no_update": True,
            "checkpoint_steps": list(TRAJECTORY_STEPS),
            "key_steps": list(KEY_STEPS),
            "bank_seed": BANK_SEED,
            "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()},
            "epsilons": list(PERTURBATION_EPSILONS),
            "historical_beta": HISTORICAL_BETA,
            "historical_gamma": HISTORICAL_GAMMA,
        },
        model_identity="LIGHTGEN_2D_POST3000_GENERATOR_SEMANTIC_REALIZATION_CEILING_ATTRIBUTION_V1",
        optimizer_identity={"created": False, "parameter_updates": 0, "adam_effective_update": "read-only checkpoint-state diagnostic"},
        sampler_identity={"kind": "accepted deterministic TRAIN-only A/B bank", "seed": BANK_SEED},
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=attempt_id,
    )


def _trajectory_records(
    records: Mapping[int, Mapping[str, Any]],
    checkpoints: Mapping[int, Mapping[str, torch.Tensor]],
    checkpoint_payloads: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    previous_step: int | None = None
    previous_generator: torch.Tensor | None = None
    for step in TRAJECTORY_STEPS:
        record = records[step]
        generator = checkpoints[step]["generator.phase"]
        item = {
            "step": step,
            "losses": dict(record["losses"]),
            "gradient_norms": {
                "E_S": record["phase_space"]["gradient_norms"]["E_S"],
                "E_I": record["phase_space"]["gradient_norms"]["E_I"],
                "E_C": record["phase_space"]["gradient_norms"]["E_C"],
            },
            "phase_pairwise_cosines": record["phase_space"]["pairwise_cosines"],
            "phase_common_alignment_index": record["phase_space"]["common_alignment_index"],
            "reconstruction": record["reconstruction"],
            "raw_validity": record["raw_validity"],
            "phase_displacement_from_clean_step0": _norm(generator - checkpoints[0]["generator.phase"], f"step {step} displacement", allow_zero=True),
            "adam_effective_update_rms_next_step": _adam_effective_update_rms(checkpoint_payloads[step]),
        }
        if previous_step is not None and previous_generator is not None:
            delta_steps = step - previous_step
            item["interval_from_previous_checkpoint"] = {
                "previous_step": previous_step,
                "delta_steps": delta_steps,
                "phase_update_rms": _rms(generator - previous_generator, f"step {step} interval phase update"),
                "semantic_slopes_per_update": {
                    name: float((record["losses"][name] - records[previous_step]["losses"][name]) / delta_steps)
                    for name in ("L_S", "L_I", "L_C")
                },
                "constraint_slopes_per_update": {
                    name: float((record["losses"][name] - records[previous_step]["losses"][name]) / delta_steps)
                    for name in ("L_rec", "L_op")
                },
            }
        result[str(step)] = item
        previous_step = step
        previous_generator = generator
    return {
        "steps": list(TRAJECTORY_STEPS),
        "records": result,
        "definitions": {
            "semantic_slope": "checkpoint loss delta divided by checkpoint update-count delta on the TRAIN A bank",
            "phase_update_rms": "RMS of generator.phase checkpoint difference over the preceding checkpoint interval",
            "adam_effective_update_rms_next_step": "bias-corrected Adam update reconstructed from the frozen checkpoint optimizer state; no optimizer constructed",
        },
    }


def _classify(
    step3000: Mapping[str, Any],
    probe3000: Mapping[str, Any],
    auxiliary3000: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str]:
    shared_si = _stable_descent(probe3000, "d_SI", "L_S") and _stable_descent(probe3000, "d_SI", "L_I")
    shared_sic = all(_stable_descent(probe3000, "d_SIC", metric) for metric in ("L_S", "L_I", "L_C"))
    output = step3000["output_space"]
    phase = step3000["phase_space"]
    output_conflict = min(output["pairwise_cosines"].values()) < 0.0
    phase_conflict = min(phase["pairwise_cosines"].values()) < 0.0
    output_more_aligned = float(output["common_alignment_index"]) > float(phase["common_alignment_index"])
    cos_c_si = float(step3000["historical_SI"]["cos_g_C_vs_v_SI"])
    projections = auxiliary3000["projection"]
    auxiliary_opposes = bool(projections["g_rec"]["projected"] or projections["g_op"]["projected"])
    flags = {
        "step3000_d_SI_improves_E_S_and_E_I": shared_si,
        "step3000_d_SIC_improves_all_three": shared_sic,
        "step3000_output_pairwise_conflict": output_conflict,
        "step3000_phase_pairwise_conflict": phase_conflict,
        "step3000_output_common_alignment_exceeds_phase": output_more_aligned,
        "step3000_E_C_alignment_sum_with_E_S_E_I": cos_c_si,
        "historical_auxiliary_projection_present": auxiliary_opposes,
    }
    if shared_sic:
        classification = CLASSIFICATION_SHARED
        recommendation = "DEFINE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1"
    elif shared_si and (not _stable_descent(probe3000, "d_SIC", "L_C") or cos_c_si <= 0.0):
        classification = CLASSIFICATION_TWO_EVALUATOR
        recommendation = "DEFINE_THREE_EVALUATOR_ROBUST_SEMANTIC_CONTINUATION_V1"
    elif output_conflict and output_more_aligned:
        classification = CLASSIFICATION_CONFLICT
        recommendation = "DEFINE_EVALUATOR_AGNOSTIC_IMAGE_MANIFOLD_OBJECTIVE_V1"
    elif output_more_aligned and not shared_si:
        classification = CLASSIFICATION_JACOBIAN
        recommendation = "DEFINE_MINIMAL_GENERATOR_FUNCTION_CLASS_EXPANSION_AFTER_SEMANTIC_CEILING_V1"
    elif auxiliary_opposes:
        classification = CLASSIFICATION_AUXILIARY
        recommendation = "DEFINE_BOUNDED_AUXILIARY_CONSTRAINT_SEMANTIC_REALIZATION_EXPERIMENT_V1"
    else:
        classification = CLASSIFICATION_MIXED
        recommendation = "NONE"
    return classification, {"flags": flags, "classification_order": ["shared", "two_evaluator", "output_conflict", "jacobian", "auxiliary", "mixed"]}, recommendation


def run_analysis(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"output already exists: {target}")

    authorities = _load_authorities(root)
    cuda_runtime = robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "ceiling attribution requires CUDA")
    device = torch.device("cuda:0")
    consumer, smooth, independent, ec = _load_models(root, authorities, device)
    bank = authorities["bank"]
    references = authorities["training"]["references"]
    images = authorities["images"]
    labels = authorities["labels"]
    ledger = DataReadLedger()
    preflight = _preflight(consumer, smooth, independent, ec, images, bank["a_indices"], references, authorities["checkpoints"][0], ledger)
    initial_runtime_state = _runtime_state(consumer)
    initial_evaluator_states = {name: _state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec))}

    source_manifest_object = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            Path(robust.__file__).resolve(),
            Path(frozen_latent.__file__).resolve(),
            Path(third_confirmation.__file__).resolve(),
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
            root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
            root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
            root / "src/ldm_project/v2/student/evaluator.py",
        ),
        config_identity={
            "authorization_mode": "IMPLEMENT_PREPARED",
            "authorization_token": AUTHORIZATION_TOKEN,
            "outer_gate": OUTER_GATE,
            "no_update": True,
            "trajectory_steps": list(TRAJECTORY_STEPS),
            "key_steps": list(KEY_STEPS),
            "bank_seed": BANK_SEED,
            "bank_indices_sha256": {name: tensor_sha256(value) for name, value in bank.items()},
            "epsilons": list(PERTURBATION_EPSILONS),
            "input_hashes": {"replay_report": EXPECTED_REPLAY_REPORT_SHA256, "post3000_report": EXPECTED_POST3000_REPORT_SHA256, "robust_report": EXPECTED_ROBUST_REPORT_SHA256, **authorities["checkpoint_hashes"]},
        },
    )
    source_manifest = source_manifest_object.to_dict()
    target.mkdir(parents=True, exist_ok=False)
    attempt_id = f"{RUN_ID}:{target.name}"
    identity = _make_identity(source_manifest, bank, attempt_id)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    receipt = control.make_ready_receipt(
        correctness={
            "passed": bool(preflight.get("status") == "READY" and preflight.get("parameter_updates") == 0 and preflight.get("runtime_state_restored") is True and preflight.get("protected_test_sealed_zero") is True and preflight.get("output_gradients") == ["E_S", "E_I", "E_C"] and preflight.get("phase_gradients") == ["E_S", "E_I", "E_C", "L_rec", "L_op"]),
            "representative_rows": int(preflight.get("representative_rows", 0)),
            "runtime_state_restored": preflight.get("runtime_state_restored"),
            "protected_test_sealed_zero": bool(preflight.get("protected_test_sealed_zero")),
        },
        resource={"passed": int(cuda_total) > 0 and 0 <= int(cuda_free) <= int(cuda_total), "free_bytes": int(cuda_free), "total_bytes": int(cuda_total), "updates": 0},
        implementation={"passed": preflight.get("optimizer_created") is False and preflight.get("parameter_updates") == 0, "selected_engine": "read_only_cuda_attribution"},
        selected_engine="read_only_cuda_attribution",
        created_at=time.time(),
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    effective_identity = control.open_scientific_attempt(target, attempt_id, receipt, resume=True)
    _write_json(target / "preflight_receipt.json", receipt.to_dict())

    _write_torch(
        target / "diagnostic_bank.pt",
        {"schema": "post3000_generator_semantic_realization_ceiling_bank.v1", "task_id": TASK_ID, "seed": BANK_SEED, "data_role": TRAIN_ROLE, **bank, "development_rows_consumed": 0},
        kind="frozen TRAIN-only A/B diagnostic bank",
    )
    _write_json(target / "source_manifest.json", source_manifest, kind="source manifest")
    _write_json(
        target / "execution_manifest.json",
        {
            "schema": "post3000_generator_semantic_realization_ceiling_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
            "input_authorities": {"replay_report": EXPECTED_REPLAY_REPORT_SHA256, "post3000_report": EXPECTED_POST3000_REPORT_SHA256, "robust_report": EXPECTED_ROBUST_REPORT_SHA256, **authorities["checkpoint_hashes"]},
            "source_manifest": source_manifest,
            "run_identity": effective_identity.to_dict(),
            "local_state": authorities["local_state"],
            "cuda_runtime": cuda_runtime,
            "scope": {"optimizer_created": False, "parameter_updates": 0, "train_rows_unique": 40, "development_rows_consumed": 0, "protected": False, "test": False, "sealed": False, "sampler_changed": False, "architecture_changed": False},
        },
        kind="execution manifest",
    )

    checkpoint_records: dict[int, dict[str, Any]] = {}
    checkpoint_tensors: dict[int, dict[str, torch.Tensor]] = {}
    for step in TRAJECTORY_STEPS:
        record, tensors = _analyze_checkpoint(consumer, smooth, independent, ec, images, labels, bank["a_indices"], references, authorities["checkpoints"][step], step, ledger)
        checkpoint_records[step] = record
        checkpoint_tensors[step] = tensors

    probes: dict[int, dict[str, Any]] = {}
    auxiliary: dict[int, dict[str, Any]] = {}
    for step in KEY_STEPS:
        probes[step] = _run_probes(
            consumer,
            smooth,
            independent,
            ec,
            images.index_select(0, bank["b_indices"]),
            references,
            authorities["checkpoints"][step],
            step,
            checkpoint_tensors[step],
            ledger,
        )
        auxiliary[step] = _auxiliary_record(checkpoint_tensors[step], step, authorities["robust_report"])

    consumer.load_trainable_state_dict(authorities["checkpoints"][0])
    _freeze_generator_only(consumer)
    _require(_state_digest(_runtime_state(consumer)) == _state_digest(initial_runtime_state), "final runtime restoration failed")
    _require(initial_evaluator_states == {name: _state_digest(model.state_dict()) for name, model in (("E_S", smooth), ("E_I", independent), ("E_C", ec))}, "frozen evaluator state changed")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "runtime gradients remained after attribution")
    _clear_evaluator_grads(smooth, independent, ec)
    ledger.assert_forbidden_zero()

    trajectory = _trajectory_records(checkpoint_records, authorities["checkpoints"], authorities["checkpoint_payloads"])
    gradient_payload: dict[str, Any] = {"schema": GRADIENT_SCHEMA, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "checkpoint_hashes": authorities["checkpoint_hashes"], "bank_indices": {"a": bank["a_indices"], "b": bank["b_indices"]}, "key_steps": {str(step): {name: value.detach().cpu() for name, value in checkpoint_tensors[step].items()} for step in KEY_STEPS}, "trajectory_phase_gradients": {str(step): {name: checkpoint_tensors[step][name].detach().cpu() for name in ("g_S", "g_I", "g_C", "g_rec", "g_op")} for step in TRAJECTORY_STEPS}}
    _write_torch(target / "gradient_attribution.pt", gradient_payload, kind="frozen gradient attribution tensors")
    _write_json(target / "directional_probes.json", {"schema": PROBE_SCHEMA, "steps": {str(step): probes[step] for step in KEY_STEPS}}, kind="symmetric held-out directional probes")

    classification, classification_evidence, recommendation = _classify(checkpoint_records[3000], probes[3000], auxiliary[3000])
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
            "accepted_replay_attribution": _artifact_ref(root, root / REPLAY_REPORT, role="accepted real-OLS replay identity-gap attribution"),
            "accepted_post3000_context": _artifact_ref(root, root / POST3000_REPORT, role="accepted post3000 real-OLS context"),
            "accepted_robust_training_report": _artifact_ref(root, root / ROBUST_REPORT, role="accepted step0-to-step3000 trajectory authority"),
            "frozen_checkpoints": {str(step): _artifact_ref(root, root / CHECKPOINTS[step][0], role=f"frozen Generator step {step}") for step in TRAJECTORY_STEPS},
            "frozen_evaluators": {name: dict(authorities["robust_report"]["evaluators"][name]) for name in ("E_S", "E_I", "E_C")},
            "accepted_A_B_bank": _artifact_ref(root, root / REPLAY_BANK_PAYLOAD, role="accepted TRAIN-only A/B bank"),
            "authority_hashes": {"replay_report": EXPECTED_REPLAY_REPORT_SHA256, "post3000_report": EXPECTED_POST3000_REPORT_SHA256, "robust_report": EXPECTED_ROBUST_REPORT_SHA256, **authorities["checkpoint_hashes"]},
        },
        "supported_prior_bottleneck": "REAL_OLS_REPLAY_CONDITIONING_IDENTITY_GAP_MIXED; canonical step3000 real-OLS Generator realization remains below replay semantic ceiling",
        "diagnostic_roles": {
            "A": {"role": "TRAIN-only gradient construction", "rows": int(bank["a_indices"].numel()), "class_balanced": True},
            "B": {"role": "disjoint TRAIN-only symmetric temporary probes", "rows": int(bank["b_indices"].numel()), "class_balanced": True},
            "DEV": {"rows_consumed": 0, "selection_or_gradient_use": False},
            "protected_test_sealed": {"rows_consumed": 0, "selection_or_gradient_use": False},
        },
        "primary_representation": {"object": "raw CMOS output and generator.phase Jacobian attribution", "generator_phase_shape": [2, 1, 100, 100], "complex_OLS_boundary": "frozen accepted Encoder/OLS input path; no OLS or sampler modification"},
        "frozen_evaluator_objective": {"L_S": "mean(||normalize(F_S(clamp_pool(CMOS)))-normalize(F_S(target))||_2^2)", "L_I": "mean(||normalize(F_I(clamp_pool(CMOS)))-normalize(F_I(target))||_2^2)", "L_C": "mean(||normalize(F_C(clamp_pool(CMOS)))-normalize(F_C(target))||_2^2)", "feature_layer": FEATURE_LAYER, "preprocessing": PREPROCESSING, "normalization_epsilon": FEATURE_EPS, "target_features_stop_gradient": True, "E_C_role": "frozen attribution evaluator only; not a training teacher"},
        "preflight": preflight,
        "stage1_residual_decomposition": {"steps": {str(step): checkpoint_records[step]["residual_decomposition"] for step in KEY_STEPS}, "bank": "TRAIN A", "metrics": ["paired_feature_loss", "class_macro_recall", "per_class_recall", "confidence", "top1_top2_margin", "target_vs_generated_feature_distance", "generated_vs_own_class_centroid_distance", "generated_vs_nearest_other_class_centroid_margin"]},
        "stage2_output_space_attribution": {"steps": {str(step): checkpoint_records[step]["output_space"] for step in KEY_STEPS}, "gradient_space": "raw CMOS tensor before exact evaluator preprocessing", "common_direction": "equal-weight sum of L2-normalized h_S,h_I,h_C; no cutoff or coefficient sweep"},
        "stage3_generator_phase_attribution": {"steps": {str(step): checkpoint_records[step]["phase_space"] for step in KEY_STEPS}, "gradient_space": "generator.phase", "common_direction": "equal-weight sum of L2-normalized g_S,g_I,g_C; RMS-normalized only for probes"},
        "stage4_historical_SI_direction": {"steps": {str(step): {"phase": checkpoint_records[step]["historical_SI"], "output": {"pairwise_cosines": checkpoint_records[step]["output_space"]["pairwise_cosines"]}} for step in KEY_STEPS}, "definition": "v_SI = g_S/||g_S||_2 + g_I/||g_I||_2; relation to g_C is the direct cosine in generator.phase space"},
        "stage5_held_out_B_symmetric_probes": {"artifact": _artifact_ref(root, target / "directional_probes.json", role="held-out B symmetric temporary phase probes"), "steps": {str(step): probes[step] for step in KEY_STEPS}, "directions": {"d_SI": "-v_SI/RMS(v_SI)", "d_C": "-u_C/RMS(u_C)", "d_SIC": "-(u_S+u_I+u_C)/RMS(u_S+u_I+u_C)"}, "temporary_parameter_restore": True, "optimizer_created": False},
        "stage6_auxiliary_gradients": {"steps": {str(step): auxiliary[step] for step in KEY_STEPS}, "artifact": _artifact_ref(root, target / "gradient_attribution.pt", role="auxiliary and semantic gradient tensors"), "historical_projection_only": True, "coefficient_changes": False},
        "stage7_trajectory": trajectory,
        "stage8_key_geometry_repeat": {"steps": list(KEY_STEPS), "source": "same frozen measurements as stages 1-6; no new checkpoint or data role", "step_records": {str(step): {"output_common_alignment_index": checkpoint_records[step]["output_space"]["common_alignment_index"], "phase_common_alignment_index": checkpoint_records[step]["phase_space"]["common_alignment_index"], "phase_pairwise_cosines": checkpoint_records[step]["phase_space"]["pairwise_cosines"], "cross_space_common_direction": checkpoint_records[step]["cross_space_common_direction"], "historical_SI": checkpoint_records[step]["historical_SI"], "losses": checkpoint_records[step]["losses"]} for step in KEY_STEPS}},
        "classification_evidence": classification_evidence,
        "recommended_single_successor": recommendation,
        "counterfactual_integrity": {"optimizer_created": False, "parameter_updates": 0, "persistent_probe_changes": False, "checkpoint_restored_after_every_probe": True, "runtime_restored_to_step0": True, "encoder_frozen": True, "architecture_changed": False, "sampler_changed": False, "evaluator_search": False, "E_S_frozen": True, "E_I_frozen": True, "E_C_frozen": True, "protected_test_sealed": False, "DEV_gradients": False},
        "claim_boundary": {"local_frozen_step0_step300_step3000_attribution_only": True, "TRAIN_A_B_only": True, "post3000_ceiling_attribution": True, "global_architecture_inadequacy": False, "sampler_selection": False, "generator_training": False, "downstream_generation_acceptance": False, "physical_hardware_validity": False, "promotion": False},
        "data_reads": ledger.to_dict(),
        "forbidden_data_reads": {"dev": ledger.counts.get("dev", 0), "protected": ledger.counts.get("protected", 0), "test": ledger.counts.get("test", 0), "sealed": ledger.counts.get("sealed", 0)},
        "files_changed": {"repository_source": ["scripts/research/lightgen/execute_lightgen_2d_attribute_post3000_generator_semantic_realization_ceiling_v1.py"], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "preflight": "READY", "trajectory_steps_complete": list(TRAJECTORY_STEPS), "key_probe_steps_complete": list(KEY_STEPS), "symmetric_epsilons": list(PERTURBATION_EPSILONS), "train_only": True, "no_dev": True, "protected_test_sealed_zero": True, "no_optimizer": True, "no_persistent_update": True},
        "unresolved": ["The attribution is local to the frozen accepted Generator, evaluator set, and deterministic TRAIN A/B bank; it does not establish a global architecture or hardware ceiling."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": effective_identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="ceiling attribution report")
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
    except (CeilingAttributionError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
