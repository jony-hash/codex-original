"""Research-only electronic Objective E and two-level O0 lifecycle.

The module deliberately does not construct an optical system.  O0-I is a
28x28 electronic image upper bound; O0-G is the current 8D command generator
followed by an ideal amplitude-square detector.  Neither candidate can
authorize the formal v2 runtime.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ...evidence import (
    EvidenceRecordV2,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ...student.conditional import (
    camera_code_fraction,
    relative_diversity_ratios,
    log_sinkhorn,
)
from ...student.evaluator import load_frozen_evaluator, load_local_mnist, SmoothMNISTEvaluator
from .measurement import validation_reference, fixed_panels
from .o0_models import build_o0_candidate
from .objective_o0_spec import (
    ElectronicO0Spec,
    load_objective_o0_spec,
    objective_o0_input_hashes,
)


OBJECTIVE_E_CONTRACT = canonical_sha256({
    "kind": "objective_e_active_set_v2",
    "activation_step": 400,
    "camera": {"target": 0.30, "penalty": "relu_square"},
    "diversity": {"lower": 0.70, "upper": 1.30, "aggregation": "zero_exact_violation_softmax", "temperature": 0.05},
    "coverage": {"target": 1.35, "penalty": "relu_square", "reference": "companion_class_matched"},
    "calibration": "scale_invariant_active_set_0.25_v2",
    "domains": {"O0-I": "electronic_image_28", "O0-G": "amplitude_square_200_to_28"},
    "measurement": "fixed_panel_dual_evaluator_v1",
    "evidence_chain": "fresh_primary_secondary_measurement_v1",
    "deterministic_runtime": "strict_cuda_v1",
    "schedule": "classwise_joint_primary_companion_v2",
    "preflight": "per_parameter_group_and_negative_control_v2",
    "checkpoint": "rng_schedule_lineage_bound_v3",
    "measurement_panel": "official_panel_content_bound_v2",
    "resume": "explicit_bound_checkpoint_v3",
})

_DETERMINISTIC_RUNTIME = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}


def deterministic_joint_companion_indices(
    labels: torch.Tensor,
    *,
    seed: int,
    global_step: int,
    primary_per_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a deterministic, classwise-disjoint primary/companion batch."""
    if labels.ndim != 1 or primary_per_class < 1 or global_step < 0:
        raise ValueError("joint companion schedule inputs are invalid")
    primary_rows: list[torch.Tensor] = []
    companion_rows: list[torch.Tensor] = []
    for label in range(10):
        choices = torch.nonzero(labels == label, as_tuple=False).flatten().cpu()
        required = primary_per_class * 2
        if choices.numel() < required:
            raise ValueError("joint companion schedule needs two disjoint class batches")
        generator = torch.Generator(device="cpu").manual_seed((seed * 100000 + global_step) * 10 + label)
        permutation = choices.index_select(0, torch.randperm(choices.numel(), generator=generator))
        primary_rows.append(permutation[:primary_per_class])
        companion_rows.append(permutation[primary_per_class:required])
    indices = torch.cat([*primary_rows, *companion_rows]).to(labels.device)
    return indices, labels.index_select(0, indices)


def objective_e2_diversity_penalty(
    features: torch.Tensor,
    real_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    lower: float = 0.70,
    upper: float = 1.30,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return exact-zero bounded diversity penalty and per-class telemetry."""
    if not (0.0 < lower < upper) or temperature <= 0.0:
        raise ValueError("invalid E2 diversity contract")
    generated = F.normalize(features, dim=1)
    real = F.normalize(real_features, dim=1)
    ratios: list[torch.Tensor] = []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("E2 diversity needs at least two samples per class")
        generated_distance = torch.pdist(generated.index_select(0, selected), p=2).median()
        real_distance = torch.pdist(real.index_select(0, selected), p=2).median().detach()
        ratios.append(generated_distance / real_distance.clamp_min(1.0e-8))
    ratio_tensor = torch.stack(ratios)
    violations = F.relu(torch.as_tensor(lower, device=ratio_tensor.device, dtype=ratio_tensor.dtype) - ratio_tensor).square()
    violations = violations + F.relu(ratio_tensor - torch.as_tensor(upper, device=ratio_tensor.device, dtype=ratio_tensor.dtype)).square()
    if bool((violations.max() <= 0).item()):
        penalty = ratio_tensor.sum() * 0.0
    else:
        weights = torch.softmax(violations / temperature, dim=0)
        penalty = (weights * violations).sum()
    worst = int(violations.argmax().item())
    return penalty, {
        "ratios": [float(value.detach()) for value in ratio_tensor],
        "violations": [float(value.detach()) for value in violations],
        "worst_class": worst,
        "penalty": float(penalty.detach()),
    }


class ObjectiveScalingNotQualified(ValueError):
    """Calibration violated a registered active-set safety bound."""

    def __init__(self, name: str, weight: float) -> None:
        super().__init__(f"active-set calibration produced unsafe {name} weight")
        self.name = name
        self.weight = weight


def objective_e_dispatch(step: int, registered_weights: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Return the active objective and consumed weights for a zero-based step."""
    if step < 0:
        raise ValueError("objective step must be non-negative")
    required = {"camera", "diversity", "coverage"}
    if set(registered_weights) != required or not all(math.isfinite(float(v)) and float(v) >= 0 for v in registered_weights.values()):
        raise ValueError("Objective E weights are malformed")
    if step < 400:
        return "A", {name: 0.0 for name in required}
    return "E", dict(registered_weights)


def camera_safety_penalty(raw: torch.Tensor, labels: torch.Tensor, *, target: float = 0.30) -> torch.Tensor:
    return F.relu(torch.as_tensor(target, device=raw.device, dtype=raw.dtype) - camera_code_fraction(raw, labels)).square()


def coverage_ratio_raw(
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
    labels: torch.Tensor,
    *,
    epsilon: float = 0.05,
    iterations: int = 20,
    companion_features: torch.Tensor | None = None,
    companion_view: torch.Tensor | None = None,
    companion_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the actual class-matched coverage ratio, separate from penalty."""
    if (companion_features is None) != (companion_view is None) or (companion_features is None) != (companion_labels is None):
        raise ValueError("coverage companion tensors must be supplied together")

    def cost(left_f: torch.Tensor, left_v: torch.Tensor, right_f: torch.Tensor, right_v: torch.Tensor) -> torch.Tensor:
        lf, rf = F.normalize(left_f, dim=1), F.normalize(right_f, dim=1)
        lp, rp = left_v.flatten(1), right_v.flatten(1)
        return log_sinkhorn(torch.cdist(lf, rf).square() + 0.25 * torch.cdist(lp, rp, p=1) / lp.shape[1], epsilon=epsilon, iterations=iterations)

    observed, baseline = [], []
    for label in range(10):
        selected = torch.nonzero(labels == label, as_tuple=False).flatten()
        if selected.numel() < 2:
            raise ValueError("coverage ratio needs at least two examples per class")
        if companion_features is None:
            half = selected.numel() // 2
            left, right = selected[:half], selected[half : 2 * half]
            base_f, base_v = real_features.index_select(0, left), real_view.index_select(0, left)
        else:
            companion_selected = torch.nonzero(companion_labels == label, as_tuple=False).flatten()
            if companion_selected.numel() != selected.numel():
                raise ValueError("coverage companion labels must match primary counts")
            right, base_f, base_v = selected, companion_features.index_select(0, companion_selected), companion_view.index_select(0, companion_selected)
        observed.append(cost(generated_features.index_select(0, right), generated_view.index_select(0, right), real_features.index_select(0, right), real_view.index_select(0, right)))
        baseline.append(cost(base_f, base_v, real_features.index_select(0, right), real_view.index_select(0, right)).detach())
    return torch.stack(observed).mean() / torch.stack(baseline).mean().clamp_min(1.0e-12)


def coverage_penalty(ratio: torch.Tensor, *, target: float = 1.35) -> torch.Tensor:
    return F.relu(ratio - torch.as_tensor(target, device=ratio.device, dtype=ratio.dtype)).square()


def objective_e_auxiliaries(
    raw: torch.Tensor,
    generated_features: torch.Tensor,
    generated_view: torch.Tensor,
    real_features: torch.Tensor,
    real_view: torch.Tensor,
    labels: torch.Tensor,
    *,
    weights: dict[str, float],
    companion_features: torch.Tensor,
    companion_view: torch.Tensor,
    companion_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = coverage_ratio_raw(generated_features, generated_view, real_features, real_view, labels, companion_features=companion_features, companion_view=companion_view, companion_labels=companion_labels)
    values = {
        "camera": camera_safety_penalty(raw, labels),
        "diversity": objective_e2_diversity_penalty(generated_features, real_features, labels)[0],
        "coverage": coverage_penalty(ratio),
        "coverage_ratio_raw": ratio,
    }
    total = sum((float(weights[name]) * values[name] for name in ("camera", "diversity", "coverage")), torch.zeros_like(values["camera"]))
    return total, values


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _provenance(spec: ElectronicO0Spec, config_path: str) -> dict[str, str | None]:
    root = _repo_root()
    status = subprocess.run(("git", "status", "--porcelain", "--untracked-files=all"), cwd=root, check=True, capture_output=True, text=True).stdout
    patch = subprocess.run(("git", "diff", "--binary"), cwd=root, check=True, capture_output=True, text=True).stdout + subprocess.run(("git", "diff", "--cached", "--binary"), cwd=root, check=True, capture_output=True, text=True).stdout
    return {
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True, capture_output=True, text=True).stdout.strip(),
        "git_status_hash": canonical_sha256(status),
        "dirty_patch_hash": None if not patch else canonical_sha256(patch),
        "config_hash": file_sha256(config_path),
        "contract_hash": OBJECTIVE_E_CONTRACT,
    }


def _require_clean(spec: ElectronicO0Spec, config_path: str) -> dict[str, str | None]:
    provenance = _provenance(spec, config_path)
    if spec.require_clean_git and provenance["git_status_hash"] != canonical_sha256(""):
        raise RuntimeError("electronic O0 evidence requires a clean worktree")
    return provenance


def _configure_deterministic_runtime(seed: int) -> None:
    configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if configured != ":4096:8":
        raise RuntimeError("electronic O0 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _training_batch(
    labels: torch.Tensor,
    *,
    run_seed: int,
    global_step: int,
    batch_size: int,
    code_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return one reproducible primary/companion pair and its bound schedule."""
    if batch_size % 10 != 0:
        raise ValueError("O0 training batch must be divisible by ten")
    per_class = batch_size // 10
    joint, joint_labels = deterministic_joint_companion_indices(
        labels,
        seed=run_seed,
        global_step=global_step,
        primary_per_class=per_class,
    )
    primary_count = batch_size
    primary_indices = joint[:primary_count]
    companion_indices = joint[primary_count:]
    code_seed = run_seed * 100000 + global_step
    schedule = {
        "global_step": global_step,
        "run_seed": run_seed,
        "code_seed": code_seed,
        "primary_indices_hash": canonical_sha256(primary_indices.detach().cpu().tolist()),
        "companion_indices_hash": canonical_sha256(companion_indices.detach().cpu().tolist()),
        "schedule_hash": canonical_sha256({
            "global_step": global_step,
            "run_seed": run_seed,
            "code_seed": code_seed,
            "primary": primary_indices.detach().cpu().tolist(),
            "companion": companion_indices.detach().cpu().tolist(),
        }),
    }
    return primary_indices, joint_labels[:primary_count], companion_indices, joint_labels[primary_count:], schedule


def _read_mapping(path: str | Path, *, description: str) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise ValueError(f"{description} is missing")
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON mapping")
    return payload


def _validate_acceptance_chain(
    args: argparse.Namespace,
    spec: ElectronicO0Spec,
    provenance: dict[str, str | None],
) -> dict[str, str]:
    inputs = objective_o0_input_hashes(
        args.config,
        evaluator_checkpoint=args.evaluator_checkpoint,
        primary_qualification_evidence=args.primary_qualification_evidence,
        secondary_evaluator_checkpoint=args.secondary_evaluator_checkpoint,
        secondary_training_evidence=args.secondary_training_evidence,
        measurement_bundle=args.measurement_bundle,
        measurement_evidence=args.measurement_evidence,
    )
    primary = EvidenceRecordV2.from_dict(_read_mapping(args.primary_qualification_evidence, description="primary qualification evidence"))
    secondary = EvidenceRecordV2.from_dict(_read_mapping(args.secondary_training_evidence, description="secondary training evidence"))
    measurement = EvidenceRecordV2.from_dict(_read_mapping(args.measurement_evidence, description="measurement evidence"))
    for record in (primary, secondary, measurement):
        record.validate()
        if record.git_commit != provenance["git_commit"] or record.dirty_patch_hash is not None:
            raise ValueError("electronic O0 requires freshly issued clean acceptance evidence")
    if primary.evidence_kind != "evaluator_qualification" or primary.status != "passed":
        raise ValueError("primary evaluator qualification must be passed")
    if secondary.evidence_kind != "acceptance_evaluator_training" or secondary.status != "passed" or secondary.permitted_next_actions != ("qualify_acceptance_measurement",):
        raise ValueError("secondary evaluator training evidence is not qualified")
    if measurement.evidence_kind != "conditional_acceptance_measurement" or measurement.status != "passed" or measurement.permitted_next_actions != ("evaluate",):
        raise ValueError("acceptance measurement evidence is not qualified")
    primary_hash = inputs["primary_evaluator"]
    secondary_hash = inputs["secondary_evaluator"]
    if primary.input_hashes.get("evaluator_checkpoint") != primary_hash or primary.context.evaluator_checkpoint_hash != primary_hash:
        raise ValueError("primary qualification does not bind the evaluator checkpoint")
    if secondary.input_hashes.get("acceptance_evaluator") != secondary_hash or secondary.context.evaluator_checkpoint_hash != secondary_hash:
        raise ValueError("secondary training evidence does not bind its checkpoint")
    if measurement.input_hashes.get("primary_evaluator") != primary_hash or measurement.input_hashes.get("secondary_evaluator") != secondary_hash:
        raise ValueError("measurement evidence does not bind both evaluators")
    if measurement.input_hashes.get("acceptance_evaluator_training_evidence") != secondary.evidence_hash:
        raise ValueError("measurement evidence does not bind secondary training evidence")
    bundle = _read_mapping(args.measurement_bundle, description="measurement bundle")
    if measurement.input_hashes.get("measurement_bundle") != inputs["measurement_bundle"]:
        raise ValueError("measurement evidence does not bind the measurement bundle")
    if bundle.get("primary_checkpoint") != primary_hash or bundle.get("secondary_checkpoint") != secondary_hash:
        raise ValueError("measurement bundle evaluator hashes mismatch")
    if bundle.get("training_evidence_hash") != secondary.evidence_hash:
        raise ValueError("measurement bundle does not bind secondary training evidence")
    split_hashes = {
        primary.context.dataset_split_manifest_hash,
        secondary.context.dataset_split_manifest_hash,
        measurement.context.dataset_split_manifest_hash,
        bundle.get("split_manifest_hash"),
    }
    if None in split_hashes or len(split_hashes) != 1:
        raise ValueError("acceptance evidence split lineage mismatch")
    if primary.config_hash != secondary.config_hash or primary.config_hash != measurement.config_hash:
        raise ValueError("acceptance evidence formal configuration mismatch")
    test_manifest = Path(args.measurement_bundle).parent / str(bundle.get("test_manifest_filename", ""))
    if not test_manifest.is_file() or file_sha256(test_manifest) != bundle.get("test_manifest_hash"):
        raise ValueError("measurement test manifest is missing or tampered")
    if measurement.input_hashes.get("test_manifest") != bundle.get("test_manifest_hash"):
        raise ValueError("measurement evidence does not bind the test manifest")
    test_payload = _read_mapping(test_manifest, description="measurement test manifest")
    if test_payload.get("manifest_hash") != bundle.get("test_content_manifest_hash") or measurement.input_hashes.get("test_content_manifest") != bundle.get("test_content_manifest_hash"):
        raise ValueError("measurement test content manifest mismatch")
    return inputs


def _validate_o0_evidence(
    path: str | Path,
    *,
    spec: ElectronicO0Spec | None,
    provenance: dict[str, str | None],
    action: str,
    state: str,
    permitted: list[str],
    expected_inputs: dict[str, str],
    candidate: str | None = None,
) -> dict[str, Any]:
    payload = _read_mapping(path, description=f"Objective O0 {action} evidence")
    content_hash = payload.pop("content_hash", None)
    if not isinstance(content_hash, str) or content_hash != canonical_sha256(payload):
        raise ValueError("Objective O0 evidence content hash mismatch")
    payload["content_hash"] = content_hash
    if payload.get("evidence_kind") != "electronic_objective_o0" or payload.get("contract") != OBJECTIVE_E_CONTRACT:
        raise ValueError("Objective O0 evidence contract mismatch")
    if payload.get("action") != action or payload.get("state") != state or payload.get("permitted_next_actions") != permitted:
        raise ValueError("Objective O0 evidence action or state mismatch")
    if payload.get("git_commit") != provenance["git_commit"] or payload.get("dirty_patch_hash") is not None:
        raise ValueError("Objective O0 evidence provenance mismatch")
    if spec is not None and (payload.get("candidate") != spec.candidate or payload.get("config_hash") != spec.config_hash):
        raise ValueError("Objective O0 evidence candidate/config mismatch")
    if candidate is not None and payload.get("candidate") != candidate:
        raise ValueError("Objective O0 evidence candidate mismatch")
    actual_inputs = payload.get("inputs")
    if not isinstance(actual_inputs, dict) or any(actual_inputs.get(name) != value for name, value in expected_inputs.items()):
        raise ValueError("Objective O0 evidence input lineage mismatch")
    if payload.get("formal_promotion_authorized") is not False:
        raise ValueError("Objective O0 evidence cannot authorize formal promotion")
    return payload


def _validate_calibration(
    args: argparse.Namespace,
    spec: ElectronicO0Spec,
    provenance: dict[str, str | None],
    inputs: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _read_mapping(args.calibration_artifact, description="Objective E calibration artifact")
    evidence = _validate_o0_evidence(
        args.calibration_evidence,
        spec=spec,
        provenance=provenance,
        action="calibrate",
        state="calibration_passed",
        permitted=["screen"],
        expected_inputs=inputs,
    )
    if artifact.get("schema_version") != 3 or artifact.get("contract") != OBJECTIVE_E_CONTRACT or artifact.get("candidate") != spec.candidate:
        raise ValueError("Objective E calibration artifact contract mismatch")
    if artifact.get("formal_promotion_authorized") is not False or artifact.get("permitted_next_actions") != ["screen"]:
        raise ValueError("Objective E calibration artifact cannot authorize promotion")
    if artifact.get("warmup_steps") != spec.warmup_steps or artifact.get("calibration_steps") != list(spec.calibration_steps):
        raise ValueError("Objective E calibration step contract mismatch")
    weights = artifact.get("weights")
    normalizers = artifact.get("normalizers")
    coefficients = artifact.get("coefficients")
    records = artifact.get("records")
    safety = artifact.get("safety")
    statuses = artifact.get("statuses")
    if not isinstance(weights, dict) or set(weights) != {"camera", "diversity", "coverage"} or not isinstance(statuses, dict) or not isinstance(normalizers, dict) or not isinstance(coefficients, dict) or not isinstance(records, list) or len(records) != 4 or not isinstance(safety, dict):
        raise ValueError("Objective E calibration weights/statuses are malformed")
    if artifact.get("warmup_schedule_hash") != canonical_sha256(artifact.get("warmup_schedule")) or artifact.get("calibration_schedule_hash") != canonical_sha256(artifact.get("calibration_schedule")):
        raise ValueError("Objective E calibration schedule hash mismatch")
    if not isinstance(artifact.get("warmup_model_state_hash"), str) or not isinstance(artifact.get("warmup_optimizer_state_hash"), str) or not isinstance(artifact.get("warmup_schedule"), list) or len(artifact["warmup_schedule"]) != spec.warmup_steps or not isinstance(artifact.get("calibration_schedule"), list) or len(artifact["calibration_schedule"]) != len(spec.calibration_steps):
        raise ValueError("Objective E calibration warmup/schedule binding is incomplete")
    for name, weight in weights.items():
        if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError("Objective E calibration contains a non-finite weight")
        if statuses.get(name) == "active" and (not math.isfinite(float(weight)) or float(weight) <= 0.0):
            raise ValueError("Objective E active weight must be finite and positive")
        if statuses.get(name) == "inactive_satisfied" and float(weight) != 0.0:
            raise ValueError("satisfied Objective E auxiliary must have zero weight")
    warmup_path = Path(str(artifact.get("warmup_checkpoint", "")))
    warmup_hash = artifact.get("warmup_checkpoint_sha256")
    if not warmup_path.is_absolute() or not warmup_path.is_file() or not isinstance(warmup_hash, str) or file_sha256(warmup_path) != warmup_hash:
        raise ValueError("Objective E calibration warmup checkpoint is missing or tampered")
    warmup_payload = safe_torch_load(warmup_path)
    if not isinstance(warmup_payload, dict) or canonical_sha256(warmup_payload.get("model_state")) != artifact["warmup_model_state_hash"] or canonical_sha256(warmup_payload.get("optimizer_state")) != artifact["warmup_optimizer_state_hash"]:
        raise ValueError("Objective E calibration warmup state hash mismatch")
    if evidence["inputs"].get("warmup_checkpoint") != warmup_hash or evidence["inputs"].get("calibration_artifact") != file_sha256(args.calibration_artifact):
        raise ValueError("Objective E calibration evidence does not bind artifact and warmup")
    return artifact, evidence


def _root(spec: ElectronicO0Spec, action: str) -> Path:
    root = Path(spec.output_root) / f"{datetime.now():%Y%m%d}_{spec.version}" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_research_{action}_{spec.candidate}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_evidence(
    root: Path,
    spec: ElectronicO0Spec,
    args: argparse.Namespace,
    action: str,
    state: str,
    metrics: dict[str, Any],
    *,
    inputs: dict[str, str],
    provenance: dict[str, str | None],
    reason: str | None = None,
    extra: dict[str, str] | None = None,
    permitted: list[str] | None = None,
) -> Path:
    payload = {
        "schema_version": 2,
        "evidence_kind": "electronic_objective_o0",
        "contract": OBJECTIVE_E_CONTRACT,
        "action": action,
        "state": state,
        "candidate": spec.candidate,
        "dataset": args.dataset,
        "seed": getattr(args, "seed", None),
        "config_hash": spec.config_hash,
        "git_commit": provenance["git_commit"],
        "dirty_patch_hash": provenance["dirty_patch_hash"],
        "inputs": {**inputs, **(extra or {})},
        "runtime": _DETERMINISTIC_RUNTIME,
        "metrics": metrics,
        "reason": reason,
        "formal_promotion_authorized": False,
        "permitted_next_actions": permitted or [],
    }
    payload["content_hash"] = canonical_sha256(payload)
    path = root / f"objective_o0_{action}_evidence.json"
    atomic_write_json(path, payload)
    return path


def _load_candidate(spec: ElectronicO0Spec, device: torch.device) -> nn.Module:
    model = build_o0_candidate(spec.candidate).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) > spec.parameter_budget:
        raise ValueError("candidate exceeds its registered parameter budget")
    return model


def _save_o0_checkpoint(
    root: Path,
    spec: ElectronicO0Spec,
    candidate: str,
    seed: int,
    completed_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    history: list[dict[str, Any]] | None = None,
    input_hashes: dict[str, str] | None = None,
    provenance: dict[str, str | None] | None = None,
    stage: str = "training",
    metadata: dict[str, Any] | None = None,
) -> Path:
    checkpoint = root / f"objective_o0_{candidate}_seed_{seed}_step_{completed_step}.pt"
    payload = {
        "schema_version": 3,
        "contract": OBJECTIVE_E_CONTRACT,
        "config_hash": spec.config_hash,
        "candidate": candidate,
        "seed": seed,
        "completed_step": completed_step,
        "stage": stage,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "rng_state": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        "history": list(history or []),
        "input_hashes": dict(input_hashes or {}),
        "provenance": dict(provenance or {}),
        "formal_promotion_authorized": False,
        "permitted_next_actions": [],
        "metadata": dict(metadata or {}),
    }
    atomic_torch_save(checkpoint, payload)
    atomic_write_json(checkpoint.with_suffix(".json"), {"schema_version": 3, "contract": OBJECTIVE_E_CONTRACT, "config_hash": spec.config_hash, "candidate": candidate, "seed": seed, "completed_step": completed_step, "stage": stage, "checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint), "input_hashes": dict(input_hashes or {}), "provenance": dict(provenance or {}), "metadata": dict(metadata or {}), "formal_promotion_authorized": False, "permitted_next_actions": []})
    return checkpoint


def _load_lifecycle_checkpoint(path: str, *, spec: ElectronicO0Spec, inputs: dict[str, str], provenance: dict[str, str | None], action: str) -> dict[str, Any]:
    checkpoint = Path(path)
    manifest_path = checkpoint.with_suffix(".json")
    if not checkpoint.is_absolute() or not checkpoint.is_file() or not manifest_path.is_file():
        raise ValueError("resume checkpoint must be an absolute path with an adjacent manifest")
    manifest = _read_mapping(manifest_path, description="resume checkpoint manifest")
    if manifest.get("schema_version") != 3 or manifest.get("checkpoint_sha256") != file_sha256(checkpoint) or manifest.get("contract") != OBJECTIVE_E_CONTRACT or manifest.get("config_hash") != spec.config_hash or manifest.get("formal_promotion_authorized") is not False or manifest.get("permitted_next_actions") != [] or manifest.get("input_hashes") != inputs or manifest.get("provenance") != provenance:
        raise ValueError("resume checkpoint manifest is invalid")
    payload = safe_torch_load(checkpoint)
    if not isinstance(payload, dict) or payload.get("schema_version") != 3 or payload.get("contract") != OBJECTIVE_E_CONTRACT or payload.get("config_hash") != spec.config_hash or payload.get("input_hashes") != inputs or payload.get("provenance") != provenance or payload.get("action") not in {None, action}:
        raise ValueError("resume checkpoint payload context is invalid")
    if not isinstance(payload.get("history"), list) or not isinstance(payload.get("rng_state"), dict):
        raise ValueError("resume checkpoint state is incomplete")
    return payload


def _render(model: nn.Module, candidate: str, code: torch.Tensor, labels: torch.Tensor, *, exposure_scale: float = 0.4913053810596466) -> torch.Tensor:
    value = model(code, labels)
    if candidate == "O0-G":
        value = (value.square() / exposure_scale).clamp(0.0, 1.0)
        return F.adaptive_avg_pool2d(value, (28, 28))
    return value


def _balanced_batch(labels: torch.Tensor, *, seed: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    per_class = batch_size // 10
    for label in range(10):
        choices = torch.nonzero(labels == label, as_tuple=False).flatten().cpu()
        rows.append(choices.index_select(0, torch.randint(choices.numel(), (per_class,), generator=generator)))
    indices = torch.cat(rows).to(device)
    return indices, labels.index_select(0, indices).to(device)


def _base_loss(model: nn.Module, candidate: str, evaluator: nn.Module, images: torch.Tensor, labels: torch.Tensor, codes: torch.Tensor, *, companion_images: torch.Tensor | None = None, companion_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    view = _render(model, candidate, codes, labels)
    features, logits = evaluator(view)
    with torch.no_grad():
        real_features, _ = evaluator(images)
    base = F.cross_entropy(logits, labels) + F.mse_loss(F.normalize(features, dim=1), F.normalize(real_features, dim=1))
    values = {"base": base, "classification": F.cross_entropy(logits, labels), "camera_fraction": camera_code_fraction(view, labels)}
    if companion_images is not None and companion_labels is not None:
        with torch.no_grad():
            companion_features, _ = evaluator(companion_images)
        aux, aux_values = objective_e_auxiliaries(view, features, view, real_features, images, labels, weights={"camera": 0.0, "diversity": 0.0, "coverage": 0.0}, companion_features=companion_features, companion_view=companion_images, companion_labels=companion_labels)
        values.update(aux_values)
    return base, values, view


def _parameter_groups(model: nn.Module, candidate: str) -> dict[str, tuple[nn.Parameter, ...]]:
    prefixes = (
        ("label_embedding", "label_embedding"),
        ("projection", "projection"),
        ("residual_blocks", "blocks"),
        ("transitions", "transitions"),
        ("output", "output"),
        ("carrier", "carrier_logits"),
    ) if candidate == "O0-I" else (
        ("carrier", "class_carrier_logits"),
        ("global_modes", "global_modes"),
        ("code_projection", "code_projection"),
        ("class_heads", "class_heads"),
    )
    named = dict(model.named_parameters())
    groups = {name: tuple(value for key, value in named.items() if key == prefix or key.startswith(prefix + ".")) for name, prefix in prefixes}
    if any(not values for values in groups.values()):
        raise ValueError("registered O0 parameter group is missing")
    return groups


def _load_evaluators(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, nn.Module]:
    primary, _ = load_frozen_evaluator(args.evaluator_checkpoint, device=device)
    secondary_payload = safe_torch_load(args.secondary_evaluator_checkpoint)
    if not isinstance(secondary_payload, dict) or secondary_payload.get("architecture_kind") != SmoothMNISTEvaluator.architecture_kind:
        raise ValueError("electronic O0 secondary evaluator architecture mismatch")
    secondary = SmoothMNISTEvaluator().to(device)
    secondary.load_state_dict(secondary_payload["state_dict"], strict=True)
    return primary.freeze(), secondary.freeze()


def _preflight_model(model: nn.Module, spec: ElectronicO0Spec, device: torch.device, primary: nn.Module, secondary: nn.Module) -> dict[str, Any]:
    labels = torch.arange(10, device=device).repeat_interleave(5)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
    before = canonical_sha256({name: value.detach().cpu().tolist() for name, value in model.state_dict().items()})
    parameter_groups = _parameter_groups(model, spec.candidate)
    group_before = {name: canonical_sha256({str(index): value.detach().cpu().tolist() for index, value in enumerate(parameters)}) for name, parameters in parameter_groups.items()}
    evaluator_before = canonical_sha256({"primary": {name: value.detach().cpu().tolist() for name, value in primary.state_dict().items()}, "secondary": {name: value.detach().cpu().tolist() for name, value in secondary.state_dict().items()}})
    started = time.perf_counter()
    loss = torch.zeros((), device=device)
    finite = True
    for update_step in range(3):
        optimizer.zero_grad(set_to_none=True)
        step_codes = torch.randn(labels.numel(), spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(1234 + update_step))
        raw = _render(model, spec.candidate, step_codes, labels, exposure_scale=spec.fixed_exposure_scale)
        primary_features, primary_logits = primary(raw)
        secondary_features, secondary_logits = secondary(raw)
        loss = F.cross_entropy(primary_logits, labels) + F.cross_entropy(secondary_logits, labels) + F.mse_loss(primary_features, primary_features.detach())
        loss.backward()
        finite = finite and bool(torch.isfinite(loss)) and all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        optimizer.step()
    elapsed_seconds = time.perf_counter() - started
    after = canonical_sha256({name: value.detach().cpu().tolist() for name, value in model.state_dict().items()})
    group_after = {name: canonical_sha256({str(index): value.detach().cpu().tolist() for index, value in enumerate(parameters)}) for name, parameters in parameter_groups.items()}
    group_updates = {name: group_before[name] != group_after[name] for name in parameter_groups}
    group_gradients = {
        name: all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and bool(parameter.grad.detach().abs().sum() > 0) for parameter in parameters)
        for name, parameters in parameter_groups.items()
    }
    evaluator_after = canonical_sha256({"primary": {name: value.detach().cpu().tolist() for name, value in primary.state_dict().items()}, "secondary": {name: value.detach().cpu().tolist() for name, value in secondary.state_dict().items()}})
    def state_hash(module: nn.Module) -> str:
        return canonical_sha256({name: value.detach().cpu().tolist() for name, value in module.state_dict().items()})

    def optimizer_hash(value: dict[str, Any]) -> str:
        def normalize(item: Any) -> Any:
            if isinstance(item, torch.Tensor):
                return item.detach().cpu().tolist()
            if isinstance(item, dict):
                return {str(key): normalize(child) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [normalize(child) for child in item]
            return item
        return canonical_sha256(normalize(value))

    def trajectory(seed: int, split_at: int | None = None) -> dict[str, Any]:
        torch.manual_seed(seed)
        local_model = _load_candidate(spec, device)
        local_optimizer = torch.optim.Adam(local_model.parameters(), lr=2.0e-4)
        losses: list[float] = []
        steps = range(10) if split_at is None else range(split_at)
        for step in steps:
            batch_labels = torch.arange(10, device=device).repeat_interleave(5)
            codes = torch.randn(batch_labels.numel(), spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(100000 + step))
            local_optimizer.zero_grad(set_to_none=True)
            local_view = _render(local_model, spec.candidate, codes, batch_labels, exposure_scale=spec.fixed_exposure_scale)
            _, local_logits = primary(local_view)
            local_loss = F.cross_entropy(local_logits, batch_labels)
            local_loss.backward()
            local_optimizer.step()
            losses.append(float(local_loss.detach()))
        if split_at is not None:
            saved_model = copy.deepcopy(local_model.state_dict())
            saved_optimizer = copy.deepcopy(local_optimizer.state_dict())
            resumed_model = _load_candidate(spec, device)
            resumed_model.load_state_dict(saved_model, strict=True)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=2.0e-4)
            resumed_optimizer.load_state_dict(saved_optimizer)
            for step in range(split_at, 10):
                codes = torch.randn(50, spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(100000 + step))
                resumed_optimizer.zero_grad(set_to_none=True)
                local_view = _render(resumed_model, spec.candidate, codes, torch.arange(10, device=device).repeat_interleave(5), exposure_scale=spec.fixed_exposure_scale)
                _, local_logits = primary(local_view)
                local_loss = F.cross_entropy(local_logits, torch.arange(10, device=device).repeat_interleave(5))
                local_loss.backward()
                resumed_optimizer.step()
                losses.append(float(local_loss.detach()))
            return {"losses": losses, "state_hash": state_hash(resumed_model), "optimizer_hash": optimizer_hash(resumed_optimizer.state_dict())}
        return {"losses": losses, "state_hash": state_hash(local_model), "optimizer_hash": optimizer_hash(local_optimizer.state_dict())}

    first = trajectory(1234)
    second = trajectory(1234)
    split = trajectory(1234, split_at=5)
    deterministic = {
        "independent_match": first == second,
        "resume_final_match": first["state_hash"] == split["state_hash"] and first["optimizer_hash"] == split["optimizer_hash"] and first["losses"] == split["losses"],
        "losses": first["losses"],
    }
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {"finite": finite, "all_parameters_updated": before != after, "parameter_group_gradients": group_gradients, "parameter_group_updates": group_updates, "all_parameter_groups_updated": all(group_updates.values()), "optimizer_created": True, "primary_evaluator_gradients_clean": all(parameter.grad is None for parameter in primary.parameters()), "secondary_evaluator_gradients_clean": all(parameter.grad is None for parameter in secondary.parameters()), "evaluator_state_unchanged": evaluator_before == evaluator_after, "loss": float(loss.detach()), "candidate": spec.candidate, "preflight_steps": 3, "elapsed_seconds": elapsed_seconds, "peak_cuda_memory_bytes": peak_memory, "budget_passed": elapsed_seconds <= 600.0 and (peak_memory == 0 or peak_memory <= int(torch.cuda.get_device_properties(device).total_memory * 0.95)), "deterministic_preflight": deterministic}


def _control_metrics(spec: ElectronicO0Spec, model: nn.Module, primary: nn.Module, secondary: nn.Module, device: torch.device) -> dict[str, Any]:
    full_images, full_labels = load_local_mnist(spec.dataset_root, train=True)
    reference, reference_labels = validation_reference(full_images, full_labels, device=device)
    images, labels = full_images[:1000].to(device), full_labels[:1000].to(device)
    with torch.no_grad():
        _, primary_logits = primary(images)
        _, secondary_logits = secondary(images)
    primary_pred, secondary_pred = primary_logits.argmax(1), secondary_logits.argmax(1)
    shuffled = (labels + 1) % 10
    template = torch.cat([images[labels == label][:1].repeat(17, 1, 1, 1) for label in range(10)])
    template_labels = torch.arange(10, device=device).repeat_interleave(17)
    with torch.no_grad():
        template_features, template_logits = primary(template)
        reference_features, reference_logits = primary(reference)
    reference_panel_indices = torch.cat([torch.nonzero(reference_labels == label, as_tuple=False).flatten()[:17] for label in range(10)])
    reference_panel = reference.index_select(0, reference_panel_indices)
    reference_panel_features = reference_features.index_select(0, reference_panel_indices)
    template_ratios = relative_diversity_ratios(template_features, reference_panel_features, template_labels)
    template_coverage = float(coverage_ratio_raw(template_features, template, reference_panel_features, reference_panel, template_labels).detach())
    template_unique = int(torch.unique(template.flatten(1), dim=0).shape[0])
    real = {"primary_accuracy": float((primary_pred == labels).float().mean()), "secondary_accuracy": float((secondary_pred == labels).float().mean()), "agreement": float((primary_pred == secondary_pred).float().mean())}
    shuffle = {"primary_accuracy": float((primary_pred == shuffled).float().mean()), "secondary_accuracy": float((secondary_pred == shuffled).float().mean())}
    template_negative = {"minimum_diversity_ratio": float(min(template_ratios)), "coverage_ratio": template_coverage, "failed_expected_gate": float(min(template_ratios)) < spec.diversity_lower or template_coverage > 1.5}
    return {"real": real, "label_shuffle": shuffle, "template_unique": template_unique, "template_negative": template_negative, "passed": min(real["primary_accuracy"], real["secondary_accuracy"]) >= 0.90 and max(shuffle.values()) < 0.95 and template_negative["failed_expected_gate"]}


def _prepare_runtime(args: argparse.Namespace, spec: ElectronicO0Spec) -> tuple[dict[str, str | None], dict[str, str]]:
    provenance = _require_clean(spec, args.config)
    inputs = _validate_acceptance_chain(args, spec, provenance)
    _configure_deterministic_runtime(int(args.seed))
    return provenance, inputs


def validate(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance = _require_clean(spec, args.config)
    inputs = {"o0_config": file_sha256(args.config)}
    root = _root(spec, "validate")
    metrics = {"candidate": spec.candidate, "parameter_budget": spec.parameter_budget, "provenance": provenance}
    path = _write_evidence(root, spec, args, "validate", "passed", metrics, inputs=inputs, provenance=provenance, permitted=["preflight"])
    print(path)
    return 0


def preflight(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    _validate_o0_evidence(
        args.validate_evidence,
        spec=spec,
        provenance=provenance,
        action="validate",
        state="passed",
        permitted=["preflight"],
        expected_inputs={"o0_config": inputs["o0_config"]},
    )
    if spec.candidate == "O0-G" and not args.o0_i_closeout:
        raise ValueError("O0-G requires a passed O0-I closeout path")
    if spec.candidate == "O0-G":
        closeout = _read_mapping(args.o0_i_closeout, description="O0-I closeout")
        closeout_hash = closeout.pop("content_hash", None)
        if not isinstance(closeout_hash, str) or closeout_hash != canonical_sha256(closeout):
            raise ValueError("O0-I closeout content hash mismatch")
        if closeout.get("status") != "closed" or closeout.get("candidate") != "O0-I" or closeout.get("formal_promotion_authorized") is not False or closeout.get("permitted_next_actions") != []:
            raise ValueError("O0-G requires a closed O0-I lifecycle")
    device = torch.device(args.device)
    model = _load_candidate(spec, device)
    primary, secondary = _load_evaluators(args, device)
    metrics = _preflight_model(model, spec, device, primary, secondary)
    metrics["controls"] = _control_metrics(spec, model, primary, secondary, device)
    metrics["provenance"] = provenance
    deterministic = metrics.get("deterministic_preflight", {})
    state = "preflight_passed" if all(metrics[k] for k in ("finite", "all_parameters_updated", "all_parameter_groups_updated", "primary_evaluator_gradients_clean", "secondary_evaluator_gradients_clean", "evaluator_state_unchanged", "budget_passed")) and all(metrics["parameter_group_gradients"].values()) and deterministic.get("independent_match") is True and deterministic.get("resume_final_match") is True and metrics["controls"]["passed"] else "preflight_failed"
    root = _root(spec, "preflight")
    path = _write_evidence(
        root,
        spec,
        args,
        "preflight",
        state,
        metrics,
        inputs=inputs,
        provenance=provenance,
        extra={"validate_evidence": file_sha256(args.validate_evidence)},
        permitted=["calibrate"] if state == "preflight_passed" else [],
    )
    print(path)
    return 0 if state == "preflight_passed" else 1


def _calibrate(args: argparse.Namespace, spec: ElectronicO0Spec) -> tuple[dict[str, float], dict[str, Any]]:
    device = torch.device(args.device)
    images, labels = load_local_mnist(spec.dataset_root, train=True)
    images, labels = images.to(device), labels.to(device)
    model = _load_candidate(spec, device)
    evaluator = _load_evaluators(args, device)[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
    warmup_schedule: list[dict[str, Any]] = []
    for step in range(spec.warmup_steps):
        indices, batch_labels, companion_indices, companion_labels, schedule = _training_batch(
            labels,
            run_seed=spec.split_seed,
            global_step=step,
            batch_size=spec.logical_batch_size,
            code_dim=spec.code_dim,
            device=device,
        )
        warmup_schedule.append(schedule)
        codes = torch.randn(spec.logical_batch_size, spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(spec.split_seed * 100000 + step))
        optimizer.zero_grad(set_to_none=True)
        base, _, _ = _base_loss(model, spec.candidate, evaluator, images.index_select(0, indices), batch_labels, codes)
        base.backward()
        optimizer.step()
    warmup_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    warmup_optimizer = optimizer.state_dict()
    parameters = tuple(model.parameters())
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    parameter_groups = _parameter_groups(model, spec.candidate)
    auxiliary_names = ("camera", "diversity", "coverage")
    raw_norms = {name: [] for name in auxiliary_names}
    gradient_snapshots: list[dict[str, tuple[torch.Tensor | None, ...]]] = []
    base_norms: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    violation_presence = {name: [] for name in auxiliary_names}
    calibration_schedule: list[dict[str, Any]] = []
    for step in spec.calibration_steps:
        indices, batch_labels, companion_indices, companion_labels, schedule = _training_batch(
            labels,
            run_seed=spec.split_seed,
            global_step=step,
            batch_size=spec.logical_batch_size,
            code_dim=spec.code_dim,
            device=device,
        )
        calibration_schedule.append(schedule)
        codes = torch.randn(spec.logical_batch_size, spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(spec.split_seed * 100000 + step))
        real = images.index_select(0, indices)
        companion = images.index_select(0, companion_indices)
        model.zero_grad(set_to_none=True)
        base, values, view = _base_loss(model, spec.candidate, evaluator, real, batch_labels, codes, companion_images=companion, companion_labels=companion_labels)
        generated_features, _ = evaluator(view)
        real_features, _ = evaluator(real)
        diversity_control, diversity_telemetry = objective_e2_diversity_penalty(
            generated_features, real_features.detach(), batch_labels
        )
        controls: dict[str, torch.Tensor] = {
            "camera": camera_safety_penalty(view, batch_labels),
            "diversity": diversity_control,
            "coverage": coverage_penalty(values["coverage_ratio_raw"]),
        }
        base_grad = torch.autograd.grad(base, parameters, retain_graph=True, allow_unused=True)
        base_norm = torch.sqrt(sum((g.detach().square().sum() for g in base_grad if g is not None), torch.zeros((), device=device)))
        if not torch.isfinite(base_norm) or float(base_norm) <= 1.0e-12:
            raise ObjectiveScalingNotQualified("base_gradient", float(base_norm.detach()))
        base_norms.append(base_norm)
        base_group_norms = {}
        for group_name, group_parameters in parameter_groups.items():
            group_grad = tuple(base_grad[parameter_indices[id(parameter)]] for parameter in group_parameters)
            base_group_norms[group_name] = float(torch.sqrt(sum((g.detach().square().sum() for g in group_grad if g is not None), torch.zeros((), device=device))).detach())
        batch_record: dict[str, Any] = {
            "step": step,
            "base_gradient_norm": float(base_norm.detach()),
            "base_group_gradient_norms": base_group_norms,
            "diversity_telemetry": diversity_telemetry,
            "coverage_ratio_raw": float(values["coverage_ratio_raw"].detach()),
            "auxiliaries": {},
        }
        batch_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for name, control in controls.items():
            if not torch.isfinite(control):
                raise ObjectiveScalingNotQualified(name, float("nan"))
            violated = bool(float(control.detach()) > 0.0)
            violation_presence[name].append(violated)
            aux_grad = torch.autograd.grad(control, parameters, retain_graph=True, allow_unused=True)
            aux_norm = torch.sqrt(sum((g.detach().square().sum() for g in aux_grad if g is not None), torch.zeros((), device=device)))
            if violated and (not torch.isfinite(aux_norm) or float(aux_norm) <= 1.0e-12):
                raise ObjectiveScalingNotQualified(name, float(aux_norm.detach()))
            if violated:
                raw_norms[name].append(aux_norm)
                batch_gradients[name] = tuple(None if gradient is None else gradient.detach().clone() for gradient in aux_grad)
            group_norms = {}
            for group_name, group_parameters in parameter_groups.items():
                group_grad = tuple(aux_grad[parameter_indices[id(parameter)]] for parameter in group_parameters)
                group_norms[group_name] = float(torch.sqrt(sum((g.detach().square().sum() for g in group_grad if g is not None), torch.zeros((), device=device))).detach())
            batch_record["auxiliaries"][name] = {"raw_penalty": float(control.detach()), "violated": violated, "raw_gradient_norm": float(aux_norm.detach()), "group_gradient_norms": group_norms}
        records.append(batch_record)
        gradient_snapshots.append(batch_gradients)
    statuses = {name: ("active" if any(violation_presence[name]) else "inactive_satisfied") for name in auxiliary_names}
    base_rms = torch.stack(base_norms).square().mean().sqrt()
    normalizers: dict[str, float] = {}
    coefficients = {name: 0.0 for name in auxiliary_names}
    weights: dict[str, float] = {name: 0.0 for name in auxiliary_names}
    safety: dict[str, Any] = {}
    for name in auxiliary_names:
        if statuses[name] == "inactive_satisfied":
            normalizers[name] = 0.0
            safety[name] = {"weighted_base_ratios": [], "combined_ratio": 0.0}
            continue
        aux_rms = torch.stack(raw_norms[name]).square().mean().sqrt()
        normalizer = base_rms / aux_rms
        if not torch.isfinite(normalizer) or float(normalizer) <= 0.0:
            raise ObjectiveScalingNotQualified(name, float(normalizer.detach()))
        normalizers[name] = float(normalizer.detach())
        coefficients[name] = 0.25
        weights[name] = coefficients[name] * normalizers[name]
        ratios = [coefficients[name] * normalizers[name] * row["auxiliaries"][name]["raw_gradient_norm"] / row["base_gradient_norm"] for row in records if row["auxiliaries"][name]["violated"]]
        if not ratios or not all(math.isfinite(value) and 0.0 < value <= 0.50 for value in ratios):
            raise ObjectiveScalingNotQualified(name, float(max(ratios, default=0.0)))
        group_ratios = {}
        for group_name in parameter_groups:
            values = []
            for row in records:
                if row["auxiliaries"][name]["violated"]:
                    base_group = row["base_group_gradient_norms"][group_name]
                    aux_group = row["auxiliaries"][name]["group_gradient_norms"][group_name]
                    if base_group <= 1.0e-12 or not math.isfinite(base_group) or aux_group <= 1.0e-12 or not math.isfinite(aux_group):
                        raise ObjectiveScalingNotQualified(f"{name}:{group_name}", base_group)
                    ratio = coefficients[name] * normalizers[name] * aux_group / base_group
                    values.append(ratio)
            group_ratios[group_name] = values
            if not values or not all(math.isfinite(value) and 0.0 < value <= 1.0 for value in values):
                raise ObjectiveScalingNotQualified(f"{name}:{group_name}", float(max(values, default=0.0)))
        safety[name] = {"weighted_base_ratios": ratios, "group_ratios": group_ratios}
    combined_ratios: list[float] = []
    combined_group_ratios: dict[str, list[float]] = {group_name: [] for group_name in parameter_groups}
    for row_index, row in enumerate(records):
        active_names = [name for name in auxiliary_names if row["auxiliaries"][name]["violated"]]
        if not active_names:
            continue
        combined_gradients: list[torch.Tensor] = []
        for parameter_index, parameter in enumerate(parameters):
            terms = [weights[name] * gradient_snapshots[row_index][name][parameter_index] for name in active_names if gradient_snapshots[row_index][name][parameter_index] is not None]
            combined_gradients.append(sum(terms, torch.zeros_like(parameter.detach())))
        combined_norm = torch.sqrt(sum((gradient.square().sum() for gradient in combined_gradients), torch.zeros((), device=device)))
        combined_ratio = float((combined_norm / row["base_gradient_norm"]).detach())
        if not math.isfinite(combined_ratio) or not 0.0 < combined_ratio <= 1.0:
            raise ObjectiveScalingNotQualified("combined", combined_ratio)
        combined_ratios.append(combined_ratio)
        for group_name, group_parameters in parameter_groups.items():
            group_norm = torch.sqrt(sum((combined_gradients[parameter_indices[id(parameter)]].square().sum() for parameter in group_parameters), torch.zeros((), device=device)))
            base_group = row["base_group_gradient_norms"][group_name]
            group_ratio = float((group_norm / base_group).detach()) if base_group > 1.0e-12 else float("nan")
            if not math.isfinite(group_ratio) or not 0.0 < group_ratio <= 1.0:
                raise ObjectiveScalingNotQualified(f"combined:{group_name}", group_ratio)
            combined_group_ratios[group_name].append(group_ratio)
    for name in auxiliary_names:
        safety.setdefault(name, {})["combined_ratio"] = max(combined_ratios, default=0.0)
        safety[name]["combined_group_ratios"] = combined_group_ratios
    run = {"model_state": warmup_state, "optimizer_state": warmup_optimizer, "warmup_steps": spec.warmup_steps, "calibration_steps": list(spec.calibration_steps), "weights": weights, "normalizers": normalizers, "coefficients": coefficients, "statuses": statuses, "records": records, "parameter_groups": list(parameter_groups), "safety": safety, "warmup_schedule": warmup_schedule, "calibration_schedule": calibration_schedule, "warmup_model_state_hash": canonical_sha256(warmup_state), "warmup_optimizer_state_hash": canonical_sha256(warmup_optimizer), "warmup_schedule_hash": canonical_sha256(warmup_schedule), "calibration_schedule_hash": canonical_sha256(calibration_schedule), "contract": OBJECTIVE_E_CONTRACT}
    return weights, run


def calibrate(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    _validate_o0_evidence(
        args.preflight_evidence,
        spec=spec,
        provenance=provenance,
        action="preflight",
        state="preflight_passed",
        permitted=["calibrate"],
        expected_inputs=inputs,
    )
    try:
        weights, payload = _calibrate(args, spec)
    except ObjectiveScalingNotQualified as error:
        root = _root(spec, "calibrate")
        path = _write_evidence(
            root,
            spec,
            args,
            "calibrate",
            "calibration_failed",
            {"classification": "objective_scaling_not_qualified", "auxiliary": error.name, "unsafe_weight": error.weight},
            inputs=inputs,
            provenance=provenance,
            reason="registered active-set weight safety bound failed",
            extra={"preflight_evidence": file_sha256(args.preflight_evidence)},
            permitted=[],
        )
        print(path)
        return 1
    root = _root(spec, "calibrate")
    checkpoint = root / "objective_o0_warmup_step_400.pt"
    atomic_torch_save(checkpoint, payload)
    artifact = {"schema_version": 3, "contract": OBJECTIVE_E_CONTRACT, "candidate": spec.candidate, "weights": weights, "normalizers": payload["normalizers"], "coefficients": payload["coefficients"], "statuses": payload["statuses"], "records": payload["records"], "parameter_groups": payload["parameter_groups"], "safety": payload["safety"], "warmup_schedule": payload["warmup_schedule"], "calibration_schedule": payload["calibration_schedule"], "warmup_model_state_hash": payload["warmup_model_state_hash"], "warmup_optimizer_state_hash": payload["warmup_optimizer_state_hash"], "warmup_schedule_hash": payload["warmup_schedule_hash"], "calibration_schedule_hash": payload["calibration_schedule_hash"], "warmup_steps": 400, "calibration_steps": [400, 401, 402, 403], "warmup_checkpoint": str(checkpoint.resolve()), "warmup_checkpoint_sha256": file_sha256(checkpoint), "formal_promotion_authorized": False, "permitted_next_actions": ["screen"]}
    atomic_write_json(root / "objective_o0_calibration.json", artifact)
    evidence = _write_evidence(
        root,
        spec,
        args,
        "calibrate",
        "calibration_passed",
        artifact,
        inputs=inputs,
        provenance=provenance,
        extra={
            "preflight_evidence": file_sha256(args.preflight_evidence),
            "warmup_checkpoint": file_sha256(checkpoint),
            "calibration_artifact": file_sha256(root / "objective_o0_calibration.json"),
        },
        permitted=["screen"],
    )
    print(evidence)
    return 0


def _measurement(model: nn.Module, spec: ElectronicO0Spec, primary: nn.Module, secondary: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    rows = []
    with torch.no_grad():
        reference_images, reference_labels = validation_reference(images.cpu(), labels.cpu(), device=device)
        reference_indices = torch.cat([torch.nonzero(reference_labels == label, as_tuple=False).flatten()[:17] for label in range(10)])
        reference_panel = reference_images.index_select(0, reference_indices)
        for _, (codes, panel_labels) in fixed_panels(device=device, code_dim=spec.code_dim).items():
            view = _render(model, spec.candidate, codes, panel_labels, exposure_scale=spec.fixed_exposure_scale)
            pf, pl = primary(view)
            sf, sl = secondary(view)
            rf, _ = primary(reference_panel)
            sf_ref, _ = secondary(reference_panel)
            primary_pred, secondary_pred = pl.argmax(1), sl.argmax(1)
            rows.append({"camera_code_fraction": float(camera_code_fraction(view, panel_labels)), "primary_accuracy": float((primary_pred == panel_labels).float().mean()), "secondary_accuracy": float((secondary_pred == panel_labels).float().mean()), "primary_min_recall": min(float((primary_pred[panel_labels == label] == label).float().mean()) for label in range(10)), "secondary_min_recall": min(float((secondary_pred[panel_labels == label] == label).float().mean()) for label in range(10)), "prediction_agreement": float((primary_pred == secondary_pred).float().mean()), "minimum_diversity_ratio": min(relative_diversity_ratios(pf, rf, panel_labels)), "secondary_minimum_diversity_ratio": min(relative_diversity_ratios(sf, sf_ref, panel_labels)), "saturation": float((view >= 1.0).float().mean()), "coverage_ratio": float(coverage_ratio_raw(pf, view, rf, reference_panel, panel_labels).item()), "secondary_coverage_ratio": float(coverage_ratio_raw(sf, view, sf_ref, reference_panel, panel_labels).item())})
    names = ("camera_code_fraction", "primary_accuracy", "secondary_accuracy", "primary_min_recall", "secondary_min_recall", "prediction_agreement", "minimum_diversity_ratio", "secondary_minimum_diversity_ratio", "saturation", "coverage_ratio", "secondary_coverage_ratio")
    return {name: (max(row[name] for row in rows) if name == "saturation" else min(row[name] for row in rows)) for name in names}


def _official_measurement(model: nn.Module, spec: ElectronicO0Spec, primary: nn.Module, secondary: nn.Module, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Evaluate the registered 10,000-image test split, not a diagnostic panel."""
    if images.shape[0] != 10_000 or labels.shape[0] != 10_000:
        raise ValueError("official O0 measurement requires exactly 10000 test samples")
    labels_device = labels.to(device)
    codes = torch.randn((10_000, spec.code_dim), device=device, generator=torch.Generator(device=device).manual_seed(spec.split_seed * 100000 + spec.final_steps))
    views, primary_features, primary_logits, secondary_features, secondary_logits = [], [], [], [], []
    reference_features, secondary_reference_features = [], []
    with torch.no_grad():
        for start in range(0, 10_000, 250):
            view = _render(model, spec.candidate, codes[start : start + 250], labels_device[start : start + 250], exposure_scale=spec.fixed_exposure_scale).clamp(0, 1)
            pf, pl = primary(view)
            sf, sl = secondary(view)
            views.append(view.cpu())
            primary_features.append(pf.cpu())
            primary_logits.append(pl.cpu())
            secondary_features.append(sf.cpu())
            secondary_logits.append(sl.cpu())
            rf, _ = primary(images[start : start + 250].to(device))
            srf, _ = secondary(images[start : start + 250].to(device))
            reference_features.append(rf.cpu())
            secondary_reference_features.append(srf.cpu())
    view = torch.cat(views)
    pf, pl = torch.cat(primary_features), torch.cat(primary_logits)
    sf, sl = torch.cat(secondary_features), torch.cat(secondary_logits)
    reference_features = torch.cat(reference_features)
    secondary_reference_features = torch.cat(secondary_reference_features)
    labels = labels.cpu()
    primary_pred, secondary_pred = pl.argmax(1), sl.argmax(1)
    primary_ratios = relative_diversity_ratios(pf, reference_features, labels)
    secondary_ratios = relative_diversity_ratios(sf, secondary_reference_features, labels)
    return {
        "official_test_count": int(images.shape[0]),
        "official_test_content_hash_bound": True,
        "camera_code_fraction": float(camera_code_fraction(view, labels)),
        "primary_accuracy": float((primary_pred == labels).float().mean()),
        "secondary_accuracy": float((secondary_pred == labels).float().mean()),
        "primary_min_recall": min(float((primary_pred[labels == label] == label).float().mean()) for label in range(10)),
        "secondary_min_recall": min(float((secondary_pred[labels == label] == label).float().mean()) for label in range(10)),
        "prediction_agreement": float((primary_pred == secondary_pred).float().mean()),
        "minimum_diversity_ratio": min(primary_ratios),
        "secondary_minimum_diversity_ratio": min(secondary_ratios),
        "primary_diversity_ratios": primary_ratios,
        "secondary_diversity_ratios": secondary_ratios,
        "saturation": float((view >= 1.0).float().mean()),
        "coverage_ratio": float(coverage_ratio_raw(pf, view, reference_features, images, labels).detach()),
        "secondary_coverage_ratio": float(coverage_ratio_raw(sf, view, secondary_reference_features, images, labels).detach()),
    }


def _run_e_steps(model: nn.Module, optimizer: torch.optim.Optimizer, spec: ElectronicO0Spec, primary: nn.Module, images: torch.Tensor, labels: torch.Tensor, *, start_step: int, end_step: int, weights: dict[str, float], active: bool, device: torch.device, seed: int = 1234, checkpoint_callback: Any | None = None) -> None:
    for step in range(start_step, end_step):
        indices, batch_labels, companion_indices, companion_labels, schedule = _training_batch(
            labels,
            run_seed=seed,
            global_step=step,
            batch_size=spec.logical_batch_size,
            code_dim=spec.code_dim,
            device=device,
        )
        codes = torch.randn(spec.logical_batch_size, spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(seed * 100000 + step))
        real = images.index_select(0, indices).to(device)
        companion = images.index_select(0, companion_indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        base, _, view = _base_loss(model, spec.candidate, primary, real, batch_labels, codes, companion_images=companion, companion_labels=companion_labels)
        if active:
            generated_features, _ = primary(view)
            real_features, _ = primary(real)
            companion_features, _ = primary(companion)
            aux, _ = objective_e_auxiliaries(view, generated_features, view, real_features.detach(), real, batch_labels, weights=weights, companion_features=companion_features.detach(), companion_view=companion, companion_labels=companion_labels)
            loss = base + aux
        else:
            loss = base
        loss.backward()
        optimizer.step()
        if checkpoint_callback is not None:
            checkpoint_callback(step + 1, schedule)


def screen(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    calibration, calibration_evidence = _validate_calibration(args, spec, provenance, inputs)
    device = torch.device(args.device)
    images, labels = load_local_mnist(spec.dataset_root, train=True)
    primary, secondary = _load_evaluators(args, device)
    reports = {}
    resume_payload = None
    root: Path | None = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_absolute() or not resume_path.is_file() or not resume_path.with_suffix(".json").is_file():
            raise ValueError("resume checkpoint must be an absolute path with an adjacent manifest")
        manifest = json.loads(resume_path.with_suffix(".json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 3 or manifest.get("checkpoint_sha256") != file_sha256(resume_path) or manifest.get("contract") != OBJECTIVE_E_CONTRACT or manifest.get("config_hash") != spec.config_hash or manifest.get("formal_promotion_authorized") is not False or manifest.get("permitted_next_actions") != []:
            raise ValueError("resume checkpoint manifest is invalid")
        resume_payload = safe_torch_load(resume_path)
        if not isinstance(resume_payload, dict) or resume_payload.get("schema_version") != 3 or resume_payload.get("config_hash") != spec.config_hash or resume_payload.get("candidate") not in {"A", "E"} or resume_payload.get("seed") != 1234 or resume_payload.get("contract") != OBJECTIVE_E_CONTRACT or resume_payload.get("input_hashes") != inputs or resume_payload.get("provenance") != provenance:
            raise ValueError("resume checkpoint payload context is invalid")
        if int(resume_payload.get("completed_step", -1)) not in {100, 200, 300, 400, 500, 600, 700, 800, 900, 984}:
            raise ValueError("resume checkpoint completed step is not registered")
        root = resume_path.parent
        if not isinstance(resume_payload.get("history"), list) or not isinstance(resume_payload.get("rng_state"), dict):
            raise ValueError("resume checkpoint history is missing")
        history_steps = [row.get("step") for row in resume_payload["history"] if isinstance(row, dict)]
        completed = int(resume_payload.get("completed_step", -1))
        if history_steps != list(range(1, completed + 1)):
            raise ValueError("resume checkpoint history is not contiguous")
        if resume_payload.get("candidate") == "E":
            baseline_path = root / "objective_o0_A_report.json"
            if not baseline_path.is_file():
                raise ValueError("E resume requires the completed A report")
            reports["A"] = json.loads(baseline_path.read_text(encoding="utf-8"))
    if root is None:
        root = _root(spec, "screen")
    model = _load_candidate(spec, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
    for candidate_kind in ("A", "E"):
        if resume_payload is not None and resume_payload.get("candidate") == "E" and candidate_kind == "A":
            continue
        model = _load_candidate(spec, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
        warmup_path = Path(calibration.get("warmup_checkpoint", ""))
        if not warmup_path.is_absolute() or not warmup_path.is_file() or file_sha256(warmup_path) != calibration.get("warmup_checkpoint_sha256"):
            raise ValueError("Objective E calibration warmup checkpoint is missing or tampered")
        warmup = safe_torch_load(warmup_path, map_location=device)
        if not isinstance(warmup, dict) or warmup.get("contract") != OBJECTIVE_E_CONTRACT or warmup.get("warmup_steps") != 400:
            raise ValueError("Objective E warmup checkpoint contract mismatch")
        # Rebuild the shared warmup from immutable initialization. Loading the
        # calibration state directly would conceal a schedule or runtime fork.
        warmup_history: list[dict[str, Any]] = []
        def save_warmup(completed_step: int, schedule: dict[str, Any]) -> None:
            if candidate_kind != "A":
                return
            warmup_history.append({"step": completed_step, "active_objective": "A", "effective_weights": {name: 0.0 for name in ("camera", "diversity", "coverage")}, "schedule": schedule})
            if completed_step not in {100, 200, 300, 400}:
                return
            _save_o0_checkpoint(root, spec, candidate_kind, 1234, completed_step, model, optimizer, history=warmup_history, input_hashes=inputs, provenance=provenance, stage="warmup")
        _run_e_steps(
            model,
            optimizer,
            spec,
            primary,
            images,
            labels,
            start_step=0,
            end_step=spec.warmup_steps,
            weights={"camera": 0.0, "diversity": 0.0, "coverage": 0.0},
            active=False,
            device=device,
            seed=spec.split_seed,
            checkpoint_callback=save_warmup,
        )
        rebuilt_state_hash = canonical_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
        rebuilt_optimizer_hash = canonical_sha256(optimizer.state_dict())
        rebuilt_schedule = []
        for warmup_step in range(spec.warmup_steps):
            _, _, _, _, schedule = _training_batch(
                labels,
                run_seed=spec.split_seed,
                global_step=warmup_step,
                batch_size=spec.logical_batch_size,
                code_dim=spec.code_dim,
                device=device,
            )
            rebuilt_schedule.append(schedule)
        if rebuilt_state_hash != calibration.get("warmup_model_state_hash") or rebuilt_optimizer_hash != calibration.get("warmup_optimizer_state_hash") or canonical_sha256(rebuilt_schedule) != calibration.get("warmup_schedule_hash"):
            raise ValueError("Objective E screen warmup does not match calibration lineage")
        history = list(resume_payload.get("history", [])) if resume_payload is not None and resume_payload.get("candidate") == candidate_kind else [
            {"step": warmup_step + 1, "active_objective": "A", "effective_weights": {name: 0.0 for name in ("camera", "diversity", "coverage")}, "schedule": rebuilt_schedule[warmup_step]}
            for warmup_step in range(spec.warmup_steps)
        ]
        start_step = 400
        if resume_payload is not None and resume_payload.get("candidate") == candidate_kind:
            model.load_state_dict(resume_payload["model_state"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            torch.set_rng_state(resume_payload["rng_state"]["cpu"])
            if torch.cuda.is_available() and resume_payload["rng_state"].get("cuda"):
                torch.cuda.set_rng_state_all(resume_payload["rng_state"]["cuda"])
            start_step = int(resume_payload["completed_step"])
        elif start_step == 400:
            _save_o0_checkpoint(root, spec, candidate_kind, 1234, 400, model, optimizer, history=history, input_hashes=inputs, provenance=provenance, stage="warmup")
        for step in range(start_step, spec.screen_steps):
            indices, batch_labels, companion_indices, companion_labels, schedule = _training_batch(
                labels,
                run_seed=1234,
                global_step=step,
                batch_size=spec.logical_batch_size,
                code_dim=spec.code_dim,
                device=device,
            )
            codes = torch.randn(spec.logical_batch_size, spec.code_dim, device=device, generator=torch.Generator(device=device).manual_seed(1234 * 100000 + step))
            real, companion = images.index_select(0, indices).to(device), images.index_select(0, companion_indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            base, values, view = _base_loss(model, spec.candidate, primary, real, batch_labels, codes, companion_images=companion, companion_labels=companion_labels)
            weights = {"camera": 0.0, "diversity": 0.0, "coverage": 0.0} if candidate_kind == "A" else calibration["weights"]
            if candidate_kind == "E":
                aux, _ = objective_e_auxiliaries(view, primary(view)[0], view, primary(real)[0].detach(), real, batch_labels, weights=weights, companion_features=primary(companion)[0].detach(), companion_view=companion, companion_labels=companion_labels)
                loss = base + aux
            else:
                loss = base
            loss.backward()
            optimizer.step()
            history.append({"step": step + 1, "active_objective": "A" if step < 400 else candidate_kind, "effective_weights": {name: (0.0 if step < 400 else float(weights[name])) for name in weights}, "loss": float(loss.detach()), "schedule": schedule})
            if step + 1 in {400, 500, 600, 700, 800, 900, 984}:
                _save_o0_checkpoint(root, spec, candidate_kind, 1234, step + 1, model, optimizer, history=history, input_hashes=inputs, provenance=provenance, stage=candidate_kind)
        reports[candidate_kind] = _measurement(model, spec, primary, secondary, images, labels, device)
        if candidate_kind == "A":
            atomic_write_json(root / "objective_o0_A_report.json", reports[candidate_kind])
            atomic_write_json(root / "objective_o0_A_complete.json", {"candidate": "A", "completed_step": 984, "report_sha256": file_sha256(root / "objective_o0_A_report.json"), "formal_promotion_authorized": False, "permitted_next_actions": []})
    baseline, candidate = reports["A"], reports["E"]
    gates = {"camera": candidate["camera_code_fraction"] >= baseline["camera_code_fraction"] + .02, "diversity": candidate["minimum_diversity_ratio"] >= baseline["minimum_diversity_ratio"] + .05 and candidate["secondary_minimum_diversity_ratio"] >= baseline["secondary_minimum_diversity_ratio"] + .05, "coverage": candidate["coverage_ratio"] <= baseline["coverage_ratio"] - .05 and candidate["secondary_coverage_ratio"] <= baseline["secondary_coverage_ratio"] - .05, "accuracy": candidate["primary_accuracy"] >= baseline["primary_accuracy"] - .03 and candidate["secondary_accuracy"] >= baseline["secondary_accuracy"] - .03, "recall": candidate["primary_min_recall"] >= baseline["primary_min_recall"] - .05 and candidate["secondary_min_recall"] >= baseline["secondary_min_recall"] - .05, "saturation": candidate["saturation"] <= .08}
    state = "screen_passed" if all(gates.values()) else "screen_failed"
    path = _write_evidence(
        root,
        spec,
        args,
        "screen",
        state,
        {"baseline": baseline, "candidate": candidate, "gates": gates, "formal_promotion_authorized": False},
        inputs=inputs,
        provenance=provenance,
        extra={"calibration_evidence": file_sha256(args.calibration_evidence), "calibration_artifact": file_sha256(args.calibration_artifact)},
        permitted=["confirm"] if state == "screen_passed" else [],
    )
    print(path)
    return 0 if state == "screen_passed" else 1


def confirm(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    calibration, _ = _validate_calibration(args, spec, provenance, inputs)
    _validate_o0_evidence(args.screen_evidence, spec=spec, provenance=provenance, action="screen", state="screen_passed", permitted=["confirm"], expected_inputs=inputs)
    device = torch.device(args.device)
    images, labels = load_local_mnist(spec.dataset_root, train=True)
    primary, secondary = _load_evaluators(args, device)
    seed_rows = []
    resume_root = None
    resume_index = 0
    if args.resume_checkpoint:
        resume_payload = _load_lifecycle_checkpoint(args.resume_checkpoint, spec=spec, inputs=inputs, provenance=provenance, action="confirm")
        if resume_payload.get("stage") != "confirm" or resume_payload.get("metadata", {}).get("next_seed_index") is None:
            raise ValueError("confirmation resume checkpoint stage is invalid")
        seed_rows = list(resume_payload.get("metadata", {}).get("seed_rows", []))
        resume_index = int(resume_payload["metadata"]["next_seed_index"])
        resume_root = Path(args.resume_checkpoint).parent
    for seed_index, seed in enumerate(spec.confirm_seeds):
        if seed_index < resume_index:
            continue
        torch.manual_seed(seed)
        model_a = _load_candidate(spec, device)
        optimizer_a = torch.optim.Adam(model_a.parameters(), lr=2.0e-4)
        _run_e_steps(model_a, optimizer_a, spec, primary, images, labels, start_step=0, end_step=400, weights={"camera": 0.0, "diversity": 0.0, "coverage": 0.0}, active=False, device=device, seed=seed)
        model_e = _load_candidate(spec, device)
        model_e.load_state_dict(model_a.state_dict(), strict=True)
        optimizer_e = torch.optim.Adam(model_e.parameters(), lr=2.0e-4)
        optimizer_e.load_state_dict(optimizer_a.state_dict())
        _run_e_steps(model_a, optimizer_a, spec, primary, images, labels, start_step=400, end_step=spec.screen_steps, weights={"camera": 0.0, "diversity": 0.0, "coverage": 0.0}, active=False, device=device, seed=seed)
        _run_e_steps(model_e, optimizer_e, spec, primary, images, labels, start_step=400, end_step=spec.screen_steps, weights=calibration["weights"], active=True, device=device, seed=seed)
        a, e = _measurement(model_a, spec, primary, secondary, images, labels, device), _measurement(model_e, spec, primary, secondary, images, labels, device)
        passed = e["camera_code_fraction"] >= a["camera_code_fraction"] + .02 and e["minimum_diversity_ratio"] >= a["minimum_diversity_ratio"] + .05 and e["secondary_minimum_diversity_ratio"] >= a["secondary_minimum_diversity_ratio"] + .05 and e["coverage_ratio"] <= a["coverage_ratio"] - .05 and e["secondary_coverage_ratio"] <= a["secondary_coverage_ratio"] - .05 and e["primary_accuracy"] >= a["primary_accuracy"] - .03 and e["secondary_accuracy"] >= a["secondary_accuracy"] - .03 and e["primary_min_recall"] >= a["primary_min_recall"] - .05 and e["secondary_min_recall"] >= a["secondary_min_recall"] - .05 and e["saturation"] <= .08
        seed_rows.append({"seed": seed, "passed": passed, "A": a, "E": e})
        root = resume_root or _root(spec, "confirm")
        _save_o0_checkpoint(root, spec, "E", seed, spec.screen_steps, model_e, optimizer_e, history=[], input_hashes=inputs, provenance=provenance, stage="confirm", metadata={"next_seed_index": seed_index + 1, "seed_rows": seed_rows})
        resume_root = root
    passed = sum(row["passed"] for row in seed_rows) >= 2
    root = _root(spec, "confirm")
    payload = {"seeds": seed_rows, "screen_evidence": file_sha256(args.screen_evidence), "formal_promotion_authorized": False, "permitted_next_actions": ["locked-final"] if passed else []}
    path = _write_evidence(root, spec, args, "confirm", "confirm_passed" if passed else "confirm_failed", payload, inputs=inputs, provenance=provenance, extra={"screen_evidence": file_sha256(args.screen_evidence), "calibration_evidence": file_sha256(args.calibration_evidence), "calibration_artifact": file_sha256(args.calibration_artifact)}, permitted=["locked-final"] if passed else [])
    print(path)
    return 0 if passed else 1


def locked_final(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    calibration, _ = _validate_calibration(args, spec, provenance, inputs)
    _validate_o0_evidence(args.confirm_evidence, spec=spec, provenance=provenance, action="confirm", state="confirm_passed", permitted=["locked-final"], expected_inputs=inputs)
    device = torch.device(args.device)
    train_images, train_labels = load_local_mnist(spec.dataset_root, train=True)
    test_images, test_labels = load_local_mnist(spec.dataset_root, train=False)
    if test_images.shape[0] != 10_000 or test_labels.shape[0] != 10_000 or tuple(test_images.shape[1:]) != (1, 28, 28):
        raise ValueError("locked-final requires the official 10000-sample test split")
    bundle = _read_mapping(args.measurement_bundle, description="measurement bundle")
    test_manifest = Path(args.measurement_bundle).parent / str(bundle["test_manifest_filename"])
    test_payload = _read_mapping(test_manifest, description="official test manifest")
    if test_payload.get("manifest_hash") != bundle.get("test_content_manifest_hash") or test_payload.get("count") != 10_000 or test_payload.get("images_sha256") != tensor_sha256(test_images) or test_payload.get("labels_sha256") != tensor_sha256(test_labels):
        raise ValueError("official test content does not match the qualified measurement manifest")
    primary, secondary = _load_evaluators(args, device)
    rows = []
    for seed in (1234,):
        torch.manual_seed(seed)
        model = _load_candidate(spec, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-4)
        zero = {"camera": 0.0, "diversity": 0.0, "coverage": 0.0}
        final_root = None
        if args.resume_checkpoint:
            resume_payload = _load_lifecycle_checkpoint(args.resume_checkpoint, spec=spec, inputs=inputs, provenance=provenance, action="locked-final")
            if resume_payload.get("stage") != "locked-final" or resume_payload.get("completed_step") != spec.final_steps:
                raise ValueError("locked-final resume checkpoint must be the completed training boundary")
            model.load_state_dict(resume_payload["model_state"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            final_root = Path(args.resume_checkpoint).parent
        else:
            final_root = _root(spec, "locked-final")
            def save_final_boundary(completed_step: int, schedule: dict[str, Any]) -> None:
                if completed_step == 400 or completed_step % 246 == 0 or completed_step == spec.final_steps:
                    _save_o0_checkpoint(final_root, spec, "E", seed, completed_step, model, optimizer, history=[], input_hashes=inputs, provenance=provenance, stage="locked-final", metadata={"measurement_ready": completed_step == spec.final_steps, "schedule": schedule})
            _run_e_steps(model, optimizer, spec, primary, train_images, train_labels, start_step=0, end_step=400, weights=zero, active=False, device=device, seed=seed, checkpoint_callback=save_final_boundary)
            _run_e_steps(model, optimizer, spec, primary, train_images, train_labels, start_step=400, end_step=spec.final_steps, weights=calibration["weights"], active=True, device=device, seed=seed, checkpoint_callback=save_final_boundary)
        measurement = _official_measurement(model, spec, primary, secondary, test_images, test_labels, device)
        gates = {"primary_accuracy": measurement["primary_accuracy"] >= .95, "secondary_accuracy": measurement["secondary_accuracy"] >= .95, "primary_recall": measurement["primary_min_recall"] >= .90, "secondary_recall": measurement["secondary_min_recall"] >= .90, "agreement": measurement["prediction_agreement"] >= .95, "camera": measurement["camera_code_fraction"] >= .25, "diversity": measurement["minimum_diversity_ratio"] >= .70 and measurement["minimum_diversity_ratio"] <= 1.30 and measurement["secondary_minimum_diversity_ratio"] >= .70 and measurement["secondary_minimum_diversity_ratio"] <= 1.30, "coverage": measurement["coverage_ratio"] <= 1.5 and measurement["secondary_coverage_ratio"] <= 1.5, "saturation": measurement["saturation"] <= .08}
        rows.append({"seed": seed, "measurement": measurement, "gates": gates, "passed": all(gates.values())})
    passed = sum(row["passed"] for row in rows) == 1
    root = _root(spec, "locked-final")
    payload = {"rows": rows, "confirm_evidence": file_sha256(args.confirm_evidence), "formal_promotion_authorized": False, "permitted_next_actions": ["closeout"] if passed else []}
    path = _write_evidence(root, spec, args, "locked-final", "locked_final_passed" if passed else "locked_final_failed", payload, inputs=inputs, provenance=provenance, extra={"confirm_evidence": file_sha256(args.confirm_evidence), "calibration_evidence": file_sha256(args.calibration_evidence), "calibration_artifact": file_sha256(args.calibration_artifact)}, permitted=["closeout"] if passed else [])
    print(path)
    return 0 if passed else 1


def closeout(args: argparse.Namespace) -> int:
    spec = load_objective_o0_spec(args.config)
    provenance, inputs = _prepare_runtime(args, spec)
    _validate_o0_evidence(args.source_evidence, spec=spec, provenance=provenance, action="locked-final", state="locked_final_passed", permitted=["closeout"], expected_inputs=inputs)
    root = _root(spec, "closeout")
    summary = {"candidate": spec.candidate, "contract": OBJECTIVE_E_CONTRACT, "status": "closed", "source_evidence": file_sha256(args.source_evidence), "formal_promotion_authorized": False, "permitted_next_actions": []}
    summary["content_hash"] = canonical_sha256(summary)
    path = root / "objective_o0_closeout.json"
    atomic_write_json(path, summary)
    print(path)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    handlers = {"validate": validate, "preflight": preflight, "calibrate": calibrate, "screen": screen, "confirm": confirm, "locked-final": locked_final, "closeout": closeout}
    return handlers[args.research_action](args)
