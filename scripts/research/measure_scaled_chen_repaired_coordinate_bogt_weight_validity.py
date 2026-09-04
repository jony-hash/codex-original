"""Validate the registered Scaled-Chen BOGT weight in the repaired coordinate.

This is a no-update diagnostic.  It reuses the exact BOGT ``_arm_loss`` helper
for the registered three panels at the frozen step-5000 parent, comparing the
immutable v1 coordinate with the hash-bound v2 coordinate.  It never creates
an optimizer, calls a training update, writes a checkpoint, or performs prior
generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import repair_scaled_chen_calibration_unit_aware as r2


TASK_ID = "H-LOMA-GEN-T1-RCWV"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
EXPECTED_HEAD = "47b9aa0896cae88516b3872e6f3355ff12434054"

CONFIG_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_manifest.v1"
PANEL_METRICS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_gradient_panel_metrics.v1"
COMPARISON_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_comparison.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_compact_report.v1"

RC_ALPHA_VALID = "RC_ALPHA_VALID"
RC_ALPHA_TOO_STRONG = "RC_ALPHA_TOO_STRONG"
RC_ALPHA_TOO_WEAK = "RC_ALPHA_TOO_WEAK"
RC_ALPHA_UNDERDETERMINED = "RC_ALPHA_UNDERDETERMINED"
BLOCKED_WEIGHT_PROVENANCE_UNRESOLVED = "BLOCKED_WEIGHT_PROVENANCE_UNRESOLVED"
BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED = "BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED"
INVALID_HARNESS = "invalid_harness"

PORT_COUNT = 100
PANEL_COUNT = 3
BATCH_SIZE = 10
MICROBATCH_SIZE = 1
SEED = 1234
ALPHA = 0.002996375650268832
OLD_GAMMA = 0.02969130038500517
GAMMA = 0.0
TARGET_RATIO = 0.10
MIN_RATIO = 0.05
MAX_RATIO = 0.40
PARENT_STEP = 5000
OLD_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v1"
NEW_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v2"
NEW_CALIBRATION_CONTRACT = "unit_aware_raw_mean_v2"
CHECKPOINT_SCHEMA = "scaled_chen_bogt_horizon_checkpoint.v1"
OLD_MANIFEST_SHA256 = "e3f9c39b9f2fc6278f93f688e6ad7735cfad9a4dfbb9e74ba32d36ac10da33c2"
REPAIRED_MANIFEST_SHA256 = "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9"
CALIBRATION_INDICES_SHA256 = "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543"
SCHEDULE_SHA256 = "73b77c379267fd4535188d96075ec14db9cfc2a53e9bdcb49fc0a0e3c4cd3145"
PANEL_INDICES_SHA256 = (
    "822948f09ebdad5f3fb36da36f45ae1dc0b9d65b6ff79675405aeec763f55861",
    "635817ff273289883881c3447115ff0db05a20aa08448e0e2095e4c420df836f",
    "2340114e34809126e683aca958f4590e195f3911aa42697ef94d6ab4940a0202",
)
IMAGES_SHA256 = "cc86acbde125e1596a9d290165c4345cff2294032e1bb5cf4aeaf8c6467322f0"
LABELS_SHA256 = "5b15b6a6d7cf9f282941280ffeb569aca7640d0d7e78be939a9b1723e4f83549"
OLD_PREFLIGHT_SHA256 = "2570df7cf4fe7a2b8fbca5e8bddce3cba259984e038347c74b5e3855b6106d9b"
OLD_PILOT_REVIEW_SHA256 = "50a4bd770a7287d8993f06d64ed925450b2d09e009ed0cacdadf4d7742ad6095"
OLD_PANEL_ARTIFACT_SHA256 = "b12d616cbb34297aed5fa7b3d941964a4ac02cb2de6c1fce1b6cf1bb5de0f7b1"
OLD_CALIBRATION_SHA256 = "d871a6319e4a4efcc7746378305b3bdc678e93c876b9af9d4e875a7e59a2ed11"
OLD_PARENT_CONFIG_SHA256 = "e3d1b536aa081a8298b21b0b608f232463444a2863e138f941765adf96d8a701"
OLD_FORMAL_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
OLD_REVIEW_SHA256 = "4978686212b164f281cf1cb5c070119929fa111ea34e92a77d4b1d5ae2ce10f6"
OLD_SOURCE_COMMIT = "cefaa19a257b836f22ad04594ec3748c027b0eec"
R2_CONFIG_SHA256 = "b8c1efab6fd49c0e68696bc7a466cf6eaeeb499a842c6efa66fe080254b867a9"
R2_CONFIG_CANONICAL_SHA256 = "c9de9505a6cf64432e4f2c350bfc5dfa188bc9563cdf8f89c274193123045deb"
R2_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
R2_PAYLOAD_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
R2_MODE_BANK_SHA256 = "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
PARENT_V1_CALIBRATION_SHA256 = "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959"
PARENT_CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"

DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_v1.yaml"
)


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching the disk."""
    target = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", target.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = target.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _json_file(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    target = _resolve(path)
    actual = file_sha256(target)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"JSON artifact hash mismatch: {_relative_path(target)}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {_relative_path(target)}")
    return payload


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "parent_bogt",
        "repaired_calibration",
        "checkpoint",
        "dataset",
        "objective",
        "execution",
        "authorization",
        "governance",
        "output",
    }
    if set(payload) != required or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("T1-RCWV config fields or schema are invalid")
    if payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("T1-RCWV task identity is invalid")

    parent = _require_mapping(payload["parent_bogt"], "parent_bogt")
    if parent.get("config_sha256") != OLD_PARENT_CONFIG_SHA256 or parent.get("source_commit") != OLD_SOURCE_COMMIT or parent.get("source_branch") != BRANCH:
        raise ValueError("old parent BOGT identity is invalid")
    if parent.get("preflight_sha256") != OLD_PREFLIGHT_SHA256 or parent.get("pilot_review_sha256") != OLD_PILOT_REVIEW_SHA256 or parent.get("gradient_panels_sha256") != OLD_PANEL_ARTIFACT_SHA256 or parent.get("calibration_sha256") != OLD_CALIBRATION_SHA256:
        raise ValueError("old alpha artifact identity is invalid")
    if parent.get("dataset_manifest_sha256") != OLD_MANIFEST_SHA256 or parent.get("schedule_sha256") != SCHEDULE_SHA256 or tuple(parent.get("panel_indices_sha256", ())) != PANEL_INDICES_SHA256:
        raise ValueError("old alpha panel identity is invalid")
    if parent.get("alpha") != ALPHA or parent.get("gamma") != OLD_GAMMA or parent.get("target_ratio") != TARGET_RATIO:
        raise ValueError("registered alpha or target ratio changed")
    if parent.get("registered_boundary") != [MIN_RATIO, MAX_RATIO]:
        raise ValueError("registered gradient-ratio boundary changed")
    if parent.get("ratio_formula") != "alpha_times_bogt_gradient_norm_divided_by_mse_gradient_norm":
        raise ValueError("registered gradient-ratio formula is invalid")

    repaired = _require_mapping(payload["repaired_calibration"], "repaired_calibration")
    if repaired.get("config_sha256") != R2_CONFIG_SHA256 or repaired.get("config_canonical_sha256") != R2_CONFIG_CANONICAL_SHA256 or repaired.get("manifest_sha256") != R2_MANIFEST_SHA256 or repaired.get("payload_sha256") != R2_PAYLOAD_SHA256:
        raise ValueError("repaired calibration artifact identity is invalid")
    if repaired.get("schema") != NEW_CALIBRATION_SCHEMA or repaired.get("contract_version") != NEW_CALIBRATION_CONTRACT or repaired.get("dataset_manifest_sha256") != REPAIRED_MANIFEST_SHA256 or repaired.get("mode_bank_sha256") != R2_MODE_BANK_SHA256 or repaired.get("parent_v1_sha256") != PARENT_V1_CALIBRATION_SHA256:
        raise ValueError("repaired calibration contract binding is invalid")

    checkpoint = _require_mapping(payload["checkpoint"], "checkpoint")
    if dict(checkpoint) != {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": PARENT_CHECKPOINT_SHA256,
        "schema": CHECKPOINT_SCHEMA,
        "global_step": PARENT_STEP,
    }:
        raise ValueError("step-5000 checkpoint binding is invalid")

    dataset = _require_mapping(payload["dataset"], "dataset")
    if dataset.get("images_sha256") != IMAGES_SHA256 or dataset.get("labels_sha256") != LABELS_SHA256 or dataset.get("calibration_indices_sha256") != CALIBRATION_INDICES_SHA256 or dataset.get("schedule_sha256") != SCHEDULE_SHA256 or dataset.get("old_manifest_sha256") != OLD_MANIFEST_SHA256 or dataset.get("repaired_manifest_sha256") != REPAIRED_MANIFEST_SHA256:
        raise ValueError("dataset content identity is invalid")
    if dataset.get("panel_count") != PANEL_COUNT or dataset.get("schedule_seed") != SEED or dataset.get("logical_batch_size") != BATCH_SIZE or dataset.get("microbatch_size") != MICROBATCH_SIZE or tuple(dataset.get("panel_indices_sha256", ())) != PANEL_INDICES_SHA256:
        raise ValueError("panel or batch contract is invalid")

    objective = _require_mapping(payload["objective"], "objective")
    if objective.get("parent") != "mse_only" or objective.get("candidate") != "mse_kl" or objective.get("alpha") != ALPHA or objective.get("gamma") != GAMMA:
        raise ValueError("objective intervention is invalid")
    if objective.get("latent") != "z_j_equals_abs_c_j_squared_divided_by_s_j_v2" or objective.get("prior") != {"kind": "truncated_normal", "mean": 1.0, "variance": 0.25, "lower": 0.0, "unconditional": True}:
        raise ValueError("objective latent or prior binding is invalid")

    execution = _require_mapping(payload["execution"], "execution")
    if dict(execution) != {
        "device": "cuda",
        "require_cublas_workspace_config": ":4096:8",
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": 536870912,
        "no_update": True,
        "optimizer_step": False,
        "backward_training": False,
        "checkpoint_created": False,
    }:
        raise ValueError("execution/no-update contract is invalid")

    if payload["authorization"] != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "support_alignment_training": False,
        "prior_generation": False,
        "decoder_generation": False,
    }:
        raise ValueError("authorization boundary is invalid")
    if payload["governance"] != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("governance boundary is invalid")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("T1-RCWV config must be a YAML mapping")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _float_equal(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-15)


def effective_ratio(alpha: float, bogt_gradient_norm: float, mse_gradient_norm: float) -> float:
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (alpha, bogt_gradient_norm, mse_gradient_norm)):
        raise ValueError("effective gradient ratio requires finite positive alpha and norms")
    return float(alpha * bogt_gradient_norm / mse_gradient_norm)


def classify_alpha_ratios(ratios: Sequence[float], *, lower: float = MIN_RATIO, upper: float = MAX_RATIO) -> str:
    if len(ratios) != PANEL_COUNT or not all(math.isfinite(float(value)) for value in ratios):
        return INVALID_HARNESS
    below = any(float(value) < lower for value in ratios)
    above = any(float(value) > upper for value in ratios)
    if below and above:
        return RC_ALPHA_UNDERDETERMINED
    if above:
        return RC_ALPHA_TOO_STRONG
    if below:
        return RC_ALPHA_TOO_WEAK
    return RC_ALPHA_VALID


def alpha_gate(ratios: Sequence[float], *, lower: float = MIN_RATIO, upper: float = MAX_RATIO) -> dict[str, Any]:
    classification = classify_alpha_ratios(ratios, lower=lower, upper=upper)
    values = [float(value) for value in ratios]
    return {
        "classification": classification,
        "rule": "all_three_panels_inclusive_registered_interval",
        "lower": lower,
        "upper": upper,
        "panel_ratios": values,
        "all_inclusive": classification == RC_ALPHA_VALID,
        "median": float(torch.tensor(values, dtype=torch.float64).median()) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def gradient_norm_from_gradients(gradients: Sequence[torch.Tensor | None]) -> float:
    values = [gradient for gradient in gradients if gradient is not None]
    if not values or not all(torch.isfinite(value).all() for value in values):
        raise ValueError("gradient vectors must be finite and non-empty")
    squared = sum(value.detach().square().sum() for value in values)
    return float(squared.sqrt().detach().cpu())


def state_digest(module: torch.nn.Module) -> str:
    return canonical_sha256(
        {
            name: tensor_sha256(value.detach().cpu())
            for name, value in module.state_dict().items()
        }
    )


def validate_gradient_parameter_subset(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    named = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    if tuple(name for name, _ in named) != bogt.EXPECTED_TRAINABLE:
        raise ValueError("gradient parameter subset does not match registered phase-only subset")
    if any(tuple(parameter.shape) != (2, 1, 100, 100) for _, parameter in named):
        raise ValueError("registered phase parameter shape is invalid")
    return named


def optimizer_state_binding(payload: Mapping[str, Any], parameter_shapes: Sequence[tuple[int, ...]]) -> dict[str, Any]:
    optimizer = payload.get("optimizer_state")
    if not isinstance(optimizer, Mapping):
        return {"status": "UNRESOLVED", "reason": "optimizer_state_missing"}
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, Mapping):
        return {"status": "UNRESOLVED", "reason": "optimizer_state_structure_invalid"}
    group = groups[0]
    if not isinstance(group, Mapping) or group.get("params") != [0, 1] or group.get("lr") != 0.001 or group.get("betas") != (0.9, 0.999) or group.get("eps") != 1.0e-8 or group.get("weight_decay") != 0:
        return {"status": "UNRESOLVED", "reason": "optimizer_group_does_not_bind_parent"}
    if set(states) != {0, 1} or len(parameter_shapes) != 2:
        return {"status": "UNRESOLVED", "reason": "optimizer_parameter_ids_do_not_bind_parent"}
    state_shapes: dict[str, list[int]] = {}
    for index, shape in enumerate(parameter_shapes):
        record = states[index]
        if not isinstance(record, Mapping) or not torch.is_tensor(record.get("exp_avg")) or not torch.is_tensor(record.get("exp_avg_sq")):
            return {"status": "UNRESOLVED", "reason": f"optimizer_state_{index}_missing_moments"}
        if tuple(record["exp_avg"].shape) != shape or tuple(record["exp_avg_sq"].shape) != shape:
            return {"status": "UNRESOLVED", "reason": f"optimizer_state_{index}_shape_mismatch"}
        if not torch.isfinite(record["exp_avg"]).all() or not torch.isfinite(record["exp_avg_sq"]).all():
            return {"status": "UNRESOLVED", "reason": f"optimizer_state_{index}_nonfinite"}
        state_shapes[str(index)] = list(shape)
    return {
        "status": "AVAILABLE_AND_BOUND",
        "parameter_ids": [0, 1],
        "parameter_shapes": state_shapes,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "state_digest": canonical_sha256(optimizer),
    }


def _load_old_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _require_mapping(config["parent_bogt"], "parent_bogt")
    parent_config_path = _resolve(parent["config"])
    if file_sha256(parent_config_path) != OLD_PARENT_CONFIG_SHA256:
        raise ValueError("old BOGT config hash is stale")
    spec = bogt._load_spec(parent_config_path)
    bogt._assert_contract(spec)
    review_path = _resolve(parent["review_contract"])
    if file_sha256(review_path) != OLD_REVIEW_SHA256:
        raise ValueError("old BOGT review hash is stale")
    bogt._load_review(review_path, spec)
    formal, formal_path = bogt._load_formal(spec)
    if file_sha256(formal_path) != OLD_FORMAL_CONFIG_SHA256:
        raise ValueError("old formal config hash is stale")

    preflight_path = _resolve(parent["preflight"])
    preflight = _json_file(preflight_path, expected_sha256=OLD_PREFLIGHT_SHA256)
    evidence = _require_mapping(preflight.get("evidence"), "old preflight evidence")
    if preflight.get("schema") != "lightgen_optical_manifold_smf_bogt_preflight.v1" or preflight.get("hypothesis") != "H-LOMA-CBOGT-S0" or evidence.get("status") != "passed" or preflight.get("training_authorized") is not False or preflight.get("formal_promotion") is not False:
        raise ValueError("old preflight is not a passed diagnostic record")
    preflight_contract = _require_mapping(preflight.get("contract"), "old preflight contract")
    if preflight_contract.get("alpha") != ALPHA or preflight_contract.get("gamma") != OLD_GAMMA or preflight_contract.get("dataset_manifest", {}).get("manifest_hash") != OLD_MANIFEST_SHA256:
        raise ValueError("old preflight weight or dataset binding is stale")
    if preflight_contract.get("dataset_manifest", {}).get("panel_indices_sha256") != list(PANEL_INDICES_SHA256) or preflight_contract.get("dataset_manifest", {}).get("schedule_sha256") != SCHEDULE_SHA256:
        raise ValueError("old preflight panel binding is stale")

    pilot_path = _resolve(parent["pilot_review"])
    pilot = _json_file(pilot_path, expected_sha256=OLD_PILOT_REVIEW_SHA256)
    approved = _require_mapping(pilot.get("approved_contract"), "old pilot approved contract")
    if pilot.get("schema") != "lightgen_optical_manifold_smf_bogt_pilot_review.v1" or pilot.get("hypothesis") != "H-LOMA-CBOGT-S0" or pilot.get("reviewer_role") != "project_owner" or pilot.get("decision") != "approve_bounded_scaled_chen_bogt_pilot" or pilot.get("bounded_diagnostic_training_authorized") is not True or pilot.get("training_authorized") is not False or pilot.get("formal_promotion") is not False:
        raise ValueError("old pilot review is not an approved diagnostic contract")
    expected_approved = {
        "arms": ["mse_only", "mse_op", "mse_kl", "mse_kl_op"],
        "steps_per_arm": 100,
        "logical_batch_size": 10,
        "microbatch_size": 1,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "zero_phase_seed": SEED,
        "posthoc_phase_quantization": [0.0, 0.82, 1.63, 2.45, 3.26, 4.08, 4.89, 5.71],
        "post_pass_next_action": "manual_review_scaled_chen_bogt_1000_step_three_seed_continuation_contract",
    }
    if dict(approved) != expected_approved:
        raise ValueError("old pilot approved contract changed")
    bindings = _require_mapping(pilot.get("source_bindings"), "old pilot source bindings")
    expected_bindings = {
        "source_commit": OLD_SOURCE_COMMIT,
        "source_branch": BRANCH,
        "source_config_sha256": OLD_PARENT_CONFIG_SHA256,
        "source_formal_config_sha256": OLD_FORMAL_CONFIG_SHA256,
        "source_lifecycle_review_sha256": OLD_REVIEW_SHA256,
        "source_validation_sha256": "ff946fbbb13c691283aa927ccc570f14f1aa3be76b9b31f673f90f39e208ccc0",
        "source_preflight_sha256": OLD_PREFLIGHT_SHA256,
        "source_dataset_manifest_hash": OLD_MANIFEST_SHA256,
        "source_schedule_sha256": SCHEDULE_SHA256,
        "source_calibration_sha256": OLD_CALIBRATION_SHA256,
        "source_gradient_panels_sha256": OLD_PANEL_ARTIFACT_SHA256,
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("old alpha provenance source bindings changed")
    measured = _require_mapping(pilot.get("measured_preflight"), "old measured preflight")
    if measured.get("alpha") != ALPHA or measured.get("gamma") != OLD_GAMMA or measured.get("gradient_panels") != preflight_contract.get("panel_norms"):
        raise ValueError("old alpha measured provenance changed")

    panel_path = _resolve(parent["gradient_panels"])
    panel_artifact = safe_torch_load(panel_path, expected_sha256=OLD_PANEL_ARTIFACT_SHA256, map_location="cpu")
    if not isinstance(panel_artifact, Mapping) or panel_artifact.get("schema") != "scaled_chen_bogt_gradient_panels.v1" or panel_artifact.get("dataset_manifest_hash") != OLD_MANIFEST_SHA256:
        raise ValueError("old gradient-panel artifact identity is invalid")
    panels = panel_artifact.get("panels")
    if not isinstance(panels, list) or len(panels) != PANEL_COUNT:
        raise ValueError("old gradient-panel count is invalid")

    registered: list[dict[str, Any]] = []
    for index, (preflight_panel, panel_record, expected_indices_sha) in enumerate(zip(preflight_contract["panel_norms"], panels, PANEL_INDICES_SHA256, strict=True)):
        panel_mapping = _require_mapping(panel_record, f"old panel {index}")
        gradient_norms = _require_mapping(panel_mapping.get("gradient_norms"), f"old panel {index} gradient norms")
        if panel_mapping.get("panel") != index or panel_mapping.get("indices_sha256") != expected_indices_sha:
            raise ValueError(f"old panel {index} identity is invalid")
        if gradient_norms.get("mse") != preflight_panel["mse"] or gradient_norms.get("kl") != preflight_panel["kl"] or gradient_norms.get("op") != preflight_panel["op"]:
            raise ValueError(f"old panel {index} gradient provenance is invalid")
        mse = float(preflight_panel["mse"])
        kl = float(preflight_panel["kl"])
        op = float(preflight_panel["op"])
        registered.append({
            "panel": index,
            "indices_sha256": expected_indices_sha,
            "raw_mse_grad_norm": mse,
            "raw_bogt_grad_norm": kl,
            "raw_op_grad_norm": op,
            "effective_bogt_ratio": effective_ratio(ALPHA, kl, mse),
            "effective_op_ratio": effective_ratio(OLD_GAMMA, op, mse),
        })
    if not all(MIN_RATIO <= row["effective_bogt_ratio"] <= MAX_RATIO and MIN_RATIO <= row["effective_op_ratio"] <= MAX_RATIO for row in registered):
        raise ValueError("old registered fixed-weight ratio Gate is not reproducible")

    calibration_path = _resolve(parent["calibration"])
    old_payload = safe_torch_load(calibration_path, expected_sha256=OLD_CALIBRATION_SHA256, map_location="cpu")
    if not isinstance(old_payload, Mapping) or old_payload.get("schema") != OLD_CALIBRATION_SCHEMA:
        raise ValueError("old calibration schema is invalid")
    old_calibration = bogt.Calibration(
        port_scale=old_payload["port_scale"],
        plane_reference=old_payload["plane_reference"],
        sample_count=int(old_payload["sample_count"]),
        plane_lower=old_payload.get("plane_lower"),
        plane_upper=old_payload.get("plane_upper"),
    )
    old_calibration.validate()
    return {
        "spec": spec,
        "formal": formal,
        "formal_path": formal_path,
        "registered": registered,
        "old_calibration": old_calibration,
        "preflight_path": preflight_path,
        "pilot_review_path": pilot_path,
        "panel_path": panel_path,
        "calibration_path": calibration_path,
    }


def _load_repaired_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    repaired = _require_mapping(config["repaired_calibration"], "repaired_calibration")
    config_path, config_payload, config_sha, canonical_sha = r2._load_config(repaired["config"])
    if config_sha != R2_CONFIG_SHA256 or canonical_sha != R2_CONFIG_CANONICAL_SHA256:
        raise ValueError("R2 config identity is stale")
    manifest_path = _resolve(repaired["manifest"])
    manifest = _json_file(manifest_path, expected_sha256=R2_MANIFEST_SHA256)
    if manifest.get("schema") != "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1" or manifest.get("task_id") != "H-LOMA-GEN-P1-R2-UACR" or manifest.get("status") != "completed" or manifest.get("gate") != "R2_CALIBRATION_REPAIR_PASS":
        raise ValueError("R2 manifest status is invalid")
    if manifest.get("cuda_replay") is not False or manifest.get("downstream_replay") is not False or manifest.get("optical_data_changed") is not False:
        raise ValueError("R2 no-replay boundary is invalid")
    if manifest.get("governance") != {
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "generative_manifold_supported": False,
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "training_authorized": False,
    }:
        raise ValueError("R2 governance binding is invalid")
    payload_path = _resolve(repaired["payload"])
    calibration = r2.load_registered_unit_aware_calibration(
        payload_path,
        expected_sha256=R2_PAYLOAD_SHA256,
        expected_dataset_manifest_hash=REPAIRED_MANIFEST_SHA256,
        expected_calibration_indices_sha256=CALIBRATION_INDICES_SHA256,
        expected_mode_bank_sha256=R2_MODE_BANK_SHA256,
        expected_parent_v1_sha256=PARENT_V1_CALIBRATION_SHA256,
    )
    calibration.validate()
    return {
        "config_path": config_path,
        "config": config_payload,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "payload_path": payload_path,
        "calibration": calibration,
    }


def _load_checkpoint(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    checkpoint_config = _require_mapping(config["checkpoint"], "checkpoint")
    path = _resolve(checkpoint_config["path"])
    payload = safe_torch_load(path, expected_sha256=PARENT_CHECKPOINT_SHA256, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA or int(payload.get("global_step", -1)) != PARENT_STEP or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("parent checkpoint schema, step, or model state is invalid")
    return dict(payload), path


def _validate_dataset(data: bogt.ResearchData, config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = data.manifest
    expected = {
        "images_sha256": IMAGES_SHA256,
        "labels_sha256": LABELS_SHA256,
        "calibration_indices_sha256": CALIBRATION_INDICES_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "panel_indices_sha256": list(PANEL_INDICES_SHA256),
        "calibration_count": 1000,
        "schedule_shape": [100, 10],
        "panel_count": PANEL_COUNT,
        "seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "heldout_excluded": True,
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("current dataset content does not match registered panel contract")
    if tuple(tensor_sha256(panel) for panel in data.panels) != PANEL_INDICES_SHA256:
        raise ValueError("current panel tensor identities do not match registration")
    return {
        "current_manifest_sha256": manifest["manifest_hash"],
        "registered_old_manifest_sha256": OLD_MANIFEST_SHA256,
        "registered_repaired_manifest_sha256": REPAIRED_MANIFEST_SHA256,
        "dataset_root_path_difference_only": manifest["manifest_hash"] != OLD_MANIFEST_SHA256,
        "panel_indices": [panel.tolist() for panel in data.panels],
        "panel_indices_sha256": list(PANEL_INDICES_SHA256),
    }


def _measure_gradient(
    model: bogt.ScaledChenBOGTModel,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: bogt.Calibration | bogt.UnitAwareCalibration,
    *,
    arm: str,
    mse_weight: float,
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    parameters = tuple(parameter for name, parameter in model.named_parameters() if name in parameter_names)
    if tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) != tuple(parameter_names):
        raise ValueError("active gradient parameter subset changed")
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("parameter gradients were not clean before no-update measurement")
    total, values = bogt._arm_loss(
        model,
        amplitudes,
        targets,
        calibration,
        arm,
        alpha=1.0,
        gamma=0.0,
        mse_weight=mse_weight,
    )
    if not torch.isfinite(total):
        raise RuntimeError("no-update gradient measurement produced a non-finite loss")
    gradients = torch.autograd.grad(total, parameters, create_graph=False, retain_graph=False, allow_unused=False)
    per_parameter = {
        name: float(gradient.detach().norm().cpu())
        for name, gradient in zip(parameter_names, gradients, strict=True)
    }
    norm = gradient_norm_from_gradients(gradients)
    if any(parameter.grad is not None for parameter in parameters):
        for parameter in parameters:
            parameter.grad = None
        raise RuntimeError("no-update gradient measurement left persistent parameter gradients")
    return {
        "loss": float(total.detach().cpu()),
        "mse_value": float(values["mse"]),
        "bogt_value": float(values["kl"]),
        "gradient_norm": norm,
        "parameter_gradient_norms": per_parameter,
    }


def _measure_coordinate(
    model: bogt.ScaledChenBOGTModel,
    data: bogt.ResearchData,
    calibration: bogt.Calibration | bogt.UnitAwareCalibration,
    *,
    device: torch.device,
    parameter_names: Sequence[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for panel_index, indices in enumerate(data.panels):
        rendered = base.render_intensity_batch(data.images.index_select(0, indices))
        amplitudes = rendered.amplitude.to(device)
        targets = rendered.target.to(device)
        before = state_digest(model)
        mse = _measure_gradient(
            model,
            amplitudes,
            targets,
            calibration,
            arm="mse_only",
            mse_weight=1.0,
            parameter_names=parameter_names,
        )
        bogt_gradient = _measure_gradient(
            model,
            amplitudes,
            targets,
            calibration,
            arm="mse_kl",
            mse_weight=0.0,
            parameter_names=parameter_names,
        )
        after = state_digest(model)
        if before != after:
            raise RuntimeError(f"learned state changed during panel {panel_index} measurement")
        records.append({
            "panel": panel_index,
            "indices_sha256": PANEL_INDICES_SHA256[panel_index],
            "raw_mse_grad_norm": mse["gradient_norm"],
            "raw_bogt_grad_norm": bogt_gradient["gradient_norm"],
            "effective_bogt_ratio": effective_ratio(ALPHA, bogt_gradient["gradient_norm"], mse["gradient_norm"]),
            "mse_gradient": mse,
            "bogt_gradient": bogt_gradient,
            "state_digest_before": before,
            "state_digest_after": after,
        })
        del rendered, amplitudes, targets
    return records


def _ratio_comparison(old: Sequence[Mapping[str, Any]], repaired: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for old_record, repaired_record in zip(old, repaired, strict=True):
        result.append({
            "panel": old_record["panel"],
            "indices_sha256": old_record["indices_sha256"],
            "mse_gradient_norm_repaired_over_old": float(repaired_record["raw_mse_grad_norm"] / old_record["raw_mse_grad_norm"]),
            "bogt_gradient_norm_repaired_over_old": float(repaired_record["raw_bogt_grad_norm"] / old_record["raw_bogt_grad_norm"]),
            "effective_ratio_repaired_over_old": float(repaired_record["effective_bogt_ratio"] / old_record["effective_bogt_ratio"]),
            "old": {
                "raw_mse_grad_norm": old_record["raw_mse_grad_norm"],
                "raw_bogt_grad_norm": old_record["raw_bogt_grad_norm"],
                "effective_bogt_ratio": old_record["effective_bogt_ratio"],
            },
            "repaired": {
                "raw_mse_grad_norm": repaired_record["raw_mse_grad_norm"],
                "raw_bogt_grad_norm": repaired_record["raw_bogt_grad_norm"],
                "effective_bogt_ratio": repaired_record["effective_bogt_ratio"],
            },
        })
    return result


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "weight_validity_manifest.json":
            refs[target.name] = {
                "path": target.name,
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path) -> None:
    expected = {
        "weight_validity_manifest.json",
        "gradient_panel_metrics.json",
        "old_new_gradient_comparison.json",
        "compact_report.json",
    }
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"T1-RCWV output tree is not exact: {sorted(actual)}")
    if any(Path(name).name.startswith(".") or ".tmp" in name for name in actual):
        raise RuntimeError("T1-RCWV output contains hidden or temporary files")


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> dict[str, int]:
    execution = _require_mapping(config["execution"], "execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["require_cublas_workspace_config"]:
        raise RuntimeError("invalid_harness: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("invalid_harness: CUDA is required for the registered gradient measurement")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < int(execution["min_free_cuda_bytes"]):
        raise RuntimeError("invalid_harness: insufficient free CUDA memory")
    disk_free = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root())).free
    if disk_free < int(execution["min_disk_free_bytes"]):
        raise RuntimeError("invalid_harness: insufficient free disk")
    if output_path.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output_path)}")
    return {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes), "disk_free_bytes": int(disk_free)}


def run(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_target, config, config_sha, config_canonical_sha = load_config(config_path)
    state = bogt._git_state()
    if state.get("branch") != BRANCH or state.get("commit") != EXPECTED_HEAD:
        raise ValueError("invalid_harness: branch or HEAD does not match T1-RCWV authority")
    requested_output = _resolve(config["output"]["base"])
    output_path = next_write_once_output(requested_output)
    resources = _resource_preflight(config, output_path)

    old = _load_old_provenance(config)
    repaired = _load_repaired_calibration(config)
    data = bogt._load_research_data(_resolve(config["dataset"]["dataset_root"]))
    dataset_identity = _validate_dataset(data, config)
    checkpoint, checkpoint_path = _load_checkpoint(config)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    formal = old["formal"]
    model = bogt.ScaledChenBOGTModel(formal)
    checkpoint_model_state = checkpoint["model_state"]
    expected_model_keys = set(model.state_dict())
    if set(checkpoint_model_state) != expected_model_keys:
        raise ValueError("parent checkpoint model-state keys do not match the current model")
    model.load_state_dict(checkpoint_model_state, strict=True)
    model = model.to(torch.device(config["execution"]["device"]))
    model.eval()
    named_parameters = validate_gradient_parameter_subset(model)
    parameter_names = tuple(name for name, _ in named_parameters)
    if bogt.mode_bank_sha256(model) != R2_MODE_BANK_SHA256:
        raise ValueError("parent checkpoint mode-bank identity does not match repaired calibration")
    optimizer_binding = optimizer_state_binding(
        checkpoint,
        [tuple(parameter.shape) for _, parameter in named_parameters],
    )
    checkpoint_file_hash_before = file_sha256(checkpoint_path)
    optimizer_state_hash_before = canonical_sha256(checkpoint["optimizer_state"])
    state_before = state_digest(model)
    torch.cuda.reset_peak_memory_stats(model.system.encoder.phase.device)
    started = time.monotonic()
    old_parent = _measure_coordinate(
        model,
        data,
        old["old_calibration"],
        device=model.system.encoder.phase.device,
        parameter_names=parameter_names,
    )
    repaired_parent = _measure_coordinate(
        model,
        data,
        repaired["calibration"],
        device=model.system.encoder.phase.device,
        parameter_names=parameter_names,
    )
    elapsed = time.monotonic() - started
    state_after = state_digest(model)
    checkpoint_file_hash_after = file_sha256(checkpoint_path)
    optimizer_state_hash_after = canonical_sha256(checkpoint["optimizer_state"])
    peak_bytes = int(torch.cuda.max_memory_allocated(model.system.encoder.phase.device))

    repaired_ratios = [float(record["effective_bogt_ratio"]) for record in repaired_parent]
    gate = alpha_gate(repaired_ratios)
    if optimizer_binding["status"] != "AVAILABLE_AND_BOUND" and gate["classification"] == RC_ALPHA_VALID:
        final_classification = BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED
        next_action = "BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED"
    else:
        final_classification = gate["classification"]
        next_action = "IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY" if final_classification == RC_ALPHA_VALID else "PLAN_REPAIRED_COORDINATE_ALIGNMENT_WEIGHT_REGISTRATION"

    comparison = _ratio_comparison(old_parent, repaired_parent)
    metrics_payload = {
        "schema": PANEL_METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "coordinate_change": "old_registered_v1_reference_to_repaired_v2_at_same_step_5000_parent_state",
        "alpha": ALPHA,
        "gamma": GAMMA,
        "panel_count": PANEL_COUNT,
        "batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "parameter_subset": list(parameter_names),
        "dataset": dataset_identity,
        "registered_old_provenance": old["registered"],
        "parent_step_5000": {
            "old_v1": old_parent,
            "repaired_v2": repaired_parent,
            "repaired_effective_ratio_gate": gate,
            "new_over_old": comparison,
        },
    }
    comparison_payload = {
        "schema": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "only_allowed_coordinate_change": "calibration_v1_to_calibration_v2",
        "same_parent_checkpoint": True,
        "same_panel_order_and_indices": True,
        "same_forward_contract": True,
        "registered_old_preflight": old["registered"],
        "step_5000_parent_replay": comparison,
        "summary": {
            "old_parent_effective_ratios": [float(record["effective_bogt_ratio"]) for record in old_parent],
            "repaired_effective_ratios": repaired_ratios,
            "repaired_median_effective_ratio": gate["median"],
            "repaired_min_effective_ratio": gate["minimum"],
            "repaired_max_effective_ratio": gate["maximum"],
        },
    }
    freeze = {
        "state_digest_before": state_before,
        "state_digest_after": state_after,
        "state_unchanged": state_before == state_after,
        "checkpoint_file_hash_before": checkpoint_file_hash_before,
        "checkpoint_file_hash_after": checkpoint_file_hash_after,
        "checkpoint_unchanged": checkpoint_file_hash_before == checkpoint_file_hash_after == PARENT_CHECKPOINT_SHA256,
        "optimizer_state_digest_before": optimizer_state_hash_before,
        "optimizer_state_digest_after": optimizer_state_hash_after,
        "optimizer_state_unchanged": optimizer_state_hash_before == optimizer_state_hash_after,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "optimizer_state_mutated": False,
        "checkpoint_created": False,
        "phase_updated": False,
        "refit": False,
        "training": False,
        "prior_generation": False,
        "decoder_generation": False,
    }
    report_payload = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": final_classification,
        "classification": final_classification,
        "alpha": ALPHA,
        "registered_boundary": [MIN_RATIO, MAX_RATIO],
        "repaired_effective_ratio_summary": {
            "median": gate["median"],
            "minimum": gate["minimum"],
            "maximum": gate["maximum"],
        },
        "optimizer_recovery_state": optimizer_binding,
        "state_isolation": freeze,
        "resources": {**resources, "peak_cuda_bytes": peak_bytes, "wall_seconds": elapsed},
        "next_action": next_action,
        "training_authorized": False,
        "formal_promotion": False,
    }

    output_path.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(output_path / "gradient_panel_metrics.json", metrics_payload, artifact_kind="T1-RCWV gradient panel metrics")
    atomic_write_new_json(output_path / "old_new_gradient_comparison.json", comparison_payload, artifact_kind="T1-RCWV old-new comparison")
    atomic_write_new_json(output_path / "compact_report.json", report_payload, artifact_kind="T1-RCWV compact report")
    manifest_payload = {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": final_classification,
        "classification": final_classification,
        "producer_git": {
            "branch": state["branch"],
            "commit": state["commit"],
            "worktree_clean_at_execution": bool(state["clean"]),
        },
        "source_bindings": {
            "parent_config": _relative_path(old["spec"].path),
            "parent_config_sha256": OLD_PARENT_CONFIG_SHA256,
            "old_preflight": _relative_path(old["preflight_path"]),
            "old_preflight_sha256": OLD_PREFLIGHT_SHA256,
            "old_pilot_review": _relative_path(old["pilot_review_path"]),
            "old_pilot_review_sha256": OLD_PILOT_REVIEW_SHA256,
            "old_gradient_panels": _relative_path(old["panel_path"]),
            "old_gradient_panels_sha256": OLD_PANEL_ARTIFACT_SHA256,
            "old_calibration": _relative_path(old["calibration_path"]),
            "old_calibration_sha256": OLD_CALIBRATION_SHA256,
            "repaired_calibration_config": _relative_path(repaired["config_path"]),
            "repaired_calibration_config_sha256": R2_CONFIG_SHA256,
            "repaired_calibration_manifest": _relative_path(repaired["manifest_path"]),
            "repaired_calibration_manifest_sha256": R2_MANIFEST_SHA256,
            "repaired_calibration_payload": _relative_path(repaired["payload_path"]),
            "repaired_calibration_payload_sha256": R2_PAYLOAD_SHA256,
            "parent_checkpoint": _relative_path(checkpoint_path),
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        },
        "old_alpha_provenance": {
            "alpha": ALPHA,
            "old_gamma_reference_only": OLD_GAMMA,
            "target_ratio": TARGET_RATIO,
            "ratio_formula": "alpha * G_BOGT / G_MSE",
            "gradient_norm": "sqrt(sum(parameter_gradient**2)) over system.encoder.phase and system.decoder.phase",
            "panels": old["registered"],
        },
        "repaired_coordinate": {
            "schema": NEW_CALIBRATION_SCHEMA,
            "contract_version": NEW_CALIBRATION_CONTRACT,
            "payload_sha256": R2_PAYLOAD_SHA256,
            "latent": "z_j=abs(c_j)^2/s_j_v2",
        },
        "checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "global_step": PARENT_STEP,
            "sha256": PARENT_CHECKPOINT_SHA256,
            "optimizer_recovery_state": optimizer_binding,
        },
        "panels": dataset_identity,
        "gradient_panel_metrics": metrics_payload["parent_step_5000"],
        "registered_gate": {
            "lower": MIN_RATIO,
            "upper": MAX_RATIO,
            "rule": "all_three_panels_inclusive_registered_interval",
            "repaired": gate,
        },
        "state_isolation": freeze,
        "artifact_layout": {
            "manifest_self_hash_excluded": True,
            "paths_are_relative_to_output": True,
        },
        "artifacts": {},
        "governance": {
            "mainline_state": "closed_at_current_evidence_boundary",
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion": False,
        },
        "next_action": next_action,
    }
    manifest_payload["artifacts"] = _artifact_refs(output_path)
    atomic_write_new_json(output_path / "weight_validity_manifest.json", manifest_payload, artifact_kind="T1-RCWV manifest")
    _assert_output_tree(output_path)
    storage_bytes = sum(target.stat().st_size for target in output_path.rglob("*") if target.is_file())
    if storage_bytes > int(config["execution"]["max_storage_bytes"]):
        raise RuntimeError("invalid_harness: T1-RCWV output exceeds storage limit")
    return {
        "output_path": str(output_path),
        "manifest": manifest_payload,
        "report": report_payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
