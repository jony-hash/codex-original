"""Run the single registered T2 support-aware alignment canary.

The canary continues the frozen step-5000 parent for exactly 1000 bounded
steps with the registered Energy-Distance objective.  Its observer is
normalized OLS power, not the complete complex semantic OLS state.  No prior
sample is decoded and no semantic-generation authority is produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research import optical_manifold_smf_bogt_horizon as horizon
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import replay_scaled_chen_repaired_coordinate_g1_sa as rcr_b
from scripts.research import run_scaled_chen_support_aware_energy_distance_preflight as pf


TASK_ID = "H-LOMA-GEN-T2-SAAC"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_v1"
PREFLIGHT_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_preflight.v1"
TRAINING_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_training.v1"
CHECKPOINT_SCHEMA = "scaled_chen_support_aware_alignment_canary_checkpoint.v1"
MRA_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_mra.v1"
SUPPORT_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_support.v1"
COMPARISON_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_parent_comparison.v1"
SAFETY_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_safety.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_support_aware_alignment_canary_compact_report.v1"
CONFIG_PATH = Path("configs/research/lightgen_scaled_chen_support_aware_alignment_canary_v1.yaml")
DEFAULT_PREFLIGHT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_support_aware_alignment_canary_v1_preflight_attempt_01"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_support_aware_alignment_canary_v1_attempt_01"
)

PORT_COUNT = 100
TRAIN_SUPPORT_COUNT = 55_000
CONFIRMATION_COUNT = 5_000
LOGICAL_BATCH_SIZE = 10
PARENT_STEP = 5_000
END_STEP = 6_000
CANARY_STEPS = 1_000
SEED = 1_234
TRAINING_BASE_SEED = 1_238
PRIOR_SEED = 1_237
BOOTSTRAP_SEED = 1_236
BOOTSTRAP_REPLICATES = 2_000
CONFIDENCE_LEVEL = 0.95
LEARNING_RATE = 1.0e-3
BETA = 0.0014299609134309638
MEAN_LOWER = 0.90
MEAN_UPPER = 1.10
RANK_RETENTION = 0.90
PARENT_R50_LOWER = 1.8142740175004783
PARENT_R95_LOWER = 1.067902362924707
PARENT_CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
PARENT_STATE_DIGEST = "cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5"
CALIBRATION_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
CALIBRATION_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
MRA_PANEL_SHA256 = "842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724"
RCR_B_DISTANCES_SHA256 = "d288beb4bd57a1db2a9a864dd8db1be5ef22f75f14702fa13f01b01623671466"
PRIOR_SAMPLE_SHA256 = "c11402193afd71d176468760024ce7dec0d4e4ee3d6d206ac7293ca8ee77fd3e"
UPSTREAM_PREFLIGHT_SHA256 = "a9055ec7dfbccad92e683e3569ba4517becbda5d05565966ab22430d806df18a"
UPSTREAM_PREFLIGHT_CONFIG_SHA256 = "f1f5e7fc948bd984722d1167defb79388c1741ca4e1ecb2a5d810095b743b157"
UPSTREAM_PREFLIGHT_CONFIG_CANONICAL_SHA256 = "be1f88e99730076618d2e3edb58d6609b995bb0d5d0af02a1dea5b2ce56bdfcc"
MRA_DATASET_MANIFEST_SHA256 = "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
CONFIRMATION_INDICES_SHA256 = "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e5e2218a"

POWER_OBSERVER_INTERPRETATION = "normalized_power_coordinate_observer"
NEXT_ACTION = "PLAN_S1_OLS_SEMANTIC_REPRESENTATION_AUDIT"
SCAM_AUTHORITY = "OLS_INPUT_CONTRACT_HIERARCHICAL_CONTROLLER_RECOMMENDED"
SCAM_PLAN = "SCAM_PLAN_READY"

T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED = "T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED"
T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED = "T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED"
T2_SUPPORT_AWARE_ALIGNMENT_TRADEOFF = "T2_SUPPORT_AWARE_ALIGNMENT_TRADEOFF"
T2_SUPPORT_AWARE_ALIGNMENT_HARMFUL = "T2_SUPPORT_AWARE_ALIGNMENT_HARMFUL"
T2_TRAINING_INVALID = "T2_TRAINING_INVALID"

GOVERNANCE = {
    "mainline_state": "closed_at_current_evidence_boundary",
    "reconstruction_manifold_supported": False,
    "generative_manifold_supported": False,
    "generation_readiness": "not_ready",
    "training_authorized": False,
    "formal_promotion": False,
}


@dataclass(frozen=True)
class ConfigSpec:
    path: Path
    payload: dict[str, Any]
    file_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class CanaryData:
    audit: Any
    calibration_indices: torch.Tensor
    parent_schedule: torch.Tensor
    schedule: torch.Tensor


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative_path(value: str | Path) -> str:
    target = Path(value).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _json_file(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    observed = file_sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"JSON artifact hash mismatch: {_relative_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {_relative_path(path)}")
    return payload


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} is not registered")


def _check_mapping_values(payload: Mapping[str, Any], expected: Mapping[str, Any], name: str) -> None:
    for key, value in expected.items():
        _require_equal(payload.get(key), value, f"{name}.{key}")


def validate_registered_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema", "task_id", "hypothesis", "source_preflight", "parent", "calibration",
        "dataset", "prior", "objective", "stream", "canary", "evaluation",
        "interpretation", "authorization", "governance", "execution", "output",
    }
    if set(config) != required:
        raise ValueError("T2-SAAC config fields are invalid")
    _check_mapping_values(config, {"schema": CONFIG_SCHEMA, "task_id": TASK_ID, "hypothesis": HYPOTHESIS}, "config")

    source = _mapping(config["source_preflight"], "source_preflight")
    _check_mapping_values(
        source,
        {
            "schema": pf.PREFLIGHT_SCHEMA,
            "task_id": pf.TASK_ID,
            "classification": "SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS",
            "sha256": UPSTREAM_PREFLIGHT_SHA256,
            "config_sha256": UPSTREAM_PREFLIGHT_CONFIG_SHA256,
            "config_canonical_sha256": UPSTREAM_PREFLIGHT_CONFIG_CANONICAL_SHA256,
        },
        "source_preflight",
    )

    objective = _mapping(config["objective"], "objective")
    if objective.get("alpha_bogt") != 0.0:
        raise ValueError("alpha_BOGT must remain zero for T2-SAAC")
    if objective.get("gamma") != 0.0:
        raise ValueError("gamma must remain zero for T2-SAAC")
    _check_mapping_values(
        objective,
        {
            "schema": "lightgen_scaled_chen_support_aware_energy_distance_objective.v1",
            "name": "mse_plus_support_aware_energy_distance",
            "formula": "L_total=L_MSE+beta*L_ED+gamma*L_OP+alpha_BOGT*L_BOGT",
            "estimator": "ordered_unbiased_u_statistic",
            "normalization": "divide_by_sqrt_100",
            "dimension": PORT_COUNT,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "prior_batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": 1,
            "prior_prior_term_included": True,
            "no_epsilon_smoothing": True,
            "allow_finite_negative": True,
            "beta": BETA,
            "alpha_bogt": 0.0,
            "gamma": 0.0,
        },
        "objective",
    )

    parent = _mapping(config["parent"], "parent")
    _check_mapping_values(
        parent,
        {
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "checkpoint_schema": "scaled_chen_bogt_horizon_checkpoint.v1",
            "global_step": PARENT_STEP,
            "learned_state_digest": PARENT_STATE_DIGEST,
            "mra_panel_sha256": MRA_PANEL_SHA256,
            "mra_parent_mse_key": "scaled_chen_step_5000",
        },
        "parent",
    )
    calibration = _mapping(config["calibration"], "calibration")
    _check_mapping_values(
        calibration,
        {
            "schema": "scaled_chen_bogt_calibration.v2",
            "contract_version": "unit_aware_raw_mean_v2",
            "payload_sha256": CALIBRATION_SHA256,
            "manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "calibration_indices_sha256": "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543",
            "dataset_manifest_sha256": "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9",
            "mode_bank_sha256": "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9",
            "zero_phase_reference": "independent_train_only_zero_phase",
            "plane_identity": "pre_ols_plane_power_reference_v1",
        },
        "calibration",
    )

    dataset = _mapping(config["dataset"], "dataset")
    _check_mapping_values(
        dataset,
        {
            "root": "data",
            "input_shape": [1, 28, 28],
            "rendered_shape": [1, 200, 200],
            "mra_manifest_sha256": MRA_DATASET_MANIFEST_SHA256,
            "confirmation_indices_sha256": CONFIRMATION_INDICES_SHA256,
            "train_support_count": TRAIN_SUPPORT_COUNT,
            "confirmation_count": CONFIRMATION_COUNT,
        },
        "dataset",
    )
    prior = _mapping(config["prior"], "prior")
    _check_mapping_values(
        prior,
        {
            "kind": "truncated_normal",
            "mean": 1.0,
            "variance": 0.25,
            "lower": 0.0,
            "unconditional": True,
            "class_conditioned": False,
            "evaluation_seed": PRIOR_SEED,
            "evaluation_count": CONFIRMATION_COUNT,
            "evaluation_identity_hash": PRIOR_SAMPLE_SHA256,
            "sampler": "bogt.truncated_normal_sample_inverse_cdf",
        },
        "prior",
    )
    stream = _mapping(config["stream"], "stream")
    _check_mapping_values(
        stream,
        {
            "training_base_seed": TRAINING_BASE_SEED,
            "training_seed_formula": "1238*100000+global_step",
            "evaluation_seed": PRIOR_SEED,
            "registration_panel_namespace": 10000,
            "namespace_separation": "training_and_registration_namespaces_are_disjoint",
        },
        "stream",
    )
    canary = _mapping(config["canary"], "canary")
    _check_mapping_values(
        canary,
        {
            "parent_step": PARENT_STEP,
            "start_step": PARENT_STEP,
            "end_step": END_STEP,
            "steps": CANARY_STEPS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "trainable_parameters": ["system.encoder.phase", "system.decoder.phase"],
            "scheduler": "none",
            "checkpoint_policy": "fixed_end_checkpoint_only",
            "endpoint_selection": "fixed_before_training_no_best_selection",
            "preflight_steps": 3,
            "logging": [
                "mse", "ed", "weighted_ed", "total", "encoder_grad_norm", "decoder_grad_norm",
                "finite", "z_mean", "z_variance", "optimizer_telemetry",
            ],
        },
        "canary",
    )
    evaluation = _mapping(config["evaluation"], "evaluation")
    _check_mapping_values(
        evaluation,
        {
            "prior_count": CONFIRMATION_COUNT,
            "prior_seed": PRIOR_SEED,
            "prior_sample_sha256": PRIOR_SAMPLE_SHA256,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "distance_metric": "exact_normalized_euclidean",
            "normalization": "divide_by_sqrt_100",
            "query_chunk_size": 128,
            "support_chunk_size": 4096,
        },
        "evaluation",
    )
    primary = _mapping(evaluation["primary"], "evaluation.primary")
    _check_mapping_values(
        primary,
        {"parent_r50_ci_lower": PARENT_R50_LOWER, "parent_r95_ci_lower": PARENT_R95_LOWER},
        "evaluation.primary",
    )
    if primary.get("rules") != [
        "upper95_end_r50_strictly_below_parent_r50_ci_lower",
        "upper95_end_r95_strictly_below_parent_r95_ci_lower",
    ]:
        raise ValueError("evaluation.primary rules are not registered")
    paired = _mapping(evaluation["paired_prior"], "evaluation.paired_prior")
    _require_equal(paired.get("rule"), "upper95_delta_p50_and_delta_p95_strictly_below_zero", "evaluation.paired_prior.rule")
    mse = _mapping(evaluation["parent_relative_mse"], "evaluation.parent_relative_mse")
    _check_mapping_values(
        mse,
        {
            "parent_panel_sha256": MRA_PANEL_SHA256,
            "parent_mse_key": "scaled_chen_step_5000",
            "rule": "upper95_mean_delta_mse_at_most_zero",
        },
        "evaluation.parent_relative_mse",
    )
    mean_safety = _mapping(evaluation["mean_safety"], "evaluation.mean_safety")
    _check_mapping_values(mean_safety, {"lower": MEAN_LOWER, "upper": MEAN_UPPER}, "evaluation.mean_safety")
    rank_safety = _mapping(evaluation["rank_safety"], "evaluation.rank_safety")
    _check_mapping_values(
        rank_safety,
        {"retention": RANK_RETENTION, "reference": "frozen_parent_rcr_b_mra_context"},
        "evaluation.rank_safety",
    )

    interpretation = _mapping(config["interpretation"], "interpretation")
    _check_mapping_values(
        interpretation,
        {
            "authority": SCAM_AUTHORITY,
            "plan": SCAM_PLAN,
            "power_observer": POWER_OBSERVER_INTERPRETATION,
            "complex_semantic_ols": "unestablished",
            "label_to_ols": "unresolved",
            "code8_to_ols": "partial_or_unresolved",
            "global_power_track": "closed_after_this_canary",
            "global_prior_family": "closed_after_this_canary",
            "next_action": NEXT_ACTION,
        },
        "interpretation",
    )
    if dict(config["governance"]) != GOVERNANCE:
        raise ValueError("T2-SAAC governance boundary is not locked")
    authorization = _mapping(config["authorization"], "authorization")
    _check_mapping_values(
        authorization,
        {
            "diagnostic_only": True,
            "bounded_canary_only": True,
            "training": "bounded_canary_only",
            "optimizer_step": "bounded_canary_only",
            "checkpoint_write": "end_checkpoint_evidence_only",
            "prior_generation": False,
            "decoder_generation": False,
            "cmos_generation": False,
            "mpcdar": False,
            "generation": False,
            "beta_sweep": False,
            "adaptive_beta": False,
            "new_global_power_objective_after_canary": False,
            "semantic_generation_authority": False,
            "formal_promotion": False,
        },
        "authorization",
    )
    execution = _mapping(config["execution"], "execution")
    _check_mapping_values(
        execution,
        {
            "device": "cuda",
            "batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": 1,
            "required_cublas_workspace_config": ":4096:8",
            "min_free_cuda_bytes": 4 * 1024**3,
            "max_peak_cuda_bytes": 5 * 1024**3,
            "min_disk_free_bytes": 10 * 1024**3,
            "preflight_max_seconds": 600.0,
            "max_storage_bytes": 2 * 1024**3,
        },
        "execution",
    )


def load_config(path: str | Path = CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("T2-SAAC config must be a YAML mapping")
    validate_registered_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def scam_interpretation() -> dict[str, str]:
    return {
        "authority": SCAM_AUTHORITY,
        "plan": SCAM_PLAN,
        "power_observer": POWER_OBSERVER_INTERPRETATION,
        "complex_semantic_ols": "unestablished",
        "label_to_ols": "unresolved",
        "code8_to_ols": "partial",
        "global_power_track": "closed_after_this_canary",
        "global_prior_family": "closed_after_this_canary",
        "next_action": NEXT_ACTION,
    }


def classify_t2_result(
    *,
    valid: bool,
    primary_pass: bool,
    paired_pass: bool,
    mse_safe: bool,
    mean_safe: bool,
    rank_safe: bool,
) -> str:
    if not valid:
        return T2_TRAINING_INVALID
    if not mse_safe or not mean_safe or not rank_safe:
        return T2_SUPPORT_AWARE_ALIGNMENT_HARMFUL
    if primary_pass and paired_pass:
        return T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED
    if primary_pass:
        return T2_SUPPORT_AWARE_ALIGNMENT_TRADEOFF
    return T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED


def _next_write_once_output(path: str | Path) -> Path:
    requested = Path(path)
    marker = "_attempt_"
    if marker not in requested.name:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = requested.name.rsplit(marker, 1)
    if not number_text.isdigit():
        raise ValueError("write-once output attempt is invalid")
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _recursive_digest(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"tensor_sha256": tensor_sha256(value.detach().cpu()), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _recursive_digest(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_recursive_digest(item) for item in value]
    return value


def _optimizer_state_digest(state: Mapping[str, Any]) -> str:
    return canonical_sha256(_recursive_digest(state))


def _cpu_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _cpu_nested(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested(item) for item in value)
    return value


def _state_digest(model: torch.nn.Module) -> str:
    return canonical_sha256({name: tensor_sha256(value.detach().cpu()) for name, value in model.state_dict().items()})


def _model_state_changes(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if set(before) != set(after):
        raise ValueError("model state keys changed")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    expected = {"system.encoder.phase", "system.decoder.phase"}
    return {
        "changed_names": changed,
        "only_registered_trainable_parameters_changed": set(changed) == expected,
        "ols_unchanged": all(torch.equal(before[name], after[name]) for name in before if name.startswith("ols.")),
    }


def _optimizer_step_values(state: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for item in _mapping(state.get("state"), "optimizer state").values():
        step = _mapping(item, "optimizer state entry").get("step")
        values.append(float(step) if torch.is_tensor(step) else float(step))
    return values


def _optimizer_telemetry(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    steps = _optimizer_step_values(state)
    return {
        "state_digest": _optimizer_state_digest(state),
        "parameter_group_count": len(state["param_groups"]),
        "learning_rates": [float(group["lr"]) for group in state["param_groups"]],
        "state_steps": steps,
        "state_step_min": min(steps),
        "state_step_max": max(steps),
    }


def _validate_optimizer_contract(state: Mapping[str, Any], *, expected_step: int) -> None:
    groups = state.get("param_groups")
    states = state.get("state")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or len(groups) != 1 or not isinstance(states, Mapping):
        raise ValueError("optimizer continuation contract is invalid")
    group = _mapping(groups[0], "optimizer parameter group")
    if list(group.get("params", ())) != [0, 1] or float(group.get("lr", math.nan)) != LEARNING_RATE:
        raise ValueError("optimizer parameter ordering or learning rate is invalid")
    if tuple(group.get("betas", ())) != (0.9, 0.999) or float(group.get("eps", math.nan)) != 1.0e-8:
        raise ValueError("optimizer Adam hyperparameters are invalid")
    if int(group.get("weight_decay", -1)) != 0 or bool(group.get("amsgrad", True)) or bool(group.get("maximize", True)):
        raise ValueError("optimizer Adam parameter group is invalid")
    if set(states) != {0, 1}:
        raise ValueError("optimizer state does not cover both registered parameters")
    for entry in states.values():
        item = _mapping(entry, "optimizer state entry")
        step = item.get("step")
        observed_step = float(step) if torch.is_tensor(step) else float(step)
        if not math.isfinite(observed_step) or observed_step != expected_step:
            raise ValueError("optimizer state step is not the registered endpoint")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = item.get(name)
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"optimizer state {name} is invalid")


def _check_resource_contract(config: Mapping[str, Any], output: Path) -> tuple[int, int, int]:
    execution = _mapping(config["execution"], "execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["required_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_REPLAY_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free, total = torch.cuda.mem_get_info(device)
    if free < int(execution["min_free_cuda_bytes"]):
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < int(execution["min_disk_free_bytes"]):
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 10 GiB free disk")
    if output.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output)}")
    return int(free), int(total), int(disk.free)


def _validate_git_state(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    if state.get("clean") is not True:
        raise RuntimeError("BLOCKED_UNEXPECTED_DIRTY_PATH: T2-SAAC requires a clean execution boundary")


def _validate_upstream_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["source_preflight"], "source_preflight")
    path = _resolve(binding["path"])
    payload = _json_file(path, expected_sha256=UPSTREAM_PREFLIGHT_SHA256)
    if payload.get("schema") != pf.PREFLIGHT_SCHEMA or payload.get("task_id") != pf.TASK_ID or payload.get("classification") != "SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS":
        raise ValueError("T2-SAAC upstream objective preflight is not the registered pass")
    if payload.get("config", {}).get("file_sha256") != UPSTREAM_PREFLIGHT_CONFIG_SHA256 or payload.get("config", {}).get("canonical_sha256") != UPSTREAM_PREFLIGHT_CONFIG_CANONICAL_SHA256:
        raise ValueError("T2-SAAC upstream preflight config binding is stale")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or any(value is not True for value in gates.values()):
        raise ValueError("T2-SAAC upstream preflight gates are not all passed")
    authorization = _mapping(payload.get("authorization"), "upstream preflight authorization")
    for name in ("training", "optimizer_step", "checkpoint_write", "generation", "decoder_generation", "cmos", "mpcdar"):
        if authorization.get(name) is not False:
            raise ValueError(f"T2-SAAC upstream preflight unexpectedly authorized {name}")
    future_path = path.parent / "future_canary_contract.json"
    future = _json_file(future_path)
    if future.get("status") != "contract_only_not_run" or future.get("objective") != "mse_plus_support_aware_energy_distance" or future.get("end_step") != END_STEP:
        raise ValueError("T2-SAAC upstream future canary contract is stale")
    future_ref = _mapping(payload.get("artifacts"), "upstream preflight artifacts").get("future_canary_contract.json")
    if _mapping(future_ref, "future canary artifact reference").get("sha256") != file_sha256(future_path):
        raise ValueError("T2-SAAC upstream future canary hash binding is stale")
    return payload


def _load_data(config: Mapping[str, Any]) -> CanaryData:
    dataset_root = _resolve(config["dataset"]["root"])
    audit = rcr_b.rcr.mra._load_data(dataset_root)
    if audit.manifest.get("manifest_hash") != MRA_DATASET_MANIFEST_SHA256:
        raise ValueError("T2-SAAC dataset manifest is stale")
    horizon_data = horizon._load_data(dataset_root)
    calibration_indices, parent_schedule = horizon._horizon_schedule(audit.labels)
    if tensor_sha256(torch.sort(calibration_indices).values) != config["calibration"]["calibration_indices_sha256"]:
        raise ValueError("T2-SAAC calibration index identity is stale")
    if not torch.equal(parent_schedule, horizon_data.schedule):
        raise ValueError("T2-SAAC parent schedule identity is stale")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train = torch.arange(TRAIN_SUPPORT_COUNT, dtype=torch.long)
    schedule_parts: list[torch.Tensor] = []
    for label in range(10):
        candidates = train[audit.labels.index_select(0, train) == label]
        shuffled = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator))
        available = shuffled[100:]
        repeats = (END_STEP + available.numel() - 1) // available.numel()
        schedule_parts.append(available.repeat(repeats)[:END_STEP])
    schedule = torch.stack(
        [torch.stack([schedule_parts[label][step] for label in range(10)]) for step in range(END_STEP)]
    )
    if not torch.equal(schedule[:PARENT_STEP], parent_schedule):
        raise ValueError("T2-SAAC continuation schedule changed the parent prefix")
    if not torch.equal(audit.labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(END_STEP)):
        raise ValueError("T2-SAAC continuation schedule is not class balanced")
    if tensor_sha256(audit.confirmation_indices) != CONFIRMATION_INDICES_SHA256:
        raise ValueError("T2-SAAC confirmation indices are stale")
    return CanaryData(audit, calibration_indices, parent_schedule, schedule)


def _load_authorities(config: Mapping[str, Any]) -> dict[str, Any]:
    _validate_upstream_preflight(config)
    pf.validate_parent_identity(config)
    pf.validate_prior_contract(config)
    sources = pf._load_registered_sources(config)
    checkpoint = sources["checkpoint"]
    calibration = pf._load_calibration(config)
    parent_dir = _resolve(config["parent"]["rcr_b_output_directory"])
    for name, expected in (
        (config["parent"]["rcr_b_manifest"], config["parent"]["rcr_b_manifest_sha256"]),
        (config["parent"]["rcr_b_metrics"], config["parent"]["rcr_b_metrics_sha256"]),
        (config["parent"]["rcr_b_report"], config["parent"]["rcr_b_report_sha256"]),
    ):
        if file_sha256(parent_dir / name) != expected:
            raise ValueError("T2-SAAC RCR-B authority hash is stale")
    distances = safe_torch_load(
        parent_dir / config["parent"]["rcr_b_distances"],
        expected_sha256=RCR_B_DISTANCES_SHA256,
        map_location="cpu",
    )
    if not isinstance(distances, Mapping):
        raise ValueError("T2-SAAC parent distance artifact is invalid")
    holdout = distances.get("d_holdout_to_train")
    prior = distances.get("d_prior_to_train")
    if not torch.is_tensor(holdout) or not torch.is_tensor(prior) or holdout.shape != (CONFIRMATION_COUNT,) or prior.shape != (CONFIRMATION_COUNT,):
        raise ValueError("T2-SAAC parent distance vectors have invalid shapes")
    panel = sources["parent"]["panel"]
    parent_mse = _mapping(_mapping(panel["per_sample_mse"], "parent MRA per-sample MSE")["confirmation"], "parent confirmation MSE").get(config["parent"]["mra_parent_mse_key"])
    if not torch.is_tensor(parent_mse) or parent_mse.shape != (CONFIRMATION_COUNT,) or not torch.isfinite(parent_mse).all():
        raise ValueError("T2-SAAC parent MSE endpoint is invalid")
    if tensor_sha256(parent_mse) != "8cffae872b7c02866e8b9f56e37ab3efaf1f8e9636d37295913e0aac87e21997":
        raise ValueError("T2-SAAC parent MSE endpoint hash is stale")
    return {
        "sources": sources,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "parent_distances": {"holdout": holdout, "prior": prior},
        "parent_mse": parent_mse,
        "parent_metrics": sources["rcr_metrics"],
    }


def _build_model(formal: Any, checkpoint: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = bogt.ScaledChenBOGTModel(formal)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model._set_trainable_contract()
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if names != bogt.EXPECTED_TRAINABLE or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise RuntimeError("T2_TRAINING_INVALID: trainable scope is not phase-only")
    if bogt.mode_bank_sha256(model) != "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9":
        raise ValueError("T2-SAAC mode bank identity is stale")
    model.train()
    return model


def _load_optimizer(model: bogt.ScaledChenBOGTModel, checkpoint: Mapping[str, Any]) -> tuple[torch.optim.Adam, str, str]:
    parent_state = checkpoint["optimizer_state"]
    _validate_optimizer_contract(parent_state, expected_step=PARENT_STEP)
    parent_digest = _optimizer_state_digest(parent_state)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if len(parameters) != 2:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED: trainable parameter count changed")
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    optimizer.load_state_dict(copy.deepcopy(parent_state))
    loaded_state = optimizer.state_dict()
    _validate_optimizer_contract(loaded_state, expected_step=PARENT_STEP)
    loaded_digest = _optimizer_state_digest(loaded_state)
    if loaded_digest != parent_digest:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED: loaded state differs from parent")
    return optimizer, parent_digest, loaded_digest


def _phase_grad_norm(parameter: torch.Tensor, name: str) -> float:
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError(f"T2_TRAINING_INVALID: missing or non-finite {name} gradient")
    return float(torch.linalg.vector_norm(parameter.grad.detach().float()).cpu())


def _training_step(
    model: bogt.ScaledChenBOGTModel,
    optimizer: torch.optim.Adam,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: Any,
    global_step: int,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    output = model(amplitudes)
    mse = torch.nn.functional.mse_loss(output.camera, targets)
    z = bogt.normalize_port_power(output.coefficients.abs().square(), calibration.port_scale)
    prior = bogt.sample_training_prior(global_step, logical_batch_size=LOGICAL_BATCH_SIZE, dtype=z.dtype).to(z.device)
    ed = bogt.two_pass_energy_distance_surrogate(z, prior, microbatch_size=1)
    weighted_ed = BETA * ed
    total = mse + weighted_ed
    if not all(torch.isfinite(value) for value in (mse, ed, weighted_ed, total)) or not torch.isfinite(z).all():
        raise RuntimeError("T2_TRAINING_INVALID: non-finite objective or power observer")
    total.backward()
    encoder = next(parameter for name, parameter in model.named_parameters() if name == "system.encoder.phase")
    decoder = next(parameter for name, parameter in model.named_parameters() if name == "system.decoder.phase")
    encoder_norm = _phase_grad_norm(encoder, "encoder")
    decoder_norm = _phase_grad_norm(decoder, "decoder")
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("T2_TRAINING_INVALID: non-finite learned state")
    return {
        "logical_step": global_step,
        "mse": float(mse.detach().cpu()),
        "ed": float(ed.detach().cpu()),
        "weighted_ed": float(weighted_ed.detach().cpu()),
        "total": float(total.detach().cpu()),
        "encoder_grad_norm": encoder_norm,
        "decoder_grad_norm": decoder_norm,
        "finite": True,
        "z_mean": float(z.detach().mean().cpu()),
        "z_variance": float(z.detach().var(unbiased=False).cpu()),
        "optimizer_telemetry": _optimizer_telemetry(optimizer),
    }


def _primary_gate(bootstrap: Mapping[str, Any]) -> bool:
    return float(bootstrap["r50_ci"][1]) < PARENT_R50_LOWER and float(bootstrap["r95_ci"][1]) < PARENT_R95_LOWER


def _paired_quantile_bootstrap_ci(parent: torch.Tensor, end: torch.Tensor, probability: float) -> tuple[float, float]:
    return pf.paired_quantile_bootstrap_ci(parent, end, probability=probability, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)


def _mean_bootstrap_ci(delta: torch.Tensor) -> tuple[float, float]:
    if delta.ndim != 1 or delta.numel() < 2 or not torch.isfinite(delta).all():
        raise ValueError("mean bootstrap input is invalid")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    values: list[torch.Tensor] = []
    for start in range(0, BOOTSTRAP_REPLICATES, 128):
        count = min(128, BOOTSTRAP_REPLICATES - start)
        indices = torch.randint(delta.numel(), (count, delta.numel()), generator=generator)
        values.append(delta.index_select(0, indices.reshape(-1)).reshape(count, -1).mean(dim=1))
    samples = torch.cat(values)
    return (
        float(torch.quantile(samples, (1.0 - CONFIDENCE_LEVEL) / 2.0)),
        float(torch.quantile(samples, 1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0)),
    )


def _collect_confirmation_mse(model: bogt.ScaledChenBOGTModel, data: CanaryData, device: torch.device) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for indices in data.audit.confirmation_indices.split(LOGICAL_BATCH_SIZE):
            rendered = base.render_intensity_batch(data.audit.images.index_select(0, indices))
            output = model(rendered.amplitude.to(device))
            target = rendered.target.to(device)
            value = torch.nn.functional.mse_loss(output.camera, target, reduction="none").flatten(1).mean(dim=1)
            chunks.append(value.cpu())
    result = torch.cat(chunks)
    if result.shape != (CONFIRMATION_COUNT,) or not torch.isfinite(result).all():
        raise RuntimeError("T2_TRAINING_INVALID: end MSE vector is invalid")
    return result


def _mra_end_metrics(model: bogt.ScaledChenBOGTModel, data: CanaryData, calibration: Any) -> dict[str, Any]:
    selection_coefficients = rcr_b.rcr._collect_coefficients(
        model, data.audit.test_images, data.audit.selection_indices, device=next(model.parameters()).device, batch_size=LOGICAL_BATCH_SIZE
    )
    confirmation_coefficients = rcr_b.rcr._collect_coefficients(
        model, data.audit.images, data.audit.confirmation_indices, device=next(model.parameters()).device, batch_size=LOGICAL_BATCH_SIZE
    )
    selection_labels = data.audit.test_labels.index_select(0, data.audit.selection_indices)
    confirmation_labels = data.audit.labels.index_select(0, data.audit.confirmation_indices)
    return {
        "schema": MRA_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "coordinate": {
            "definition": "z_j=abs(c_j)^2/s_j",
            "interpretation": POWER_OBSERVER_INTERPRETATION,
            "dimension": PORT_COUNT,
            "physical_state": "c_in_C^100_contains_amplitude_and_phase",
            "complex_semantic_ols": "unestablished",
            "calibration_sha256": CALIBRATION_SHA256,
        },
        "selection": rcr_b.rcr.coordinate_metrics(selection_coefficients, selection_labels, calibration),
        "confirmation": rcr_b.rcr.coordinate_metrics(confirmation_coefficients, confirmation_labels, calibration),
        "panel_identity": {
            "selection_indices_sha256": tensor_sha256(data.audit.selection_indices),
            "confirmation_indices_sha256": tensor_sha256(data.audit.confirmation_indices),
            "dataset_manifest_sha256": data.audit.manifest["manifest_hash"],
            "panel_reused_without_resampling": True,
        },
        "evaluation": {"no_update": True, "prior_generation": False, "semantic_ols_audit": False},
    }


def _mean_safety(mra: Mapping[str, Any]) -> dict[str, Any]:
    values = {split: float(_mapping(mra[split], split)["z_mean"]) for split in ("selection", "confirmation")}
    passed = all(math.isfinite(value) and MEAN_LOWER <= value <= MEAN_UPPER for value in values.values())
    return {"lower": MEAN_LOWER, "upper": MEAN_UPPER, "values": values, "passed": passed}


def _rank_safety(mra: Mapping[str, Any], parent_metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"retention": RANK_RETENTION, "reference": "frozen_parent_rcr_b_mra_context", "panels": {}}
    passed = True
    for split in ("selection", "confirmation"):
        end = _mapping(mra[split], split)
        parent = _mapping(_mapping(parent_metrics["mra_context"], "parent mra_context")[split], f"parent {split}")
        checks: dict[str, Any] = {}
        for name in ("complex_effective_rank", "power_effective_rank"):
            end_value = float(end[name])
            parent_value = float(parent[name])
            ok = math.isfinite(end_value) and end_value >= RANK_RETENTION * parent_value
            checks[name] = {"end": end_value, "parent": parent_value, "threshold": RANK_RETENTION * parent_value, "passed": ok}
            passed = passed and ok
        result["panels"][split] = checks
    result["passed"] = passed
    return result


def _support_end_metrics(
    model: bogt.ScaledChenBOGTModel,
    data: CanaryData,
    calibration: Any,
    config: Mapping[str, Any],
    parent_distances: Mapping[str, torch.Tensor],
    parent_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    train_indices = torch.arange(TRAIN_SUPPORT_COUNT, dtype=torch.long)
    heldout_indices = data.audit.confirmation_indices.cpu().contiguous()
    train_coefficients = rcr_b.rcr._collect_coefficients(model, data.audit.images, train_indices, device=device, batch_size=LOGICAL_BATCH_SIZE)
    heldout_coefficients = rcr_b.rcr._collect_coefficients(model, data.audit.images, heldout_indices, device=device, batch_size=LOGICAL_BATCH_SIZE)
    train_z = bogt.normalize_port_power(train_coefficients.abs().square(), calibration.port_scale).cpu().contiguous()
    heldout_z = bogt.normalize_port_power(heldout_coefficients.abs().square(), calibration.port_scale).cpu().contiguous()
    prior_z = rcr_b.sample_registered_prior(CONFIRMATION_COUNT, seed=PRIOR_SEED)
    if tensor_sha256(prior_z) != PRIOR_SAMPLE_SHA256 or tensor_sha256(heldout_indices) != CONFIRMATION_INDICES_SHA256:
        raise RuntimeError("T2_TRAINING_INVALID: registered endpoint sample identity changed")
    distance = _mapping(config["evaluation"], "evaluation")
    holdout_distances = rcr_b.exact_normalized_nearest_distances(
        heldout_z.to(device), train_z.to(device), query_chunk_size=int(distance["query_chunk_size"]), support_chunk_size=int(distance["support_chunk_size"])
    ).cpu()
    prior_distances = rcr_b.exact_normalized_nearest_distances(
        prior_z.to(device), train_z.to(device), query_chunk_size=int(distance["query_chunk_size"]), support_chunk_size=int(distance["support_chunk_size"])
    ).cpu()
    bootstrap = rcr_b.bootstrap_ratio_ci(
        holdout_distances,
        prior_distances,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    support = {
        "schema": SUPPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "coordinate": {
            "definition": "z_j=abs(c_j)^2/s_j",
            "interpretation": POWER_OBSERVER_INTERPRETATION,
            "dimension": PORT_COUNT,
            "normalization": "divide_by_sqrt_100",
            "calibration_sha256": CALIBRATION_SHA256,
        },
        "dataset": {
            "train_support_start": 0,
            "train_support_count": TRAIN_SUPPORT_COUNT,
            "heldout_count": CONFIRMATION_COUNT,
            "heldout_indices_sha256": tensor_sha256(heldout_indices),
            "dataset_manifest_sha256": data.audit.manifest["manifest_hash"],
        },
        "prior": {
            "count": CONFIRMATION_COUNT,
            "seed": PRIOR_SEED,
            "kind": "truncated_normal",
            "mean": 1.0,
            "variance": 0.25,
            "lower": 0.0,
            "unconditional": True,
            "sample_tensor_sha256": tensor_sha256(prior_z),
            "empirical_mean": float(prior_z.mean()),
            "empirical_variance": float(prior_z.var(unbiased=False)),
        },
        "holdout": rcr_b._summary_stats(holdout_distances),
        "prior_distances": rcr_b._summary_stats(prior_distances),
        "ratios": {"r50": bootstrap["r50"], "r95": bootstrap["r95"]},
        "bootstrap": bootstrap,
        "distance": {
            "metric": "exact_normalized_euclidean",
            "normalization": "divide_by_sqrt_100",
            "query_chunk_size": int(distance["query_chunk_size"]),
            "support_chunk_size": int(distance["support_chunk_size"]),
            "approximate_search": False,
        },
        "parent_endpoint_identity": {
            "holdout_vector_sha256": tensor_sha256(parent_distances["holdout"]),
            "prior_vector_sha256": tensor_sha256(parent_distances["prior"]),
            "rcr_b_distances_sha256": RCR_B_DISTANCES_SHA256,
            "parent_r50": parent_metrics["ratios"]["r50"],
            "parent_r95": parent_metrics["ratios"]["r95"],
        },
    }
    return support, {"holdout": holdout_distances, "prior": prior_distances, "prior_z": prior_z}


def _paired_prior_gate(parent: torch.Tensor, end: torch.Tensor) -> dict[str, Any]:
    delta = end - parent
    p50 = float(torch.quantile(delta, 0.50))
    p95 = float(torch.quantile(delta, 0.95))
    ci50 = _paired_quantile_bootstrap_ci(parent, end, 0.50)
    ci95 = _paired_quantile_bootstrap_ci(parent, end, 0.95)
    passed = ci50[1] < 0.0 and ci95[1] < 0.0
    return {
        "rule": "upper95_delta_p50_and_delta_p95_strictly_below_zero",
        "delta_p50": p50,
        "delta_p95": p95,
        "delta_p50_ci": list(ci50),
        "delta_p95_ci": list(ci95),
        "passed": passed,
    }


def _mse_gate(parent: torch.Tensor, end: torch.Tensor) -> dict[str, Any]:
    delta = end - parent
    ci = _mean_bootstrap_ci(delta)
    return {
        "rule": "upper95_mean_delta_mse_at_most_zero",
        "mean_delta": float(delta.mean()),
        "mean_delta_ci": list(ci),
        "parent_vector_sha256": tensor_sha256(parent),
        "end_vector_sha256": tensor_sha256(end),
        "passed": ci[1] <= 0.0,
    }


def _write_json(path: Path, payload: Mapping[str, Any], kind: str = "T2-SAAC artifact") -> None:
    atomic_write_new_json(path, payload, artifact_kind=kind)


def _assert_output_tree(output: Path) -> None:
    expected = {
        "training_manifest.json", "objective_contract.json", "training_metrics.json", "end_checkpoint.pt",
        "end_checkpoint_identity.json", "mra_end_metrics.json", "g1_end_support_metrics.json",
        "parent_end_comparison.json", "safety_metrics.json", "compact_report.json",
    }
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != expected or any(path.name.startswith(".") or ".tmp" in path.name or path.suffix.lower() in {".tif", ".tiff"} for path in output.rglob("*")):
        raise RuntimeError(f"T2-SAAC output tree is not exact: {sorted(actual)}")


def _validate_canary_preflight(path: str | Path, spec: ConfigSpec, state: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_file(_resolve(path))
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("status") != "passed":
        raise RuntimeError("T2_TRAINING_INVALID: same-entrypoint canary preflight did not pass")
    config_binding = _mapping(payload.get("config"), "canary preflight config")
    if config_binding.get("file_sha256") != spec.file_sha256 or config_binding.get("canonical_sha256") != spec.canonical_sha256:
        raise RuntimeError("T2_TRAINING_INVALID: canary preflight config is stale")
    git_binding = _mapping(payload.get("git"), "canary preflight git")
    if git_binding.get("commit") != state.get("commit") or git_binding.get("clean") is not True:
        raise RuntimeError("T2_TRAINING_INVALID: canary preflight git boundary is stale")
    if payload.get("upstream_preflight", {}).get("sha256") != UPSTREAM_PREFLIGHT_SHA256:
        raise RuntimeError("T2_TRAINING_INVALID: upstream preflight binding is stale")
    gates = payload.get("gates")
    if not isinstance(gates, Sequence) or isinstance(gates, (str, bytes)) or any(_mapping(item, "canary preflight gate").get("passed") is not True for item in gates):
        raise RuntimeError("T2_TRAINING_INVALID: canary preflight gates are not all passed")
    negative = _mapping(payload.get("negative_control"), "canary preflight negative control")
    if negative.get("expected_decision") != "reject_primary_gate" or negative.get("passed") is not True:
        raise RuntimeError("T2_TRAINING_INVALID: canary preflight negative control is invalid")
    return payload


def preflight(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = DEFAULT_PREFLIGHT_OUTPUT_PATH,
) -> dict[str, Any]:
    config_target, config, config_file_hash, config_canonical_hash = load_config(config_path)
    state = base._git_state()
    _validate_git_state(state)
    authorities = _load_authorities(config)
    data = _load_data(config)
    actual_output = _next_write_once_output(_resolve(output_path))
    free_before, total_cuda, disk_free_before = _check_resource_contract(config, actual_output)
    device = torch.device(config["execution"]["device"])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    formal = authorities["sources"]["parent"]["formal"]
    model = _build_model(formal, authorities["checkpoint"], device)
    optimizer, parent_optimizer_digest, loaded_optimizer_digest = _load_optimizer(model, authorities["checkpoint"])
    state_before = _state_digest(model)
    initial_ols = _cpu_state_dict(model.ols.state_dict())
    history: list[dict[str, Any]] = []
    for step in range(PARENT_STEP + 1, PARENT_STEP + 1 + int(config["canary"]["preflight_steps"])):
        indices = data.schedule[step - 1]
        rendered = base.render_intensity_batch(data.audit.images.index_select(0, indices))
        history.append(_training_step(model, optimizer, rendered.amplitude.to(device), rendered.target.to(device), authorities["calibration"], step))
        del rendered
    torch.cuda.synchronize(device)
    state_after = _state_digest(model)
    final_ols = _cpu_state_dict(model.ols.state_dict())
    ols_unchanged = all(torch.equal(initial_ols[name], final_ols[name]) for name in initial_ols)
    parent_metrics = authorities["parent_metrics"]
    negative_pass = not _primary_gate(parent_metrics["bootstrap"])
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = [
        {"name": "three_step_real_update", "passed": len(history) == 3 and state_after != state_before and all(row["finite"] for row in history)},
        {"name": "registered_trainable_scope", "passed": tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) == bogt.EXPECTED_TRAINABLE},
        {"name": "parent_optimizer_continuation", "passed": parent_optimizer_digest == loaded_optimizer_digest and _optimizer_step_values(optimizer.state_dict()) == [5003.0, 5003.0]},
        {"name": "frozen_ols", "passed": ols_unchanged},
        {"name": "finite_values", "passed": all(torch.isfinite(parameter).all() for parameter in model.parameters())},
        {"name": "cuda_peak_memory", "passed": peak <= int(config["execution"]["max_peak_cuda_bytes"]), "bytes": peak},
        {"name": "preflight_wall_time", "passed": elapsed <= float(config["execution"]["preflight_max_seconds"]), "seconds": elapsed},
    ]
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "passed" if all(item["passed"] for item in gates) and negative_pass else "invalid_harness",
        "git": state,
        "config": {"path": _relative_path(config_target), "file_sha256": config_file_hash, "canonical_sha256": config_canonical_hash},
        "upstream_preflight": {"path": _relative_path(config["source_preflight"]["path"]), "sha256": UPSTREAM_PREFLIGHT_SHA256, "classification": "SUPPORT_AWARE_OBJECTIVE_PREFLIGHT_PASS"},
        "parent": {"checkpoint_sha256": PARENT_CHECKPOINT_SHA256, "step": PARENT_STEP, "state_digest": state_before, "optimizer_state_digest": parent_optimizer_digest},
        "objective": {"name": "mse_plus_support_aware_energy_distance", "beta": BETA, "alpha_bogt": 0.0, "gamma": 0.0, "power_observer": POWER_OBSERVER_INTERPRETATION},
        "schedule": {"seed": SEED, "parent_schedule_sha256": tensor_sha256(data.parent_schedule), "continuation_schedule_sha256": tensor_sha256(data.schedule), "steps_exercised": len(history)},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "parent_state_digest": parent_optimizer_digest, "loaded_state_digest": loaded_optimizer_digest, "telemetry": _optimizer_telemetry(optimizer)},
        "gates": gates,
        "negative_control": {"kind": "frozen_parent_rcr_b_endpoint", "expected_decision": "reject_primary_gate", "primary_gate": False, "passed": negative_pass},
        "history": history,
        "resource": {"device": str(device), "free_cuda_bytes_before": free_before, "total_cuda_bytes": total_cuda, "disk_free_bytes_before": disk_free_before, "peak_cuda_allocated_bytes": peak, "wall_clock_seconds": elapsed},
        "authorization": {"three_step_real_update": True, "formal_1000_step_canary_started": False, "generation": False, "semantic_generation_authority": False},
        "governance": GOVERNANCE,
    }
    actual_output.mkdir(parents=True, exist_ok=False)
    _write_json(actual_output / "preflight.json", payload)
    if {path.name for path in actual_output.iterdir()} != {"preflight.json"}:
        raise RuntimeError("T2-SAAC preflight output tree is not exact")
    if payload["status"] != "passed":
        raise RuntimeError("T2_TRAINING_INVALID: same-entrypoint canary preflight failed")
    return {"output_directory": actual_output, "payload": payload}


def run(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    preflight_evidence: str | Path | None = None,
) -> dict[str, Any]:
    if preflight_evidence is None:
        raise ValueError("T2-SAAC formal run requires --preflight-evidence")
    config_target, config, config_file_hash, config_canonical_hash = load_config(config_path)
    state = base._git_state()
    _validate_git_state(state)
    spec = ConfigSpec(config_target, config, config_file_hash, config_canonical_hash)
    preflight_payload = _validate_canary_preflight(preflight_evidence, spec, state)
    authorities = _load_authorities(config)
    data = _load_data(config)
    actual_output = _next_write_once_output(_resolve(output_path))
    free_before, total_cuda, disk_free_before = _check_resource_contract(config, actual_output)
    device = torch.device(config["execution"]["device"])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    formal = authorities["sources"]["parent"]["formal"]
    model = _build_model(formal, authorities["checkpoint"], device)
    start_state_digest = _state_digest(model)
    if start_state_digest != PARENT_STATE_DIGEST:
        raise RuntimeError("T2_TRAINING_INVALID: parent learned-state digest mismatch")
    start_state = _cpu_state_dict(model.state_dict())
    optimizer, parent_optimizer_digest, loaded_optimizer_digest = _load_optimizer(model, authorities["checkpoint"])
    history: list[dict[str, Any]] = []
    for step in range(PARENT_STEP + 1, END_STEP + 1):
        indices = data.schedule[step - 1]
        rendered = base.render_intensity_batch(data.audit.images.index_select(0, indices))
        history.append(_training_step(model, optimizer, rendered.amplitude.to(device), rendered.target.to(device), authorities["calibration"], step))
        del rendered
    torch.cuda.synchronize(device)
    end_state_digest = _state_digest(model)
    end_state = _cpu_state_dict(model.state_dict())
    changes = _model_state_changes(start_state, end_state)
    optimizer_end = _optimizer_telemetry(optimizer)
    if end_state_digest == start_state_digest or not changes["only_registered_trainable_parameters_changed"] or not changes["ols_unchanged"]:
        raise RuntimeError("T2_TRAINING_INVALID: learned-state identity or frozen-state contract failed")
    if optimizer_end["state_steps"] != [float(END_STEP), float(END_STEP)]:
        raise RuntimeError("T2_TRAINING_INVALID: optimizer step did not reach 6000")

    actual_output.mkdir(parents=True, exist_ok=False)
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "global_step": END_STEP,
        "parent_checkpoint": {"step": PARENT_STEP, "sha256": PARENT_CHECKPOINT_SHA256},
        "model_state": end_state,
        "optimizer_state": _cpu_nested(optimizer.state_dict()),
        "optimizer_identity": {"name": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest, "end_state_digest": optimizer_end["state_digest"]},
        "trainable_parameters": ["system.encoder.phase", "system.decoder.phase"],
        "objective": {"name": "mse_plus_support_aware_energy_distance", "beta": BETA, "alpha_bogt": 0.0, "gamma": 0.0},
        "endpoint_fixed_before_training": True,
    }
    atomic_torch_save_new(actual_output / "end_checkpoint.pt", checkpoint_payload, artifact_kind="T2-SAAC end checkpoint")
    checkpoint_hash = file_sha256(actual_output / "end_checkpoint.pt")
    loaded_checkpoint = safe_torch_load(actual_output / "end_checkpoint.pt", expected_sha256=checkpoint_hash, map_location="cpu")
    if loaded_checkpoint.get("schema") != CHECKPOINT_SCHEMA or loaded_checkpoint.get("global_step") != END_STEP:
        raise RuntimeError("T2_TRAINING_INVALID: end checkpoint identity is invalid")
    eval_model = bogt.ScaledChenBOGTModel(formal)
    eval_model.load_state_dict(loaded_checkpoint["model_state"], strict=True)
    eval_model.to(device)
    for parameter in eval_model.parameters():
        parameter.requires_grad_(False)
    eval_model.eval()
    eval_state_before = _state_digest(eval_model)
    mra_metrics = _mra_end_metrics(eval_model, data, authorities["calibration"])
    support_metrics, end_vectors = _support_end_metrics(
        eval_model, data, authorities["calibration"], config, authorities["parent_distances"], authorities["parent_metrics"]
    )
    end_mse = _collect_confirmation_mse(eval_model, data, device)
    eval_state_after = _state_digest(eval_model)
    if eval_state_before != eval_state_after or eval_state_before != end_state_digest:
        raise RuntimeError("T2_TRAINING_INVALID: end evaluation changed learned state")
    primary_pass = _primary_gate(support_metrics["bootstrap"])
    paired_prior = _paired_prior_gate(authorities["parent_distances"]["prior"], end_vectors["prior"])
    mse_gate = _mse_gate(authorities["parent_mse"], end_mse)
    mean_safety = _mean_safety(mra_metrics)
    rank_safety = _rank_safety(mra_metrics, authorities["parent_metrics"])
    paired_pass = paired_prior["passed"]
    valid = (
        all(row["finite"] for row in history)
        and eval_state_before == eval_state_after
        and changes["only_registered_trainable_parameters_changed"]
        and changes["ols_unchanged"]
        and optimizer_end["state_steps"] == [float(END_STEP), float(END_STEP)]
    )
    final_classification = classify_t2_result(
        valid=valid,
        primary_pass=primary_pass,
        paired_pass=paired_pass,
        mse_safe=mse_gate["passed"],
        mean_safe=mean_safety["passed"],
        rank_safe=rank_safety["passed"],
    )
    interpretation = scam_interpretation()
    scam_authority = "PASS" if valid and final_classification == T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED else "FAIL"
    global_conclusion = "GLOBAL_POWER_GEOMETRY_CONTROLLABILITY_SUPPORTED" if final_classification == T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED else final_classification
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    training_metrics = {
        "schema": TRAINING_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "endpoint": {"start_step": PARENT_STEP, "end_step": END_STEP, "steps": CANARY_STEPS, "selection": "fixed_before_training_no_best_selection"},
        "objective": {"name": "mse_plus_support_aware_energy_distance", "formula": "L_total=L_MSE+beta*L_ED", "beta": BETA, "alpha_bogt": 0.0, "gamma": 0.0, "power_observer": POWER_OBSERVER_INTERPRETATION},
        "history": history,
        "summary": {"start": history[0], "end": history[-1], "mean_mse": float(sum(row["mse"] for row in history) / len(history)), "mean_ed": float(sum(row["ed"] for row in history) / len(history)), "mean_weighted_ed": float(sum(row["weighted_ed"] for row in history) / len(history)), "mean_total": float(sum(row["total"] for row in history) / len(history))},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest, "loaded_state_digest": loaded_optimizer_digest, "end": optimizer_end},
        "learned_state": {"start_digest": start_state_digest, "end_digest": end_state_digest, "changed_names": changes["changed_names"], "only_registered_trainable_parameters_changed": changes["only_registered_trainable_parameters_changed"], "ols_unchanged": changes["ols_unchanged"]},
        "schedule": {"seed": SEED, "parent_schedule_sha256": tensor_sha256(data.parent_schedule), "continuation_schedule_sha256": tensor_sha256(data.schedule), "shape": list(data.schedule.shape)},
        "resource": {"device": str(device), "free_cuda_bytes_before": free_before, "total_cuda_bytes": total_cuda, "disk_free_bytes_before": disk_free_before, "peak_cuda_allocated_bytes": peak, "wall_clock_seconds": elapsed},
    }
    primary = {
        "name": "T2_GLOBAL_NORMALIZED_POWER_GEOMETRY_CONTROLLABILITY_CANARY",
        "passed": primary_pass,
        "r50_end_ci": support_metrics["bootstrap"]["r50_ci"],
        "r95_end_ci": support_metrics["bootstrap"]["r95_ci"],
        "parent_r50_ci_lower": PARENT_R50_LOWER,
        "parent_r95_ci_lower": PARENT_R95_LOWER,
        "r50_ci_separated_improvement": float(support_metrics["bootstrap"]["r50_ci"][1]) < PARENT_R50_LOWER,
        "r95_ci_separated_improvement": float(support_metrics["bootstrap"]["r95_ci"][1]) < PARENT_R95_LOWER,
    }
    parent_comparison = {
        "schema": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "parent": {"r50": authorities["parent_metrics"]["ratios"]["r50"], "r50_ci": authorities["parent_metrics"]["bootstrap"]["r50_ci"], "r95": authorities["parent_metrics"]["ratios"]["r95"], "r95_ci": authorities["parent_metrics"]["bootstrap"]["r95_ci"]},
        "end": {"r50": support_metrics["ratios"]["r50"], "r50_ci": support_metrics["bootstrap"]["r50_ci"], "r95": support_metrics["ratios"]["r95"], "r95_ci": support_metrics["bootstrap"]["r95_ci"]},
        "primary_gate": primary,
        "paired_prior_gate": paired_prior,
        "parent_relative_mse_gate": mse_gate,
        "mra_mean_safety": mean_safety,
        "mra_rank_safety": rank_safety,
    }
    safety_metrics = {
        "schema": SAFETY_SCHEMA,
        "task_id": TASK_ID,
        "paired_prior": paired_prior,
        "parent_relative_mse": mse_gate,
        "mean_scale": mean_safety,
        "rank": rank_safety,
        "passed": mse_gate["passed"] and mean_safety["passed"] and rank_safety["passed"],
    }
    compact_report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "gate": final_classification,
        "primary_gate": primary,
        "paired_prior_gate": paired_prior,
        "parent_relative_mse_gate": mse_gate,
        "end": {"r50": support_metrics["ratios"]["r50"], "r50_ci": support_metrics["bootstrap"]["r50_ci"], "r95": support_metrics["ratios"]["r95"], "r95_ci": support_metrics["bootstrap"]["r95_ci"]},
        "scam_authority": scam_authority,
        "power_observer_interpretation": POWER_OBSERVER_INTERPRETATION,
        "complex_semantic_ols_status": "UNESTABLISHED",
        "global_power_geometry_conclusion": global_conclusion,
        "semantic_ols_conclusion": "UNESTABLISHED",
        "label_to_ols_conclusion": "UNRESOLVED",
        "code8_to_ols_conclusion": "PARTIAL / UNRESOLVED",
        "global_prior_family_status": "CLOSED_AFTER_THIS_CANARY",
        "interpretation": interpretation,
        "governance": GOVERNANCE,
        "next_action": NEXT_ACTION if valid else "HANDLE_T2_TRAINING_INVALIDITY",
    }
    objective_contract = {
        "schema": "lightgen_scaled_chen_support_aware_alignment_canary_objective.v1",
        "task_id": TASK_ID,
        "parent_checkpoint": {"path": _relative_path(config["parent"]["checkpoint_path"]), "sha256": PARENT_CHECKPOINT_SHA256, "step": PARENT_STEP},
        "calibration": {"schema": authorities["calibration"].schema, "contract_version": authorities["calibration"].contract_version, "payload_sha256": CALIBRATION_SHA256, "mode_bank_sha256": config["calibration"]["mode_bank_sha256"]},
        "prior": dict(config["prior"]),
        "objective": {"name": "mse_plus_support_aware_energy_distance", "formula": "L_total=L_MSE+beta*L_ED", "beta": BETA, "alpha_bogt": 0.0, "gamma": 0.0, "estimator": "ordered_unbiased_u_statistic", "normalization": "divide_by_sqrt_100", "power_observer": POWER_OBSERVER_INTERPRETATION},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest},
        "trainable_parameters": ["system.encoder.phase", "system.decoder.phase"],
        "horizon": {"start_step": PARENT_STEP, "end_step": END_STEP, "steps": CANARY_STEPS, "endpoint_fixed_before_training": True},
        "no_prior_decoding": True,
        "no_semantic_generation_authority": True,
    }
    end_identity = {
        "schema": "lightgen_scaled_chen_support_aware_alignment_canary_end_checkpoint_identity.v1",
        "task_id": TASK_ID,
        "path": "end_checkpoint.pt",
        "sha256": checkpoint_hash,
        "schema_bound": CHECKPOINT_SCHEMA,
        "global_step": END_STEP,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "learned_state_start_sha256": start_state_digest,
        "learned_state_end_sha256": end_state_digest,
        "optimizer_start_digest": parent_optimizer_digest,
        "optimizer_end_digest": optimizer_end["state_digest"],
        "only_registered_trainable_parameters_changed": changes["only_registered_trainable_parameters_changed"],
        "ols_unchanged": changes["ols_unchanged"],
    }
    _write_json(actual_output / "objective_contract.json", objective_contract)
    _write_json(actual_output / "training_metrics.json", training_metrics)
    _write_json(actual_output / "end_checkpoint_identity.json", end_identity)
    _write_json(actual_output / "mra_end_metrics.json", mra_metrics)
    _write_json(actual_output / "g1_end_support_metrics.json", support_metrics)
    _write_json(actual_output / "parent_end_comparison.json", parent_comparison)
    _write_json(actual_output / "safety_metrics.json", safety_metrics)
    _write_json(actual_output / "compact_report.json", compact_report)
    storage = sum(path.stat().st_size for path in actual_output.rglob("*") if path.is_file())
    if storage > int(config["execution"]["max_storage_bytes"]):
        raise RuntimeError("T2_TRAINING_INVALID: output exceeds registered storage bound")
    manifest = {
        "schema": TRAINING_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "attempt": actual_output.name,
        "output_directory": _relative_path(actual_output),
        "git": state,
        "config": {"path": _relative_path(config_target), "file_sha256": config_file_hash, "canonical_sha256": config_canonical_hash},
        "upstream_preflight": {"path": _relative_path(config["source_preflight"]["path"]), "sha256": UPSTREAM_PREFLIGHT_SHA256, "status": "passed"},
        "same_entrypoint_preflight": {"path": _relative_path(preflight_evidence), "sha256": file_sha256(_resolve(preflight_evidence)), "status": preflight_payload["status"]},
        "parent": {"checkpoint": PARENT_CHECKPOINT_SHA256, "step": PARENT_STEP, "state_digest": PARENT_STATE_DIGEST, "rcr_b_distances": RCR_B_DISTANCES_SHA256, "mra_panel": MRA_PANEL_SHA256},
        "calibration": {"payload": CALIBRATION_SHA256, "manifest": CALIBRATION_MANIFEST_SHA256, "contract_version": authorities["calibration"].contract_version},
        "prior": {"evaluation_seed": PRIOR_SEED, "evaluation_count": CONFIRMATION_COUNT, "sample_sha256": PRIOR_SAMPLE_SHA256, "training_base_seed": TRAINING_BASE_SEED},
        "training": {"start_step": PARENT_STEP, "end_step": END_STEP, "steps": CANARY_STEPS, "objective": objective_contract["objective"], "optimizer": objective_contract["optimizer"], "trainable_parameters": objective_contract["trainable_parameters"]},
        "evaluation": {"mra": "mra_end_metrics.json", "support": "g1_end_support_metrics.json", "comparison": "parent_end_comparison.json", "safety": "safety_metrics.json", "no_prior_decoding": True},
        "gate": final_classification,
        "scam_authority": scam_authority,
        "interpretation": {"power_observer": POWER_OBSERVER_INTERPRETATION, "complex_semantic_ols": "UNESTABLISHED", "global_power_geometry": global_conclusion, "semantic_ols": "UNESTABLISHED", "label_to_ols": "UNRESOLVED", "code8_to_ols": "PARTIAL / UNRESOLVED", "global_prior_family": "CLOSED_AFTER_THIS_CANARY"},
        "governance": GOVERNANCE,
        "next_action": compact_report["next_action"],
        "storage_bytes": storage,
        "artifact_refs": {path.relative_to(actual_output).as_posix(): {"path": path.relative_to(actual_output).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size} for path in sorted(actual_output.iterdir(), key=lambda item: item.name) if path.is_file()},
    }
    _write_json(actual_output / "training_manifest.json", manifest)
    _assert_output_tree(actual_output)
    return {"output_directory": actual_output, "manifest": manifest, "report": compact_report, "training_metrics": training_metrics, "mra": mra_metrics, "support": support_metrics, "safety": safety_metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--preflight-evidence", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "preflight":
        result = preflight(args.config, args.output or DEFAULT_PREFLIGHT_OUTPUT_PATH)
        print(json.dumps(result["payload"], indent=2, sort_keys=True))
        return 0
    result = run(args.config, args.output or DEFAULT_OUTPUT_PATH, args.preflight_evidence)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
