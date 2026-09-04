"""Run the one-shot repaired-coordinate T1 generative alignment canary.

The entrypoint is deliberately narrower than the historical horizon trainer:
it restores the registered step-5000 Adam state, performs exactly one fixed
1000-step continuation with the registered repaired-coordinate BOGT weight,
then evaluates only the registered MRA, support-accessibility, and safety
boundaries.  It never samples a decoded prior or enters a generation track.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
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


TASK_ID = "H-LOMA-GEN-T1-GAC"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1"
PREFLIGHT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_preflight.v1"
TRAINING_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_training.v1"
END_CHECKPOINT_SCHEMA = "scaled_chen_repaired_coordinate_generative_alignment_canary_checkpoint.v1"
MRA_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_mra.v1"
G1_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_g1.v1"
SAFETY_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_safety.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_compact_report.v1"

CANARY_START_STEP = 5000
CANARY_STEPS = 1000
CANARY_END_STEP = 6000
SEED = 1234
BATCH_SIZE = 10
LATENT_DIM = 100
PRIOR_COUNT = 5000
PRIOR_SEED = 1237
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
RANK_RETENTION = 0.90
MEAN_LOWER = 0.90
MEAN_UPPER = 1.10
LEARNING_RATE = 1.0e-3
GAMMA = 0.0
ALPHA_TEXT = "0.00017905883237862847"
ALPHA_REPAIRED = float(ALPHA_TEXT)
OLD_ALPHA = 0.002996375650268832
PARENT_CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
PARENT_STATE_DIGEST = "cdbf06f03b611f4df38324c1546686bc5fc614644a64130c95cd3a76012e27b5"
MRA_DATASET_MANIFEST_SHA256 = "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
PARENT_R50_LOWER = 1.8142740175004783
PARENT_R95_LOWER = 1.067902362924707

T1_ALIGNMENT_DIRECTION_SUPPORTED = "T1_ALIGNMENT_DIRECTION_SUPPORTED"
T1_ALIGNMENT_NOT_SUPPORTED = "T1_ALIGNMENT_NOT_SUPPORTED"
T1_ALIGNMENT_HARMFUL = "T1_ALIGNMENT_HARMFUL"
T1_TRAINING_INVALID = "T1_TRAINING_INVALID"

GOVERNANCE = {
    "mainline_state": "closed_at_current_evidence_boundary",
    "reconstruction_manifold_supported": False,
    "generative_manifold_supported": False,
    "generation_readiness": "not_ready",
    "training_authorized_after_canary": False,
    "formal_promotion": False,
}

DEFAULT_CONFIG_PATH = Path(
    "configs/research/"
    "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1.yaml"
)
DEFAULT_PREFLIGHT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_preflight_attempt_01"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_v1_attempt_01"
)


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
    schedule: torch.Tensor
    parent_schedule: torch.Tensor


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
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


def validate_alpha(value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError("registered alpha must be finite")
    if float(value) == OLD_ALPHA:
        raise ValueError("old alpha is forbidden")
    if float(value) != ALPHA_REPAIRED:
        raise ValueError("registered alpha does not match RCAWR attempt_02")


def validate_fixed_end_policy(start_step: int, end_step: int, steps: int) -> None:
    if int(steps) != CANARY_STEPS:
        raise ValueError("endpoint policy requires exactly 1000 steps")
    if int(start_step) != CANARY_START_STEP or int(end_step) != CANARY_END_STEP:
        raise ValueError("endpoint policy is fixed at step 5000 to step 6000 with exactly 1000 steps")
    if int(end_step) - int(start_step) != int(steps):
        raise ValueError("horizon must contain exactly 1000 steps")


def validate_checkpoint_identity(
    payload: Mapping[str, Any], *, expected_schema: str, expected_step: int
) -> None:
    if (
        payload.get("schema") != expected_schema
        or int(payload.get("global_step", -1)) != int(expected_step)
        or not isinstance(payload.get("model_state"), Mapping)
        or not isinstance(payload.get("optimizer_state"), Mapping)
    ):
        raise ValueError("checkpoint identity is invalid")


def validate_optimizer_contract(
    state: Mapping[str, Any], *, expected_parameter_count: int, learning_rate: float
) -> None:
    groups = state.get("param_groups")
    states = state.get("state")
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or len(groups) != 1 or not isinstance(states, Mapping):
        raise ValueError("optimizer continuation contract is invalid")
    group = _require_mapping(groups[0], "optimizer parameter group")
    if list(group.get("params", ())) != list(range(expected_parameter_count)):
        raise ValueError("optimizer parameter ordering is invalid")
    if float(group.get("lr", float("nan"))) != float(learning_rate):
        raise ValueError("optimizer learning rate is invalid")
    if tuple(group.get("betas", ())) != (0.9, 0.999) or float(group.get("eps", float("nan"))) != 1.0e-8:
        raise ValueError("optimizer Adam hyperparameters are invalid")
    if int(group.get("weight_decay", -1)) != 0 or bool(group.get("amsgrad", True)) or bool(group.get("maximize", True)):
        raise ValueError("optimizer Adam parameter group is invalid")
    if set(states) != set(range(expected_parameter_count)):
        raise ValueError("optimizer state does not cover every trainable parameter")
    for parameter_id, value in states.items():
        entry = _require_mapping(value, f"optimizer state {parameter_id}")
        step = entry.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1 or not torch.isfinite(step).all() or float(step) <= 0.0:
                raise ValueError("optimizer state step is invalid")
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)) or float(step) <= 0.0:
            raise ValueError("optimizer state step is invalid")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = entry.get(name)
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"optimizer state {name} is invalid")


def validate_trainable_parameter_names(names: Sequence[str]) -> None:
    expected = ("system.encoder.phase", "system.decoder.phase")
    if tuple(names) != expected:
        raise ValueError("trainable parameter scope is invalid")


def classify_primary_gate(
    r50_ci: Sequence[float],
    r95_ci: Sequence[float],
    parent_r50_lower: float,
    parent_r95_lower: float,
) -> bool:
    if len(r50_ci) != 2 or len(r95_ci) != 2:
        raise ValueError("primary gate confidence intervals must have two bounds")
    values = tuple(float(value) for value in (*r50_ci, *r95_ci, parent_r50_lower, parent_r95_lower))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("primary gate confidence intervals must be finite")
    return float(r50_ci[1]) < float(parent_r50_lower) and float(r95_ci[1]) < float(parent_r95_lower)


def classify_final_result(
    primary_pass: bool,
    mean_safe: bool,
    rank_safe: bool,
    current_sample_safe: bool,
    *,
    valid: bool = True,
    ratio_worsened: bool = False,
) -> str:
    if not valid:
        return T1_TRAINING_INVALID
    if ratio_worsened or not mean_safe or not rank_safe or not current_sample_safe:
        return T1_ALIGNMENT_HARMFUL
    return T1_ALIGNMENT_DIRECTION_SUPPORTED if primary_pass else T1_ALIGNMENT_NOT_SUPPORTED


def next_write_once_output(base_path: str | Path) -> Path:
    target = Path(base_path)
    if re.fullmatch(r".+_attempt_\d+", target.name) is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    match = re.fullmatch(r"(.+)_attempt_(\d+)", target.name)
    assert match is not None
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = target.parent / f"{prefix}_attempt_{number:02d}"
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


def _cpu_state_dict(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
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


def _model_state_changes(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
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
    for item in _require_mapping(state.get("state"), "optimizer state").values():
        step = _require_mapping(item, "optimizer state entry").get("step")
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


def _gate(name: str, passed: bool, **metrics: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "metrics": metrics}


def _assert_config(spec: ConfigSpec) -> None:
    payload = spec.payload
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "parent",
        "calibration",
        "dataset",
        "prior",
        "training",
        "evaluation",
        "execution",
        "authorization",
        "governance",
    }
    if set(payload) != required or payload["schema"] != CONFIG_SCHEMA or payload["task_id"] != TASK_ID or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("GAC config identity or fields are invalid")
    training = _require_mapping(payload["training"], "training")
    validate_fixed_end_policy(training["start_step"], training["end_step"], training["steps"])
    if training["parent_step"] != CANARY_START_STEP or training["objective"] != "mse_plus_repaired_coordinate_bogt" or training["arm"] != "mse_kl" or training["optimizer"] != "Adam" or training["scheduler"] != "none" or training["seed"] != SEED or training["logical_batch_size"] != BATCH_SIZE or training["microbatch_size"] != 1 or training["gradient_accumulation_steps"] != 1 or training["checkpoint_policy"] != "fixed_end_checkpoint_only" or training["endpoint_selection"] != "fixed_before_training_no_best_selection":
        raise ValueError("GAC training contract is not locked")
    validate_alpha(float(training["alpha"]))
    if float(training["gamma"]) != GAMMA or float(training["mse_weight"]) != 1.0 or float(training["learning_rate"]) != LEARNING_RATE or float(training["old_alpha"]) != OLD_ALPHA:
        raise ValueError("GAC objective or optimizer contract is not locked")
    calibration = _require_mapping(payload["calibration"], "calibration")
    if calibration["schema"] != "scaled_chen_bogt_calibration.v2" or calibration["contract_version"] != "unit_aware_raw_mean_v2" or calibration["payload_sha256"] != "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07" or calibration["calibration_indices_sha256"] != "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543" or calibration["dataset_manifest_sha256"] != "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9" or calibration["mode_bank_sha256"] != "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9" or calibration["zero_phase_reference"] != "independent_train_only_zero_phase" or calibration["plane_identity"] != "pre_ols_plane_power_reference_v1":
        raise ValueError("repaired calibration contract is not locked")
    dataset = _require_mapping(payload["dataset"], "dataset")
    if dataset["root"] != "data" or dataset["mra_manifest_sha256"] != MRA_DATASET_MANIFEST_SHA256 or dataset["horizon_manifest_sha256"] != "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9" or dataset["confirmation_indices_sha256"] != "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a" or dataset["input_shape"] != [1, 28, 28] or dataset["rendered_shape"] != [1, 200, 200]:
        raise ValueError("GAC dataset contract is not locked")
    prior = _require_mapping(payload["prior"], "prior")
    if dict(prior) != {
        "kind": "truncated_normal",
        "mean": 1.0,
        "variance": 0.25,
        "lower": 0.0,
        "unconditional": True,
        "class_conditioned": False,
        "phase": "fixed_calibrated_zero_residual",
        "sample_count": PRIOR_COUNT,
        "seed": PRIOR_SEED,
        "sampler": "bogt.truncated_normal_sample_inverse_cdf",
    }:
        raise ValueError("prior contract is not locked")
    evaluation = _require_mapping(payload["evaluation"], "evaluation")
    g1 = _require_mapping(evaluation["g1"], "evaluation.g1")
    if g1["train_support_count"] != 55000 or g1["heldout_count"] != 5000 or g1["prior_count"] != PRIOR_COUNT or g1["prior_seed"] != PRIOR_SEED or g1["bootstrap_seed"] != BOOTSTRAP_SEED or g1["bootstrap_replicates"] != BOOTSTRAP_REPLICATES or float(g1["confidence_level"]) != CONFIDENCE_LEVEL or g1["distance_metric"] != "exact_normalized_euclidean" or g1["normalization"] != "divide_by_sqrt_100" or g1["approximate_search"] is not False:
        raise ValueError("G1 endpoint contract is not locked")
    mean_safety = _require_mapping(evaluation["mean_safety"], "evaluation.mean_safety")
    if float(mean_safety["lower"]) != MEAN_LOWER or float(mean_safety["upper"]) != MEAN_UPPER:
        raise ValueError("mean safety window is not locked")
    rank_safety = _require_mapping(evaluation["rank_safety"], "evaluation.rank_safety")
    if float(rank_safety["retention"]) != RANK_RETENTION or rank_safety["reference"] != "frozen_parent_rcr_b_mra_context":
        raise ValueError("rank safety contract is not locked")
    authorization = _require_mapping(payload["authorization"], "authorization")
    if authorization["diagnostic_only"] is not True or authorization["training_scope"] != "one_bounded_1000_step_repaired_coordinate_canary" or authorization["training_authorized"] != "bounded_canary_only" or any(authorization[name] is not False for name in ("prior_generation", "decoder_generation", "cmos_generation", "mpcdar", "g2", "formal_promotion")):
        raise ValueError("GAC authorization boundary is invalid")
    if dict(payload["governance"]) != GOVERNANCE:
        raise ValueError("GAC governance boundary is invalid")


def _load_config(path: str | Path) -> ConfigSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GAC config must be a YAML mapping")
    spec = ConfigSpec(target, payload, file_sha256(target), canonical_sha256(payload))
    _assert_config(spec)
    return spec


def _validate_parent_authority(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent = _require_mapping(config["parent"], "parent")
    checkpoint_config = _require_mapping(parent["checkpoint"], "parent.checkpoint")
    if dict(checkpoint_config) != {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": PARENT_CHECKPOINT_SHA256,
        "schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": CANARY_START_STEP,
    }:
        raise ValueError("parent step-5000 checkpoint binding is not locked")
    rcr_binding = _require_mapping(parent["rcr_b"], "parent.rcr_b")
    rcr_config_path, rcr_config, rcr_file_hash, rcr_canonical_hash = rcr_b._load_config(rcr_binding["config"])
    if rcr_file_hash != rcr_binding["config_sha256"] or rcr_canonical_hash != rcr_binding["config_canonical_sha256"]:
        raise ValueError("RCR-B config identity is stale")
    sources = rcr_b._validate_replay_sources(rcr_config)
    rcr_directory = _resolve(rcr_binding["output_directory"])
    manifest = _json_file(rcr_directory / rcr_binding["manifest"], expected_sha256=rcr_binding["manifest_sha256"])
    metrics = _json_file(rcr_directory / rcr_binding["metrics"], expected_sha256=rcr_binding["metrics_sha256"])
    report = _json_file(rcr_directory / rcr_binding["compact_report"], expected_sha256=rcr_binding["compact_report_sha256"])
    if manifest.get("task_id") != "H-LOMA-GEN-RCR-B" or manifest.get("status") != "completed" or manifest.get("gate", {}).get("classification") != rcr_binding["gate"]:
        raise ValueError("RCR-B parent manifest authority is invalid")
    if metrics.get("schema") != rcr_b.METRICS_SCHEMA or metrics.get("task_id") != "H-LOMA-GEN-RCR-B" or metrics.get("status") != "completed" or metrics.get("gate", {}).get("classification") != rcr_binding["gate"]:
        raise ValueError("RCR-B parent support authority is invalid")
    if report.get("task_id") != "H-LOMA-GEN-RCR-B" or report.get("gate") != rcr_binding["gate"]:
        raise ValueError("RCR-B parent report authority is invalid")
    if dict(metrics.get("governance", {})) != rcr_b.GOVERNANCE:
        raise ValueError("RCR-B parent governance boundary is invalid")
    artifact = _require_mapping(_require_mapping(manifest.get("artifacts"), "RCR-B artifacts").get("support_distances.pt"), "RCR-B distance artifact")
    if artifact.get("sha256") != rcr_binding["support_distances_sha256"]:
        raise ValueError("RCR-B support-distance artifact binding is stale")
    baseline = _require_mapping(parent["baseline"], "parent.baseline")
    for name in ("r50", "r95"):
        observed = float(metrics["ratios"][name])
        if observed != float(baseline[name]):
            raise ValueError(f"RCR-B parent {name} authority changed")
    for name in ("r50_ci", "r95_ci"):
        observed = tuple(float(value) for value in metrics["bootstrap"][name])
        expected = tuple(float(value) for value in baseline[name])
        if observed != expected:
            raise ValueError(f"RCR-B parent {name} authority changed")
    if manifest["checkpoint"]["sha256"] != PARENT_CHECKPOINT_SHA256 or manifest["checkpoint"]["global_step"] != CANARY_START_STEP or metrics["learned_state_freeze"]["state_digest_before"] != PARENT_STATE_DIGEST or metrics["learned_state_freeze"]["state_digest_after"] != PARENT_STATE_DIGEST:
        raise ValueError("RCR-B parent learned-state authority changed")
    return {
        "rcr_config_path": rcr_config_path,
        "rcr_config": rcr_config,
        "rcr_file_sha256": rcr_file_hash,
        "rcr_canonical_sha256": rcr_canonical_hash,
        "manifest": manifest,
        "metrics": metrics,
        "report": report,
        "directory": rcr_directory,
        "baseline": baseline,
    }, sources, rcr_config


def _load_parent_checkpoint(
    config: Mapping[str, Any], sources: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, str]:
    checkpoint, path = rcr_b.rcr._load_checkpoint(config["parent"], sources["parent"]["source"])
    observed = file_sha256(path)
    if observed != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint SHA-256 is stale")
    validate_checkpoint_identity(checkpoint, expected_schema="scaled_chen_bogt_horizon_checkpoint.v1", expected_step=CANARY_START_STEP)
    validate_optimizer_contract(checkpoint["optimizer_state"], expected_parameter_count=2, learning_rate=LEARNING_RATE)
    return checkpoint, path, observed


def _load_data(dataset_root: str | Path, config: Mapping[str, Any]) -> CanaryData:
    audit = rcr_b.rcr.mra._load_data(_resolve(dataset_root))
    horizon_data = horizon._load_data(_resolve(dataset_root))
    if audit.manifest["manifest_hash"] != MRA_DATASET_MANIFEST_SHA256:
        raise ValueError("GAC dataset manifest is not registered")
    calibration_indices, parent_schedule = horizon._horizon_schedule(audit.labels)
    if tensor_sha256(torch.sort(calibration_indices).values) != config["calibration"]["calibration_indices_sha256"]:
        raise ValueError("GAC calibration index identity is stale")
    if tensor_sha256(parent_schedule) != horizon_data.manifest["schedule_sha256"] or not torch.equal(parent_schedule, horizon_data.schedule):
        raise ValueError("GAC parent schedule identity is stale")
    schedule_parts: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train = torch.arange(55000, dtype=torch.long)
    for label in range(10):
        candidates = train[audit.labels.index_select(0, train) == label]
        shuffled = candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator))
        available = shuffled[100:]
        repeats = (CANARY_END_STEP + available.numel() - 1) // available.numel()
        schedule_parts.append(available.repeat(repeats)[:CANARY_END_STEP])
    schedule = torch.stack([torch.stack([schedule_parts[label][step] for label in range(10)]) for step in range(CANARY_END_STEP)])
    if not torch.equal(schedule[:CANARY_START_STEP], parent_schedule) or not torch.equal(audit.labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(CANARY_END_STEP)):
        raise ValueError("GAC continuation schedule is not an exact balanced extension")
    return CanaryData(audit, calibration_indices, schedule, parent_schedule)


def _build_training_model(formal: Any, checkpoint: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = bogt.ScaledChenBOGTModel(formal)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model._set_trainable_contract()
    validate_trainable_parameter_names(tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad))
    if any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise RuntimeError("T1_TRAINING_INVALID: OLS became trainable")
    model.train()
    return model


def _load_parent_optimizer(
    model: bogt.ScaledChenBOGTModel,
    checkpoint: Mapping[str, Any],
) -> tuple[torch.optim.Adam, str, str]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if len(parameters) != 2:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED: trainable parameter count changed")
    parent_state = checkpoint["optimizer_state"]
    validate_optimizer_contract(parent_state, expected_parameter_count=2, learning_rate=LEARNING_RATE)
    parent_digest = _optimizer_state_digest(parent_state)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    try:
        optimizer.load_state_dict(copy.deepcopy(parent_state))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED") from exc
    loaded_state = optimizer.state_dict()
    validate_optimizer_contract(loaded_state, expected_parameter_count=2, learning_rate=LEARNING_RATE)
    loaded_digest = _optimizer_state_digest(loaded_state)
    if loaded_digest != parent_digest:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED: loaded state differs from parent")
    if _optimizer_step_values(loaded_state) != [float(CANARY_START_STEP), float(CANARY_START_STEP)]:
        raise RuntimeError("BLOCKED_PARENT_OPTIMIZER_STATE_UNRESOLVED: parent step is not 5000")
    return optimizer, parent_digest, loaded_digest


def _batch_z_telemetry(
    model: bogt.ScaledChenBOGTModel,
    amplitudes: torch.Tensor,
    calibration: Any,
) -> tuple[float, float]:
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(amplitudes.shape[0]):
            output = model(amplitudes[index : index + 1])
            values.append(bogt.normalize_port_power(output.coefficients.abs().square(), calibration.port_scale))
    z = torch.cat(values, dim=0)
    if not torch.isfinite(z).all() or (z < 0).any():
        raise RuntimeError("T1_TRAINING_INVALID: non-finite batch latent telemetry")
    return float(z.mean().detach().cpu()), float(z.var(unbiased=False).detach().cpu())


def _training_step(
    model: bogt.ScaledChenBOGTModel,
    optimizer: torch.optim.Adam,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: Any,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    total, values = bogt._arm_loss(
        model,
        amplitudes,
        targets,
        calibration,
        "mse_kl",
        alpha=ALPHA_REPAIRED,
        gamma=GAMMA,
        mse_weight=1.0,
    )
    if not torch.isfinite(total):
        raise RuntimeError("T1_TRAINING_INVALID: non-finite total loss")
    total.backward()
    gradients: dict[str, float] = {}
    for group_name, prefix in (("encoder", "system.encoder.phase"), ("decoder", "system.decoder.phase")):
        tensors = [parameter.grad for name, parameter in model.named_parameters() if name == prefix]
        if len(tensors) != 1 or tensors[0] is None or not torch.isfinite(tensors[0]).all():
            raise RuntimeError("T1_TRAINING_INVALID: missing or non-finite phase gradient")
        gradients[group_name] = float(torch.linalg.vector_norm(tensors[0].detach().float()).cpu())
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("T1_TRAINING_INVALID: non-finite learned state")
    z_mean, z_variance = _batch_z_telemetry(model, amplitudes, calibration)
    return {
        "mse": float(values["mse"]),
        "bogt": float(values["kl"]),
        "weighted_bogt": float(ALPHA_REPAIRED * values["kl"]),
        "total": float(total.detach().cpu()),
        "encoder_grad_norm": gradients["encoder"],
        "decoder_grad_norm": gradients["decoder"],
        "finite": True,
        "z_mean": z_mean,
        "z_variance": z_variance,
        "optimizer_step": _optimizer_step_values(optimizer.state_dict()),
    }


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> tuple[int, int, int]:
    execution = _require_mapping(config["execution"], "execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["required_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_REPLAY_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    device = torch.device(execution["device"])
    free, total = torch.cuda.mem_get_info(device)
    if free < int(execution["min_free_cuda_bytes"]):
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < int(execution["min_disk_free_bytes"]):
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 10 GiB free disk")
    if output_path.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output_path)}")
    return int(free), int(total), int(disk.free)


def _load_calibration(sources: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    calibration = sources["r2"]["calibration"]
    expected = config["calibration"]
    if calibration.schema != expected["schema"] or calibration.contract_version != expected["contract_version"] or sources["r2"]["payload_sha256"] != expected["payload_sha256"]:
        raise RuntimeError("T1_TRAINING_INVALID: repaired calibration binding failed")
    calibration.validate()
    return calibration


def _validate_preflight_evidence(
    path: str | Path,
    spec: ConfigSpec,
    state: Mapping[str, Any],
    checkpoint_hash: str,
) -> dict[str, Any]:
    payload = _json_file(_resolve(path))
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("status") != "passed" or payload.get("config", {}).get("file_sha256") != spec.file_sha256 or payload.get("config", {}).get("canonical_sha256") != spec.canonical_sha256 or payload.get("git", {}).get("commit") != state.get("commit") or payload.get("git", {}).get("clean") is not True or payload.get("parent_checkpoint", {}).get("sha256") != checkpoint_hash or payload.get("objective", {}).get("alpha") != ALPHA_REPAIRED or payload.get("objective", {}).get("gamma") != GAMMA:
        raise RuntimeError("T1_TRAINING_INVALID: preflight evidence is stale or unauthorized")
    gates = payload.get("gates")
    if not isinstance(gates, Sequence) or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in gates):
        raise RuntimeError("T1_TRAINING_INVALID: preflight gates are not passed")
    negative = _require_mapping(payload.get("negative_control"), "preflight negative control")
    if negative.get("expected_decision") != "reject_primary_gate" or negative.get("passed") is not True:
        raise RuntimeError("T1_TRAINING_INVALID: preflight negative control is invalid")
    return payload


def _load_end_model(formal: Any, checkpoint: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = rcr_b.rcr.mra._load_scaled_model(formal, checkpoint, device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RuntimeError("T1_TRAINING_INVALID: end checkpoint is not frozen")
    return model


def _mra_end_metrics(
    model: bogt.ScaledChenBOGTModel,
    data: CanaryData,
    calibration: Any,
    device: torch.device,
    calibration_hash: str,
) -> dict[str, Any]:
    audit = data.audit
    selection_coefficients = rcr_b.rcr._collect_coefficients(
        model,
        audit.test_images,
        audit.selection_indices,
        device=device,
        batch_size=BATCH_SIZE,
    )
    confirmation_coefficients = rcr_b.rcr._collect_coefficients(
        model,
        audit.images,
        audit.confirmation_indices,
        device=device,
        batch_size=BATCH_SIZE,
    )
    selection_labels = audit.test_labels.index_select(0, audit.selection_indices)
    confirmation_labels = audit.labels.index_select(0, audit.confirmation_indices)
    result = {
        "schema": MRA_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "coordinate": {
            "schema": calibration.schema,
            "contract_version": calibration.contract_version,
            "definition": "z_j=abs(c_j)^2/s_j_v2",
            "dimension": LATENT_DIM,
            "calibration_sha256": calibration_hash,
        },
        "selection": rcr_b.rcr.coordinate_metrics(selection_coefficients, selection_labels, calibration),
        "confirmation": rcr_b.rcr.coordinate_metrics(confirmation_coefficients, confirmation_labels, calibration),
        "panel_identity": {
            "selection_indices_sha256": tensor_sha256(audit.selection_indices),
            "confirmation_indices_sha256": tensor_sha256(audit.confirmation_indices),
            "dataset_manifest_sha256": audit.manifest["manifest_hash"],
            "panel_reused_without_resampling": True,
        },
        "evaluation": {"no_update": True, "decoder_generation": False, "prior_generation": False},
    }
    del selection_coefficients, confirmation_coefficients
    return result


def _mean_safety(metrics: Mapping[str, Any]) -> dict[str, Any]:
    values = {split: float(_require_mapping(metrics[split], split)["z_mean"]) for split in ("selection", "confirmation")}
    passed = all(math.isfinite(value) and MEAN_LOWER <= value <= MEAN_UPPER for value in values.values())
    return {"lower": MEAN_LOWER, "upper": MEAN_UPPER, "values": values, "passed": passed}


def _rank_safety(metrics: Mapping[str, Any], parent_metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"retention": RANK_RETENTION, "reference": "frozen_parent_rcr_b_mra_context", "panels": {}}
    passed = True
    for split in ("selection", "confirmation"):
        end = _require_mapping(metrics[split], split)
        parent = _require_mapping(_require_mapping(parent_metrics["mra_context"], "parent mra_context")[split], f"parent {split}")
        checks = {}
        for name in ("complex_effective_rank", "power_effective_rank"):
            end_value = float(end[name])
            parent_value = float(parent[name])
            ok = math.isfinite(end_value) and end_value >= RANK_RETENTION * parent_value
            checks[name] = {"end": end_value, "parent": parent_value, "threshold": RANK_RETENTION * parent_value, "passed": ok}
            passed = passed and ok
        result["panels"][split] = checks
    result["passed"] = passed
    return result


def _current_sample_safety(
    model: bogt.ScaledChenBOGTModel,
    data: CanaryData,
    calibration: Any,
    device: torch.device,
    calibration_hash: str,
) -> dict[str, Any]:
    research_data = bogt.ResearchData(
        data.audit.images,
        data.audit.labels,
        data.calibration_indices,
        data.parent_schedule,
        data.audit.confirmation_indices,
        tuple(data.parent_schedule[index].clone() for index in range(3)),
        {"manifest_hash": data.audit.manifest["manifest_hash"]},
    )
    templates = bogt._class_templates(research_data)
    evaluation = bogt._evaluate_model(
        model,
        research_data,
        calibration,
        data.audit.confirmation_indices,
        device,
        templates,
        calibration_hash,
        split="heldout",
    )["metrics"]
    paired = float(evaluation["paired_mse"])
    cyclic = float(evaluation["cyclic_mse"])
    template = float(evaluation["template_control_mse"])
    passed = all(math.isfinite(value) for value in (paired, cyclic, template)) and paired < cyclic and paired < template
    return {
        "route": "confirmation_same_route_full_forward",
        "paired_mse": paired,
        "cyclic_mse": cyclic,
        "template_control_mse": template,
        "passed": passed,
        "evaluation_finite": bool(evaluation["finite"]),
    }


def _g1_end_metrics(
    model: bogt.ScaledChenBOGTModel,
    data: CanaryData,
    calibration: Any,
    device: torch.device,
    config: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    audit = data.audit
    train_indices = torch.arange(0, 55000, dtype=torch.long)
    heldout_indices = audit.confirmation_indices.cpu().contiguous()
    train_coefficients = rcr_b.rcr._collect_coefficients(model, audit.images, train_indices, device=device, batch_size=BATCH_SIZE)
    heldout_coefficients = rcr_b.rcr._collect_coefficients(model, audit.images, heldout_indices, device=device, batch_size=BATCH_SIZE)
    train_z = bogt.normalize_port_power(train_coefficients.abs().square(), calibration.port_scale).detach().cpu().contiguous()
    heldout_z = bogt.normalize_port_power(heldout_coefficients.abs().square(), calibration.port_scale).detach().cpu().contiguous()
    prior_z = rcr_b.sample_registered_prior(PRIOR_COUNT, seed=PRIOR_SEED)
    expected_prior_hash = parent_metrics["prior"]["sample_tensor_sha256"]
    if tensor_sha256(prior_z) != expected_prior_hash or float(prior_z.mean()) != float(parent_metrics["prior"]["empirical_mean"]):
        raise RuntimeError("T1_TRAINING_INVALID: registered prior sample identity changed")
    distance = _require_mapping(config["evaluation"]["g1"], "evaluation.g1")
    support = train_z.to(device)
    heldout_query = heldout_z.to(device)
    prior_query = prior_z.to(device)
    holdout_distances = rcr_b.exact_normalized_nearest_distances(
        heldout_query,
        support,
        query_chunk_size=int(distance["query_chunk_size"]),
        support_chunk_size=int(distance["support_chunk_size"]),
    ).cpu()
    prior_distances = rcr_b.exact_normalized_nearest_distances(
        prior_query,
        support,
        query_chunk_size=int(distance["query_chunk_size"]),
        support_chunk_size=int(distance["support_chunk_size"]),
    ).cpu()
    bootstrap = rcr_b.bootstrap_ratio_ci(
        holdout_distances,
        prior_distances,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence_level=CONFIDENCE_LEVEL,
    )
    result = {
        "schema": G1_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "coordinate": {
            "schema": calibration.schema,
            "contract_version": calibration.contract_version,
            "definition": "z_j=abs(c_j)^2/s_j_v2",
            "dimension": LATENT_DIM,
            "normalization": "divide_by_sqrt_100",
        },
        "dataset": {
            "train_support_start": 0,
            "train_support_count": 55000,
            "heldout_count": 5000,
            "heldout_indices_sha256": tensor_sha256(heldout_indices),
            "dataset_manifest_sha256": audit.manifest["manifest_hash"],
        },
        "prior": {
            "count": PRIOR_COUNT,
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
        "endpoint_primary_inputs": {
            "r50_upper": float(bootstrap["r50_ci"][1]),
            "r95_upper": float(bootstrap["r95_ci"][1]),
            "parent_r50_ci_lower": PARENT_R50_LOWER,
            "parent_r95_ci_lower": PARENT_R95_LOWER,
        },
    }
    del train_coefficients, heldout_coefficients, train_z, heldout_z, support, heldout_query, prior_query
    return result


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "training_manifest.json":
            refs[target.name] = {"path": target.name, "sha256": file_sha256(target), "size_bytes": target.stat().st_size}
    return refs


def _assert_output_tree(output_path: Path) -> None:
    expected = {
        "training_manifest.json",
        "objective_contract.json",
        "training_metrics.json",
        "end_checkpoint.pt",
        "end_checkpoint_identity.json",
        "mra_end_metrics.json",
        "g1_end_support_metrics.json",
        "parent_end_comparison.json",
        "safety_metrics.json",
        "compact_report.json",
    }
    actual = {path.relative_to(output_path).as_posix() for path in output_path.rglob("*") if path.is_file()}
    if actual != expected or any(Path(name).name.startswith(".") or ".tmp" in name or name.lower().endswith((".tif", ".tiff")) for name in actual):
        raise RuntimeError(f"GAC output tree is not exact: {sorted(actual)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="T1-GAC artifact")


def preflight(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_PREFLIGHT_OUTPUT_PATH,
) -> dict[str, Any]:
    spec = _load_config(config_path)
    state = base._git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: expected clean branch {BRANCH}; status={state.get('status')}")
    parent_authority, sources, rcr_config = _validate_parent_authority(spec.payload)
    checkpoint, checkpoint_path, checkpoint_hash = _load_parent_checkpoint(spec.payload, sources)
    formal = sources["parent"]["formal"]
    data = _load_data(spec.payload["dataset"]["root"], spec.payload)
    calibration = _load_calibration(sources, spec.payload)
    actual_output = next_write_once_output(_resolve(output_path))
    free_before, total_cuda, disk_free_before = _resource_preflight(spec.payload, actual_output)
    device = torch.device(spec.payload["execution"]["device"])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = _build_training_model(formal, checkpoint, device)
    optimizer, parent_optimizer_digest, loaded_optimizer_digest = _load_parent_optimizer(model, checkpoint)
    state_before = rcr_b.state_digest(model)
    initial_ols = _cpu_state_dict(model.ols.state_dict())
    history: list[dict[str, Any]] = []
    for logical_step in range(CANARY_START_STEP + 1, CANARY_START_STEP + int(spec.payload["execution"]["preflight_steps"]) + 1):
        rendered = base.render_intensity_batch(data.audit.images.index_select(0, data.schedule[logical_step - 1]))
        row = _training_step(model, optimizer, rendered.amplitude.to(device), rendered.target.to(device), calibration)
        row["logical_step"] = logical_step
        history.append(row)
        del rendered
    state_after = rcr_b.state_digest(model)
    final_ols = _cpu_state_dict(model.ols.state_dict())
    changes = {"ols_unchanged": all(torch.equal(initial_ols[name], final_ols[name]) for name in initial_ols)}
    negative_pass = not classify_primary_gate(parent_authority["baseline"]["r50_ci"], parent_authority["baseline"]["r95_ci"], PARENT_R50_LOWER, PARENT_R95_LOWER)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    gates = [
        _gate("three_step_real_update", len(history) == 3 and state_after != state_before and all(row["finite"] for row in history), state_before=state_before, state_after=state_after),
        _gate("registered_trainable_scope", tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad) == ("system.encoder.phase", "system.decoder.phase")),
        _gate("parent_optimizer_continuation", parent_optimizer_digest == loaded_optimizer_digest and _optimizer_step_values(optimizer.state_dict()) == [5003.0, 5003.0], parent_digest=parent_optimizer_digest, loaded_digest=loaded_optimizer_digest),
        _gate("frozen_ols", changes["ols_unchanged"]),
        _gate("finite_values", all(torch.isfinite(parameter).all() for parameter in model.parameters())),
        _gate("cuda_peak_memory", peak <= int(spec.payload["execution"]["max_peak_cuda_bytes"]), bytes=peak),
        _gate("preflight_wall_time", elapsed <= float(spec.payload["execution"]["preflight_max_seconds"]), seconds=elapsed),
    ]
    payload = {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "passed" if all(item["passed"] for item in gates) and negative_pass else "invalid_harness",
        "git": state,
        "config": {"path": _relative_path(spec.path), "file_sha256": spec.file_sha256, "canonical_sha256": spec.canonical_sha256},
        "parent_checkpoint": {"path": _relative_path(checkpoint_path), "sha256": checkpoint_hash, "step": CANARY_START_STEP},
        "objective": {"arm": "mse_kl", "alpha": ALPHA_REPAIRED, "gamma": GAMMA, "old_alpha_rejected": True},
        "schedule": {"seed": SEED, "shape": list(data.schedule.shape), "parent_schedule_sha256": tensor_sha256(data.parent_schedule), "continuation_schedule_sha256": tensor_sha256(data.schedule), "steps_exercised": len(history)},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "parent_state_digest": parent_optimizer_digest, "loaded_state_digest": loaded_optimizer_digest, "telemetry": _optimizer_telemetry(optimizer)},
        "gates": gates,
        "negative_control": {"kind": "frozen_parent_rcr_b_endpoint", "expected_decision": "reject_primary_gate", "primary_gate": False, "passed": negative_pass},
        "recovery": {"claimed": False, "interrupted_resumed_segment": False},
        "history": history,
        "resource": {"device": str(device), "free_cuda_bytes_before": free_before, "total_cuda_bytes": total_cuda, "disk_free_bytes_before": disk_free_before, "peak_cuda_allocated_bytes": peak, "wall_clock_seconds": elapsed},
        "authorization": {"training_scope": "preflight_only_for_one_bounded_1000_step_canary", "formal_training_started": False, "checkpoint_created": False, "generation": False},
    }
    actual_output.mkdir(parents=True, exist_ok=False)
    _write_json(actual_output / "preflight.json", payload)
    if {path.name for path in actual_output.iterdir()} != {"preflight.json"}:
        raise RuntimeError("GAC preflight output tree is not exact")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return {"output_directory": str(actual_output), "payload": payload}


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    preflight_evidence: str | Path | None = None,
) -> dict[str, Any]:
    if preflight_evidence is None:
        raise ValueError("GAC formal run requires --preflight-evidence")
    spec = _load_config(config_path)
    state = base._git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: expected clean branch {BRANCH}; status={state.get('status')}")
    parent_authority, sources, rcr_config = _validate_parent_authority(spec.payload)
    checkpoint, checkpoint_path, checkpoint_hash = _load_parent_checkpoint(spec.payload, sources)
    _validate_preflight_evidence(preflight_evidence, spec, state, checkpoint_hash)
    formal = sources["parent"]["formal"]
    data = _load_data(spec.payload["dataset"]["root"], spec.payload)
    calibration = _load_calibration(sources, spec.payload)
    actual_output = next_write_once_output(_resolve(output_path))
    free_before, total_cuda, disk_free_before = _resource_preflight(spec.payload, actual_output)
    device = torch.device(spec.payload["execution"]["device"])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = _build_training_model(formal, checkpoint, device)
    start_state_digest = rcr_b.state_digest(model)
    if start_state_digest != PARENT_STATE_DIGEST:
        raise RuntimeError("T1_TRAINING_INVALID: parent learned-state digest mismatch")
    start_state = _cpu_state_dict(model.state_dict())
    optimizer, parent_optimizer_digest, loaded_optimizer_digest = _load_parent_optimizer(model, checkpoint)
    history: list[dict[str, Any]] = []
    validate_fixed_end_policy(CANARY_START_STEP, CANARY_END_STEP, CANARY_STEPS)
    for logical_step in range(CANARY_START_STEP + 1, CANARY_END_STEP + 1):
        rendered = base.render_intensity_batch(data.audit.images.index_select(0, data.schedule[logical_step - 1]))
        row = _training_step(model, optimizer, rendered.amplitude.to(device), rendered.target.to(device), calibration)
        row["logical_step"] = logical_step
        history.append(row)
        del rendered
    torch.cuda.synchronize(device)
    end_state_digest = rcr_b.state_digest(model)
    end_state = _cpu_state_dict(model.state_dict())
    changes = _model_state_changes(start_state, end_state)
    if end_state_digest == start_state_digest or not changes["only_registered_trainable_parameters_changed"] or not changes["ols_unchanged"]:
        raise RuntimeError("T1_TRAINING_INVALID: learned-state identity or frozen-state contract failed")
    optimizer_end = _optimizer_telemetry(optimizer)
    if optimizer_end["state_steps"] != [float(CANARY_END_STEP), float(CANARY_END_STEP)]:
        raise RuntimeError("T1_TRAINING_INVALID: optimizer step did not reach 6000")
    actual_output.mkdir(parents=True, exist_ok=False)
    end_checkpoint_payload = {
        "schema": END_CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "global_step": CANARY_END_STEP,
        "parent_checkpoint": {"step": CANARY_START_STEP, "sha256": checkpoint_hash},
        "model_state": end_state,
        "optimizer_state": _cpu_nested(optimizer.state_dict()),
        "optimizer_identity": {"name": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest, "end_state_digest": optimizer_end["state_digest"]},
        "trainable_parameters": ["system.encoder.phase", "system.decoder.phase"],
        "endpoint_fixed_before_training": True,
    }
    atomic_torch_save_new(actual_output / "end_checkpoint.pt", end_checkpoint_payload, artifact_kind="T1-GAC end checkpoint")
    end_checkpoint_hash = file_sha256(actual_output / "end_checkpoint.pt")
    loaded_end_checkpoint = safe_torch_load(actual_output / "end_checkpoint.pt", expected_sha256=end_checkpoint_hash, map_location="cpu")
    validate_checkpoint_identity(loaded_end_checkpoint, expected_schema=END_CHECKPOINT_SCHEMA, expected_step=CANARY_END_STEP)
    eval_model = _load_end_model(formal, loaded_end_checkpoint, device)
    eval_state_before = rcr_b.state_digest(eval_model)
    calibration_hash = spec.payload["calibration"]["payload_sha256"]
    mra_metrics = _mra_end_metrics(eval_model, data, calibration, device, calibration_hash)
    current_sample_safety = _current_sample_safety(eval_model, data, calibration, device, calibration_hash)
    g1_metrics = _g1_end_metrics(eval_model, data, calibration, device, spec.payload, parent_authority["metrics"])
    eval_state_after = rcr_b.state_digest(eval_model)
    if eval_state_before != eval_state_after or eval_state_before != end_state_digest:
        raise RuntimeError("T1_TRAINING_INVALID: end evaluation changed learned state")
    mean_safety = _mean_safety(mra_metrics)
    rank_safety = _rank_safety(mra_metrics, parent_authority["metrics"])
    primary_pass = classify_primary_gate(g1_metrics["bootstrap"]["r50_ci"], g1_metrics["bootstrap"]["r95_ci"], PARENT_R50_LOWER, PARENT_R95_LOWER)
    ratio_worsened = float(g1_metrics["ratios"]["r50"]) > float(parent_authority["metrics"]["ratios"]["r50"]) and float(g1_metrics["ratios"]["r95"]) > float(parent_authority["metrics"]["ratios"]["r95"])
    final_classification = classify_final_result(primary_pass, mean_safety["passed"], rank_safety["passed"], current_sample_safety["passed"], ratio_worsened=ratio_worsened)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    training_metrics = {
        "schema": TRAINING_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "endpoint": {"start_step": CANARY_START_STEP, "end_step": CANARY_END_STEP, "steps": CANARY_STEPS, "selection": "fixed_before_training_no_best_selection"},
        "objective": {"arm": "mse_kl", "mse_weight": 1.0, "alpha": ALPHA_REPAIRED, "gamma": GAMMA, "old_alpha": OLD_ALPHA, "weighted_term": "alpha_times_bogt"},
        "history": history,
        "summary": {"start": history[0], "end": history[-1], "mean_mse": float(sum(row["mse"] for row in history) / len(history)), "mean_bogt": float(sum(row["bogt"] for row in history) / len(history)), "mean_weighted_bogt": float(sum(row["weighted_bogt"] for row in history) / len(history)), "mean_total": float(sum(row["total"] for row in history) / len(history))},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest, "loaded_state_digest": loaded_optimizer_digest, "end": optimizer_end},
        "learned_state": {"start_digest": start_state_digest, "end_digest": end_state_digest, "changed_names": changes["changed_names"], "only_registered_trainable_parameters_changed": changes["only_registered_trainable_parameters_changed"], "ols_unchanged": changes["ols_unchanged"]},
        "schedule": {"seed": SEED, "parent_schedule_sha256": tensor_sha256(data.parent_schedule), "continuation_schedule_sha256": tensor_sha256(data.schedule), "shape": list(data.schedule.shape)},
        "resource": {"device": str(device), "free_cuda_bytes_before": free_before, "total_cuda_bytes": total_cuda, "disk_free_bytes_before": disk_free_before, "peak_cuda_allocated_bytes": peak, "wall_clock_seconds": elapsed},
    }
    mra_metrics["evaluation"].update({"state_digest_before": eval_state_before, "state_digest_after": eval_state_after, "no_update": True})
    primary = {
        "name": "SUPPORT_ALIGNMENT_DIRECTION_SUPPORTED",
        "passed": primary_pass,
        "rule": "upper95(end R50) < parent R50 CI lower AND upper95(end R95) < parent R95 CI lower",
        "end_r50_ci": g1_metrics["bootstrap"]["r50_ci"],
        "end_r95_ci": g1_metrics["bootstrap"]["r95_ci"],
        "parent_r50_ci_lower": PARENT_R50_LOWER,
        "parent_r95_ci_lower": PARENT_R95_LOWER,
    }
    parent_comparison = {
        "schema": "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_parent_comparison.v1",
        "task_id": TASK_ID,
        "parent": {"attempt": "lightgen_scaled_chen_g1_prior_support_accessibility_repaired_coordinate_v1_attempt_01", "r50": parent_authority["metrics"]["ratios"]["r50"], "r50_ci": parent_authority["metrics"]["bootstrap"]["r50_ci"], "r95": parent_authority["metrics"]["ratios"]["r95"], "r95_ci": parent_authority["metrics"]["bootstrap"]["r95_ci"]},
        "end": {"r50": g1_metrics["ratios"]["r50"], "r50_ci": g1_metrics["bootstrap"]["r50_ci"], "r95": g1_metrics["ratios"]["r95"], "r95_ci": g1_metrics["bootstrap"]["r95_ci"]},
        "primary_gate": primary,
        "mra": {split: {name: {"parent": float(parent_authority["metrics"]["mra_context"][split][name]), "end": float(mra_metrics[split][name])} for name in ("z_mean", "z_variance", "gaussian_kl", "numerical_truncated_normal_kl", "complex_effective_rank", "power_effective_rank")} for split in ("selection", "confirmation")},
        "mse_parent_relative": "descriptive_only_no_registered_percentage_gate",
    }
    safety_metrics = {
        "schema": SAFETY_SCHEMA,
        "task_id": TASK_ID,
        "mean_scale": mean_safety,
        "rank": rank_safety,
        "current_sample": current_sample_safety,
        "ratio_worsened": ratio_worsened,
        "finite_training": all(row["finite"] for row in history),
        "end_evaluation_no_update": eval_state_before == eval_state_after,
        "passed": mean_safety["passed"] and rank_safety["passed"] and current_sample_safety["passed"] and not ratio_worsened,
    }
    compact_report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "completed",
        "gate": final_classification,
        "primary_gate": primary,
        "end": {"r50": g1_metrics["ratios"]["r50"], "r50_ci": g1_metrics["bootstrap"]["r50_ci"], "r95": g1_metrics["ratios"]["r95"], "r95_ci": g1_metrics["bootstrap"]["r95_ci"]},
        "parent": parent_comparison["parent"],
        "safety": safety_metrics,
        "interpretation": "bounded training-direction evidence only; no generative manifold, prior decoder accessibility, generation, or formal promotion authority is established",
        "governance": GOVERNANCE,
        "next_action": {T1_ALIGNMENT_DIRECTION_SUPPORTED: "REVIEW_GENERATIVE_ALIGNMENT_CANARY", T1_ALIGNMENT_NOT_SUPPORTED: "PLAN_ALIGNMENT_OBJECTIVE_MECHANISM_REVIEW", T1_ALIGNMENT_HARMFUL: "REVIEW_ALIGNMENT_TRADEOFF_FAILURE"}[final_classification],
    }
    objective_contract = {
        "schema": "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_objective.v1",
        "task_id": TASK_ID,
        "parent_checkpoint": {"path": _relative_path(checkpoint_path), "sha256": checkpoint_hash, "step": CANARY_START_STEP},
        "calibration": {"schema": calibration.schema, "contract_version": calibration.contract_version, "payload_sha256": calibration_hash, "mode_bank_sha256": spec.payload["calibration"]["mode_bank_sha256"]},
        "prior": dict(spec.payload["prior"]),
        "objective": {"arm": "mse_kl", "formula": "L_total=L_MSE+alpha_repaired*L_BOGT", "alpha_repaired": ALPHA_REPAIRED, "gamma": GAMMA, "old_alpha_rejected": OLD_ALPHA},
        "optimizer": {"identity": "Adam", "learning_rate": LEARNING_RATE, "scheduler": "none", "parent_state_digest": parent_optimizer_digest},
        "trainable_parameters": ["system.encoder.phase", "system.decoder.phase"],
        "horizon": {"start_step": CANARY_START_STEP, "end_step": CANARY_END_STEP, "steps": CANARY_STEPS, "endpoint_fixed_before_training": True},
        "no_generation": True,
    }
    end_identity = {
        "schema": "lightgen_scaled_chen_repaired_coordinate_generative_alignment_canary_end_checkpoint_identity.v1",
        "task_id": TASK_ID,
        "path": "end_checkpoint.pt",
        "sha256": end_checkpoint_hash,
        "schema_bound": END_CHECKPOINT_SCHEMA,
        "global_step": CANARY_END_STEP,
        "parent_checkpoint_sha256": checkpoint_hash,
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
    _write_json(actual_output / "g1_end_support_metrics.json", g1_metrics)
    _write_json(actual_output / "parent_end_comparison.json", parent_comparison)
    _write_json(actual_output / "safety_metrics.json", safety_metrics)
    _write_json(actual_output / "compact_report.json", compact_report)
    storage = sum(path.stat().st_size for path in actual_output.rglob("*") if path.is_file())
    if storage > int(spec.payload["execution"]["max_storage_bytes"]):
        raise RuntimeError("T1_TRAINING_INVALID: GAC output exceeds registered storage bound")
    manifest = {
        "schema": TRAINING_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "attempt": actual_output.name,
        "output_directory": str(actual_output),
        "git": state,
        "config": {"path": _relative_path(spec.path), "file_sha256": spec.file_sha256, "canonical_sha256": spec.canonical_sha256},
        "parent": {"rcr_b_config": _relative_path(parent_authority["rcr_config_path"]), "rcr_b_config_sha256": parent_authority["rcr_file_sha256"], "rcr_b_config_canonical_sha256": parent_authority["rcr_canonical_sha256"], "rcr_b_output": _relative_path(parent_authority["directory"]), "rcr_b_manifest_sha256": file_sha256(parent_authority["directory"] / "observer_manifest.json"), "rcr_b_metrics_sha256": file_sha256(parent_authority["directory"] / "support_metrics.json"), "checkpoint": end_identity["parent_checkpoint_sha256"], "step": CANARY_START_STEP},
        "calibration": dict(objective_contract["calibration"]),
        "prior": dict(spec.payload["prior"]),
        "training": {"start_step": CANARY_START_STEP, "end_step": CANARY_END_STEP, "steps": CANARY_STEPS, "objective": objective_contract["objective"], "optimizer": objective_contract["optimizer"], "trainable_parameters": objective_contract["trainable_parameters"]},
        "evaluation": {"mra": "mra_end_metrics.json", "g1": "g1_end_support_metrics.json", "safety": "safety_metrics.json", "no_optimizer_step_after_end_checkpoint": True},
        "gate": final_classification,
        "primary_gate": primary,
        "governance": GOVERNANCE,
        "next_action": compact_report["next_action"],
        "storage_bytes": storage,
        "artifact_refs": _artifact_refs(actual_output),
    }
    _write_json(actual_output / "training_manifest.json", manifest)
    _assert_output_tree(actual_output)
    result = {"output_directory": str(actual_output), "manifest": manifest, "report": compact_report, "training_metrics": training_metrics, "mra": mra_metrics, "g1": g1_metrics, "safety": safety_metrics}
    print(json.dumps(compact_report, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--preflight-evidence", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "preflight":
        preflight(args.config, args.output or DEFAULT_PREFLIGHT_OUTPUT_PATH)
        return 0
    run(args.config, args.output or DEFAULT_OUTPUT_PATH, args.preflight_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
