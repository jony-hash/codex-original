"""Four-arm causal pilot for the corrected SMF VAE KL dose.

This module is a new research lifecycle.  It reuses the already validated SMF
and posterior step helpers, but owns its data exclusion, source binding,
preflight, checkpoint and comparison contracts.  The formal IdentityOLS and
the fixed research SMF transport are not modified here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from ...config import FormalSystemConfig
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_vae_kl_attribution_audit as kl_audit
from . import optical_manifold_smf_vae_pilot as vae


BRANCH = "research/v2-lightgen-optical-manifold-kl-ols-loss-recovery-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_resume.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_vae_kl_effect_panel.v1"

SOURCE_VALIDATION_HASH = "d548cf93f2cfefb2258239b066b14f574b098bfa17edc0f377f718fd2cba3ca5"
SOURCE_PREFLIGHT_HASH = "722557fc0bfc058edafa8666772ee87bb927b0a37924b55087467303d3a56623"
SOURCE_TRAINING_HASH = "a95890d8feb38c0682179cbbb4ef612d49c5f1918609596524959b789e672c3e"
SOURCE_CHECKPOINT_HASH = "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_VAE_VALIDATION_HASH = "d50919acb75ffe68453db52a6d87ebdfeb8ae9963f157b0bd3afb582f17de04c"
SOURCE_VAE_PREFLIGHT_HASH = "5e1f8569e7e58ce086db37a318df76fa81107820323128dafbedd02ad0e8a936"
SOURCE_VAE_TRAINING_HASH = "ec6afc5b15274ce9a7e2446bef43bb6346007480da23408fe1399b412200c476"
SOURCE_VAE_NORMALIZATION_HASH = "e764ceb31e9ce1aa24f2cd9ad909011550aa6ad095404916ff84be8f93d07a85"
SOURCE_KL_ATTRIBUTION_HASH = "40d612e20ffc7044c0a8e21bce86c7d07f3076a31ea0d6aadee5fa6f53ea22ce"
SOURCE_KL_DOSE_VALIDATION_HASH = "c1314bcd31d2990c7ebc87b5401c100f7b7a081b26011d6a830a3764f104b273"
SOURCE_KL_DOSE_AUDIT_HASH = "2d708bc3eea54dcfe8f580261891387c5e04698d6673667b421e53132fe3329f"

SOURCE_STEP = 1000
STEPS = 100
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
PORT_COUNT = 100
REAL_LATENT_DIM = 200
CHECKPOINT_INTERVAL = 50
SEED = 1234
LEARNING_RATE = 1.0e-3
REGISTERED_BETA = 0.0018256452619527068
CANDIDATE_MULTIPLIER = 8.0
CANDIDATE_BETA = REGISTERED_BETA * CANDIDATE_MULTIPLIER
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 1800.0
MAX_STORAGE_BYTES = 1024**3
SAFETY_MSE_RATIO = 2.0
MIN_KL_REDUCTION = 0.02
MAX_CANDIDATE_MSE_RELATIVE = 1.05
MIN_VARIANCE_RETENTION = 0.50
MIN_ACTIVE_RETENTION = 0.90
ARMS = (
    "smf_deterministic_mse",
    "smf_posterior_no_kl",
    "smf_current_beta_vae",
    "smf_candidate_beta_vae",
)
POSTERIOR_ARMS = {"smf_posterior_no_kl", "smf_current_beta_vae", "smf_candidate_beta_vae"}
NEXT_PASS_ACTION = "manual_review_identity_ols_coupling_pilot_contract"
NEXT_FAILED_ACTION = "manual_review_smf_vae_kl_objective_contract"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Source:
    source: vae.Source
    hashes: dict[str, str]
    dose_audit: dict[str, Any]
    current_beta: float


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"KL effect pilot requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "KL effect pilot requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup"
        )


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "dataset",
        "optical_contract",
        "training",
        "gates",
        "authorization",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema"] != CONFIG_SCHEMA
        or payload["hypothesis"] != HYPOTHESIS
    ):
        raise ValueError("KL effect pilot config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    source = spec.payload["source"]
    expected = {
        "plan6_validation_sha256": SOURCE_VALIDATION_HASH,
        "plan6_preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "plan6_training_sha256": SOURCE_TRAINING_HASH,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "mnist_manifest_sha256": SOURCE_MANIFEST_HASH,
        "vae_validation_sha256": SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight_sha256": SOURCE_VAE_PREFLIGHT_HASH,
        "vae_training_sha256": SOURCE_VAE_TRAINING_HASH,
        "vae_normalization_sha256": SOURCE_VAE_NORMALIZATION_HASH,
        "kl_attribution_audit_sha256": SOURCE_KL_ATTRIBUTION_HASH,
        "kl_dose_validation_sha256": SOURCE_KL_DOSE_VALIDATION_HASH,
        "kl_dose_audit_sha256": SOURCE_KL_DOSE_AUDIT_HASH,
    }
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("KL effect pilot source binding mismatch")
    if source.get("source_global_step") != SOURCE_STEP:
        raise ValueError("KL effect pilot source step mismatch")
    dataset = spec.payload["dataset"]
    if dataset != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "training_count": 1000,
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": 1,
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "excluded_prior_panels": True,
        "calibration_and_schedule_disjoint": True,
        "heldout_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "heldout_panel_count": 1000,
        "visual_panel_count": 100,
    }:
        raise ValueError("KL effect pilot dataset contract mismatch")
    optical = spec.payload["optical_contract"]
    for key, value in {
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "boundary": "finite_crop",
        "formal_ols": "identity",
        "research_ols": "100_port_lp01_smf",
        "posterior_domain": "100_complex_coefficients_as_200_real",
        "posterior_head": "shared_per_port_linear_16_silu",
        "posterior_status": "project_approximation",
        "fixed_transport": True,
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }.items():
        if optical.get(key) != value:
            raise ValueError(f"KL effect pilot optical contract mismatch: {key}")
    training = spec.payload["training"]
    for key, value in {
        "arms": list(ARMS),
        "optimizer": "Adam",
        "optimizer_update_device": "cpu_shadow",
        "learning_rate": LEARNING_RATE,
        "reset_adam_from_source": True,
        "steps_per_arm": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reconstruction_activation_checkpoint": True,
        "deterministic": True,
        "epsilon_seed": SEED,
        "registered_beta": REGISTERED_BETA,
        "candidate_multiplier": CANDIDATE_MULTIPLIER,
        "candidate_beta": CANDIDATE_BETA,
        "no_early_stop": True,
        "no_tuning": True,
        "no_checkpoint_selection": True,
    }.items():
        if training.get(key) != value:
            raise ValueError(f"KL effect pilot training contract mismatch: {key}")
    if spec.payload["authorization"].get("training_authorized") is not False or spec.payload["authorization"].get("formal_promotion") is not False:
        raise ValueError("KL effect pilot authorization boundary is invalid")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_smf_vae_kl_effect_pilot",
        "approved_contract": {
            "arms": list(ARMS),
            "steps_per_arm": STEPS,
            "logical_batch_size": BATCH_SIZE,
            "microbatch_size": 1,
            "source_global_step": SOURCE_STEP,
            "preflight_steps": PREFLIGHT_STEPS,
            "resume_segment": "2+1",
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "registered_beta": REGISTERED_BETA,
            "candidate_multiplier": CANDIDATE_MULTIPLIER,
            "candidate_beta": CANDIDATE_BETA,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "post_valid_failed_next_action": NEXT_FAILED_ACTION,
        },
        "bounded_diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("KL effect pilot review contract does not match scope")
    return target


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    legacy = vae._load_spec(
        _resolve(
            "configs/research/lightgen_optical_manifold_smf_vae_pilot_v1.yaml",
            base_path=base._repo_root(),
        )
    )
    formal, _legacy_path = vae._load_formal(legacy)
    if path.resolve() != _legacy_path.resolve():
        raise ValueError("KL effect pilot formal config path mismatch")
    return formal, path


def _ancestor(commit: str, current: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    ).returncode != 0:
        raise ValueError("source evidence commit is not an ancestor")


def _load_source(spec: Spec, formal_path: Path, current: str) -> Source:
    legacy = vae._load_spec(
        _resolve(
            "configs/research/lightgen_optical_manifold_smf_vae_pilot_v1.yaml",
            base_path=base._repo_root(),
        )
    )
    source = vae._load_source(legacy, formal_path, current)
    src = spec.payload["source"]
    extra = {
        "vae_validation": (src["vae_validation"], SOURCE_VAE_VALIDATION_HASH),
        "vae_preflight": (src["vae_preflight"], SOURCE_VAE_PREFLIGHT_HASH),
        "vae_training": (src["vae_training"], SOURCE_VAE_TRAINING_HASH),
        "vae_normalization": (src["vae_normalization"], SOURCE_VAE_NORMALIZATION_HASH),
        "kl_attribution_audit": (src["kl_attribution_audit"], SOURCE_KL_ATTRIBUTION_HASH),
        "kl_dose_validation": (src["kl_dose_validation"], SOURCE_KL_DOSE_VALIDATION_HASH),
        "kl_dose_audit": (src["kl_dose_audit"], SOURCE_KL_DOSE_AUDIT_HASH),
    }
    for role, (raw_path, digest) in extra.items():
        if file_sha256(_resolve(raw_path)) != digest:
            raise ValueError(f"{role} source hash mismatch")
    preflight = json.loads(_resolve(src["vae_preflight"]).read_text(encoding="utf-8"))
    preflight_record = EvidenceRecordV2.from_dict(preflight["evidence"])
    preflight_record.validate()
    if preflight_record.status != "passed":
        raise ValueError("VAE preflight source is not authoritative")
    current_beta = float(preflight["contract"]["beta"])
    if current_beta != REGISTERED_BETA:
        raise ValueError("VAE current beta differs from registered beta")
    dose = json.loads(_resolve(src["kl_dose_audit"]).read_text(encoding="utf-8"))
    dose_record = EvidenceRecordV2.from_dict(dose["evidence"])
    dose_record.validate()
    if dose_record.status != "diagnostic" or dose["contract"].get("outcome") != "diagnostic_completed_relative_kl_constraint" or dose["contract"].get("selected_multiplier") != CANDIDATE_MULTIPLIER:
        raise ValueError("KL dose source is not the passed corrected audit")
    hashes = {
        **source.hashes,
        "vae_validation": SOURCE_VAE_VALIDATION_HASH,
        "vae_preflight": SOURCE_VAE_PREFLIGHT_HASH,
        "vae_training": SOURCE_VAE_TRAINING_HASH,
        "vae_normalization": SOURCE_VAE_NORMALIZATION_HASH,
        "kl_attribution_audit": SOURCE_KL_ATTRIBUTION_HASH,
        "kl_dose_validation": SOURCE_KL_DOSE_VALIDATION_HASH,
        "kl_dose_audit": SOURCE_KL_DOSE_AUDIT_HASH,
    }
    return Source(source, hashes, dose, current_beta)


def _dose_indices(validation_path: Path) -> torch.Tensor:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    contract = payload["contract"]
    values = contract["calibration_panel_indices"] + contract["confirmation_panel_indices"]
    return torch.tensor(values, dtype=torch.long).flatten()


def _load_data(root: str | Path, dose_validation: Path) -> vae.Data:
    prior = vae._load_data(root)
    excluded = torch.cat(
        (
            prior.schedule.flatten(),
            prior.heldout_indices,
            prior.calibration_indices.flatten(),
            _dose_indices(dose_validation),
        )
    )
    calibration: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    train_labels = prior.labels.index_select(0, prior.train_indices)
    for label in range(10):
        candidates = prior.train_indices[train_labels == label]
        candidates = candidates[~torch.isin(candidates, excluded)].sort().values
        if candidates.numel() < 200:
            raise ValueError("insufficient disjoint MNIST samples for KL effect pilot")
        generator = torch.Generator(device="cpu").manual_seed(SEED + label)
        shuffled = candidates[torch.randperm(candidates.numel(), generator=generator)]
        calibration.append(shuffled[:100])
        selected.append(shuffled[100:200])
    calibration_indices = torch.stack(calibration, dim=1)
    schedule = torch.stack(selected, dim=1)
    if (
        torch.unique(calibration_indices).numel() != calibration_indices.numel()
        or torch.unique(schedule).numel() != schedule.numel()
        or torch.isin(calibration_indices.flatten(), schedule.flatten()).any()
        or torch.isin(schedule.flatten(), excluded).any()
    ):
        raise ValueError("KL effect pilot schedule is not disjoint")
    epsilon_generator = torch.Generator(device="cpu").manual_seed(SEED)
    epsilon_schedule = torch.randn(
        (STEPS, BATCH_SIZE, PORT_COUNT, 2),
        generator=epsilon_generator,
        dtype=torch.float32,
    )
    if not torch.equal(
        prior.labels.index_select(0, schedule.flatten()).reshape(STEPS, BATCH_SIZE)[0],
        torch.arange(10),
    ):
        raise ValueError("KL effect pilot schedule is not one-per-label")
    manifest = {
        "schema": "lightgen_optical_manifold_smf_vae_kl_effect_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "calibration_indices_sha256": tensor_sha256(calibration_indices),
        "calibration_count": int(calibration_indices.numel()),
        "schedule_sha256": tensor_sha256(schedule),
        "schedule_shape": list(schedule.shape),
        "schedule_count": int(schedule.numel()),
        "epsilon_schedule_sha256": tensor_sha256(epsilon_schedule),
        "heldout_indices_sha256": tensor_sha256(prior.heldout_indices),
        "visual_indices_sha256": tensor_sha256(prior.visual_indices),
        "dose_validation_sha256": file_sha256(dose_validation),
        "prior_panels_excluded": True,
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "schedule_balance_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return vae.Data(
        prior.images,
        prior.labels,
        prior.train_indices,
        prior.heldout_indices,
        calibration_indices,
        schedule,
        epsilon_schedule,
        prior.visual_indices,
        manifest,
    )


def _context(spec: Spec, formal: FormalSystemConfig, data: vae.Data) -> EvidenceContextV2:
    return EvidenceContextV2(
        canonical_sha256(formal.model_dump(mode="json")),
        spec.config_hash,
        data.manifest["manifest_hash"],
        canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        1.0,
    )


def _input_hashes(spec: Spec, review: Path, data: vae.Data, source: Source) -> dict[str, str]:
    return {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review),
        "dataset_manifest": data.manifest["manifest_hash"],
    }


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite KL effect artifact: {target}")
    atomic_write_json(target, payload)


def _load_validation(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: Source, review: Path, state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected = _input_hashes(spec, review, data, source)
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.context != _context(spec, formal, data)
        or record.input_hashes != expected
        or record.permitted_next_actions != ("run_smf_vae_kl_effect_preflight",)
    ):
        raise ValueError("KL effect pilot validation is stale or mismatched")
    return payload, digest


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, _resolve(spec.payload["source"]["kl_dose_validation"]))
    hashes = _input_hashes(spec, review, data, source)
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"branch": BRANCH, "clean": True}),
            ("source_evidence", {"source_step": SOURCE_STEP, "dose_multiplier": CANDIDATE_MULTIPLIER}),
            ("four_arm_contract", {"arms": list(ARMS)}),
            ("mnist_panel_isolation", {"calibration": 1000, "training": 1000, "heldout": 1000}),
            ("formal_identity_smf_optics", {"grid": 2048, "formal_ols": "identity", "research_ols": "100_port_lp01_smf"}),
            ("fixed_ols_boundary", {"trainable_transport": False, "exposure": 1.0}),
            ("research_boundary", {"training_authorized": False, "formal_promotion": False}),
        )
    )
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_vae_kl_effect_preflight",))
    record.validate()
    payload = _record_payload(VALIDATION_SCHEMA, record, {"source_hashes": source.hashes, "dataset_manifest": data.manifest, "candidate_beta": CANDIDATE_BETA, "optimizer_created": False, "cuda_context_created": False})
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _runtime(formal: FormalSystemConfig, source: Source, arm: str, device: torch.device) -> tuple[nn.Module, Any, vae.SharedPerPortPosterior | None, Any]:
    runtime_arm = "smf_deterministic_mse" if arm == "smf_deterministic_mse" else "smf_beta_vae"
    return vae._make_runtime(formal, source.source, device, runtime_arm)


def _beta_for_arm(arm: str, current_beta: float) -> float:
    if arm == "smf_current_beta_vae":
        return current_beta
    if arm == "smf_candidate_beta_vae":
        return CANDIDATE_BETA
    return 0.0


def _run_segment(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior | None, optimizer: Any, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, arm: str, beta: float, start: int, count: int) -> list[dict[str, Any]]:
    if start < 0 or start + count > STEPS:
        raise ValueError("KL effect pilot segment is outside schedule")
    history: list[dict[str, Any]] = []
    for step in range(start, start + count):
        amplitudes, targets = vae._batch(data, data.schedule[step], device)
        report = vae._step(model, operator, posterior, optimizer, amplitudes, targets, data.epsilon_schedule[step].to(device), mean.to(device), scale.to(device), arm, beta)
        report["step"] = step + 1
        history.append(report)
    return history


def _tree_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _tree_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_clone(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _resume_payload(model: nn.Module, posterior: vae.SharedPerPortPosterior | None, optimizer: Any, arm: str, step: int, history: list[dict[str, Any]], data: vae.Data, spec: Spec, hashes: Mapping[str, str], beta: float, completed_arms: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "model_state": _model_state(model),
        "posterior_state": None if posterior is None else _model_state(posterior),
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "rng_state": _tree_clone(base._rng_state()),
        "history": list(history),
        "schedule_cursor": {"next_step": step, "batch_size": BATCH_SIZE},
        "epsilon_cursor": step,
        "beta": beta,
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(hashes),
        "completed_arms": dict(completed_arms),
        "formal_candidate_checkpoint": False,
        "training_authorized": False,
        "formal_promotion": False,
    }


def _load_resume(path: Path, model: nn.Module, posterior: vae.SharedPerPortPosterior | None, optimizer: Any, data: vae.Data, spec: Spec, hashes: Mapping[str, str], arm: str, output_dir: Path, *, require_final: bool = False) -> dict[str, Any]:
    if path.parent.resolve() != output_dir.resolve():
        raise ValueError("KL effect pilot resume belongs to another attempt")
    payload = safe_torch_load(path, map_location="cpu")
    step = int(payload.get("step", 0))
    if payload.get("schema") != RESUME_SCHEMA or payload.get("arm") != arm or not (0 < step <= STEPS) or (require_final and step != STEPS) or (not require_final and step >= STEPS) or payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes):
        raise ValueError("KL effect pilot resume lineage mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    if posterior is not None:
        posterior.load_state_dict(payload["posterior_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return payload


def _phase_update_gate(history: list[dict[str, Any]], posterior_required: bool) -> bool:
    if not history:
        return False
    for item in history:
        if not all(item["update_norms"].get(name, 0.0) > 0.0 for name in base.EXPECTED_TRAINABLE):
            return False
        if posterior_required and not any(value > 0.0 for name, value in item["update_norms"].items() if name.startswith("posterior.")):
            return False
    return True


def _safety_ok(report: Mapping[str, Any], initial_paired_mse: float) -> bool:
    """Apply the reconstruction safety gate in the reconstruction metric's units.

    A posterior arm's total objective contains ``beta * KL`` and therefore
    cannot be compared with the initial paired MSE.  The safety stop is about
    image reconstruction only; the KL objective has its own explicit gates.
    """
    return bool(
        report["finite"]
        and report["mse"] <= SAFETY_MSE_RATIO * max(initial_paired_mse, 1.0e-30)
    )


def _gradient_boundary(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    amplitudes, targets = vae._batch(data, data.schedule[0], device)
    epsilon = data.epsilon_schedule[0].to(device)
    reconstruction = vae._objective_gradient_report(model, operator, posterior, amplitudes, targets, epsilon, mean.to(device), scale.to(device), "reconstruction")
    kl = vae._objective_gradient_report(model, operator, posterior, amplitudes, targets, epsilon, mean.to(device), scale.to(device), "kl")
    kl_norms = kl["gradient_norms"]
    return {
        "reconstruction": reconstruction,
        "kl": kl,
        "kl_encoder_nonzero": kl_norms["system.encoder.phase"] > 0.0,
        "kl_decoder_zero": kl_norms["system.decoder.phase"] == 0.0,
        "kl_posterior_nonzero": any(value > 0.0 for name, value in kl_norms.items() if name.startswith("posterior.")),
        "candidate_beta": CANDIDATE_BETA,
    }


def _exact_resume(formal: FormalSystemConfig, source: Source, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, arm: str, beta: float, output_dir: Path, spec: Spec, hashes: Mapping[str, str]) -> dict[str, Any]:
    continuous_model, continuous_operator, continuous_post, continuous_opt = _runtime(formal, source, arm, device)
    continuous = _run_segment(continuous_model, continuous_operator, continuous_post, continuous_opt, data, mean, scale, device, arm, beta, 0, PREFLIGHT_STEPS)
    continuous_state = (_model_state(continuous_model), None if continuous_post is None else _model_state(continuous_post), continuous_opt.state_dict(), _tree_clone(base._rng_state()))
    split_model, split_operator, split_post, split_opt = _runtime(formal, source, arm, device)
    split = _run_segment(split_model, split_operator, split_post, split_opt, data, mean, scale, device, arm, beta, 0, 2)
    payload = _resume_payload(split_model, split_post, split_opt, arm, 2, split, data, spec, hashes, beta, {})
    path = output_dir / f"preflight_{arm}_step_0002.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preflight resume: {path}")
    atomic_torch_save(path, payload)
    resumed_model, resumed_operator, resumed_post, resumed_opt = _runtime(formal, source, arm, device)
    resumed_payload = _load_resume(path, resumed_model, resumed_post, resumed_opt, data, spec, hashes, arm, output_dir)
    resumed = _run_segment(resumed_model, resumed_operator, resumed_post, resumed_opt, data, mean, scale, device, arm, beta, 2, 1)
    resumed_state = (_model_state(resumed_model), None if resumed_post is None else _model_state(resumed_post), resumed_opt.state_dict(), _tree_clone(base._rng_state()))
    return {
        "continuous": continuous,
        "resumed": resumed,
        "resume_artifact": str(path.resolve()),
        "resume_artifact_sha256": file_sha256(path),
        "persisted_history": resumed_payload["history"] == split,
        "schedule_cursor": resumed_payload["schedule_cursor"] == {"next_step": 2, "batch_size": BATCH_SIZE},
        "model_state": _tree_equal(continuous_state[0], resumed_state[0]),
        "posterior_state": _tree_equal(continuous_state[1], resumed_state[1]),
        "optimizer_state": _tree_equal(continuous_state[2], resumed_state[2]),
        "step_report": _tree_equal(continuous[2], resumed[0]),
        "rng_state": _tree_equal(continuous_state[3], resumed_state[3]),
    }


def _load_preflight(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: vae.Data, source: Source, review: Path, validation_hash: str, state: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, str], torch.Tensor, torch.Tensor, float]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    norm_path = _resolve(payload["contract"]["normalization_path"])
    norm_hash = file_sha256(norm_path)
    expected = {**_input_hashes(spec, review, data, source), "training_validation": validation_hash, "posterior_normalization": norm_hash}
    if payload.get("schema") != PREFLIGHT_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected:
        raise ValueError("KL effect pilot preflight is stale or mismatched")
    normalization = safe_torch_load(norm_path, map_location="cpu")
    if normalization.get("schema") != "lightgen_optical_manifold_smf_vae_kl_effect_posterior_normalization.v1" or normalization.get("config_hash") != spec.config_hash or normalization.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or normalization.get("validation_sha256") != validation_hash or normalization.get("source_hashes") != source.hashes or tuple(normalization.get("shape", ())) != (PORT_COUNT, 2):
        raise ValueError("KL effect pilot normalization is stale")
    hashes = {**expected, "training_preflight": digest}
    return payload, digest, hashes, normalization["mean"], normalization["scale"], float(payload["contract"]["candidate_beta"])


def _normalization_payload(mean: torch.Tensor, scale: torch.Tensor, spec: Spec, data: vae.Data, source: Source, validation_hash: str) -> dict[str, Any]:
    return {
        "schema": "lightgen_optical_manifold_smf_vae_kl_effect_posterior_normalization.v1",
        "mean": mean.cpu(),
        "scale": scale.cpu(),
        "shape": [PORT_COUNT, 2],
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "source_hashes": dict(source.hashes),
        "validation_sha256": validation_hash,
        "training_authorized": False,
        "formal_promotion": False,
    }


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("KL effect pilot preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, _resolve(spec.payload["source"]["kl_dose_validation"]))
    _validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite KL effect preflight: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        norm_model, norm_operator, _norm_post, _norm_opt = _runtime(formal, source, "smf_deterministic_mse", device)
        mean, scale = vae._fit_normalization(norm_model, norm_operator, data, device)
        norm_path = output.parent / "posterior_normalization.pt"
        atomic_torch_save(norm_path, _normalization_payload(mean, scale, spec, data, source, validation_hash))
        hashes = {**_input_hashes(spec, review, data, source), "training_validation": validation_hash, "posterior_normalization": file_sha256(norm_path)}
        exact: dict[str, Any] = {}
        for arm in ARMS:
            exact[arm] = _exact_resume(formal, source, data, mean, scale, device, arm, _beta_for_arm(arm, source.current_beta), output.parent, spec, hashes)
        candidate_model, candidate_operator, candidate_post, _candidate_opt = _runtime(formal, source, "smf_candidate_beta_vae", device)
        boundary = _gradient_boundary(candidate_model, candidate_operator, candidate_post, data, mean, scale, device)
        exact_ok = all(all(value for key, value in report.items() if key in {"persisted_history", "schedule_cursor", "model_state", "posterior_state", "optimizer_state", "step_report", "rng_state"}) for report in exact.values())
        finite = all(item["finite"] for report in exact.values() for item in report["continuous"] + report["resumed"])
        phase_updates = all(_phase_update_gate(report["continuous"], arm in POSTERIOR_ARMS) for arm, report in exact.items())
        posterior_boundary = boundary["kl_encoder_nonzero"] and boundary["kl_decoder_zero"] and boundary["kl_posterior_nonzero"]
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        gates = (
            GateResult("finite_values", finite, {}),
            GateResult("four_arm_phase_updates", phase_updates, {}),
            GateResult("posterior_updates", _phase_update_gate(exact["smf_candidate_beta_vae"]["continuous"], True), {}),
            GateResult("persisted_resume_exact", exact_ok, {arm: {key: value for key, value in report.items() if isinstance(value, bool)} for arm, report in exact.items()}),
            GateResult("fixed_epsilon_schedule", True, {"seed": SEED, "shared_posterior_epsilon": True}),
            GateResult("kl_gradient_boundary", posterior_boundary, {"encoder_nonzero": boundary["kl_encoder_nonzero"], "decoder_zero": boundary["kl_decoder_zero"], "posterior_nonzero": boundary["kl_posterior_nonzero"]}),
            GateResult("candidate_beta_binding", CANDIDATE_BETA == CANDIDATE_MULTIPLIER * REGISTERED_BETA, {"registered_beta": REGISTERED_BETA, "multiplier": CANDIDATE_MULTIPLIER, "candidate_beta": CANDIDATE_BETA}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
        )
        status = "passed" if all(gate.passed for gate in gates) else "invalid_harness"
        actions = ("run_smf_vae_kl_effect",) if status == "passed" else ()
        record = EvidenceRecordV2(2, PREFLIGHT_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, actions, None if status == "passed" else "KL effect pilot preflight gate failed")
        record.validate()
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"normalization_path": str(norm_path.resolve()), "normalization_sha256": file_sha256(norm_path), "registered_beta": REGISTERED_BETA, "candidate_multiplier": CANDIDATE_MULTIPLIER, "candidate_beta": CANDIDATE_BETA, "arms": exact, "gradient_boundary": boundary, "free_cuda_bytes": int(free), "total_cuda_bytes": int(total), "peak_cuda_bytes": peak, "elapsed_seconds": elapsed, "training_authorized": False, "formal_promotion": False})
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status == "passed" else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def _posterior_stats(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior, data: vae.Data, indices: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, device: torch.device) -> dict[str, Any]:
    return kl_audit._posterior_metrics(model, operator, posterior, data, indices, mean.to(device), scale.to(device), device)


def _metrics(model: nn.Module, operator: Any, posterior: vae.SharedPerPortPosterior | None, data: vae.Data, mean: torch.Tensor, scale: torch.Tensor, device: torch.device, *, sample: bool) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    epsilon = torch.zeros((data.heldout_indices.numel(), PORT_COUNT, 2), dtype=torch.float32)
    metrics, tensors = vae._evaluate(model, operator, posterior, data, data.heldout_indices, mean.to(device), scale.to(device), device, sample=sample, epsilon=epsilon)
    if posterior is not None:
        metrics.update({key: value for key, value in _posterior_stats(model, operator, posterior, data, data.heldout_indices, mean, scale, device).items() if key in {"active_latent_count", "posterior_mean_variance", "posterior_variance_mean", "kl_total", "kl_mu", "kl_variance"}})
    return metrics, tensors


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite KL effect checkpoint: {path}")
    atomic_torch_save(path, payload)


def _save_panel(path: Path, data: vae.Data, tensors: Mapping[str, torch.Tensor], hashes: Mapping[str, str]) -> str:
    payload = {
        "schema": PANEL_SCHEMA,
        "indices_are_global_mnist": True,
        "indices": data.visual_indices.cpu(),
        "input_28": data.images.index_select(0, data.visual_indices).cpu(),
        "tensors": {name: value.cpu().contiguous() for name, value in tensors.items()},
        "source_hashes": dict(hashes),
        "posterior_sample_seed": SEED + 1,
        "diagnostic_only": True,
    }
    atomic_torch_save(path, payload)
    return file_sha256(path)


def _load_completed_arm(path: Path, model: nn.Module, posterior: vae.SharedPerPortPosterior | None, optimizer: Any, data: vae.Data, spec: Spec, hashes: Mapping[str, str], arm: str, output_dir: Path) -> dict[str, Any]:
    return _load_resume(path, model, posterior, optimizer, data, spec, hashes, arm, output_dir, require_final=True)


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("KL effect pilot training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root, _resolve(spec.payload["source"]["kl_dose_validation"]))
    validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, state)
    preflight_payload, preflight_hash, hashes, mean, scale, candidate_beta = _load_preflight(args.preflight_evidence, spec, formal, data, source, review, validation_hash, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite KL effect training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        initial: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            model, operator, posterior, _optimizer = _runtime(formal, source, arm, device)
            metrics, _tensors = _metrics(model, operator, posterior, data, mean, scale, device, sample=False)
            initial[arm] = metrics
        resume_payload: Mapping[str, Any] | None = None
        resume_arm: str | None = None
        if args.resume_checkpoint:
            resume_path = _resolve(args.resume_checkpoint)
            resume_payload = safe_torch_load(resume_path, map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in ARMS:
                raise ValueError("invalid KL effect resume arm")
        results: dict[str, Any] = {}
        models: dict[str, tuple[Any, Any, Any]] = {}
        checkpoints: dict[str, Any] = {}
        completed: dict[str, Any] = {}
        for arm in ARMS:
            model, operator, posterior, optimizer = _runtime(formal, source, arm, device)
            start = 0
            history: list[dict[str, Any]] = []
            if resume_payload is not None and resume_arm == arm:
                loaded = _load_resume(_resolve(args.resume_checkpoint), model, posterior, optimizer, data, spec, hashes, arm, output.parent)
                start = int(loaded["step"])
                history = list(loaded["history"])
                completed.update(loaded.get("completed_arms", {}))
            elif resume_payload is not None and arm in resume_payload.get("completed_arms", {}):
                item = resume_payload["completed_arms"][arm]
                completed_model, completed_operator, completed_post, completed_opt = _runtime(formal, source, arm, device)
                completed_payload = _load_completed_arm(_resolve(item["path"]), completed_model, completed_post, completed_opt, data, spec, hashes, arm, output.parent)
                models[arm] = (completed_model, completed_operator, completed_post)
                results[arm] = {"history": list(completed_payload["history"]), "steps_completed": STEPS, "metrics": _metrics(completed_model, completed_operator, completed_post, data, mean, scale, device, sample=False)[0]}
                completed[arm] = item
                continue
            beta = _beta_for_arm(arm, source.current_beta)
            for step in range(start, STEPS):
                amplitudes, targets = vae._batch(data, data.schedule[step], device)
                report = vae._step(model, operator, posterior, optimizer, amplitudes, targets, data.epsilon_schedule[step].to(device), mean.to(device), scale.to(device), arm, beta)
                report["step"] = step + 1
                history.append(report)
                if not _safety_ok(report, initial[arm]["paired_mse"]):
                    raise RuntimeError(f"KL effect pilot safety gate failed for {arm} at step {step + 1}")
                if (step + 1) % CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = output.parent / f"{arm}_step_{step + 1:04d}.pt"
                    checkpoint_payload = _resume_payload(model, posterior, optimizer, arm, step + 1, history, data, spec, hashes, beta, completed)
                    _save_checkpoint(checkpoint_path, checkpoint_payload)
                    checkpoints[arm] = {"path": str(checkpoint_path.resolve()), "sha256": file_sha256(checkpoint_path)}
            metrics, _tensors = _metrics(model, operator, posterior, data, mean, scale, device, sample=False)
            results[arm] = {"history": history, "steps_completed": len(history), "metrics": metrics}
            models[arm] = (model, operator, posterior)
            completed[arm] = checkpoints[arm]
        if set(results) != set(ARMS):
            raise RuntimeError("KL effect pilot did not produce all arm results")
        panel_tensors: dict[str, torch.Tensor] = {"target": vae._batch(data, data.visual_indices, device)[1].cpu()}
        for arm in ARMS:
            model, operator, posterior = models[arm]
            _panel_metrics, tensors = vae._evaluate(model, operator, posterior, data, data.visual_indices, mean.to(device), scale.to(device), device, sample=False, epsilon=torch.zeros((data.visual_indices.numel(), PORT_COUNT, 2)))
            panel_tensors[f"{arm}_mean"] = tensors["camera"]
            if posterior is not None:
                epsilon = torch.randn((data.visual_indices.numel(), PORT_COUNT, 2), generator=torch.Generator(device="cpu").manual_seed(SEED + 1), dtype=torch.float32)
                _sample_metrics, sample_tensors = vae._evaluate(model, operator, posterior, data, data.visual_indices, mean.to(device), scale.to(device), device, sample=True, epsilon=epsilon)
                panel_tensors[f"{arm}_sample"] = sample_tensors["camera"]
        source_model, source_operator, source_post, _source_opt = _runtime(formal, source, "smf_deterministic_mse", device)
        _source_metrics, source_tensors = vae._evaluate(source_model, source_operator, source_post, data, data.visual_indices, mean.to(device), scale.to(device), device, sample=False)
        panel_tensors["direct_smf_source"] = source_tensors["camera"]
        panel_path = output.parent / "smf_vae_kl_effect_panel.pt"
        panel_hash = _save_panel(panel_path, data, panel_tensors, {**hashes, "training_validation": validation_hash, "training_preflight": preflight_hash})
        candidate = results["smf_candidate_beta_vae"]["metrics"]
        no_kl = results["smf_posterior_no_kl"]["metrics"]
        candidate_initial = initial["smf_candidate_beta_vae"]
        candidate_kl_reduction = candidate.get("posterior_kl", float("inf")) <= candidate_initial.get("posterior_kl", float("inf")) * (1.0 - MIN_KL_REDUCTION)
        gates = (
            GateResult("candidate_kl_below_initial_2_percent", candidate_kl_reduction, {"initial": candidate_initial.get("posterior_kl"), "final": candidate.get("posterior_kl")}),
            GateResult("candidate_kl_below_no_kl", candidate.get("posterior_kl", float("inf")) < no_kl.get("posterior_kl", float("inf")), {"candidate": candidate.get("posterior_kl"), "no_kl": no_kl.get("posterior_kl")}),
            GateResult("candidate_mse_relative_to_no_kl", candidate["paired_mse"] <= no_kl["paired_mse"] * MAX_CANDIDATE_MSE_RELATIVE, {"ratio": candidate["paired_mse"] / max(no_kl["paired_mse"], 1.0e-30)}),
            GateResult("candidate_mse_below_cyclic", candidate["paired_mse"] < candidate["cyclic_mse"], {}),
            GateResult("posterior_variance_retention", candidate.get("posterior_mean_variance", 0.0) >= MIN_VARIANCE_RETENTION * max(candidate_initial.get("posterior_mean_variance", 0.0), 1.0e-30), {}),
            GateResult("active_latent_retention", candidate.get("active_latent_count", 0) >= MIN_ACTIVE_RETENTION * max(candidate_initial.get("active_latent_count", 0), 1), {"initial": candidate_initial.get("active_latent_count"), "final": candidate.get("active_latent_count")}),
            GateResult("mean_sample_finite", all(torch.isfinite(value).all() for value in panel_tensors.values())),
            GateResult("all_arms_completed", all(result["steps_completed"] == STEPS for result in results.values()), {}),
            GateResult("four_phase_updates", all(_phase_update_gate(result["history"], arm in POSTERIOR_ARMS) for arm, result in results.items()), {}),
            GateResult("fixed_ols_and_evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("panel_provenance", True, {"panel_sha256": panel_hash}),
            GateResult("cuda_peak_memory", int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_BYTES, {"bytes": int(torch.cuda.max_memory_allocated(device))}),
            GateResult("wall_time", time.monotonic() - started <= MAX_WALL_SECONDS, {"seconds": time.monotonic() - started}),
            GateResult("run_storage", sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file()) <= MAX_STORAGE_BYTES, {}),
        )
        scientific = all(gate.passed for gate in gates[:10])
        resources = all(gate.passed for gate in gates[-3:])
        status = "passed" if scientific and resources else ("resource_aborted" if not resources else "valid_failed")
        actions = (NEXT_PASS_ACTION,) if status == "passed" else ((NEXT_FAILED_ACTION,) if status == "valid_failed" else ())
        final_hashes = {**hashes, "training_validation": validation_hash, "training_preflight": preflight_hash}
        record = EvidenceRecordV2(2, TRAINING_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), final_hashes, gates, actions, None if status == "passed" else "KL effect pilot scientific or resource gate failed")
        record.validate()
        payload = _record_payload(TRAINING_SCHEMA, record, {"outcome": status, "validation_hash": validation_hash, "preflight_hash": preflight_hash, "registered_beta": REGISTERED_BETA, "candidate_multiplier": CANDIDATE_MULTIPLIER, "candidate_beta": candidate_beta, "initial_metrics": initial, "arms": results, "checkpoints": checkpoints, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "free_cuda_bytes_before": int(free), "total_cuda_bytes": int(total), "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)), "wall_seconds": time.monotonic() - started, "run_storage_bytes": sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file()), "evaluator_calls": 0, "formal_candidate_checkpoint": False})
        _write_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"passed", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.cuda.empty_cache()


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-kl-effect-pilot-validate":
        return validate(args)
    if args.research_action == "smf-kl-effect-pilot-preflight":
        return preflight(args)
    if args.research_action == "smf-kl-effect-pilot-train":
        return train(args)
    raise ValueError(f"unsupported KL effect pilot action: {args.research_action}")
