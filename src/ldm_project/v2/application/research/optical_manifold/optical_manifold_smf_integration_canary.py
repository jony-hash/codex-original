"""Diagnostic Identity-to-100-port-SMF integration canary for H-LOMA-I0.

This lifecycle exists to exercise the complete research optical path after the
PLAN 5c stable-projection gate failed.  It deliberately does not relax that
gate and its short training result is never a scientific SMF or projection
claim.  The formal v2 system remains IdentityOLS; the SMF is inserted only in
this research helper at the complex encoder/decoder boundary.
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
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ldm_project.v2.config import FormalSystemConfig, load_config
from ldm_project.v2.contracts import ComplexField, IntensityFrame
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ldm_project.v2.optics.devices import IdentityOLS
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_adaptive_projection_audit as adaptive
from ldm_project.v2.application.research.optical_manifold import optical_manifold_full_epoch as full_epoch
from ldm_project.v2.application.research.optical_manifold import optical_manifold_regularization_audit as regularization


BRANCH = "research/v2-lightgen-optical-manifold-smf-integration-canary-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_smf_integration_canary_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_smf_integration_canary_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_canary_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_canary_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_canary_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_smf_canary_resume.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_canary_panel.v1"
PROJECTION_SCHEMA = "lightgen_optical_manifold_smf_canary_projection.v1"

SOURCE_VALIDATION_HASH = "7e11e3d989e2dd558d11f4d13d96b2cbd33c628e337806ef27d888ff071d99b5"
SOURCE_AUDIT_HASH = "98f40cc3312c01f22056424e6a4baa25d6bfd618bb0eb4f3dcbab75934fbb7a3"
SOURCE_CHECKPOINT_HASH = "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_STEP = 5500
STEPS = 20
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
MICROBATCH = 1
RANK = 32
DIRECTIONS = 64
SEED = 1234
LEARNING_RATE = 1.0e-3
PORT_COUNT = 100
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 1800.0
MAX_STORAGE_BYTES = 512 * 1024**2
SAFETY_MSE_RATIO = 2.0
NEXT_ACTION = "manual_review_smf_canary_visual_and_metric_trends"
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


class _SafetyAbort(RuntimeError):
    def __init__(self, step: int, result: Mapping[str, Any]) -> None:
        super().__init__(f"SMF canary safety MSE bound exceeded at step {step}")
        self.step = step
        self.result = dict(result)


@dataclass(frozen=True)
class CanarySpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class CanaryData:
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    heldout_indices: torch.Tensor
    schedule: torch.Tensor
    visual_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("SMF integration canary requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"SMF integration canary requires branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "SMF integration canary requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _load_spec(path: str | Path) -> CanarySpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema", "hypothesis", "formal_system_config", "source", "dataset",
        "optical_contract", "projection", "training", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("SMF canary config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported SMF canary config")
    return CanarySpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: CanarySpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_diagnostic_smf_integration_canary",
        "approved_contract": {
            "steps_per_arm": STEPS,
            "logical_batch_size": BATCH_SIZE,
            "microbatch_size": MICROBATCH,
            "arms": ["smf_mse_only", "smf_rank32_swd"],
            "smf_port_count": PORT_COUNT,
            "projection_rank": RANK,
            "projection_status": "diagnostic_unstable_projection",
            "preflight_steps": PREFLIGHT_STEPS,
            "resume_segment": "2+1",
            "trend_metrics_non_gating": True,
            "post_completion_next_action": NEXT_ACTION,
        },
        "diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("SMF canary review contract does not match scope")
    return payload, target


def _assert_contract(spec: CanarySpec) -> None:
    payload = spec.payload
    source = payload["source"]
    if source["source_global_step"] != SOURCE_STEP or source["source_checkpoint_sha256"] != SOURCE_CHECKPOINT_HASH:
        raise ValueError("SMF canary source checkpoint contract is not locked")
    if source["mnist_manifest_sha256"] != SOURCE_MANIFEST_HASH:
        raise ValueError("SMF canary MNIST source manifest contract is not locked")
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "schedule_steps": STEPS,
        "schedule_batch_size": BATCH_SIZE,
        "schedule": "balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "excluded_prior_panels": True,
        "heldout_excluded": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "visual_panel_count": 100,
    }:
        raise ValueError("SMF canary dataset contract is not locked")
    if payload["projection"] != {
        "source_family": "plan_5c_cross_fitted_calibration_panels",
        "rank": RANK,
        "status": "diagnostic_unstable_projection",
        "fit": "consensus_from_three_calibration_panels",
        "whitening": "calibration_score_covariance_eigenvalue",
        "directions": DIRECTIONS,
        "direction_seed": SEED,
        "rank_assignment": "fixed_per_direction_permutation",
    }:
        raise ValueError("SMF canary projection contract is not locked")
    if payload["training"]["arms"] != ["smf_mse_only", "smf_rank32_swd"] or payload["training"]["steps_per_arm"] != STEPS:
        raise ValueError("SMF canary training contract is not locked")
    if payload["optical_contract"]["research_ols"] != "100_port_lp01_smf":
        raise ValueError("SMF canary optical contract is not locked")
    if payload["authorization"]["training_authorized"] is not False or payload["authorization"]["formal_promotion"] is not False:
        raise ValueError("SMF canary authorization boundary is invalid")


def _load_formal(spec: CanarySpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("SMF canary requires current Identity lineage")
    if formal.system.kind != base.EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("SMF canary requires the formal Identity optical system")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _load_source(spec: CanarySpec, formal_path: Path, current_commit: str) -> SourceBundle:
    source = spec.payload["source"]
    paths = {
        "adaptive_projection_validation": (_resolve(source["adaptive_projection_validation"]), SOURCE_VALIDATION_HASH),
        "adaptive_projection_audit": (_resolve(source["adaptive_projection_audit"]), SOURCE_AUDIT_HASH),
        "source_checkpoint": (_resolve(source["source_checkpoint"]), SOURCE_CHECKPOINT_HASH),
    }
    for role, (path, expected) in paths.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    validation_payload = json.loads(paths["adaptive_projection_validation"][0].read_text(encoding="utf-8"))
    audit_payload = json.loads(paths["adaptive_projection_audit"][0].read_text(encoding="utf-8"))
    validation_record = EvidenceRecordV2.from_dict(validation_payload["evidence"])
    audit_record = EvidenceRecordV2.from_dict(audit_payload["evidence"])
    validation_record.validate()
    audit_record.validate()
    if validation_record.evidence_kind != adaptive.VALIDATION_SCHEMA or validation_record.status != "passed":
        raise ValueError("PLAN 5c validation source is not passed")
    if audit_record.evidence_kind != adaptive.AUDIT_SCHEMA or audit_record.status != "valid_failed":
        raise ValueError("PLAN 5c audit must be complete valid_failed evidence")
    if validation_payload.get("training_authorized") is not False or audit_payload.get("training_authorized") is not False:
        raise ValueError("PLAN 5c source training boundary is invalid")
    if validation_payload.get("formal_promotion") is not False or audit_payload.get("formal_promotion") is not False:
        raise ValueError("PLAN 5c source promotion boundary is invalid")
    for commit in (validation_record.git_commit, audit_record.git_commit):
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, current_commit], cwd=base._repo_root(), check=False)
        if result.returncode != 0:
            raise ValueError("PLAN 5c source commit is not an ancestor")
    checkpoint = safe_torch_load(paths["source_checkpoint"][0], expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if (
        checkpoint.get("schema") != "lightgen_optical_manifold_identity_full_epoch_resume.v1"
        or checkpoint.get("global_step") != SOURCE_STEP
        or not isinstance(checkpoint.get("model_state"), Mapping)
        or not isinstance(checkpoint.get("optimizer_state"), Mapping)
        or not isinstance(checkpoint.get("rng_state"), Mapping)
    ):
        raise ValueError("SMF canary source checkpoint is incomplete")
    configured_formal = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    if configured_formal != formal_path:
        raise ValueError("SMF canary formal source path mismatch")
    hashes = {
        "canary_config": spec.config_hash,
        "formal_config": file_sha256(formal_path),
        "adaptive_projection_validation": SOURCE_VALIDATION_HASH,
        "adaptive_projection_audit": SOURCE_AUDIT_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "mnist_manifest": SOURCE_MANIFEST_HASH,
    }
    return SourceBundle(checkpoint, hashes, formal_path)


def _load_data(dataset_root: str | Path) -> CanaryData:
    adaptive_data = adaptive._load_data(dataset_root)
    pilot_data = adaptive_data.objective_data.full_data.pilot_data
    schedule = adaptive.pilot._pilot_schedule(pilot_data.train_indices, pilot_data.labels)
    regularization_data = adaptive_data.objective_data.regularization_data
    excluded = torch.cat(
        (
            regularization_data.calibration_indices,
            *regularization_data.gradient_batches,
            adaptive_data.objective_data.projection_indices,
            *adaptive_data.objective_data.gradient_batches,
            schedule.reshape(-1),
            adaptive_data.calibration_panels[0],
            adaptive_data.calibration_panels[1],
            adaptive_data.calibration_panels[2],
            adaptive_data.assessment_indices,
            *adaptive_data.gradient_batches,
            pilot_data.heldout_indices,
        )
    )
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("SMF canary prior exclusion set contains duplicates")
    train_labels = pilot_data.labels.index_select(0, pilot_data.train_indices)
    per_label: list[torch.Tensor] = []
    for label in range(10):
        class_indices = pilot_data.train_indices[train_labels == label]
        remaining = class_indices[~torch.isin(class_indices, excluded)]
        if remaining.numel() < STEPS:
            raise ValueError("SMF canary lacks fresh per-label training samples")
        per_label.append(remaining[:STEPS])
    canary_schedule = torch.cat(
        [torch.cat([per_label[label][step : step + 1] for label in range(10)]) for step in range(STEPS)]
    ).reshape(STEPS, BATCH_SIZE)
    if torch.unique(canary_schedule).numel() != canary_schedule.numel() or torch.isin(canary_schedule, excluded).any():
        raise ValueError("SMF canary schedule is not fresh and without replacement")
    expected_labels = torch.arange(10).repeat(STEPS)
    if not torch.equal(pilot_data.labels.index_select(0, canary_schedule.reshape(-1)), expected_labels):
        raise ValueError("SMF canary schedule is not balanced one-per-label")
    visual_parts = []
    heldout_labels = pilot_data.labels.index_select(0, pilot_data.heldout_indices)
    for label in range(10):
        values = pilot_data.heldout_indices[heldout_labels == label]
        visual_parts.append(values[:10])
    visual_indices = torch.cat(visual_parts)
    manifest = {
        "schema": "lightgen_optical_manifold_smf_canary_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "prior_exclusion_indices_sha256": tensor_sha256(torch.sort(excluded).values),
        "schedule_sha256": tensor_sha256(canary_schedule),
        "schedule_shape": list(canary_schedule.shape),
        "schedule_count": int(canary_schedule.numel()),
        "schedule_seed": SEED,
        "visual_indices_sha256": tensor_sha256(visual_indices),
        "visual_count": int(visual_indices.numel()),
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return CanaryData(
        pilot_data.images,
        pilot_data.labels,
        pilot_data.train_indices,
        pilot_data.heldout_indices,
        canary_schedule,
        visual_indices,
        manifest,
    )


def _context(spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _model_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("SMF canary trainable parameters are not exactly four phase tensors")
    return {
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(trainable),
        "optimizer_parameter_names": list(trainable),
        "frozen_buffers": True,
    }


def _validation_record(state: Mapping[str, Any], spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, source: SourceBundle, model_contract: Mapping[str, Any]) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_canary_branch", {"branch_matches": True}),
            ("plan_5c_valid_failed_source_consumed_as_diagnostic", {"source_audit": SOURCE_AUDIT_HASH}),
            ("source_checkpoint_step", {"global_step": SOURCE_STEP}),
            ("mnist_schedule_fresh_balanced", {"steps": STEPS, "batch": BATCH_SIZE}),
            ("formal_geometry", {"grid": 2048, "identity_formal_ols": True}),
            ("research_smf_contract", {"ports": PORT_COUNT, "mode_model": "step_index_lp01_ports"}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("four_phase_parameter_scope", {"parameters": model_contract["trainable_parameter_names"]}),
            ("diagnostic_training_boundary", {"training_authorized": False, "formal_promotion": False}),
        )
    )
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=VALIDATION_SCHEMA,
        tier="diagnostic",
        status="passed",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes={**source.hashes, "dataset_manifest": data.manifest["manifest_hash"]},
        gates=gates,
        permitted_next_actions=("run_smf_integration_canary_preflight",),
    )
    record.validate()
    return record


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite SMF canary artifact: {target}")
    atomic_write_json(target, payload)


def _load_authoritative_pilot_data(dataset_root: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load the 55k/5k split manifest consumed by the canary."""
    pilot_data = full_epoch.pilot._load_pilot_data(dataset_root)
    if not full_epoch.pilot.manifest_matches_hash(
        pilot_data.manifest, SOURCE_MANIFEST_HASH
    ):
        raise ValueError("SMF canary MNIST source manifest changed")
    manifest = dict(pilot_data.manifest)
    manifest["manifest_hash"] = SOURCE_MANIFEST_HASH
    return pilot_data.images, pilot_data.labels, manifest


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    # The canary consumes the authoritative pilot split manifest.  The base
    # representative manifest is a different schema and is intentionally not
    # a source-lineage input for this lifecycle.
    images, labels, manifest = _load_authoritative_pilot_data(args.dataset_root)
    if not full_epoch.pilot.manifest_matches_hash(manifest, SOURCE_MANIFEST_HASH):
        raise ValueError("SMF canary MNIST manifest mismatch")
    data = _load_data(args.dataset_root)
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    contract = _model_contract(model)
    if tuple(images.shape) != (60000, 1, 28, 28) or tuple(labels.shape) != (60000,):
        raise ValueError("SMF canary requires complete MNIST train data")
    record = _validation_record(state, spec, formal, data, source, contract)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "formal_config": str(formal_path.resolve()),
            "source_hashes": source.hashes,
            "dataset_manifest": manifest,
            "canary_manifest": data.manifest,
            "model_contract": contract,
            "optimizer_created": False,
            "cuda_context_created": False,
        },
    )
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, state: Mapping[str, Any], source: SourceBundle) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("SMF canary validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.status != "passed" or record.evidence_kind != VALIDATION_SCHEMA:
        raise ValueError("SMF canary validation must be passed")
    if state["branch"] != BRANCH or not state["clean"] or record.git_commit != state["commit"]:
        raise ValueError("SMF canary validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("SMF canary validation context mismatch")
    expected = {**source.hashes, "dataset_manifest": data.manifest["manifest_hash"]}
    if record.input_hashes != expected:
        raise ValueError("SMF canary validation source hashes do not match")
    if record.permitted_next_actions != ("run_smf_integration_canary_preflight",):
        raise ValueError("SMF canary validation next action is invalid")
    return payload, digest


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _module_state_cpu(module: torch.nn.Module | IdentityOLS) -> dict[str, Any]:
    if isinstance(module, torch.nn.Module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    if isinstance(module, IdentityOLS):
        return {"type": type(module).__qualname__, "attributes": copy.deepcopy(vars(module))}
    raise TypeError(f"unsupported state snapshot type: {type(module)!r}")


def _nonphase_state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name not in EXPECTED_TRAINABLE
    }


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


def _smf_operator(system: Any, device: torch.device, *, zero_transport: bool = False) -> FiberArrayOLS:
    transport = torch.zeros((PORT_COUNT, PORT_COUNT), dtype=torch.complex64) if zero_transport else None
    operator = FiberArrayOLS(
        system.encoder_plane,
        FiberArraySpec(
            grid_shape=(10, 10),
            pitch_m=2.0e-6,
            wavelength_m=532.0e-9,
            waist_m=7.111e-6,
            fill=0.8,
            v_number=2.2,
            facet_magnification=1.0,
            mode_model="step_index_lp01_ports",
            expected_fov_m=(200.0e-6, 200.0e-6),
        ),
        transport=transport,
    )
    return operator.to(device)


def _relay_stats(system: Any) -> dict[str, torch.Tensor]:
    values = {
        "front_source_pupil_retention": system.front.last_diagnostic.get("source_pupil_retention"),
        "front_target_pupil_retention": system.front.last_diagnostic.get("target_pupil_retention"),
        "back_source_pupil_retention": system.back.last_diagnostic.get("source_pupil_retention"),
        "back_target_pupil_retention": system.back.last_diagnostic.get("target_pupil_retention"),
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or (value < 0).any() or (value > 1.0001).any():
            raise ValueError(f"SMF canary relay retention is invalid: {name}")
    return values


def _smf_forward(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, amplitude: torch.Tensor, *, trace: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    system = model.system
    slm_field = system.slm(system.command(amplitude))
    front = system.front(slm_field)
    encoded, encoder_records, encoder_stats = system.encoder(front, "encoder")
    smf_output = smf(encoded)
    decoded, decoder_records, decoder_stats = system.decoder(smf_output, "decoder")
    chip = ComplexField(decoded.tensor, system.chip_plane, decoded.provenance)
    camera_field = system.back(chip)
    relay = _relay_stats(system)
    raw = IntensityFrame(camera_field.tensor.abs().square(), system.camera_plane, camera_field.provenance, model.exposure.calibration_id)
    pre = encoder_records[-1].tensor
    diagnostics = {
        "smf_provenance": smf_output.provenance.to_dict(),
        "port_coefficients": smf.last_diagnostics.get("port_input_coefficients"),
        "coupling_efficiency": smf.last_diagnostics.get("coupling_efficiency"),
        "input_energy": smf.last_diagnostics.get("input_energy"),
        "output_energy": smf.last_diagnostics.get("output_energy"),
        "relay": relay,
        "encoder_crop_retention": encoder_stats["crop_retention"],
        "decoder_crop_retention": decoder_stats["crop_retention"],
    }
    return raw.tensor, pre if trace else None, diagnostics


def _identity_trace_has_ols(model: base.IdentityOpticalManifoldAutoencoder, amplitude: torch.Tensor) -> bool:
    result = model.system(
        model.system.command(amplitude),
        exposure=model.exposure,
        return_optical_trace=True,
    )
    matches = [
        field
        for field in result.optical_trace or ()
        if field.plane.plane_id == "ols_output" and field.provenance.operation == "identity_ols"
    ]
    return len(matches) == 1


def _batch(data: CanaryData, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _runtime(formal: FormalSystemConfig, source: SourceBundle, device: torch.device) -> tuple[base.IdentityOpticalManifoldAutoencoder, full_epoch._DeterministicCpuAdam, FiberArrayOLS]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    optimizer = full_epoch._make_full_optimizer(model)
    base._restore_rng(source.checkpoint["rng_state"])
    return model, optimizer, _smf_operator(model.system, device)


def _grad_zeros(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value, device="cpu") for name, value in model.named_parameters()}


def _clear_grad(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for value in model.parameters():
        value.grad = None


def _add_grads(total: dict[str, torch.Tensor], model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for name, value in model.named_parameters():
        if value.grad is not None:
            total[name].add_(value.grad.detach().cpu())


def _layer_norms(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        result[name] = float(value.norm().item())
        for index in range(value.shape[0]):
            result[f"{name}[{index}]"] = float(value[index].norm().item())
    return result


def _phase_layer_keys(model: base.IdentityOpticalManifoldAutoencoder) -> tuple[str, ...]:
    parameters = dict(model.named_parameters())
    return tuple(
        f"{name}[{index}]"
        for name in EXPECTED_TRAINABLE
        for index in range(parameters[name].shape[0])
    )


def _four_phase_nonzero(values: Mapping[str, float]) -> bool:
    expected = {
        "system.encoder.phase[0]",
        "system.encoder.phase[1]",
        "system.decoder.phase[0]",
        "system.decoder.phase[1]",
    }
    return set(values).issuperset(expected) and all(float(values[name]) > 0.0 for name in expected)


def _swd_encoder_only(values: Mapping[str, float]) -> bool:
    encoder = {"system.encoder.phase[0]", "system.encoder.phase[1]"}
    decoder = {"system.decoder.phase[0]", "system.decoder.phase[1]"}
    return set(values).issuperset(encoder | decoder) and all(float(values[name]) > 0.0 for name in encoder) and all(float(values[name]) == 0.0 for name in decoder)


def _mse_gradients(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, amplitudes: torch.Tensor, targets: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gradients = _grad_zeros(model)
    losses: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    port_shape: list[int] | None = None
    _clear_grad(model)
    for start in range(0, BATCH_SIZE, MICROBATCH):
        raw = checkpoint(
            lambda value: _smf_forward(model, smf, value)[0],
            amplitudes[start : start + 1],
            use_reentrant=False,
        )
        diagnostics = smf.last_diagnostics
        coefficients = diagnostics.get("port_input_coefficients")
        if isinstance(coefficients, torch.Tensor):
            port_shape = list(coefficients.shape)
        loss = F.mse_loss(raw, targets[start : start + 1])
        losses.append(loss.detach())
        cameras.append(raw.detach())
        (loss / BATCH_SIZE).backward()
    _add_grads(gradients, model)
    _clear_grad(model)
    camera = torch.cat(cameras)
    return gradients, {
        "loss": float(torch.stack(losses).mean().cpu()),
        "camera_finite": bool(torch.isfinite(camera).all()),
        "camera_min": float(camera.min().cpu()),
        "camera_max": float(camera.max().cpu()),
        "port_coefficients_shape": port_shape,
        "gradient_norms": _layer_norms(gradients),
    }


def _projection_on_device(projection: regularization.Projection, device: torch.device) -> regularization.Projection:
    return regularization.Projection(
        mean_field=projection.mean_field.to(device),
        basis=projection.basis.to(device),
        whitening=projection.whitening.to(device),
        eigenvalues=projection.eigenvalues.to(device),
        explained_variance=projection.explained_variance.to(device),
    )


def _rank_coefficients(scores: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != (BATCH_SIZE, RANK) or directions.shape != (DIRECTIONS, RANK):
        raise ValueError("SMF canary rank replay shapes are invalid")
    projected = scores @ directions.T
    order = projected.argsort(dim=0)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(BATCH_SIZE, device=scores.device).reshape(-1, 1).expand_as(order))
    expected = torch.arange(BATCH_SIZE, device=scores.device).reshape(-1, 1).expand_as(ranks)
    if not torch.equal(torch.sort(ranks, dim=0).values, expected):
        raise ValueError("SMF canary rank assignment is not a per-direction permutation")
    quantiles = regularization._normal_midpoint_quantiles(BATCH_SIZE, scores.device)
    target = quantiles.index_select(0, ranks.reshape(-1)).reshape_as(ranks)
    coefficients = 2.0 * (projected - target).div(float(BATCH_SIZE * DIRECTIONS))
    return coefficients, ranks


def _swd_gradients(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, amplitudes: torch.Tensor, projection: regularization.Projection, directions: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    projection = _projection_on_device(projection, amplitudes.device)
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, MICROBATCH):
            _raw, pre, _diagnostics = _smf_forward(model, smf, amplitudes[start : start + 1], trace=True)
            scores.append(regularization._project_fields(pre, projection).detach())
    score_batch = torch.cat(scores)
    coefficients, ranks = _rank_coefficients(score_batch, directions)
    gradients = _grad_zeros(model)
    for start in range(0, BATCH_SIZE, MICROBATCH):
        _clear_grad(model)
        _raw, pre, _diagnostics = _smf_forward(model, smf, amplitudes[start : start + 1], trace=True)
        score = regularization._project_fields(pre, projection)
        (score @ directions.T * coefficients[start : start + 1]).sum().backward()
        _add_grads(gradients, model)
    _clear_grad(model)
    swd = float(regularization._sliced_wasserstein(score_batch, directions).detach().cpu())
    return gradients, swd, {
        "score_shape": list(score_batch.shape),
        "rank_shape": list(ranks.shape),
        "rank_assignment_is_permutation": bool(torch.equal(torch.sort(ranks, dim=0).values, torch.arange(BATCH_SIZE, device=ranks.device).reshape(-1, 1).expand_as(ranks))),
    }


def _swd_observation(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, amplitudes: torch.Tensor, projection: regularization.Projection, directions: torch.Tensor) -> tuple[float, dict[str, Any]]:
    projection = _projection_on_device(projection, amplitudes.device)
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, MICROBATCH):
            _raw, pre, _diagnostics = _smf_forward(model, smf, amplitudes[start : start + 1], trace=True)
            scores.append(regularization._project_fields(pre, projection).detach())
    score_batch = torch.cat(scores)
    _coefficients, ranks = _rank_coefficients(score_batch, directions)
    return float(regularization._sliced_wasserstein(score_batch, directions).detach().cpu()), {
        "score_shape": list(score_batch.shape),
        "rank_shape": list(ranks.shape),
        "rank_assignment_is_permutation": bool(torch.equal(torch.sort(ranks, dim=0).values, torch.arange(BATCH_SIZE, device=ranks.device).reshape(-1, 1).expand_as(ranks))),
    }


def _step(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, smf: FiberArrayOLS, amplitudes: torch.Tensor, targets: torch.Tensor, projection: regularization.Projection, directions: torch.Tensor, arm: str, lambda_value: float) -> dict[str, Any]:
    before = {name: value.detach().clone() for name, value in model.named_parameters() if name in EXPECTED_TRAINABLE}
    mse_grad, mse_info = _mse_gradients(model, smf, amplitudes, targets)
    if arm == "smf_rank32_swd":
        swd_grad, swd_value, replay = _swd_gradients(model, smf, amplitudes, projection, directions)
    else:
        swd_grad = _grad_zeros(model)
        swd_value, replay = _swd_observation(model, smf, amplitudes, projection, directions)
    combined = {name: mse_grad[name] + lambda_value * swd_grad[name] for name in mse_grad}
    for name, value in model.named_parameters():
        value.grad = combined[name].to(value.device)
    optimizer.step()
    parameters = dict(model.named_parameters())
    updates: dict[str, float] = {}
    for name in EXPECTED_TRAINABLE:
        updates[name] = float((parameters[name].detach() - before[name]).abs().sum().cpu())
        for index in range(parameters[name].shape[0]):
            updates[f"{name}[{index}]"] = float((parameters[name][index].detach() - before[name][index]).abs().sum().cpu())
    combined_norms = _layer_norms(combined)
    _clear_grad(model)
    finite_values = [mse_info["loss"], mse_info["camera_min"], mse_info["camera_max"], swd_value, *combined_norms.values(), *updates.values()]
    return {
        "arm": arm,
        "loss": mse_info["loss"],
        "swd": swd_value,
        "camera_finite": mse_info["camera_finite"],
        "camera_min": mse_info["camera_min"],
        "camera_max": mse_info["camera_max"],
        "port_coefficients_shape": mse_info["port_coefficients_shape"],
        "reconstruction_gradient_norms": mse_info["gradient_norms"],
        "regularizer_gradient_norms": _layer_norms(swd_grad),
        "combined_gradient_norms": combined_norms,
        "update_norms": updates,
        "replay": replay,
        "finite": mse_info["camera_finite"] and all(bool(torch.isfinite(torch.tensor(value))) for value in finite_values),
    }


def _capture_projection_fields(model: base.IdentityOpticalManifoldAutoencoder, data: CanaryData, panels: tuple[torch.Tensor, ...], device: torch.device, *, spec: CanarySpec, validation_hash: str, source: SourceBundle) -> tuple[regularization.Projection, dict[str, Any]]:
    fields: list[torch.Tensor] = []
    provenance: dict[str, Any] | None = None
    for panel in panels:
        panel_fields: list[torch.Tensor] = []
        with torch.no_grad():
            for index in panel:
                amplitude, _target = _batch(data, index.reshape(1), device)
                _raw, pre, plane, pre_provenance = regularization._forward_trace(model, amplitude)
                panel_fields.append(pre.detach().cpu().to(torch.complex64))
                current = {"pre_ols_plane": plane, "pre_ols_provenance": pre_provenance, "post_ols": {"plane_id": "ols_output", "operation": "identity_ols"}}
                if provenance is None:
                    provenance = current
                elif provenance != current:
                    raise ValueError("SMF canary projection provenance changed")
        fields.append(torch.cat(panel_fields))
    fits = tuple(adaptive._fit_projection_rank(field, RANK) for field in fields)
    projection = adaptive._consensus_projection(fits, torch.cat(fields), RANK)
    payload = {
        "schema": PROJECTION_SCHEMA,
        "config_hash": spec.config_hash,
        "validation_hash": validation_hash,
        "source_hashes": dict(source.hashes),
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "rank": RANK,
        "basis": projection.basis,
        "mean_field": projection.mean_field,
        "whitening": projection.whitening,
        "eigenvalues": projection.eigenvalues,
        "explained_variance": projection.explained_variance,
        "source_panel_indices_sha256": [tensor_sha256(panel) for panel in panels],
        "status": "diagnostic_unstable_projection",
        "optical_provenance": provenance,
        "source_projection_audit_sha256": SOURCE_AUDIT_HASH,
    }
    return projection, payload


def _lambda(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, data: CanaryData, projection: regularization.Projection, directions: torch.Tensor, device: torch.device) -> float:
    amplitudes, targets = _batch(data, data.schedule[0], device)
    reconstruction, _ = _mse_gradients(model, smf, amplitudes, targets)
    regularizer_grad, _swd, _replay = _swd_gradients(model, smf, amplitudes, projection, directions)
    rec_norm = sum(float(reconstruction[name].norm()) for name in EXPECTED_TRAINABLE)
    reg_norm = sum(float(regularizer_grad[name].norm()) for name in EXPECTED_TRAINABLE if name.startswith("system.encoder.phase"))
    if rec_norm <= 0.0 or reg_norm <= 0.0:
        raise ValueError("SMF canary cannot calibrate a finite nonzero lambda")
    return 0.1 * rec_norm / reg_norm


def _probe_metrics(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, data: CanaryData, indices: torch.Tensor, projection: regularization.Projection, directions: torch.Tensor, device: torch.device) -> dict[str, Any]:
    projection = _projection_on_device(projection, device)
    losses: list[torch.Tensor] = []
    cyclic: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    fields: list[torch.Tensor] = []
    labels = data.labels.index_select(0, indices)
    with torch.no_grad():
        for position, index in enumerate(indices):
            amplitude, target = _batch(data, index.reshape(1), device)
            raw, pre, diagnostics = _smf_forward(model, smf, amplitude, trace=True)
            losses.append(F.mse_loss(raw, target).cpu())
            cyclic_target = _batch(data, indices[(position + 1) % len(indices)].reshape(1), device)[1]
            cyclic.append(F.mse_loss(raw, cyclic_target).cpu())
            scores.append(regularization._project_fields(pre, projection).cpu())
            fields.append(regularization._unit_power_field(pre).flatten(1).cpu())
    score_tensor = torch.cat(scores)
    field_tensor = torch.cat(fields)
    ratio = _structure_ratio(torch.cat((field_tensor.real, field_tensor.imag), dim=1), labels)
    coupling_values: list[float] = []
    active_port_values: list[float] = []
    for position, index in enumerate(indices):
        amplitude, _target = _batch(data, index.reshape(1), device)
        _raw, _pre, current = _smf_forward(model, smf, amplitude, trace=True)
        coupling = current.get("coupling_efficiency")
        coefficients = current.get("port_coefficients")
        if isinstance(coupling, torch.Tensor):
            coupling_values.extend(float(item) for item in coupling.reshape(-1).detach().cpu())
        if isinstance(coefficients, torch.Tensor):
            active_port_values.append(float((coefficients.abs().square().sum(dim=0) > 1.0e-12).float().mean().cpu()))
    return {
        "paired_mse": float(torch.stack(losses).mean()),
        "cyclic_mse": float(torch.stack(cyclic).mean()),
        "swd": float(regularization._sliced_wasserstein(score_tensor.to(device), directions).cpu()),
        "normalized_inter_intra": ratio,
        "coupling_efficiency_mean": float(sum(coupling_values) / len(coupling_values)) if coupling_values else 0.0,
        "active_port_fraction_mean": float(sum(active_port_values) / len(active_port_values)) if active_port_values else 0.0,
        "finite": bool(torch.isfinite(torch.stack(losses)).all() and torch.isfinite(torch.stack(cyclic)).all() and torch.isfinite(score_tensor).all()),
    }


def _structure_ratio(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    within: list[torch.Tensor] = []
    between: list[torch.Tensor] = []
    means = []
    for label in range(10):
        group = scores[labels == label]
        means.append(group.mean(dim=0))
        if group.shape[0] > 1:
            within.append(torch.pdist(group).mean())
    for left in range(10):
        for right in range(left + 1, 10):
            between.append((means[left] - means[right]).norm())
    intra = torch.stack(within).mean().clamp_min(1.0e-12)
    inter = torch.stack(between).mean()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    permuted = labels[torch.randperm(labels.numel(), generator=generator)]
    permuted_within: list[torch.Tensor] = []
    permuted_means = []
    for label in range(10):
        group = scores[permuted == label]
        permuted_means.append(group.mean(dim=0))
        if group.shape[0] > 1:
            permuted_within.append(torch.pdist(group).mean())
    permuted_intra = torch.stack(permuted_within).mean().clamp_min(1.0e-12)
    permuted_inter = torch.stack([permuted_means[left].sub(permuted_means[right]).norm() for left in range(10) for right in range(left + 1, 10)]).mean()
    return {
        "true_inter_intra": float((inter / intra).item()),
        "permuted_inter_intra": float((permuted_inter / permuted_intra).item()),
    }


def _resume_payload(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, arm: str, step: int, history: list[dict[str, Any]], spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], lambda_value: float, completed_arms: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "model_state": _state_cpu(model),
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "rng_state": _tree_clone(base._rng_state()),
        "history": list(history),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(hashes),
        "projection_hash": hashes.get("projection"),
        "lambda": lambda_value,
        "data_cursor": {"arm": arm, "next_step": step, "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH},
        "completed_arms": dict(completed_arms or {}),
        "formal_candidate_checkpoint": False,
        "training_authorized": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite SMF canary checkpoint: {path}")
    atomic_torch_save(path, payload)


def _load_resume(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], lambda_value: float, arm: str, output_dir: Path) -> dict[str, Any]:
    target = _resolve(path)
    if target.parent.resolve() != output_dir.resolve():
        raise ValueError("explicit SMF canary resume checkpoint must belong to the requested attempt")
    payload = safe_torch_load(target, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid SMF canary resume artifact")
    if payload.get("arm") != arm or not 0 < int(payload.get("step", 0)) < STEPS:
        raise ValueError("SMF canary resume arm or step is invalid")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"]:
        raise ValueError("SMF canary resume data lineage mismatch")
    if payload.get("evidence_hashes") != dict(hashes) or payload.get("projection_hash") != hashes.get("projection") or float(payload.get("lambda")) != lambda_value:
        raise ValueError("SMF canary resume source lineage mismatch")
    cursor = payload.get("data_cursor")
    if cursor != {"arm": arm, "next_step": int(payload["step"]), "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH}:
        raise ValueError("SMF canary resume data cursor is invalid")
    if not isinstance(payload.get("history"), list) or len(payload["history"]) != int(payload["step"]):
        raise ValueError("SMF canary resume history is inconsistent")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return dict(payload)


def _run_steps(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, smf: FiberArrayOLS, data: CanaryData, projection: regularization.Projection, directions: torch.Tensor, arm: str, lambda_value: float, start: int, history: list[dict[str, Any]], output_dir: Path, spec: CanarySpec, hashes: Mapping[str, str], *, save_checkpoints: bool, completed_arms: Mapping[str, Any] | None = None, initial_batch_losses: list[float] | None = None) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    final_path: Path | None = None
    final_hash: str | None = None
    for step in range(start, STEPS):
        amplitudes, targets = _batch(data, data.schedule[step], next(model.parameters()).device)
        result = _step(model, optimizer, smf, amplitudes, targets, projection, directions, arm, lambda_value)
        result["step"] = step + 1
        history.append(result)
        if not result["finite"]:
            raise ValueError("SMF canary encountered non-finite step values")
        if initial_batch_losses is not None and result["loss"] > initial_batch_losses[step] * SAFETY_MSE_RATIO:
            raise _SafetyAbort(step + 1, result)
        if save_checkpoints and ((step + 1) % 10 == 0):
            path = output_dir / f"{arm}_step_{step + 1:05d}.pt"
            payload = _resume_payload(model, optimizer, arm=arm, step=step + 1, history=history, spec=spec, data=data, hashes=hashes, lambda_value=lambda_value, completed_arms=completed_arms)
            _save_checkpoint(path, payload)
            final_path, final_hash = path, file_sha256(path)
    return history, final_path, final_hash


def _preflight_record(state: Mapping[str, Any], spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, hashes: Mapping[str, str], gates: tuple[GateResult, ...], status: str, actions: tuple[str, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(schema_version=2, evidence_kind=PREFLIGHT_SCHEMA, tier="diagnostic", status=status, git_commit=str(state["commit"]), dirty_patch_hash=None, config_hash=spec.config_hash, context=_context(spec, formal, data), input_hashes=dict(hashes), gates=gates, permitted_next_actions=actions, invalidation_reason=reason)
    record.validate()
    return record


def _exact_resume(formal: FormalSystemConfig, source: SourceBundle, data: CanaryData, projection: regularization.Projection, directions: torch.Tensor, arm: str, lambda_value: float, device: torch.device, spec: CanarySpec, hashes: Mapping[str, str], output_dir: Path) -> dict[str, Any]:
    continuous_model, continuous_opt, continuous_smf = _runtime(formal, source, device)
    split_model, split_opt, split_smf = _runtime(formal, source, device)
    continuous_nonphase = _nonphase_state_cpu(continuous_model)
    split_nonphase = _nonphase_state_cpu(split_model)
    continuous_smf_state = _module_state_cpu(continuous_smf)
    split_smf_state = _module_state_cpu(split_smf)
    continuous_history: list[dict[str, Any]] = []
    split_history: list[dict[str, Any]] = []
    for step in range(3):
        amplitudes, targets = _batch(data, data.schedule[step], device)
        continuous_history.append(_step(continuous_model, continuous_opt, continuous_smf, amplitudes, targets, projection, directions, arm, lambda_value))
    continuous_rng = _tree_clone(base._rng_state())
    for step in range(2):
        amplitudes, targets = _batch(data, data.schedule[step], device)
        split_history.append(_step(split_model, split_opt, split_smf, amplitudes, targets, projection, directions, arm, lambda_value))
    resume_path = output_dir / f"preflight_{arm}_step_00002.pt"
    resume = _resume_payload(split_model, split_opt, arm=arm, step=2, history=split_history, spec=spec, data=data, hashes=hashes, lambda_value=lambda_value, completed_arms={})
    _save_checkpoint(resume_path, resume)
    resumed_model, resumed_opt, resumed_smf = _runtime(formal, source, device)
    _load_resume(resume_path, resumed_model, resumed_opt, spec=spec, data=data, hashes=hashes, lambda_value=lambda_value, arm=arm, output_dir=output_dir)
    amplitudes, targets = _batch(data, data.schedule[2], device)
    split_history.append(_step(resumed_model, resumed_opt, resumed_smf, amplitudes, targets, projection, directions, arm, lambda_value))
    # The resumed run has just consumed the same third batch.
    resumed_rng = _tree_clone(base._rng_state())
    return {
        "continuous": continuous_history,
        "resumed": split_history,
        "model_exact": all(torch.equal(continuous_model.state_dict()[name].detach().cpu(), resumed_model.state_dict()[name].detach().cpu()) for name in continuous_model.state_dict()),
        "optimizer_exact": _tree_equal(continuous_opt.state_dict(), resumed_opt.state_dict()),
        "rng_exact": _tree_equal(continuous_rng, resumed_rng),
        "frozen_state_exact": _tree_equal(continuous_nonphase, _nonphase_state_cpu(continuous_model)) and _tree_equal(split_nonphase, _nonphase_state_cpu(resumed_model)),
        "smf_state_exact": _tree_equal(continuous_smf_state, _module_state_cpu(continuous_smf)) and _tree_equal(split_smf_state, _module_state_cpu(resumed_smf)),
        "resume_artifact": str(resume_path.resolve()),
        "resume_artifact_sha256": file_sha256(resume_path),
        "finite": all(item["finite"] for item in continuous_history + split_history),
    }


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF canary preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SMF canary preflight: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for SMF canary preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        source_model, _source_opt, source_smf = _runtime(formal, source, device)
        source_model_state = _state_cpu(source_model)
        source_formal_ols_state = _module_state_cpu(source_model.system.ols)
        source_smf_state = _module_state_cpu(source_smf)
        calibration_data = adaptive._load_data(args.dataset_root)
        panels = calibration_data.calibration_panels
        projection, projection_payload = _capture_projection_fields(source_model, data, panels, device, spec=spec, validation_hash=validation_hash, source=source)
        directions = regularization._fixed_directions(device)
        projection_path = output.parent / "diagnostic_unstable_rank32_projection.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_checkpoint(projection_path, projection_payload)
        projection_hash = file_sha256(projection_path)
        hashes = {**source.hashes, "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash, "projection": projection_hash}
        lambda_value = _lambda(source_model, source_smf, data, projection, directions, device)
        exact = {arm: _exact_resume(formal, source, data, projection, directions, arm, lambda_value, device, spec, hashes, output.parent) for arm in ("smf_rank32_swd", "smf_mse_only")}
        reports: dict[str, Any] = {}
        for arm in ("smf_rank32_swd", "smf_mse_only"):
            model, optimizer, smf = _runtime(formal, source, device)
            batch_amplitudes, batch_targets = _batch(data, data.schedule[0], device)
            initial = _step(model, optimizer, smf, batch_amplitudes, batch_targets, projection, directions, arm, lambda_value)
            reports[arm] = {"step": initial, "exact_resume": exact[arm]}
        zero_model, _zero_opt, zero_smf = _runtime(formal, source, device)
        zero_smf = _smf_operator(zero_model.system, device, zero_transport=True)
        zero_amplitude, _zero_target = _batch(data, data.schedule[0, :1], device)
        zero_raw, _zero_pre, zero_diag = _smf_forward(zero_model, zero_smf, zero_amplitude, trace=True)
        identity_amplitude, _identity_target = _batch(data, data.schedule[0, :1], device)
        identity_raw, _identity_pre, identity_plane, identity_provenance = regularization._forward_trace(source_model, identity_amplitude)
        smf_raw, _smf_pre, smf_diag = _smf_forward(source_model, source_smf, identity_amplitude, trace=True)
        identity_has_ols = _identity_trace_has_ols(source_model, identity_amplitude)
        smf_has_transport = smf_diag["smf_provenance"].get("operation") == "synthetic_smf_lp01_port_transport" and smf_diag["port_coefficients"].shape == (1, PORT_COUNT)
        frozen_formal = _tree_equal(source_model_state, _state_cpu(source_model)) and _tree_equal(source_formal_ols_state, _module_state_cpu(source_model.system.ols))
        frozen_smf = _tree_equal(source_smf_state, _module_state_cpu(source_smf))
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        regularized = reports["smf_rank32_swd"]["step"]
        mse_only = reports["smf_mse_only"]["step"]
        gates = (
            GateResult("finite_values", all(item["finite"] for item in (regularized, mse_only)), {}),
            GateResult("four_phase_reconstruction_updates", _four_phase_nonzero(regularized["update_norms"]), {"layers": 4}),
            GateResult("swd_rank_permutation", regularized["replay"]["rank_assignment_is_permutation"], {}),
            GateResult("swd_encoder_only", _swd_encoder_only(regularized["regularizer_gradient_norms"]), {}),
            GateResult("exact_resume_both_arms", all(exact[arm]["model_exact"] and exact[arm]["optimizer_exact"] and exact[arm]["rng_exact"] and exact[arm]["frozen_state_exact"] and exact[arm]["smf_state_exact"] and exact[arm]["finite"] for arm in exact), {}),
            GateResult("smf_100_port_coefficients", regularized["port_coefficients_shape"] == [1, PORT_COUNT], {"shape": regularized["port_coefficients_shape"], "ports": PORT_COUNT}),
            GateResult("smf_passive_zero_transport_control", bool(torch.isfinite(zero_raw).all()) and float(zero_raw.abs().max()) <= 1.0e-12 and bool(torch.isfinite(zero_diag["output_energy"]).all()) and bool((zero_diag["output_energy"] <= zero_diag["input_energy"] * (1.0 + 1.0e-5) + 1.0e-12).all()), {"max_abs": float(zero_raw.abs().max())}),
            GateResult("identity_and_smf_are_not_bypassed", identity_has_ols and smf_has_transport and not torch.equal(identity_raw, smf_raw), {"identity_has_identity_ols": identity_has_ols, "smf_has_transport": smf_has_transport}),
            GateResult("frozen_formal_state", frozen_formal and frozen_smf, {"formal_identity_ols_unchanged": frozen_formal, "smf_buffers_unchanged": frozen_smf}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        passed = all(gate.passed for gate in gates)
        record = _preflight_record(state, spec, formal, data, hashes, gates, "passed" if passed else "invalid_harness", ("run_smf_integration_canary",) if passed else (), None if passed else "SMF canary preflight harness gate failed")
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"validation_hash": validation_hash, "projection_hash": projection_hash, "projection_path": str(projection_path.resolve()), "lambda": lambda_value, "arms": reports, "zero_transport_control": {"finite": bool(torch.isfinite(zero_raw).all()), "max_abs": float(zero_raw.abs().max()), "input_energy_max": float(zero_diag["input_energy"].max().cpu()), "output_energy_max": float(zero_diag["output_energy"].max().cpu()), "smf_provenance": zero_diag["smf_provenance"]}, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "trend_metrics_non_gating": True})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if passed else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def _load_preflight(path: str | Path, spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, state: Mapping[str, Any], validation_hash: str, source: SourceBundle) -> tuple[dict[str, Any], str, dict[str, str]]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("SMF canary preflight schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.status != "passed" or record.evidence_kind != PREFLIGHT_SCHEMA:
        raise ValueError("SMF canary preflight must be passed")
    if record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("SMF canary preflight is stale")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("SMF canary preflight contract is missing")
    projection_hash = contract.get("projection_hash")
    expected = {**source.hashes, "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash, "projection": projection_hash}
    if record.input_hashes != expected or not isinstance(projection_hash, str):
        raise ValueError("SMF canary preflight source hashes do not match")
    projection_path = _resolve(contract["projection_path"])
    if file_sha256(projection_path) != projection_hash:
        raise ValueError("SMF canary projection artifact hash mismatch")
    if record.permitted_next_actions != ("run_smf_integration_canary",):
        raise ValueError("SMF canary preflight next action is invalid")
    return payload, digest, expected


def _training_record(state: Mapping[str, Any], spec: CanarySpec, formal: FormalSystemConfig, data: CanaryData, hashes: Mapping[str, str], status: str, actions: tuple[str, ...], gates: tuple[GateResult, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(schema_version=2, evidence_kind=TRAINING_SCHEMA, tier="diagnostic", status=status, git_commit=str(state["commit"]), dirty_patch_hash=None, config_hash=spec.config_hash, context=_context(spec, formal, data), input_hashes=dict(hashes), gates=gates, permitted_next_actions=actions, invalidation_reason=reason)
    record.validate()
    return record


def _save_panel(path: Path, model_identity: base.IdentityOpticalManifoldAutoencoder, model_mse: base.IdentityOpticalManifoldAutoencoder, model_reg: base.IdentityOpticalManifoldAutoencoder, smf_identity: FiberArrayOLS, smf_mse: FiberArrayOLS, smf_reg: FiberArrayOLS, data: CanaryData, device: torch.device, source_hashes: Mapping[str, str]) -> str:
    inputs, targets = _batch(data, data.visual_indices, device)
    values: dict[str, torch.Tensor] = {"indices": data.visual_indices.cpu(), "input_28": data.images.index_select(0, data.visual_indices).cpu(), "target_200": targets.cpu()}
    with torch.no_grad():
        identity = []
        smf_initial = []
        mse = []
        reg = []
        coupling = []
        for start in range(0, len(data.visual_indices), BATCH_SIZE):
            amp = inputs[start : start + BATCH_SIZE]
            identity.append(model_identity(amp, activation_checkpoint=False).raw_camera.detach().cpu())
            raw_initial, _pre_initial, _diag_initial = _smf_forward(model_identity, smf_identity, amp)
            smf_initial.append(raw_initial.detach().cpu())
            raw_mse, _pre_mse, _diag_mse = _smf_forward(model_mse, smf_mse, amp)
            raw_reg, _pre_reg, diag_reg = _smf_forward(model_reg, smf_reg, amp)
            mse.append(raw_mse.detach().cpu())
            reg.append(raw_reg.detach().cpu())
            if isinstance(diag_reg.get("coupling_efficiency"), torch.Tensor):
                coupling.append(diag_reg["coupling_efficiency"].detach().cpu())
    values.update({"identity_source": torch.cat(identity), "smf_initial": torch.cat(smf_initial), "smf_mse_only_step20": torch.cat(mse), "smf_rank32_swd_step20": torch.cat(reg), "port_coupling_efficiency": torch.cat(coupling) if coupling else torch.empty(0)})
    payload = {"schema": PANEL_SCHEMA, "source_hashes": dict(source_hashes), "indices_are_global_mnist": True, "tensors": values}
    atomic_torch_save(path, payload)
    try:
        from PIL import Image
        rows = []
        for key in ("target_200", "smf_initial", "smf_mse_only_step20", "smf_rank32_swd_step20"):
            images = values[key].squeeze(1).clamp(0.0, 1.0).mul(255).to(torch.uint8)
            thumb = F.interpolate(images.unsqueeze(1).float(), size=(32, 32), mode="bilinear", align_corners=False).squeeze(1).to(torch.uint8)
            rows.append(torch.cat([thumb[index] for index in range(100)], dim=1))
        grid = torch.cat(rows, dim=0).numpy()
        Image.fromarray(grid, mode="L").save(path.with_suffix(".png"))
    except ImportError:
        pass
    return file_sha256(path)


def _initial_batch_losses(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, data: CanaryData, device: torch.device) -> list[float]:
    values: list[float] = []
    with torch.no_grad():
        for step in range(STEPS):
            amplitudes, targets = _batch(data, data.schedule[step], device)
            raw, _pre, _diagnostics = _smf_forward(model, smf, amplitudes)
            values.append(float(F.mse_loss(raw, targets).cpu()))
    return values


def _load_completed_arm(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, spec: CanarySpec, data: CanaryData, hashes: Mapping[str, str], lambda_value: float, arm: str, output_dir: Path, expected_sha256: str) -> dict[str, Any]:
    target = _resolve(path)
    if file_sha256(target) != expected_sha256:
        raise ValueError("completed SMF canary arm checkpoint hash mismatch")
    payload = safe_torch_load(target, expected_sha256=expected_sha256, map_location="cpu")
    if payload.get("arm") != arm or int(payload.get("step", 0)) != STEPS:
        raise ValueError("completed SMF canary arm checkpoint is not a final checkpoint")
    # Reuse the same lineage fields as an incomplete resume, while allowing
    # the terminal step for the completed arm.
    if payload.get("schema") != RESUME_SCHEMA or payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes) or payload.get("projection_hash") != hashes.get("projection") or float(payload.get("lambda")) != lambda_value:
        raise ValueError("completed SMF canary arm checkpoint lineage mismatch")
    if target.parent.resolve() != output_dir.resolve():
        raise ValueError("completed SMF canary arm checkpoint belongs to a different attempt")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    if not isinstance(payload.get("history"), list) or len(payload["history"]) != STEPS:
        raise ValueError("completed SMF canary arm history is inconsistent")
    return dict(payload)


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF canary training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source)
    preflight_payload, preflight_hash, hashes = _load_preflight(args.preflight_evidence, spec, formal, data, state, validation_hash, source)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SMF canary training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for SMF canary training")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    preflight_contract = preflight_payload["contract"]
    projection_payload = safe_torch_load(preflight_contract["projection_path"], expected_sha256=preflight_contract["projection_hash"], map_location="cpu")
    if projection_payload.get("schema") != PROJECTION_SCHEMA or projection_payload.get("config_hash") != spec.config_hash or projection_payload.get("validation_hash") != validation_hash or projection_payload.get("source_hashes") != source.hashes or projection_payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]:
        raise ValueError("SMF canary projection provenance does not match training inputs")
    projection = regularization.Projection(projection_payload["mean_field"], projection_payload["basis"], projection_payload["whitening"], projection_payload["eigenvalues"], projection_payload["explained_variance"])
    lambda_value = float(preflight_contract["lambda"])
    directions = regularization._fixed_directions(device)
    torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        source_identity, _source_identity_opt, source_identity_smf = _runtime(formal, source, device)
        source_identity_state = _state_cpu(source_identity)
        source_identity_ols_state = _module_state_cpu(source_identity.system.ols)
        source_identity_smf_state = _module_state_cpu(source_identity_smf)
        baseline = _probe_metrics(source_identity, source_identity_smf, data, data.visual_indices, projection, directions, device)
        safety_baselines = _initial_batch_losses(source_identity, source_identity_smf, data, device)
        arms: dict[str, Any] = {}
        checkpoints: dict[str, Any] = {}
        trained_models: dict[str, base.IdentityOpticalManifoldAutoencoder] = {}
        trained_smfs: dict[str, FiberArrayOLS] = {}
        training_hashes = {**hashes, "training_preflight": preflight_hash}
        resume_payload: Mapping[str, Any] | None = None
        resume_arm: str | None = None
        if args.resume_checkpoint:
            resume_target = _resolve(args.resume_checkpoint)
            if resume_target.parent.resolve() != output.parent.resolve():
                raise ValueError("explicit SMF canary resume checkpoint must belong to the requested attempt")
            resume_payload = safe_torch_load(resume_target, map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in {"smf_rank32_swd", "smf_mse_only"}:
                raise ValueError("explicit SMF canary resume checkpoint has an invalid arm")

        def run_arm(arm: str, *, resume_path: Path | None = None, completed_arms: Mapping[str, Any] | None = None) -> None:
            model, optimizer, smf = _runtime(formal, source, device)
            history: list[dict[str, Any]] = []
            start = 0
            if resume_path is not None:
                resume = _load_resume(resume_path, model, optimizer, spec=spec, data=data, hashes=training_hashes, lambda_value=lambda_value, arm=arm, output_dir=output.parent)
                start = int(resume["step"])
                history = list(resume["history"])
            try:
                history, final_path, final_hash = _run_steps(model, optimizer, smf, data, projection, directions, arm, lambda_value, start, history, output.parent, spec, training_hashes, save_checkpoints=True, completed_arms=completed_arms, initial_batch_losses=safety_baselines)
            except _SafetyAbort:
                arms[arm] = {"steps_completed": len(history), "history": history, "metrics": None, "final_checkpoint": None, "final_checkpoint_sha256": None, "model_state": _state_cpu(model)}
                trained_models[arm] = model
                trained_smfs[arm] = smf
                raise
            metrics = _probe_metrics(model, smf, data, data.visual_indices, projection, directions, device)
            arms[arm] = {"steps_completed": len(history), "history": history, "metrics": metrics, "final_checkpoint": str(final_path.resolve()) if final_path else None, "final_checkpoint_sha256": final_hash, "model_state": _state_cpu(model)}
            checkpoints[arm] = {"path": arms[arm]["final_checkpoint"], "sha256": final_hash}
            trained_models[arm] = model
            trained_smfs[arm] = smf
            _clear_grad(model)
            torch.cuda.empty_cache()

        safety_abort: _SafetyAbort | None = None
        try:
            if resume_arm == "smf_mse_only":
                if not isinstance(resume_payload, Mapping) or not isinstance(resume_payload.get("completed_arms"), Mapping):
                    raise ValueError("MSE-only resume is missing completed regularized-arm binding")
                completed = resume_payload["completed_arms"].get("smf_rank32_swd")
                if not isinstance(completed, Mapping) or not isinstance(completed.get("path"), str) or not isinstance(completed.get("sha256"), str):
                    raise ValueError("MSE-only resume has no completed regularized-arm checkpoint")
                reg_model, reg_optimizer, reg_smf = _runtime(formal, source, device)
                completed_payload = _load_completed_arm(completed["path"], reg_model, reg_optimizer, spec=spec, data=data, hashes=training_hashes, lambda_value=lambda_value, arm="smf_rank32_swd", output_dir=output.parent, expected_sha256=completed["sha256"])
                trained_models["smf_rank32_swd"] = reg_model
                trained_smfs["smf_rank32_swd"] = reg_smf
                arms["smf_rank32_swd"] = {"steps_completed": STEPS, "history": list(completed_payload["history"]), "metrics": _probe_metrics(reg_model, reg_smf, data, data.visual_indices, projection, directions, device), "final_checkpoint": str(_resolve(completed["path"]).resolve()), "final_checkpoint_sha256": completed["sha256"], "model_state": _state_cpu(reg_model)}
                checkpoints["smf_rank32_swd"] = {"path": arms["smf_rank32_swd"]["final_checkpoint"], "sha256": completed["sha256"]}
                run_arm("smf_mse_only", resume_path=_resolve(args.resume_checkpoint), completed_arms={"smf_rank32_swd": checkpoints["smf_rank32_swd"]})
            else:
                run_arm("smf_rank32_swd", resume_path=_resolve(args.resume_checkpoint) if resume_arm == "smf_rank32_swd" else None)
                run_arm("smf_mse_only", completed_arms={"smf_rank32_swd": checkpoints["smf_rank32_swd"]})
        except _SafetyAbort as exc:
            safety_abort = exc

        panel_path: Path | None = None
        panel_hash: str | None = None
        if safety_abort is None:
            panel_path = output.parent / "smf_canary_visual_panel.pt"
            panel_hash = _save_panel(panel_path, source_identity, trained_models["smf_mse_only"], trained_models["smf_rank32_swd"], source_identity_smf, trained_smfs["smf_mse_only"], trained_smfs["smf_rank32_swd"], data, device, training_hashes)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        finite = all(item["finite"] for arm in arms.values() for item in arm["history"]) and all(item["metrics"] is not None and item["metrics"]["finite"] for item in arms.values())
        updates = all(_four_phase_nonzero(item["update_norms"]) for arm in arms.values() for item in arm["history"])
        frozen = _tree_equal(source_identity_state, _state_cpu(source_identity)) and _tree_equal(source_identity_ols_state, _module_state_cpu(source_identity.system.ols)) and _tree_equal(source_identity_smf_state, _module_state_cpu(source_identity_smf))
        gates = (
            GateResult("finite_values", finite, {}),
            GateResult("both_arms_completed_20_steps", all(item["steps_completed"] == STEPS for item in arms.values()), {}),
            GateResult("four_phase_updates_nonzero", updates, {"layers": 4}),
            GateResult("frozen_formal_state", frozen, {"formal_identity_ols_unchanged": _tree_equal(source_identity_ols_state, _module_state_cpu(source_identity.system.ols)), "source_smf_buffers_unchanged": _tree_equal(source_identity_smf_state, _module_state_cpu(source_identity_smf))}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("source_and_panel_provenance", panel_hash is not None and panel_path is not None and panel_path.is_file(), {"panel_sha256": panel_hash}),
            GateResult("safety_mse_bound", all(all(item["loss"] <= safety_baselines[item["step"] - 1] * SAFETY_MSE_RATIO for item in arm["history"]) for arm in arms.values()), {"ratio": SAFETY_MSE_RATIO}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        harness_ok = safety_abort is None and all(gate.passed for gate in gates)
        resource_failed = any(gate.name in {"cuda_peak_memory", "wall_time", "run_storage"} and not gate.passed for gate in gates)
        status = "diagnostic" if harness_ok else ("failed" if safety_abort is not None else ("resource_aborted" if resource_failed else "invalid_harness"))
        outcome = "diagnostic_completed" if harness_ok else ("safety_aborted" if safety_abort is not None else ("resource_aborted" if resource_failed else "invalid_harness"))
        reason = None if harness_ok else (str(safety_abort) if safety_abort is not None else "SMF canary harness or resource gate failed")
        record = _training_record(state, spec, formal, data, training_hashes, status, (NEXT_ACTION,) if harness_ok else (), gates, reason)
        payload = _record_payload(TRAINING_SCHEMA, record, {"outcome": outcome, "validation_hash": validation_hash, "preflight_hash": preflight_hash, "source_hashes": source.hashes, "baseline": baseline, "safety_baseline_batch_mse": safety_baselines, "safety_abort": {"step": safety_abort.step, "result": safety_abort.result} if safety_abort is not None else None, "arms": {arm: {key: value for key, value in result.items() if key != "model_state"} for arm, result in arms.items()}, "checkpoints": checkpoints, "panel_path": str(panel_path.resolve()) if panel_path is not None else None, "panel_sha256": panel_hash, "lambda": lambda_value, "formal_candidate_checkpoint": False, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "evaluator_calls": 0, "training_authorized": False, "formal_promotion": False})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if harness_ok else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-integration-canary-validate":
        return validate(args)
    if args.research_action == "smf-integration-canary-preflight":
        return preflight(args)
    if args.research_action == "smf-integration-canary-train":
        return train(args)
    raise ValueError(f"unsupported SMF canary action: {args.research_action}")
