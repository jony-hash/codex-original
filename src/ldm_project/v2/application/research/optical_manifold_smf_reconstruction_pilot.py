"""Matched Identity-versus-SMF reconstruction pilot for H-LOMA-I0.

This lifecycle is the first scientific PLAN 6 comparison.  It changes only
the research OLS between the unchanged formal 2+2 encoder and decoder; the
formal v2 IdentityOLS, current conditional runtime, and the PLAN 5c evidence
remain untouched.
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

from ...config import FormalSystemConfig, load_config
from ...contracts import ComplexField, OperationProvenance
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
from ...optics.devices import IdentityOLS
from ...optics.fiber_operator import FiberArrayOLS
from . import optical_manifold as base
from . import optical_manifold_adaptive_projection_audit as adaptive
from . import optical_manifold_full_epoch as full_epoch
from . import optical_manifold_regularization_audit as regularization
from . import optical_manifold_smf_integration_canary as canary


BRANCH = "research/v2-lightgen-optical-manifold-smf-reconstruction-pilot-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_smf_reconstruction_pilot_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_smf_reconstruction_pilot_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_reconstruction_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_reconstruction_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_reconstruction_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_smf_reconstruction_resume.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_identity_smf_comparison_panel.v1"

SOURCE_CANARY_VALIDATION_HASH = "bc87f1c996331dbe88f059b71faac2da0ce891c32969dd195c881a9eeff93bc2"
SOURCE_CANARY_PREFLIGHT_HASH = "70fc113ca98a4776d88837e3aab4293d0676d965ab98dbcc19f631820d1fe70f"
SOURCE_CANARY_TRAINING_HASH = "271f934515bde2fddaab413aba05e82472751b1664053c3a27d28de3c76a1860"
SOURCE_CANARY_PANEL_HASH = "2b08105870853a81bc8cc59c866de411b8c78cd06cc0da1bac408fc89500058f"
SOURCE_CHECKPOINT_HASH = "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_STEP = 5500
STEPS = 1000
PREFLIGHT_STEPS = 3
BATCH_SIZE = 10
MICROBATCH = 1
CHECKPOINT_INTERVAL = 100
SEED = 1234
LEARNING_RATE = 1.0e-3
PORT_COUNT = 100
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 3600.0
MAX_STORAGE_BYTES = 1024**3
SAFETY_MSE_RATIO = 2.0
SMF_REDUCTION_MIN = 0.05
IDENTITY_MAX_DEGRADATION = 0.02
NEXT_PASS_ACTION = "manual_review_fmf_m2_integration_canary_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_smf_bottleneck_attribution_contract"
ARMS = ("smf_mse_reconstruction", "identity_mse_control")
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


class _SafetyAbort(RuntimeError):
    def __init__(self, arm: str, step: int, result: Mapping[str, Any]) -> None:
        super().__init__(f"SMF reconstruction safety bound exceeded for {arm} at step {step}")
        self.arm = arm
        self.step = step
        self.result = dict(result)


@dataclass(frozen=True)
class PilotSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class PilotData:
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
        raise ValueError("SMF reconstruction pilot requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"SMF reconstruction pilot requires branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "SMF reconstruction pilot requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _load_spec(path: str | Path) -> PilotSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema", "hypothesis", "formal_system_config", "source", "dataset",
        "optical_contract", "training", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("SMF reconstruction pilot config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported SMF reconstruction pilot config")
    return PilotSpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: PilotSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    approved = {
        "arms": list(ARMS),
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH,
        "source_global_step": SOURCE_STEP,
        "preflight_steps": PREFLIGHT_STEPS,
        "resume_segment": "2+1",
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "ols_variable_only": True,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
    }
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_identity_smf_reconstruction_pilot",
        "approved_contract": approved,
        "bounded_diagnostic_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("SMF reconstruction pilot review contract does not match scope")
    return payload, target


def _assert_contract(spec: PilotSpec) -> None:
    payload = spec.payload
    source = payload["source"]
    for key, expected in (
        ("canary_validation_sha256", SOURCE_CANARY_VALIDATION_HASH),
        ("canary_preflight_sha256", SOURCE_CANARY_PREFLIGHT_HASH),
        ("canary_training_sha256", SOURCE_CANARY_TRAINING_HASH),
        ("canary_panel_sha256", SOURCE_CANARY_PANEL_HASH),
        ("source_checkpoint_sha256", SOURCE_CHECKPOINT_HASH),
        ("mnist_manifest_sha256", SOURCE_MANIFEST_HASH),
    ):
        if source[key] != expected:
            raise ValueError(f"SMF reconstruction source binding is not locked for {key}")
    if source["source_global_step"] != SOURCE_STEP:
        raise ValueError("SMF reconstruction source step is not locked")
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH,
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "excluded_prior_panels": True,
        "excluded_canary_schedule": True,
        "heldout_excluded": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
        "heldout_panel_count": 1000,
        "visual_panel_count": 100,
    }:
        raise ValueError("SMF reconstruction dataset contract is not locked")
    expected_optical = {
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
        "front_relay": "coherent_imaging_4f",
        "back_relay": "free_space",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }
    if payload["optical_contract"] != expected_optical:
        raise ValueError("SMF reconstruction optical contract is not locked")
    if payload["training"] != {
        "arms": list(ARMS),
        "optimizer": "Adam",
        "optimizer_update_device": "cpu_shadow",
        "learning_rate": LEARNING_RATE,
        "reset_adam_from_source": True,
        "steps_per_arm": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reconstruction_activation_checkpoint": True,
        "deterministic": True,
        "fixed_execution_order": list(ARMS),
        "no_early_stop": True,
        "no_tuning": True,
        "no_checkpoint_selection": True,
    }:
        raise ValueError("SMF reconstruction training contract is not locked")
    if payload["gates"] != {
        "smf_relative_mse_reduction_min": SMF_REDUCTION_MIN,
        "smf_paired_mse_below_cyclic": True,
        "smf_centered_correlation_improves": True,
        "smf_variance_retention_improves": True,
        "identity_max_relative_degradation": IDENTITY_MAX_DEGRADATION,
        "identity_paired_mse_below_cyclic": True,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_preflight_wall_seconds": MAX_PREFLIGHT_SECONDS,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "safety_mse_ratio": SAFETY_MSE_RATIO,
        "evaluator_calls": 0,
    }:
        raise ValueError("SMF reconstruction gate contract is not locked")
    auth = payload["authorization"]
    if (
        auth["bounded_diagnostic_training_authorized"] is not True
        or auth["training_authorized"] is not False
        or auth["formal_promotion"] is not False
        or auth["post_pass_next_action"] != NEXT_PASS_ACTION
        or auth["post_valid_failed_next_action"] != NEXT_VALID_FAILED_ACTION
    ):
        raise ValueError("SMF reconstruction authorization boundary is invalid")


def _load_formal(spec: PilotSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("SMF reconstruction requires current Identity lineage")
    if formal.system.kind != base.EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("SMF reconstruction requires formal Identity optics")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("SMF reconstruction source commit is not an ancestor")


def _load_source(spec: PilotSpec, formal_path: Path, current_commit: str) -> SourceBundle:
    source = spec.payload["source"]
    paths = {
        "canary_validation": (_resolve(source["canary_validation"]), SOURCE_CANARY_VALIDATION_HASH),
        "canary_preflight": (_resolve(source["canary_preflight"]), SOURCE_CANARY_PREFLIGHT_HASH),
        "canary_training": (_resolve(source["canary_training"]), SOURCE_CANARY_TRAINING_HASH),
        "canary_panel": (_resolve(source["canary_panel"]), SOURCE_CANARY_PANEL_HASH),
        "source_checkpoint": (_resolve(source["source_checkpoint"]), SOURCE_CHECKPOINT_HASH),
    }
    for role, (path, expected) in paths.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    records: dict[str, EvidenceRecordV2] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    for role in ("canary_validation", "canary_preflight", "canary_training"):
        value = json.loads(paths[role][0].read_text(encoding="utf-8"))
        record = EvidenceRecordV2.from_dict(value["evidence"])
        record.validate()
        records[role] = record
        payloads[role] = value
        _assert_ancestor(record.git_commit, current_commit)
        if value.get("training_authorized") is not False or value.get("formal_promotion") is not False:
            raise ValueError("canary source authorization boundary is invalid")
    if records["canary_validation"].evidence_kind != canary.VALIDATION_SCHEMA or records["canary_validation"].status != "passed":
        raise ValueError("canary validation source is not passed")
    if records["canary_preflight"].evidence_kind != canary.PREFLIGHT_SCHEMA or records["canary_preflight"].status != "passed":
        raise ValueError("canary preflight source is not passed")
    training_contract = payloads["canary_training"].get("contract")
    if (
        records["canary_training"].evidence_kind != canary.TRAINING_SCHEMA
        or records["canary_training"].status != "diagnostic"
        or not isinstance(training_contract, Mapping)
        or training_contract.get("outcome") != "diagnostic_completed"
    ):
        raise ValueError("canary training source is not a completed diagnostic")
    panel = safe_torch_load(
        paths["canary_panel"][0],
        expected_sha256=SOURCE_CANARY_PANEL_HASH,
        map_location="cpu",
    )
    if (
        panel.get("schema") != canary.PANEL_SCHEMA
        or panel.get("indices_are_global_mnist") is not True
        or not isinstance(panel.get("source_hashes"), Mapping)
        or not isinstance(panel.get("tensors"), Mapping)
    ):
        raise ValueError("canary source panel is incomplete or not global-indexed")
    source_checkpoint = safe_torch_load(paths["source_checkpoint"][0], expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if (
        source_checkpoint.get("schema") != "lightgen_optical_manifold_identity_full_epoch_resume.v1"
        or source_checkpoint.get("global_step") != SOURCE_STEP
        or not isinstance(source_checkpoint.get("model_state"), Mapping)
        or not isinstance(source_checkpoint.get("optimizer_state"), Mapping)
        or not isinstance(source_checkpoint.get("rng_state"), Mapping)
    ):
        raise ValueError("SMF reconstruction source checkpoint is incomplete")
    if _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent) != formal_path:
        raise ValueError("SMF reconstruction formal path mismatch")
    return SourceBundle(
        checkpoint=source_checkpoint,
        hashes={
            "canary_validation": SOURCE_CANARY_VALIDATION_HASH,
            "canary_preflight": SOURCE_CANARY_PREFLIGHT_HASH,
            "canary_training": SOURCE_CANARY_TRAINING_HASH,
            "canary_panel": SOURCE_CANARY_PANEL_HASH,
            "source_checkpoint": SOURCE_CHECKPOINT_HASH,
            "mnist_manifest": SOURCE_MANIFEST_HASH,
            "formal_config": file_sha256(formal_path),
        },
        formal_path=formal_path,
    )


def _load_data(dataset_root: str | Path) -> PilotData:
    images, labels, manifest = canary._load_authoritative_pilot_data(dataset_root)
    if manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH or manifest.get("train_count") != 55000 or manifest.get("validation_count") != 5000:
        raise ValueError("SMF reconstruction requires the authoritative 55k/5k MNIST split")
    adaptive_data = adaptive._load_data(dataset_root)
    pilot_data = adaptive_data.objective_data.full_data.pilot_data
    prior_schedule = adaptive.pilot._pilot_schedule(pilot_data.train_indices, pilot_data.labels)
    regularization_data = adaptive_data.objective_data.regularization_data
    excluded = torch.cat(
        (
            regularization_data.calibration_indices,
            *regularization_data.gradient_batches,
            adaptive_data.objective_data.projection_indices,
            *adaptive_data.objective_data.gradient_batches,
            prior_schedule.reshape(-1),
            adaptive_data.calibration_panels[0],
            adaptive_data.calibration_panels[1],
            adaptive_data.calibration_panels[2],
            adaptive_data.assessment_indices,
            *adaptive_data.gradient_batches,
            pilot_data.heldout_indices,
            canary._load_data(dataset_root).schedule.reshape(-1),
        )
    )
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("SMF reconstruction exclusion set contains duplicates")
    schedule = _build_schedule(
        labels,
        pilot_data.train_indices,
        excluded,
        steps=STEPS,
        seed=SEED,
    )
    train_labels = labels.index_select(0, pilot_data.train_indices)
    remaining_counts: list[int] = []
    for label in range(10):
        class_indices = pilot_data.train_indices[train_labels == label]
        remaining_counts.append(int(class_indices[~torch.isin(class_indices, excluded)].numel()))
    if any(count < STEPS for count in remaining_counts):
        raise ValueError("SMF reconstruction lacks 1000 fresh samples for a label")

    if torch.unique(schedule).numel() != schedule.numel() or torch.isin(schedule.reshape(-1), excluded).any():
        raise ValueError("SMF reconstruction schedule is not fresh and without replacement")
    if not torch.equal(labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(STEPS)):
        raise ValueError("SMF reconstruction schedule is not balanced one-per-label")
    heldout_labels = labels.index_select(0, pilot_data.heldout_indices)
    visual = torch.cat([pilot_data.heldout_indices[heldout_labels == label][:10] for label in range(10)])
    heldout = torch.cat([pilot_data.heldout_indices[heldout_labels == label] for label in range(10)])
    data_manifest = {
        "schema": "lightgen_optical_manifold_smf_reconstruction_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "excluded_indices_sha256": tensor_sha256(torch.sort(excluded).values),
        "excluded_count": int(excluded.numel()),
        "schedule_sha256": tensor_sha256(schedule),
        "schedule_shape": list(schedule.shape),
        "schedule_count": int(schedule.numel()),
        "schedule_seed": SEED,
        "per_label_remaining_counts": remaining_counts,
        "heldout_indices_sha256": tensor_sha256(heldout),
        "heldout_count": int(heldout.numel()),
        "visual_indices_sha256": tensor_sha256(visual),
        "visual_count": int(visual.numel()),
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
    }
    data_manifest["manifest_hash"] = canonical_sha256(data_manifest)
    return PilotData(images, labels, pilot_data.train_indices, heldout, schedule, visual, data_manifest)


def _build_schedule(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    excluded: torch.Tensor,
    *,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Build the shared per-step, one-example-per-class schedule."""
    if labels.ndim != 1 or train_indices.ndim != 1 or excluded.ndim != 1:
        raise ValueError("SMF reconstruction schedule inputs must be vectors")
    train_labels = labels.index_select(0, train_indices)
    per_label: list[torch.Tensor] = []
    for label in range(10):
        class_indices = train_indices[train_labels == label]
        remaining = class_indices[~torch.isin(class_indices, excluded)].sort().values
        if remaining.numel() < steps:
            raise ValueError("SMF reconstruction lacks fresh samples for a label")
        generator = torch.Generator(device="cpu").manual_seed(seed + label)
        per_label.append(remaining[torch.randperm(remaining.numel(), generator=generator)[:steps]])
    schedule = torch.stack(per_label, dim=1)
    if torch.unique(schedule).numel() != schedule.numel():
        raise ValueError("SMF reconstruction schedule contains repeated indices")
    return schedule


def _context(spec: PilotSpec, formal: FormalSystemConfig, data: PilotData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _model_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    contract = base._parameter_contract(model)
    if tuple(contract["trainable_parameter_names"]) != EXPECTED_TRAINABLE:
        raise ValueError("SMF reconstruction trainable parameter contract failed")
    return contract


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


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite SMF reconstruction artifact: {target}")
    atomic_write_json(target, payload)


def _validation_record(state: Mapping[str, Any], spec: PilotSpec, formal: FormalSystemConfig, data: PilotData, source: SourceBundle, review_path: Path, model_contract: Mapping[str, Any]) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_pilot_branch", {"branch_matches": True}),
            ("canary_source_complete", {"training_outcome": "diagnostic_completed"}),
            ("source_checkpoint_step", {"global_step": SOURCE_STEP}),
            ("mnist_schedule_fresh_balanced", {"steps": STEPS, "batch": BATCH_SIZE, "count": STEPS * BATCH_SIZE}),
            ("heldout_excluded", {"count": 1000}),
            ("formal_geometry", {"grid": 2048, "identity_formal_ols": True}),
            ("research_smf_contract", {"ports": PORT_COUNT, "mode_model": "step_index_lp01_ports"}),
            ("ols_variable_only", {"identity_control_and_smf_arm": True}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("four_phase_parameter_scope", {"parameters": model_contract["trainable_parameter_names"]}),
            ("diagnostic_training_boundary", {"training_authorized": False, "formal_promotion": False}),
        )
    )
    hashes = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"]}
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_reconstruction_preflight",))
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    contract = _model_contract(model)
    if tuple(data.images.shape) != (60000, 1, 28, 28) or tuple(data.labels.shape) != (60000,):
        raise ValueError("SMF reconstruction requires complete MNIST tensors")
    record = _validation_record(state, spec, formal, data, source, review_path, contract)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "formal_config": str(formal_path.resolve()),
            "source_hashes": source.hashes,
            "dataset_manifest": data.manifest,
            "model_contract": contract,
            "optimizer_created": False,
            "cuda_context_created": False,
        },
    )
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: PilotSpec, formal: FormalSystemConfig, data: PilotData, state: Mapping[str, Any], source: SourceBundle, review_path: Path) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("SMF reconstruction validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("SMF reconstruction validation must be passed")
    if state["branch"] != BRANCH or not state["clean"] or record.git_commit != state["commit"]:
        raise ValueError("SMF reconstruction validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("SMF reconstruction validation context mismatch")
    expected = {
        **source.hashes,
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
    }
    if record.input_hashes != expected:
        raise ValueError("SMF reconstruction validation input hashes do not match")
    if record.permitted_next_actions != ("run_smf_reconstruction_preflight",):
        raise ValueError("SMF reconstruction validation next action is invalid")
    return payload, digest


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _module_state_cpu(module: torch.nn.Module | IdentityOLS | FiberArrayOLS) -> Any:
    if module is None:
        return None
    if isinstance(module, torch.nn.Module):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    if isinstance(module, IdentityOLS):
        return {"type": type(module).__qualname__, "attributes": copy.deepcopy(vars(module))}
    if isinstance(module, FiberArrayOLS):
        return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    raise TypeError(f"unsupported state snapshot type: {type(module)!r}")


def _nonphase_state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name not in EXPECTED_TRAINABLE}


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


def _batch(data: PilotData, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _runtime(formal: FormalSystemConfig, source: SourceBundle, device: torch.device, arm: str) -> tuple[base.IdentityOpticalManifoldAutoencoder, full_epoch._DeterministicCpuAdam, FiberArrayOLS | None]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    optimizer = full_epoch._make_full_optimizer(model)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    base._restore_rng(source.checkpoint["rng_state"])
    smf = canary._smf_operator(model.system, device) if arm == "smf_mse_reconstruction" else None
    return model, optimizer, smf


def _grad_norms(gradients: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return canary._layer_norms(gradients)


def _identity_gradients(model: base.IdentityOpticalManifoldAutoencoder, amplitudes: torch.Tensor, targets: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    gradients = {name: torch.zeros_like(value, device="cpu") for name, value in model.named_parameters()}
    losses: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    canary._clear_grad(model)
    for start in range(0, BATCH_SIZE, MICROBATCH):
        output = model(amplitudes[start : start + 1], activation_checkpoint=True)
        loss = F.mse_loss(output.reconstruction, targets[start : start + 1])
        losses.append(loss.detach())
        cameras.append(output.raw_camera.detach())
        (loss / BATCH_SIZE).backward()
    canary._add_grads(gradients, model)
    canary._clear_grad(model)
    camera = torch.cat(cameras)
    return gradients, {
        "loss": float(torch.stack(losses).mean().cpu()),
        "camera_finite": bool(torch.isfinite(camera).all()),
        "camera_min": float(camera.min().cpu()),
        "camera_max": float(camera.max().cpu()),
        "port_coefficients_shape": None,
        "gradient_norms": _grad_norms(gradients),
    }


def _step(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, smf: FiberArrayOLS | None, amplitudes: torch.Tensor, targets: torch.Tensor, arm: str) -> dict[str, Any]:
    before = {name: value.detach().clone() for name, value in model.named_parameters() if name in EXPECTED_TRAINABLE}
    if smf is None:
        gradients, info = _identity_gradients(model, amplitudes, targets)
    else:
        gradients, info = canary._mse_gradients(model, smf, amplitudes, targets)
    for name, value in model.named_parameters():
        value.grad = gradients[name].to(value.device)
    optimizer.step()
    parameters = dict(model.named_parameters())
    updates: dict[str, float] = {}
    for name in EXPECTED_TRAINABLE:
        updates[name] = float((parameters[name].detach() - before[name]).abs().sum().cpu())
        for index in range(parameters[name].shape[0]):
            updates[f"{name}[{index}]"] = float((parameters[name][index].detach() - before[name][index]).abs().sum().cpu())
    canary._clear_grad(model)
    finite_values = [info["loss"], info["camera_min"], info["camera_max"], *_grad_norms(gradients).values(), *updates.values()]
    return {
        "arm": arm,
        "loss": info["loss"],
        "camera_finite": info["camera_finite"],
        "camera_min": info["camera_min"],
        "camera_max": info["camera_max"],
        "port_coefficients_shape": info["port_coefficients_shape"],
        "gradient_norms": _grad_norms(gradients),
        "update_norms": updates,
        "activation_checkpoint": True,
        "finite": info["camera_finite"] and all(bool(torch.isfinite(torch.tensor(value))) for value in finite_values),
    }


def _structure_metrics(features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < 2:
        raise ValueError("structure feature shape is invalid")
    work = torch.cat((features.real, features.imag), dim=1) if torch.is_complex(features) else features
    work = work.to(torch.float64)
    centered = work - work.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(1.0e-30)
    probabilities = eigenvalues / total
    entropy_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1.0e-30).log()).sum()))
    participation_rank = float(1.0 / probabilities.square().sum().clamp_min(1.0e-30))
    means = []
    intra_values = []
    for label in range(10):
        group = work[labels == label]
        if group.numel() == 0:
            raise ValueError("structure panel is missing a label")
        means.append(group.mean(dim=0))
        intra_values.append((group - group.mean(dim=0, keepdim=True)).norm(dim=1).mean())
    means_tensor = torch.stack(means)
    intra = torch.stack(intra_values).mean().clamp_min(1.0e-30)
    inter = torch.pdist(means_tensor).mean().clamp_min(1.0e-30)
    distances = torch.cdist(work, work)
    silhouettes: list[torch.Tensor] = []
    for index in range(work.shape[0]):
        same = labels == labels[index]
        same[index] = False
        a = distances[index, same].mean() if same.any() else torch.tensor(0.0, dtype=distances.dtype)
        other = []
        for label in range(10):
            if label != int(labels[index]):
                mask = labels == label
                if mask.any():
                    other.append(distances[index, mask].mean())
        b = torch.stack(other).min()
        silhouettes.append((b - a) / torch.maximum(a, b).clamp_min(1.0e-30))
    return {
        "sample_count": int(work.shape[0]),
        "feature_dimension_real_imag": int(work.shape[1]),
        "entropy_effective_rank": entropy_rank,
        "participation_rank": participation_rank,
        "inter_class_distance": float(inter),
        "intra_class_distance": float(intra),
        "inter_intra_ratio": float(inter / intra),
        "silhouette": float(torch.stack(silhouettes).mean()),
        "finite": bool(torch.isfinite(work).all() and torch.isfinite(distances).all()),
    }


def _decode_pre_ols(model: base.IdentityOpticalManifoldAutoencoder, field: torch.Tensor, smf: FiberArrayOLS | None) -> torch.Tensor:
    system = model.system
    provenance = OperationProvenance("interpolation_pre_ols", "encoder_layer_2", None)
    value = ComplexField(field, system.encoder_plane, provenance)
    transformed = system.ols(value) if smf is None else smf(value)
    decoded, _, _ = system.decoder(transformed, "decoder")
    chip = ComplexField(decoded.tensor, system.chip_plane, decoded.provenance)
    camera_field = system.back(chip)
    return camera_field.tensor.abs().square()


def _interpolation_metrics(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS | None, pre_visual: torch.Tensor, camera_visual: torch.Tensor, device: torch.device) -> dict[str, Any]:
    if pre_visual.shape[0] != 100 or camera_visual.shape[0] != 100:
        raise ValueError("interpolation requires the fixed 100-image visual panel")
    values: list[torch.Tensor] = []
    endpoint_errors: list[torch.Tensor] = []
    for label in range(10):
        first = pre_visual[label * 10].to(device)
        last = pre_visual[label * 10 + 9].to(device)
        ts = (0.0, 0.25, 0.5, 0.75, 1.0)
        outputs = [_decode_pre_ols(model, first.mul(1.0 - t) + last.mul(t), smf).detach() for t in ts]
        endpoint_errors.append(F.mse_loss(outputs[0], camera_visual[label * 10].to(device)).detach())
        endpoint_errors.append(F.mse_loss(outputs[-1], camera_visual[label * 10 + 9].to(device)).detach())
        values.extend(outputs)
    stacked = torch.stack(values)
    adjacent = (stacked[1:] - stacked[:-1]).flatten(1).norm(dim=1)
    return {
        "endpoint_max_mse": float(torch.stack(endpoint_errors).max().cpu()),
        "adjacent_mean_l2": float(adjacent.mean().cpu()),
        "finite": bool(torch.isfinite(stacked).all()),
        "endpoint_count": 20,
        "interpolation_count": 50,
        "t_values": [0.0, 0.25, 0.5, 0.75, 1.0],
    }


def _evaluate(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS | None, data: PilotData, indices: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    cameras: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    couplings: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    active_ports: list[torch.Tensor] = []
    pre_visual: list[torch.Tensor] = []
    camera_visual: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            amplitudes, target = _batch(data, batch_indices, device)
            if smf is None:
                raw, pre, _plane, _provenance = regularization._forward_trace(model, amplitudes)
                coefficient = pre.flatten(1)
            else:
                raw, pre, diagnostics = canary._smf_forward(model, smf, amplitudes, trace=True)
                coefficient = diagnostics.get("port_coefficients")
                if not isinstance(coefficient, torch.Tensor) or tuple(coefficient.shape[-1:]) != (PORT_COUNT,):
                    raise ValueError("SMF evaluation did not expose 100 complex coefficients")
                coupling = diagnostics.get("coupling_efficiency")
                power_in = diagnostics.get("input_energy")
                power_out = diagnostics.get("output_energy")
                active = diagnostics.get("active_port_fraction")
                if isinstance(coupling, torch.Tensor):
                    couplings.append(coupling.detach().cpu().reshape(-1))
                if isinstance(power_in, torch.Tensor) and isinstance(power_out, torch.Tensor):
                    powers.append((power_out / power_in.clamp_min(1.0e-30)).detach().cpu().reshape(-1))
                if isinstance(active, torch.Tensor):
                    active_ports.append(active.detach().cpu().reshape(-1))
            cameras.append(raw.detach().cpu())
            targets.append(target.detach().cpu())
            features.append(coefficient.detach().cpu())
            if len(torch.cat(cameras)) <= 100:
                pre_visual.append(pre.detach().cpu())
                camera_visual.append(raw.detach().cpu())
    camera = torch.cat(cameras)
    target = torch.cat(targets)
    feature = torch.cat(features)
    if camera.shape[0] != indices.numel():
        raise ValueError("evaluation output count mismatch")
    flat_camera = camera.flatten(1)
    flat_target = target.flatten(1)
    camera_centered = flat_camera - flat_camera.mean(dim=1, keepdim=True)
    target_centered = flat_target - flat_target.mean(dim=1, keepdim=True)
    correlation = (camera_centered * target_centered).sum(dim=1) / (camera_centered.norm(dim=1) * target_centered.norm(dim=1)).clamp_min(1.0e-30)
    variance_retention = flat_camera.var(dim=0, unbiased=False).mean() / flat_target.var(dim=0, unbiased=False).mean().clamp_min(1.0e-30)
    visual_pre = torch.cat(pre_visual)[:100]
    visual_camera = torch.cat(camera_visual)[:100]
    visual_labels = data.labels.index_select(0, data.visual_indices).cpu()
    metrics = {
        "paired_mse": float(F.mse_loss(camera, target)),
        "cyclic_mse": float(F.mse_loss(camera, target.roll(1, dims=0))),
        "centered_correlation": float(correlation.mean()),
        "variance_retention": float(variance_retention),
        "finite": bool(torch.isfinite(camera).all() and torch.isfinite(target).all() and torch.isfinite(feature).all()),
        "structure": _structure_metrics(feature[:100], visual_labels),
        "evaluation_count": int(indices.numel()),
        "feature_kind": "smf_100_complex_coefficients" if smf is not None else "pre_ols_complex_field",
        "optical_power_retention": float(torch.cat(powers).mean()) if powers else None,
        "coupling_efficiency_mean": float(torch.cat(couplings).mean()) if couplings else None,
        "active_port_fraction_mean": float(torch.cat(active_ports).mean()) if active_ports else None,
    }
    metrics["interpolation"] = _interpolation_metrics(model, smf, visual_pre, visual_camera, device)
    return metrics, {
        "camera": camera,
        "target": target,
        "pre_visual": visual_pre,
        "camera_visual": visual_camera,
    }


def _initial_losses(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS | None, data: PilotData, device: torch.device) -> list[float]:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for schedule_indices in data.schedule:
            amplitudes, target = _batch(data, schedule_indices, device)
            if smf is None:
                raw = model(amplitudes, activation_checkpoint=False).raw_camera
            else:
                raw, _pre, _diagnostics = canary._smf_forward(model, smf, amplitudes)
            values.append(float(F.mse_loss(raw, target).cpu()))
    return values


def _resume_payload(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, arm: str, step: int, history: list[dict[str, Any]], initial_losses: list[float], spec: PilotSpec, data: PilotData, hashes: Mapping[str, str], completed_arms: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "model_state": _state_cpu(model),
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "rng_state": _tree_clone(base._rng_state()),
        "history": list(history),
        "initial_batch_losses": list(initial_losses),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(hashes),
        "data_cursor": {"arm": arm, "next_step": step, "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH},
        "completed_arms": dict(completed_arms or {}),
        "formal_candidate_checkpoint": False,
        "training_authorized": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite SMF reconstruction checkpoint: {path}")
    atomic_torch_save(path, payload)


def _load_resume(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, arm: str, spec: PilotSpec, data: PilotData, hashes: Mapping[str, str], output_dir: Path) -> dict[str, Any]:
    target = _resolve(path)
    if target.parent.resolve() != output_dir.resolve():
        raise ValueError("explicit resume checkpoint belongs to a different attempt")
    payload = safe_torch_load(target, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid SMF reconstruction resume artifact")
    step = payload.get("step")
    if payload.get("arm") != arm or not isinstance(step, int) or not 0 < step < STEPS:
        raise ValueError("resume arm or step is invalid")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes):
        raise ValueError("resume lineage mismatch")
    if payload.get("data_cursor") != {"arm": arm, "next_step": step, "logical_batch_size": BATCH_SIZE, "microbatch_size": MICROBATCH}:
        raise ValueError("resume data cursor mismatch")
    if not isinstance(payload.get("history"), list) or len(payload["history"]) != step or not isinstance(payload.get("initial_batch_losses"), list) or len(payload["initial_batch_losses"]) != STEPS:
        raise ValueError("resume history or safety baseline mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return dict(payload)


def _preflight_record(state: Mapping[str, Any], spec: PilotSpec, formal: FormalSystemConfig, data: PilotData, hashes: Mapping[str, str], gates: tuple[GateResult, ...], status: str, actions: tuple[str, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(2, PREFLIGHT_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), dict(hashes), gates, actions, reason)
    record.validate()
    return record


def _exact_resume(formal: FormalSystemConfig, source: SourceBundle, data: PilotData, arm: str, spec: PilotSpec, hashes: Mapping[str, str], output_dir: Path, device: torch.device) -> dict[str, Any]:
    continuous_model, continuous_optimizer, continuous_smf = _runtime(formal, source, device, arm)
    split_model, split_optimizer, split_smf = _runtime(formal, source, device, arm)
    continuous_history: list[dict[str, Any]] = []
    split_history: list[dict[str, Any]] = []
    for step in range(PREFLIGHT_STEPS):
        amplitudes, targets = _batch(data, data.schedule[step], device)
        continuous_history.append(_step(continuous_model, continuous_optimizer, continuous_smf, amplitudes, targets, arm))
    continuous_rng = _tree_clone(base._rng_state())
    for step in range(PREFLIGHT_STEPS - 1):
        amplitudes, targets = _batch(data, data.schedule[step], device)
        split_history.append(_step(split_model, split_optimizer, split_smf, amplitudes, targets, arm))
    resume_path = output_dir / f"preflight_{arm}_step_00002.pt"
    _save_checkpoint(resume_path, _resume_payload(split_model, split_optimizer, arm=arm, step=2, history=split_history, initial_losses=[0.0] * STEPS, spec=spec, data=data, hashes=hashes))
    resumed_model, resumed_optimizer, resumed_smf = _runtime(formal, source, device, arm)
    payload = _load_resume(resume_path, resumed_model, resumed_optimizer, arm=arm, spec=spec, data=data, hashes=hashes, output_dir=output_dir)
    if payload["step"] != 2:
        raise ValueError("preflight resume checkpoint did not restore step 2")
    amplitudes, targets = _batch(data, data.schedule[2], device)
    split_history.append(_step(resumed_model, resumed_optimizer, resumed_smf, amplitudes, targets, arm))
    resumed_rng = _tree_clone(base._rng_state())
    return {
        "continuous": continuous_history,
        "resumed": split_history,
        "model_exact": all(torch.equal(continuous_model.state_dict()[name].detach().cpu(), resumed_model.state_dict()[name].detach().cpu()) for name in continuous_model.state_dict()),
        "optimizer_exact": _tree_equal(continuous_optimizer.state_dict(), resumed_optimizer.state_dict()),
        "rng_exact": _tree_equal(continuous_rng, resumed_rng),
        "frozen_state_exact": _tree_equal(_nonphase_state_cpu(continuous_model), _nonphase_state_cpu(resumed_model)),
        "smf_state_exact": _tree_equal(_module_state_cpu(continuous_smf), _module_state_cpu(resumed_smf)),
        "resume_artifact": str(resume_path.resolve()),
        "resume_artifact_sha256": file_sha256(resume_path),
        "finite": all(item["finite"] for item in continuous_history + split_history),
    }


def _preflight_controls(model: base.IdentityOpticalManifoldAutoencoder, smf: FiberArrayOLS, data: PilotData, device: torch.device) -> dict[str, Any]:
    amplitudes, _target = _batch(data, data.schedule[0, :1], device)
    identity_raw, _pre, _plane, _provenance = regularization._forward_trace(model, amplitudes)
    smf_raw, _smf_pre, smf_diag = canary._smf_forward(model, smf, amplitudes, trace=True)
    zero_smf = canary._smf_operator(model.system, device, zero_transport=True)
    zero_raw, _zero_pre, zero_diag = canary._smf_forward(model, zero_smf, amplitudes, trace=True)
    identity_ols = canary._identity_trace_has_ols(model, amplitudes)
    smf_provenance = smf_diag["smf_provenance"].get("operation")
    coefficients = smf_diag.get("port_coefficients")
    input_energy = smf_diag.get("input_energy")
    output_energy = smf_diag.get("output_energy")
    passivity = isinstance(input_energy, torch.Tensor) and isinstance(output_energy, torch.Tensor) and bool((output_energy <= input_energy * (1.0 + 1.0e-5) + 1.0e-12).all())
    return {
        "identity_has_identity_ols": identity_ols,
        "smf_has_transport": smf_provenance == "synthetic_smf_lp01_port_transport",
        "smf_coefficients_shape": list(coefficients.shape) if isinstance(coefficients, torch.Tensor) else None,
        "identity_smf_not_equal": not torch.equal(identity_raw, smf_raw),
        "zero_port_finite": bool(torch.isfinite(zero_raw).all()),
        "zero_port_max_abs": float(zero_raw.abs().max().cpu()),
        "zero_port_is_zero": bool(zero_raw.abs().max() <= 1.0e-12),
        "passivity": passivity,
        "input_energy_max": float(input_energy.max().cpu()) if isinstance(input_energy, torch.Tensor) else None,
        "output_energy_max": float(output_energy.max().cpu()) if isinstance(output_energy, torch.Tensor) else None,
        "zero_output_energy_max": float(zero_diag["output_energy"].max().cpu()),
    }


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF reconstruction preflight requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source, review_path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SMF reconstruction preflight: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for SMF reconstruction preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash}
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        source_model, _source_optimizer, source_smf = _runtime(formal, source, device, "smf_mse_reconstruction")
        source_state = _state_cpu(source_model)
        source_formal_state = _module_state_cpu(source_model.system.ols)
        source_smf_state = _module_state_cpu(source_smf)
        exact = {arm: _exact_resume(formal, source, data, arm, spec, hashes, output.parent, device) for arm in ARMS}
        reports: dict[str, Any] = {}
        for arm in ARMS:
            model, optimizer, smf = _runtime(formal, source, device, arm)
            amplitudes, targets = _batch(data, data.schedule[0], device)
            reports[arm] = {"step": _step(model, optimizer, smf, amplitudes, targets, arm), "exact_resume": exact[arm]}
        controls = _preflight_controls(source_model, source_smf, data, device)
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        smf_step = reports["smf_mse_reconstruction"]["step"]
        identity_step = reports["identity_mse_control"]["step"]
        gates = (
            GateResult("finite_values", all(item["step"]["finite"] and item["exact_resume"]["finite"] for item in reports.values()), {}),
            GateResult("four_phase_reconstruction_updates", canary._four_phase_nonzero(smf_step["update_norms"]) and canary._four_phase_nonzero(identity_step["update_norms"]), {"layers": 4}),
            GateResult("activation_checkpoint", all(item["step"]["activation_checkpoint"] for item in reports.values()), {}),
            GateResult("exact_resume_both_arms", all(item["model_exact"] and item["optimizer_exact"] and item["rng_exact"] and item["frozen_state_exact"] and item["smf_state_exact"] for item in exact.values()), {}),
            GateResult("smf_100_port_coefficients", smf_step["port_coefficients_shape"] == [1, PORT_COUNT], {"shape": smf_step["port_coefficients_shape"]}),
            GateResult("smf_passive_zero_transport_control", controls["smf_has_transport"] and controls["passivity"] and controls["zero_port_finite"] and controls["zero_port_is_zero"], controls),
            GateResult("identity_and_smf_are_not_bypassed", controls["identity_has_identity_ols"] and controls["identity_smf_not_equal"], controls),
            GateResult("frozen_formal_state", _tree_equal(source_state, _state_cpu(source_model)) and _tree_equal(source_formal_state, _module_state_cpu(source_model.system.ols)) and _tree_equal(source_smf_state, _module_state_cpu(source_smf)), {}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        passed = all(gate.passed for gate in gates)
        record = _preflight_record(state, spec, formal, data, hashes, gates, "passed" if passed else "invalid_harness", ("run_smf_reconstruction_pilot",) if passed else (), None if passed else "SMF reconstruction preflight gate failed")
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"validation_hash": validation_hash, "source_hashes": source.hashes, "training_hashes": hashes, "arms": reports, "controls": controls, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "optimizer_created": True, "training_authorized": False})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if passed else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def _load_preflight(path: str | Path, spec: PilotSpec, formal: FormalSystemConfig, data: PilotData, state: Mapping[str, Any], validation_hash: str, source: SourceBundle, review_path: Path) -> tuple[dict[str, Any], str, dict[str, str]]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("SMF reconstruction preflight schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != PREFLIGHT_SCHEMA or record.status != "passed":
        raise ValueError("SMF reconstruction preflight must be passed")
    if record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("SMF reconstruction preflight is stale")
    expected = {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "training_validation": validation_hash}
    if record.input_hashes != expected:
        raise ValueError("SMF reconstruction preflight source hashes do not match")
    if record.permitted_next_actions != ("run_smf_reconstruction_pilot",):
        raise ValueError("SMF reconstruction preflight next action is invalid")
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or contract.get("training_hashes") != expected:
        raise ValueError("SMF reconstruction preflight training provenance is invalid")
    return payload, digest, {**expected, "training_preflight": digest}


def _run_steps(model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, smf: FiberArrayOLS | None, data: PilotData, arm: str, start: int, history: list[dict[str, Any]], initial_losses: list[float], output_dir: Path, spec: PilotSpec, hashes: Mapping[str, str], *, completed_arms: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], Path | None, str | None]:
    final_path: Path | None = None
    final_hash: str | None = None
    for step in range(start, STEPS):
        amplitudes, targets = _batch(data, data.schedule[step], next(model.parameters()).device)
        result = _step(model, optimizer, smf, amplitudes, targets, arm)
        result["step"] = step + 1
        history.append(result)
        if not result["finite"]:
            raise ValueError("SMF reconstruction encountered non-finite step values")
        if result["loss"] > initial_losses[step] * SAFETY_MSE_RATIO:
            raise _SafetyAbort(arm, step + 1, result)
        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            path = output_dir / f"{arm}_step_{step + 1:04d}.pt"
            payload = _resume_payload(model, optimizer, arm=arm, step=step + 1, history=history, initial_losses=initial_losses, spec=spec, data=data, hashes=hashes, completed_arms=completed_arms)
            _save_checkpoint(path, payload)
            final_path, final_hash = path, file_sha256(path)
    return history, final_path, final_hash


def _load_completed_arm(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: full_epoch._DeterministicCpuAdam, *, spec: PilotSpec, data: PilotData, hashes: Mapping[str, str], output_dir: Path, expected_sha256: str) -> dict[str, Any]:
    target = _resolve(path)
    if target.parent.resolve() != output_dir.resolve() or file_sha256(target) != expected_sha256:
        raise ValueError("completed SMF arm checkpoint path or hash is invalid")
    payload = safe_torch_load(target, expected_sha256=expected_sha256, map_location="cpu")
    if payload.get("schema") != RESUME_SCHEMA or payload.get("arm") != "smf_mse_reconstruction" or payload.get("step") != STEPS:
        raise ValueError("completed SMF arm checkpoint is invalid")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != data.manifest["schedule_sha256"] or payload.get("evidence_hashes") != dict(hashes):
        raise ValueError("completed SMF arm checkpoint lineage mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return dict(payload)


def _save_panel(path: Path, data: PilotData, device: torch.device, source_hashes: Mapping[str, str], source_identity: Mapping[str, torch.Tensor], source_smf: Mapping[str, torch.Tensor], final_identity: Mapping[str, torch.Tensor], final_smf: Mapping[str, torch.Tensor], coupling: Mapping[str, Any]) -> str:
    values = {
        "indices": data.visual_indices.cpu(),
        "input_28": data.images.index_select(0, data.visual_indices).cpu(),
        "target_200": source_identity["target"][:100].cpu(),
        "identity_source": source_identity["camera"][:100].cpu(),
        "identity_final": final_identity["camera"][:100].cpu(),
        "smf_initial": source_smf["camera"][:100].cpu(),
        "smf_final": final_smf["camera"][:100].cpu(),
    }
    payload = {"schema": PANEL_SCHEMA, "source_hashes": dict(source_hashes), "indices_are_global_mnist": True, "tensors": values, "coupling": coupling}
    atomic_torch_save(path, payload)
    try:
        from PIL import Image

        rows = []
        for key in ("target_200", "identity_source", "identity_final", "smf_initial", "smf_final"):
            images = values[key].squeeze(1).clamp(0.0, 1.0).mul(255).to(torch.uint8)
            thumb = F.interpolate(images.unsqueeze(1).float(), size=(32, 32), mode="bilinear", align_corners=False).squeeze(1).to(torch.uint8)
            rows.append(torch.cat([thumb[index] for index in range(100)], dim=1))
        Image.fromarray(torch.cat(rows, dim=0).numpy(), mode="L").save(path.with_suffix(".png"))
    except ImportError:
        pass
    return file_sha256(path)


def _training_record(state: Mapping[str, Any], spec: PilotSpec, formal: FormalSystemConfig, data: PilotData, hashes: Mapping[str, str], status: str, actions: tuple[str, ...], gates: tuple[GateResult, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(2, TRAINING_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), dict(hashes), gates, actions, reason)
    record.validate()
    return record


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("SMF reconstruction training requires CUDA")
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    _validation_payload, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, source, review_path)
    preflight_payload, preflight_hash, training_hashes = _load_preflight(args.preflight_evidence, spec, formal, data, state, validation_hash, source, review_path)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SMF reconstruction training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for SMF reconstruction training")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    started = time.monotonic()
    torch.use_deterministic_algorithms(True)
    try:
        source_identity_model, _source_identity_optimizer, source_identity_smf = _runtime(formal, source, device, "smf_mse_reconstruction")
        source_smf_model, _source_smf_optimizer, source_smf_operator = _runtime(formal, source, device, "smf_mse_reconstruction")
        source_identity_metrics, source_identity_panel = _evaluate(source_identity_model, None, data, data.heldout_indices, device)
        source_smf_metrics, source_smf_panel = _evaluate(source_smf_model, source_smf_operator, data, data.heldout_indices, device)
        initial_losses: dict[str, list[float]] = {
            "identity_mse_control": _initial_losses(source_identity_model, None, data, device),
            "smf_mse_reconstruction": _initial_losses(source_smf_model, source_smf_operator, data, device),
        }
        source_model_state = _state_cpu(source_identity_model)
        source_formal_state = _module_state_cpu(source_identity_model.system.ols)
        source_smf_state = _module_state_cpu(source_smf_operator)
        trained_models: dict[str, base.IdentityOpticalManifoldAutoencoder] = {}
        trained_operators: dict[str, FiberArrayOLS | None] = {}
        arm_results: dict[str, Any] = {}
        checkpoints: dict[str, Any] = {}
        resume_payload: Mapping[str, Any] | None = None
        resume_arm: str | None = None
        if args.resume_checkpoint:
            resume_target = _resolve(args.resume_checkpoint)
            if resume_target.parent.resolve() != output.parent.resolve():
                raise ValueError("explicit SMF reconstruction resume belongs to a different attempt")
            resume_payload = safe_torch_load(resume_target, map_location="cpu")
            resume_arm = str(resume_payload.get("arm"))
            if resume_arm not in ARMS:
                raise ValueError("explicit SMF reconstruction resume has an invalid arm")

        def run_arm(arm: str, *, resume_path: Path | None = None, completed_arms: Mapping[str, Any] | None = None) -> None:
            model, optimizer, smf = _runtime(formal, source, device, arm)
            history: list[dict[str, Any]] = []
            start = 0
            safety = initial_losses[arm]
            if resume_path is not None:
                resume = _load_resume(resume_path, model, optimizer, arm=arm, spec=spec, data=data, hashes=training_hashes, output_dir=output.parent)
                start = int(resume["step"])
                history = list(resume["history"])
                safety = [float(value) for value in resume["initial_batch_losses"]]
            try:
                history, final_path, final_hash = _run_steps(model, optimizer, smf, data, arm, start, history, safety, output.parent, spec, training_hashes, completed_arms=completed_arms)
            except _SafetyAbort:
                arm_results[arm] = {"steps_completed": len(history), "history": history, "metrics": None, "final_checkpoint": None, "final_checkpoint_sha256": None}
                trained_models[arm] = model
                trained_operators[arm] = smf
                raise
            metrics, panel = _evaluate(model, smf, data, data.heldout_indices, device)
            arm_results[arm] = {"steps_completed": len(history), "history": history, "metrics": metrics, "final_checkpoint": str(final_path.resolve()) if final_path else None, "final_checkpoint_sha256": final_hash}
            checkpoints[arm] = {"path": arm_results[arm]["final_checkpoint"], "sha256": final_hash}
            trained_models[arm] = model
            trained_operators[arm] = smf
            arm_results[arm]["panel"] = panel
            canary._clear_grad(model)
            torch.cuda.empty_cache()

        safety_abort: _SafetyAbort | None = None
        try:
            if resume_arm == "identity_mse_control":
                if not isinstance(resume_payload, Mapping) or not isinstance(resume_payload.get("completed_arms"), Mapping):
                    raise ValueError("Identity resume is missing completed SMF binding")
                completed = resume_payload["completed_arms"].get("smf_mse_reconstruction")
                if not isinstance(completed, Mapping):
                    raise ValueError("Identity resume has no completed SMF checkpoint")
                smf_model, smf_optimizer, smf_operator = _runtime(formal, source, device, "smf_mse_reconstruction")
                completed_payload = _load_completed_arm(completed["path"], smf_model, smf_optimizer, spec=spec, data=data, hashes=training_hashes, output_dir=output.parent, expected_sha256=completed["sha256"])
                smf_metrics, smf_panel = _evaluate(smf_model, smf_operator, data, data.heldout_indices, device)
                arm_results["smf_mse_reconstruction"] = {"steps_completed": STEPS, "history": list(completed_payload["history"]), "metrics": smf_metrics, "final_checkpoint": str(_resolve(completed["path"]).resolve()), "final_checkpoint_sha256": completed["sha256"], "panel": smf_panel}
                checkpoints["smf_mse_reconstruction"] = {"path": arm_results["smf_mse_reconstruction"]["final_checkpoint"], "sha256": completed["sha256"]}
                trained_models["smf_mse_reconstruction"] = smf_model
                trained_operators["smf_mse_reconstruction"] = smf_operator
                run_arm("identity_mse_control", resume_path=_resolve(args.resume_checkpoint), completed_arms={"smf_mse_reconstruction": checkpoints["smf_mse_reconstruction"]})
            else:
                run_arm("smf_mse_reconstruction", resume_path=_resolve(args.resume_checkpoint) if resume_arm == "smf_mse_reconstruction" else None)
                run_arm("identity_mse_control", completed_arms={"smf_mse_reconstruction": checkpoints["smf_mse_reconstruction"]})
        except _SafetyAbort as exc:
            safety_abort = exc

        panel_path: Path | None = None
        panel_hash: str | None = None
        if safety_abort is None:
            panel_path = output.parent / "identity_smf_comparison_panel.pt"
            panel_hash = _save_panel(panel_path, data, device, training_hashes, source_identity_panel, source_smf_panel, arm_results["identity_mse_control"]["panel"], arm_results["smf_mse_reconstruction"]["panel"], {arm: arm_results[arm]["metrics"].get("coupling_efficiency_mean") for arm in ARMS})
        smf_final = arm_results.get("smf_mse_reconstruction", {}).get("metrics")
        identity_final = arm_results.get("identity_mse_control", {}).get("metrics")
        finite = all(item.get("metrics") is not None and item["metrics"]["finite"] and all(step["finite"] for step in item["history"]) for item in arm_results.values())
        all_steps = all(item.get("steps_completed") == STEPS for item in arm_results.values())
        updates = all(canary._four_phase_nonzero(step["update_norms"]) for item in arm_results.values() for step in item["history"])
        smf_science = bool(smf_final and (source_smf_metrics["paired_mse"] - smf_final["paired_mse"]) / source_smf_metrics["paired_mse"] >= SMF_REDUCTION_MIN and smf_final["paired_mse"] < smf_final["cyclic_mse"] and smf_final["centered_correlation"] > source_smf_metrics["centered_correlation"] and smf_final["variance_retention"] > source_smf_metrics["variance_retention"])
        identity_science = bool(identity_final and identity_final["paired_mse"] <= source_identity_metrics["paired_mse"] * (1.0 + IDENTITY_MAX_DEGRADATION) and identity_final["paired_mse"] < identity_final["cyclic_mse"])
        source_unchanged = _tree_equal(source_model_state, _state_cpu(source_identity_model)) and _tree_equal(source_formal_state, _module_state_cpu(source_identity_model.system.ols)) and _tree_equal(source_smf_state, _module_state_cpu(source_smf_operator))
        elapsed = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
        gates = (
            GateResult("finite_values", finite, {}),
            GateResult("both_arms_completed_1000_steps", all_steps, {"steps_per_arm": STEPS}),
            GateResult("four_phase_updates_nonzero", updates, {"layers": 4}),
            GateResult("smf_reconstruction_gate", smf_science, {"initial": source_smf_metrics.get("paired_mse"), "final": smf_final.get("paired_mse") if smf_final else None, "minimum_relative_reduction": SMF_REDUCTION_MIN}),
            GateResult("identity_control_gate", identity_science, {"initial": source_identity_metrics.get("paired_mse"), "final": identity_final.get("paired_mse") if identity_final else None, "maximum_relative_degradation": IDENTITY_MAX_DEGRADATION}),
            GateResult("source_state_unchanged", source_unchanged, {}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("source_and_panel_provenance", panel_hash is not None, {"panel_sha256": panel_hash}),
            GateResult("safety_mse_bound", safety_abort is None, {"ratio": SAFETY_MSE_RATIO}),
            GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
            GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
            GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
        )
        resource_failed = any(gate.name in {"cuda_peak_memory", "wall_time", "run_storage"} and not gate.passed for gate in gates)
        harness_failed = safety_abort is not None or not finite or not all_steps or not updates or not source_unchanged or panel_hash is None
        if safety_abort is not None:
            status, outcome, actions, reason = "failed", "safety_aborted", (), str(safety_abort)
        elif resource_failed:
            status, outcome, actions, reason = "resource_aborted", "resource_aborted", (), "SMF reconstruction pilot resource gate failed"
        elif harness_failed:
            status, outcome, actions, reason = "invalid_harness", "invalid_harness", (), "SMF reconstruction pilot harness gate failed"
        elif smf_science and identity_science:
            status, outcome, actions, reason = "passed", "passed", (NEXT_PASS_ACTION,), None
        else:
            status, outcome, actions, reason = "valid_failed", "valid_failed", (NEXT_VALID_FAILED_ACTION,), "complete valid run did not satisfy the registered SMF reconstruction gate"
        record = _training_record(state, spec, formal, data, training_hashes, status, actions, gates, reason)
        payload = _record_payload(TRAINING_SCHEMA, record, {"outcome": outcome, "validation_hash": validation_hash, "preflight_hash": preflight_hash, "source_hashes": source.hashes, "training_hashes": training_hashes, "initial_metrics": {"identity_mse_control": source_identity_metrics, "smf_mse_reconstruction": source_smf_metrics}, "arms": {arm: {key: value for key, value in result.items() if key != "panel"} for arm, result in arm_results.items()}, "checkpoints": checkpoints, "panel_path": str(panel_path.resolve()) if panel_path else None, "panel_sha256": panel_hash, "safety_abort": {"arm": safety_abort.arm, "step": safety_abort.step, "result": safety_abort.result} if safety_abort else None, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "evaluator_calls": 0, "formal_candidate_checkpoint": False, "training_authorized": False, "formal_promotion": False})
        _write_new(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status in {"passed", "valid_failed"} else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-reconstruction-validate":
        return validate(args)
    if args.research_action == "smf-reconstruction-preflight":
        return preflight(args)
    if args.research_action == "smf-reconstruction-train":
        return train(args)
    raise ValueError(f"unsupported SMF reconstruction action: {args.research_action}")
