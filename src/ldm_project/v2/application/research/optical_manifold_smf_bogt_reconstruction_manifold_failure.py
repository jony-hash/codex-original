"""Frozen bounded-recoverability audit for the Scaled-Chen reconstruction manifold.

This is a research-only consumer.  It fits one preregistered, minimum-norm
linear readout on a disjoint training subset and evaluates that frozen readout
against the unchanged physical forward on the registered selection and
confirmation panels.  The probe is diagnostic: it is never inserted into the
optical model, checkpoint, evaluator, or generation path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...config import FormalSystemConfig
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_reconstruction_audit as reconstruction
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_horizon as horizon
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_task_relevant_bottleneck as trba


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S3-RMFA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

PRIMARY_CHECKPOINT = "scaled_chen_step_5000"
SELECTION_COUNT = 1000
SELECTION_PER_LABEL = 100
CONFIRMATION_COUNT = 5000
FIT_COUNT = 1000
FIT_PER_LABEL = 100
FIT_TRAIN_LIMIT = 55000
FIT_EXCLUSION_SEED = 1235
MICROBATCH_SIZE = 10
PINV_RCOND = 1.0e-12
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
PHYSICAL_BASELINE_TOLERANCE = 1.0e-6
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

NEXT_ACTION_R1 = "manual_review_reconstruction_mapping_alignment_contract"
NEXT_ACTION_R2 = "manual_review_reconstruction_representation_sufficiency_contract"
NEXT_ACTION_R3 = "manual_review_reconstruction_failure_metric_contract"
NEXT_ACTION_RUN = "run_scaled_chen_reconstruction_manifold_failure"

PARENT_TRBA = {
    "config": "configs/research/lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_v1.yaml",
    "config_sha256": "23029689416358b5003974bfb74817cebf3791e03764b5124f1644e4f22d5095",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_review_v1.json",
    "review_sha256": "934e52354968db04aa1305d946138ae8958afab06e6f2d5f844d07cb731f3351",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/validation.json",
    "validation_sha256": "3bec36f315c872a05bcb3381929987d8fff57914351a2d5bef4ff1eebda0c659",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/audit.json",
    "audit_sha256": "b6be2c6f97dbdcb1d02ad0f2395eca438ebeb619196758a57eb8cc26f31745c0",
    "compact_report": "outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/compact_report.json",
    "compact_report_sha256": "a6015a3fa594515fc24ac1af6cb684d3aa70cd703ae22ae9e8ab3924c86f633a",
    "panel": "outputs/20260901_v2/runs/lightgen_scaled_chen_task_relevant_bottleneck_v1_attempt_02/panel.pt",
    "panel_sha256": "ec76f0760a743ef8fd845b6b93969fac727619e439b4f03109fb40218cc46319",
    "implementation_commit": "7c1c0f8dcfb878049260043441a117456e5ebe6a",
    "status_commit": "0ec8b664935050ca27bc4944ccc66680461510b7",
    "audit_status": "valid_failed",
    "audit_classification": "task_relevant_bottleneck_not_supported",
    "generation_readiness": "not_ready",
}

REFERENCES = {
    "scaled_chen_checkpoint": {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
        "producer_commit": "019b76ef911e0bd16b8d5d4169807323427684af",
        "global_step": 5000,
        "role": "frozen_scaled_chen_reference",
    },
    "identity_checkpoint_report_only": {
        "path": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/full_epoch_final_step_005500.pt",
        "sha256": "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f",
        "producer_commit": "4bbe4383f07e03572e154854cea8da6b5d24022e",
        "global_step": 5500,
        "role": "report_only_context",
    },
}

FIT_PANEL = {
    "source": "mnist_train_indices_0_54999",
    "count": FIT_COUNT,
    "per_label": FIT_PER_LABEL,
    "construction": "first_100_sorted_per_label_after_excluding_calibration_confirmation_plan3_plan4",
    "calibration_excluded": True,
    "confirmation_excluded": True,
    "plan3_plan4_excluded": True,
    "schedule_overlap_allowed_for_fit_only": True,
    "indices_sha256": "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d",
}

EVALUATION_PANELS = {
    "selection": {
        "source": "mnist_test_indices_5000_9999",
        "count": SELECTION_COUNT,
        "per_label": SELECTION_PER_LABEL,
        "parent": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/audit.json",
    },
    "confirmation": {
        "source": "horizon_heldout_train_indices_55000_59999",
        "count": CONFIRMATION_COUNT,
        "parent": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/audit.json",
    },
    "candidate_prefix": {"source": "mnist_test_indices_0_4999", "count": 5000, "consumed": False},
    "no_new_samples": True,
}

REPRESENTATION = {
    "plane": "pre_ols",
    "feature_projection": "adaptive_avg_pool(real,16x16)+adaptive_avg_pool(imag,16x16)",
    "feature_dim": 512,
    "bias": True,
    "input_dim": 513,
    "label_independent": True,
    "target_independent": True,
}

TARGET = {
    "kind": "camera_intensity_same_fov",
    "shape": [1, 200, 200],
    "preprocessing": "base.render_intensity_batch",
    "paired_target": True,
    "class_template_comparator_only": True,
}

PROBE = {
    "family": "moore_penrose_minimum_norm_linear_readout",
    "solver": "torch.linalg.pinv",
    "rcond": PINV_RCOND,
    "fit_dtype": "float64",
    "storage_dtype": "float32",
    "regularization": "none_parameter_free",
    "adaptive_rank": False,
    "nonlinear": False,
    "lambda": None,
    "fit_once_before_selection": True,
    "refit_selection": False,
    "refit_confirmation": False,
}

MEASUREMENT = {
    "objective": "bounded_frozen_recoverability",
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "report_only_checkpoints": ["step_0", "step_1000"],
    "primary_metric": "probe_paired_MSE",
    "comparators": ["physical_paired_MSE", "class_template_MSE", "cyclic_MSE"],
    "context_metrics": ["instance_retrieval", "within_label_distance_correlation", "rank"],
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "no_new_panel": True,
    "physical_forward_unchanged": True,
    "mechanism_evidence_only": True,
    "no_update": True,
}

GATES = {
    "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "physical_baseline_tolerance": PHYSICAL_BASELINE_TOLERANCE,
    "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "backward": False,
    "evaluator_calls": 0,
    "checkpoint_written": False,
    "prior_sampled": False,
    "generation_started": False,
    "physical_forward_modified": False,
    "formal_promotion": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_reconstruction_manifold_failure_attribution",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "prohibited": [
        "optimizer",
        "backward",
        "checkpoint_write",
        "prior_sampling",
        "generation",
        "evaluator",
        "fmf",
        "training",
        "promotion",
        "physical_forward_modification",
        "new_samples",
        "second_probe",
        "hyperparameter_sweep",
        "nonlinear_probe",
    ],
}

PHYSICAL_ANCHORS = {
    "selection": {"paired_mse": 0.06204846128821373, "class_template_mse": 0.032479993999004364, "cyclic_mse": 0.07995091378688812},
    "confirmation": {"paired_mse": 0.05778951570391655, "class_template_mse": 0.027741199359297752, "cyclic_mse": 0.07560829818248749},
}

PLAN3 = {
    "path": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_reconstruction_audit_v1_attempt_03/reconstruction_audit_panel.pt",
    "sha256": "6842b3e49da0cda84cac3499240a6d276ae018eb545860330c457ec457d6af25",
    "indices_sha256": "bd1ecf7433715eced813edee46a5038688ce62d30c8e51b74d6e4aaf992d7d90",
    "count": 100,
}
PLAN4 = {
    "path": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_latent_observer_v1_attempt_03/pre_ols_field.pt",
    "sha256": "ebf6af4a9efb7a9e8e91127c85566f27caa5f4f26e361dc9a3660e916ae91580",
    "indices_sha256": "9ffe5fd2df41a1a7627efd926e04159506d8aabb228799d9c384d9cb34565f03",
    "count": 1000,
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    trba_spec: trba.Spec
    trba_review_path: Path
    trba_validation_path: Path
    trba_audit_path: Path
    trba_compact_path: Path
    trba_panel_path: Path
    trba_audit: dict[str, Any]
    trba_compact: dict[str, Any]
    parent: trba.ParentBundle


@dataclass(frozen=True)
class Probe:
    weights: torch.Tensor
    fit_indices: torch.Tensor
    input_dim: int
    output_dim: int
    rank: int
    effective_rank: int
    condition_number: float
    cutoff: float
    feature_contract_hash: str
    fit_panel_hash: str
    probe_hash: str
    frozen: bool
    state_unchanged: bool
    ols_state_unchanged: bool


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"RMFA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema", "hypothesis", "formal_system_config", "parent_trba", "references",
        "fit_panel", "evaluation_panels", "representation", "target", "probe",
        "measurement", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("RMFA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RMFA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("RMFA formal config is not locked")
    expected = {
        "parent_trba": PARENT_TRBA,
        "references": REFERENCES,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "representation": REPRESENTATION,
        "target": TARGET,
        "probe": PROBE,
        "measurement": MEASUREMENT,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if payload[name] != value:
            raise ValueError(f"RMFA {name} contract is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "parent_trba": PARENT_TRBA,
            "references": {
                "primary_checkpoint": PRIMARY_CHECKPOINT,
                "checkpoint_sha256": REFERENCES["scaled_chen_checkpoint"]["sha256"],
                "report_only_identity_sha256": REFERENCES["identity_checkpoint_report_only"]["sha256"],
            },
            "fit_panel": {key: FIT_PANEL[key] for key in ("source", "count", "per_label", "construction", "indices_sha256")},
            "evaluation_panels": {
                "selection": EVALUATION_PANELS["selection"]["source"],
                "confirmation": EVALUATION_PANELS["confirmation"]["source"],
                "candidate_prefix_consumed": EVALUATION_PANELS["candidate_prefix"]["consumed"],
            },
            "probe": {key: PROBE[key] for key in ("family", "solver", "rcond", "adaptive_rank", "nonlinear", "refit_selection", "refit_confirmation")},
            "r1": "reconstruction_mapping_misalignment_supported",
            "r2": "bounded_reconstruction_recoverability_not_supported",
            "r3": "reconstruction_failure_attribution_underdetermined",
            "gates": {
                "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
                "strict_template_crossing": True,
                "instance_sensitive_control": True,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_confidence": CONFIDENCE_LEVEL,
                "physical_baseline_tolerance": PHYSICAL_BASELINE_TOLERANCE,
            },
            "no_update": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }
    if payload != expected:
        raise ValueError("RMFA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return payload


def _load_parent(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    trba_path = _resolve(PARENT_TRBA["config"], base_path=spec.path.parent)
    if file_sha256(trba_path) != PARENT_TRBA["config_sha256"]:
        raise ValueError("TRBA parent config is missing, stale, or tampered")
    trba_spec = trba._load_spec(trba_path)
    trba._assert_contract(trba_spec)
    review_path = _resolve(PARENT_TRBA["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != PARENT_TRBA["review_sha256"]:
        raise ValueError("TRBA parent review contract is stale")
    trba._load_review(review_path, trba_spec)
    bundle = trba._validate_rmri_parent(trba_spec, dataset_root, load_checkpoints=load_checkpoints)
    validation_path = _resolve(PARENT_TRBA["validation"], base_path=spec.path.parent)
    validation = _read_json_hash(validation_path, PARENT_TRBA["validation_sha256"], "TRBA validation")
    validation_record = EvidenceRecordV2.from_dict(validation["evidence"])
    validation_record.validate()
    if validation.get("status") != "passed" or validation_record.git_commit != PARENT_TRBA["implementation_commit"]:
        raise ValueError("TRBA validation terminal state is stale")
    audit_path = _resolve(PARENT_TRBA["audit"], base_path=spec.path.parent)
    audit = _read_json_hash(audit_path, PARENT_TRBA["audit_sha256"], "TRBA audit")
    audit_record = EvidenceRecordV2.from_dict(audit["evidence"])
    audit_record.validate()
    if (
        audit.get("schema") != trba.AUDIT_SCHEMA
        or audit.get("hypothesis") != trba.HYPOTHESIS
        or audit.get("status") != PARENT_TRBA["audit_status"]
        or audit.get("classification") != PARENT_TRBA["audit_classification"]
        or audit_record.git_commit != PARENT_TRBA["implementation_commit"]
    ):
        raise ValueError("TRBA audit terminal state is stale or mismatched")
    compact_path = _resolve(PARENT_TRBA["compact_report"], base_path=spec.path.parent)
    compact = _read_json_hash(compact_path, PARENT_TRBA["compact_report_sha256"], "TRBA compact report")
    if compact.get("classification") != PARENT_TRBA["audit_classification"] or compact.get("generation_readiness") != "not_ready":
        raise ValueError("TRBA compact terminal interpretation changed")
    panel_path = _resolve(PARENT_TRBA["panel"], base_path=spec.path.parent)
    if not panel_path.is_file() or file_sha256(panel_path) != PARENT_TRBA["panel_sha256"]:
        raise ValueError("TRBA panel is missing, stale, or tampered")
    return ParentBundle(trba_spec, review_path, validation_path, audit_path, compact_path, panel_path, audit, compact, bundle)


def _load_formal(bundle: ParentBundle) -> tuple[FormalSystemConfig, Path]:
    formal = bundle.parent.parent.formal
    path = bundle.parent.parent.formal_path
    if formal.kind != bogt.EXPECTED_FORMAL_KIND or not formal.is_formal:
        raise ValueError("RMFA requires the current formal Identity geometry")
    if tuple(formal.slm_shape) != (200, 200) or tuple(formal.metasurface_shape) != (100, 100):
        raise ValueError("RMFA formal geometry shape changed")
    return formal, path


def _load_source(bundle: ParentBundle, *, load_checkpoints: bool) -> mra.SourceBundle:
    return mra._load_source(bundle.parent.parent.mra_spec, load_checkpoints=load_checkpoints)


def _input_hashes(spec: Spec, review_path: Path, bundle: ParentBundle, source: mra.SourceBundle, fit_indices: torch.Tensor | None = None) -> dict[str, str]:
    parent = bundle.parent
    parent_hashes = trba._input_hashes(bundle.trba_spec, bundle.trba_review_path, parent, parent.parent.source)
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "formal_config": file_sha256(parent.parent.formal_path),
        "dataset_manifest": parent.parent.data.manifest["manifest_hash"],
        "selection_indices": parent.parent.data.manifest["selection_indices_sha256"],
        "confirmation_indices": parent.parent.data.manifest["confirmation_indices_sha256"],
        "parent_trba_config": file_sha256(bundle.trba_spec.path),
        "parent_trba_review_contract": file_sha256(bundle.trba_review_path),
        "parent_trba_validation": file_sha256(bundle.trba_validation_path),
        "parent_trba_audit": file_sha256(bundle.trba_audit_path),
        "parent_trba_compact_report": file_sha256(bundle.trba_compact_path),
        "parent_trba_panel": file_sha256(bundle.trba_panel_path),
        "plan3_panel": file_sha256(_resolve(PLAN3["path"])),
        "plan4_pre_ols": file_sha256(_resolve(PLAN4["path"])),
        "fit_panel": FIT_PANEL["indices_sha256"],
        "feature_contract": canonical_sha256(REPRESENTATION),
        "target_contract": canonical_sha256(TARGET),
        "probe_contract": canonical_sha256(PROBE),
        "implementation_module": file_sha256(Path(__file__).resolve()),
    }
    hashes.update({f"parent_{name}": value for name, value in parent_hashes.items()})
    if fit_indices is not None:
        hashes["fit_indices"] = tensor_sha256(fit_indices)
    if source.hashes.get("scaled_chen_step_5000"):
        hashes["scaled_chen_checkpoint"] = source.hashes["scaled_chen_step_5000"]
    return hashes


def _context(spec: Spec, bundle: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"measurement": MEASUREMENT, "representation": REPRESENTATION, "probe": PROBE, "hypothesis": HYPOTHESIS})
    manifest = bundle.parent.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, dataset_split_manifest_hash=manifest)


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _state_equal(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> bool:
    return set(before) == set(after) and all(torch.equal(before[name], after[name]) for name in before)


def _load_plan_indices(specification: Mapping[str, Any]) -> torch.Tensor:
    path = _resolve(specification["path"])
    payload = safe_torch_load(path, expected_sha256=specification["sha256"], map_location="cpu")
    if not isinstance(payload, Mapping) or not torch.is_tensor(payload.get("indices")):
        raise ValueError("reserved panel artifact does not expose indices")
    indices = payload["indices"].detach().cpu().to(torch.long)
    if indices.ndim != 1 or indices.numel() != specification["count"] or torch.unique(indices).numel() != indices.numel():
        raise ValueError("reserved panel indices are malformed")
    if tensor_sha256(indices) != specification["indices_sha256"]:
        raise ValueError("reserved panel index digest changed")
    return indices


def _fit_indices(data: mra.AuditData, horizon_data: horizon.HorizonData, plan3_indices: torch.Tensor, plan4_indices: torch.Tensor) -> torch.Tensor:
    if data.images.shape[0] != 60000 or data.labels.shape[0] != 60000:
        raise ValueError("RMFA fit requires the complete MNIST train split")
    train = torch.arange(FIT_TRAIN_LIMIT, dtype=torch.long)
    excluded = torch.cat((horizon_data.calibration_indices, data.confirmation_indices, plan3_indices, plan4_indices))
    excluded = torch.unique(excluded[(excluded >= 0) & (excluded < FIT_TRAIN_LIMIT)])
    available = train[~torch.isin(train, excluded)]
    pieces: list[torch.Tensor] = []
    for label in range(10):
        pieces.append(available[data.labels.index_select(0, available) == label][:FIT_PER_LABEL])
    if any(piece.numel() != FIT_PER_LABEL for piece in pieces):
        raise ValueError("RMFA fit panel lacks a complete balanced class")
    result = torch.cat(pieces)
    counts = torch.bincount(data.labels.index_select(0, result), minlength=10)
    if result.numel() != FIT_COUNT or not torch.equal(counts, torch.full((10,), FIT_PER_LABEL, dtype=torch.long)):
        raise ValueError("RMFA fit panel is not balanced")
    if tensor_sha256(result) != FIT_PANEL["indices_sha256"]:
        raise ValueError("RMFA fit panel construction hash changed")
    if torch.isin(result, data.confirmation_indices).any() or torch.isin(result, horizon_data.calibration_indices).any():
        raise ValueError("RMFA fit panel overlaps reserved data")
    return result


def _feature_projection(value: torch.Tensor) -> torch.Tensor:
    return mra._feature_projection(value, complex_value=True).float()


def _capture(model: bogt.ScaledChenBOGTModel, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(images)
    with torch.no_grad():
        stages, camera = mra._scaled_trace_stages(model, rendered.amplitude.to(device))
    pre = stages["pre_ols"].detach()
    return _feature_projection(pre).cpu(), camera.detach().cpu().float()


def _solve_minimum_norm(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve the one registered readout without selecting any fit parameter."""
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("RMFA design and target matrices must have equal row counts")
    if features.shape[1] + 1 != REPRESENTATION["input_dim"] or targets.shape[1] != 40000:
        raise ValueError("RMFA design or target dimensions changed")
    augmented = torch.cat((features.float(), torch.ones((features.shape[0], 1), dtype=torch.float32)), dim=1).double()
    target_double = targets.float().double()
    singular = torch.linalg.svdvals(augmented)
    if singular.numel() == 0 or not torch.isfinite(singular).all() or float(singular.max()) <= 0.0:
        raise ValueError("RMFA fit design is not finite")
    cutoff = float(singular.max()) * PINV_RCOND
    retained = singular >= cutoff
    rank = int(retained.sum())
    condition = float(singular.max() / singular[retained][-1]) if rank else float("inf")
    weights = torch.linalg.pinv(augmented, rcond=PINV_RCOND).mm(target_double).float().contiguous()
    if not torch.isfinite(weights).all():
        raise ValueError("RMFA probe coefficients are not finite")
    return weights, {"rank": rank, "effective_rank": rank, "condition_number": condition, "cutoff": cutoff}


def _fit_probe(model: bogt.ScaledChenBOGTModel, data: mra.AuditData, fit_indices: torch.Tensor, device: torch.device) -> Probe:
    model_before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    fit_images = data.images.index_select(0, fit_indices)
    for indices in torch.arange(fit_indices.numel(), dtype=torch.long).split(MICROBATCH_SIZE):
        feature, _ = _capture(model, fit_images.index_select(0, indices), device)
        target = base.render_intensity_batch(fit_images.index_select(0, indices)).target.float()
        features.append(feature)
        targets.append(target.flatten(1))
    x = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    if x.shape != (FIT_COUNT, REPRESENTATION["feature_dim"]) or y.shape[1] != 40000:
        raise ValueError("RMFA probe dimensions changed")
    weights, solver = _solve_minimum_norm(x, y)
    if not _state_equal(model_before, _state_cpu(model)) or not _state_equal(ols_before, _state_cpu(model.ols)):
        raise ValueError("RMFA model state changed during probe fitting")
    feature_hash = canonical_sha256(REPRESENTATION)
    fit_hash = tensor_sha256(fit_indices)
    return Probe(
        weights=weights,
        fit_indices=fit_indices.clone(),
        input_dim=REPRESENTATION["input_dim"],
        output_dim=int(y.shape[1]),
        rank=int(solver["rank"]),
        effective_rank=int(solver["effective_rank"]),
        condition_number=float(solver["condition_number"]),
        cutoff=float(solver["cutoff"]),
        feature_contract_hash=feature_hash,
        fit_panel_hash=fit_hash,
        probe_hash=tensor_sha256(weights),
        frozen=True,
        state_unchanged=True,
        ols_state_unchanged=True,
    )


def _apply_probe(probe: Probe, features: torch.Tensor) -> torch.Tensor:
    if not probe.frozen or features.ndim != 2 or features.shape[1] + 1 != probe.input_dim:
        raise ValueError("RMFA probe is not frozen or has an invalid feature shape")
    augmented = torch.cat((features.float(), torch.ones((features.shape[0], 1))), dim=1)
    result = augmented.mm(probe.weights).reshape(-1, 1, 200, 200)
    if not torch.isfinite(result).all():
        raise ValueError("RMFA probe output is not finite")
    return result


def _evaluate_panel(model: bogt.ScaledChenBOGTModel, probe: Probe, images: torch.Tensor, labels: torch.Tensor, device: torch.device, name: str) -> dict[str, Any]:
    model_before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    target = base.render_intensity_batch(images).target.float()
    templates = torch.stack([target[labels == label].mean(dim=0) for label in range(10)])
    cyclic_target = target.roll(1, 0)
    physical_errors: list[torch.Tensor] = []
    probe_errors: list[torch.Tensor] = []
    cyclic_errors: list[torch.Tensor] = []
    template_errors: list[torch.Tensor] = []
    physical_features: list[torch.Tensor] = []
    probe_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    first_target = target[:1].clone()
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        feature, physical = _capture(model, images.index_select(0, indices), device)
        predicted = _apply_probe(probe, feature)
        target_batch = target.index_select(0, indices)
        cyclic_batch = cyclic_target.index_select(0, indices)
        template_batch = templates.index_select(0, labels.index_select(0, indices))
        physical_errors.append((physical - target_batch).square().mean(dim=(1, 2, 3)))
        probe_errors.append((predicted - target_batch).square().mean(dim=(1, 2, 3)))
        cyclic_errors.append((physical - cyclic_batch).square().mean(dim=(1, 2, 3)))
        template_errors.append((physical - template_batch).square().mean(dim=(1, 2, 3)))
        physical_features.append(mra._feature_projection(physical))
        probe_features.append(mra._feature_projection(predicted))
        target_features.append(mra._feature_projection(target_batch))
    physical_mse = torch.cat(physical_errors)
    probe_mse = torch.cat(probe_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    physical_compact = torch.cat(physical_features)
    probe_compact = torch.cat(probe_features)
    target_compact = torch.cat(target_features)
    probe_context = _context_metrics_from_compact(probe_compact, target_compact, labels)
    physical_context = _context_metrics_from_compact(physical_compact, target_compact, labels)
    delta = physical_mse - probe_mse
    ci = _stratified_bootstrap_ci(delta, labels)
    anchor = PHYSICAL_ANCHORS[name]
    physical_values = {"paired_mse": float(physical_mse.mean()), "class_template_mse": float(template_mse.mean()), "cyclic_mse": float(cyclic_mse.mean())}
    replay = {key: abs(physical_values[key] - value) <= PHYSICAL_BASELINE_TOLERANCE for key, value in anchor.items()}
    report = {
        "panel": name,
        "evaluation_count": int(images.shape[0]),
        "physical": {**physical_values, "mse_by_label": [float(physical_mse[labels == label].mean()) for label in range(10)], **physical_context},
        "probe": {"paired_mse": float(probe_mse.mean()), "mse_by_label": [float(probe_mse[labels == label].mean()) for label in range(10)], **_context_metrics_from_compact(probe_compact, target_compact, labels)},
        "cyclic_mse": float(cyclic_mse.mean()),
        "class_template_mse": float(template_mse.mean()),
        "relative_improvement": float(delta.mean() / max(float(physical_mse.mean()), 1.0e-12)),
        "bootstrap": {"delta_definition": "physical_paired_mse - probe_paired_mse", "mean_delta": float(delta.mean()), "ci": list(ci), "replicates": BOOTSTRAP_REPLICATES, "confidence_level": CONFIDENCE_LEVEL},
        "native_physical_anchor": anchor,
        "native_replay": replay,
        "finite": bool(torch.isfinite(physical_mse).all() and torch.isfinite(probe_mse).all() and torch.isfinite(cyclic_mse).all() and torch.isfinite(template_mse).all()),
        "instance_sensitive": bool(float(probe_mse.mean()) < float(cyclic_mse.mean()) and float(probe_context["same_label_top1"]) >= float(probe_context["class_template_same_label_top1"])),
        "state_unchanged": _state_equal(model_before, _state_cpu(model)),
        "ols_state_unchanged": _state_equal(ols_before, _state_cpu(model.ols)),
        "sample_mse": {"physical": physical_mse, "probe": probe_mse, "cyclic": cyclic_mse, "class_template": template_mse},
        "target_first": first_target,
    }
    report["selection_gate"] = _panel_gate(report, require_bootstrap=False)
    return report


def _context_metrics_from_compact(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    retrieval = reconstruction._retrieval_metrics(prediction, target, labels)
    geometry, geometry_by_label = reconstruction._pairwise_distance_correlation(prediction, target, labels)
    templates = torch.stack([target[labels == label].mean(dim=0) for label in range(10)])
    distances = prediction.square().sum(dim=1, keepdim=True) + templates.square().sum(dim=1).unsqueeze(0) - 2.0 * prediction.mm(templates.T)
    distances = distances.clamp_min(0.0)
    return {**retrieval, "within_label_distance_correlation": geometry, "within_label_distance_correlation_by_label": geometry_by_label, "class_template_same_label_top1": float((distances.argmin(dim=1) == labels).float().mean())}


def _stratified_bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor, *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    if delta.ndim != 1 or labels.ndim != 1 or delta.numel() != labels.numel() or not torch.isfinite(delta).all():
        raise ValueError("RMFA bootstrap input is invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means: list[torch.Tensor] = []
    for _ in range(replicates):
        rows: list[torch.Tensor] = []
        for label in range(10):
            values = delta[labels == label]
            rows.append(values.index_select(0, torch.randint(values.numel(), (values.numel(),), generator=generator)))
        means.append(torch.cat(rows).mean())
    quantiles = torch.stack(means).quantile(torch.tensor([(1.0 - CONFIDENCE_LEVEL) / 2.0, 1.0 - (1.0 - CONFIDENCE_LEVEL) / 2.0]))
    return float(quantiles[0]), float(quantiles[1])


def _panel_gate(report: Mapping[str, Any], *, require_bootstrap: bool) -> dict[str, Any]:
    probe = report["probe"]
    bootstrap = report["bootstrap"]
    checks = {
        "finite": report.get("finite") is True,
        "physical_anchor_replay": all(report.get("native_replay", {}).values()),
        "relative_improvement": float(report.get("relative_improvement", float("nan"))) >= MATERIAL_RELATIVE_IMPROVEMENT,
        "strict_template_crossing": float(probe.get("paired_mse", float("inf"))) < float(report.get("class_template_mse", float("-inf"))),
        "paired_below_cyclic": float(probe.get("paired_mse", float("inf"))) < float(report.get("cyclic_mse", float("-inf"))),
        "instance_sensitive": report.get("instance_sensitive") is True,
        "frozen": report.get("state_unchanged") is True and report.get("ols_state_unchanged") is True,
    }
    if require_bootstrap:
        checks["bootstrap_lower_bound"] = float(bootstrap["ci"][0]) > 0.0
    return {"passed": bool(all(checks.values())), "checks": checks}


def _classify(selection: Mapping[str, Any], confirmation: Mapping[str, Any] | None = None) -> tuple[str, str]:
    if selection.get("probe_contract_valid") is False or selection.get("interpretation_stable") is False:
        return "reconstruction_failure_attribution_underdetermined", NEXT_ACTION_R3
    if not selection.get("selection_gate", {}).get("passed", False):
        return "bounded_reconstruction_recoverability_not_supported", NEXT_ACTION_R2
    if confirmation is None:
        return "bounded_reconstruction_recoverability_not_supported", NEXT_ACTION_R2
    if confirmation.get("selection_gate", {}).get("passed", False) and confirmation.get("confirmation_gate", {}).get("passed", False):
        return "reconstruction_mapping_misalignment_supported", NEXT_ACTION_R1
    return "bounded_reconstruction_recoverability_not_supported", NEXT_ACTION_R2


def _record(schema: str, status: str, classification: str | None, spec: Spec, state: Mapping[str, Any], bundle: ParentBundle, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, bundle), dict(hashes), tuple(gates), tuple(actions), reason)
    record.validate()
    return {
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "generation_started": False,
        "physical_forward_modified": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse RMFA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("RMFA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"RMFA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if not existing.issubset({validation_path.name}):
        raise FileExistsError("RMFA attempt directory contains unexpected artifacts")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RMFA audit artifact: {output_path}")


def _validation_gates(spec: Spec, bundle: ParentBundle, output_path: Path, fit_indices: torch.Tensor) -> tuple[GateResult, ...]:
    data = bundle.parent.parent.data
    labels = data.test_labels.index_select(0, data.selection_indices)
    counts = torch.bincount(labels, minlength=10)
    operator = mra._build_research_ols(bundle.parent.parent.formal)
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("parent_trba_terminal", bundle.trba_audit.get("classification") == PARENT_TRBA["audit_classification"], {}),
        GateResult("selection_balanced", bool(torch.equal(counts, torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long))), {"counts": counts.tolist()}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_confirmation_disjoint", not bool(torch.isin(fit_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_panel_locked", fit_indices.numel() == FIT_COUNT and tensor_sha256(fit_indices) == FIT_PANEL["indices_sha256"], {}),
        GateResult("mode_bank_rank_100", mra._mode_rank(operator) == 100, {"rank": mra._mode_rank(operator)}),
        GateResult("finite_mode_bank", bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all()), {}),
        GateResult("formal_identity_boundary", bundle.parent.parent.formal.kind == bogt.EXPECTED_FORMAL_KIND and bundle.parent.parent.formal.is_formal, {}),
        GateResult("candidate_prefix_not_consumed", EVALUATION_PANELS["candidate_prefix"]["consumed"] is False, {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0}),
        GateResult("output_path_absent", not output_path.exists(), {"path": str(output_path)}),
        GateResult("disk_free_precondition", disk.free > MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES}),
    )


def _load_validation(path: str | Path, spec: Spec, review_path: Path, bundle: ParentBundle, source: mra.SourceBundle, state: Mapping[str, Any], fit_indices: torch.Tensor) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, bundle, source, fit_indices)
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("status") != "passed"
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.input_hashes != expected_hashes
        or record.permitted_next_actions != (NEXT_ACTION_RUN,)
    ):
        raise ValueError("RMFA validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    bundle = _load_parent(spec, args.dataset_root, load_checkpoints=False)
    horizon_data = horizon._load_data(args.dataset_root)
    plan3_indices = _load_plan_indices(PLAN3)
    plan4_indices = _load_plan_indices(PLAN4)
    fit_indices = _fit_indices(bundle.parent.parent.data, horizon_data, plan3_indices, plan4_indices)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, bundle, output_path, fit_indices)
    source = bundle.parent.parent.source
    hashes = _input_hashes(spec, review_path, bundle, source, fit_indices)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, state, bundle, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), {"fit_indices": fit_indices.tolist(), "fit_panel": FIT_PANEL, "evaluation_panels": EVALUATION_PANELS, "representation": REPRESENTATION, "target": TARGET, "probe": PROBE, "measurement": MEASUREMENT}, None if valid else "RMFA static validation failed")
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="RMFA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RMFA requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise RuntimeError("RMFA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("RMFA requires 10 GiB free storage")
    return int(free), int(total)


def _probe_preflight(probe: Probe, model: bogt.ScaledChenBOGTModel, data: mra.AuditData, fit_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    if not probe.frozen or probe.weights.shape != (513, 40000) or probe.fit_panel_hash != tensor_sha256(fit_indices):
        raise ValueError("RMFA probe preflight contract failed")
    before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    features, _ = _capture(model, data.images.index_select(0, fit_indices[:16]), device)
    output = _apply_probe(probe, features)
    return {"finite": bool(torch.isfinite(output).all()), "feature_dim": int(features.shape[1]), "output_shape": list(output.shape), "probe_frozen": probe.frozen, "state_unchanged": _state_equal(before, _state_cpu(model)), "ols_state_unchanged": _state_equal(ols_before, _state_cpu(model.ols)), "optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False, "generation_started": False, "physical_forward_modified": False}


def _probe_artifact(probe: Probe, selection: Mapping[str, Any], confirmation: Mapping[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scaled_chen_reconstruction_manifold_failure_probe.v1",
        "weights": probe.weights,
        "fit_indices": probe.fit_indices,
        "metadata": {"input_dim": probe.input_dim, "output_dim": probe.output_dim, "rank": probe.rank, "effective_rank": probe.effective_rank, "condition_number": probe.condition_number, "cutoff": probe.cutoff, "feature_contract_hash": probe.feature_contract_hash, "fit_panel_hash": probe.fit_panel_hash, "probe_hash": probe.probe_hash, "frozen": probe.frozen},
        "selection_sample_mse": {key: value for key, value in selection["sample_mse"].items()},
        "confirmation_sample_mse": {key: value for key, value in (confirmation or {}).get("sample_mse", {}).items()},
        "mechanism_evidence_only": True,
        "physical_forward_modified": False,
    }
    return payload


def _json_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key not in {"sample_mse", "target_first"}}


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("RMFA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    bundle = _load_parent(spec, args.dataset_root, load_checkpoints=False)
    horizon_data = horizon._load_data(args.dataset_root)
    fit_indices = _fit_indices(bundle.parent.parent.data, horizon_data, _load_plan_indices(PLAN3), _load_plan_indices(PLAN4))
    source = _load_source(bundle, load_checkpoints=False)
    _load_validation(args.validation_evidence, spec, review_path, bundle, source, state, fit_indices)
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _check_run_slot(output_path, validation_path)
    compact_path = output_path.with_name("compact_report.json")
    probe_path = output_path.with_name("probe.pt")
    if compact_path.exists() or probe_path.exists():
        raise FileExistsError("refusing to overwrite RMFA companion artifact")
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    source = _load_source(bundle, load_checkpoints=True)
    checkpoint = source.checkpoints[PRIMARY_CHECKPOINT]
    formal, _ = _load_formal(bundle)
    model = mra._load_scaled_model(formal, checkpoint, device)
    state_before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    probe = _fit_probe(model, bundle.parent.parent.data, fit_indices, device)
    preflight = _probe_preflight(probe, model, bundle.parent.parent.data, fit_indices, device)
    selection_images = bundle.parent.parent.data.test_images.index_select(0, bundle.parent.parent.data.selection_indices)
    selection_labels = bundle.parent.parent.data.test_labels.index_select(0, bundle.parent.parent.data.selection_indices)
    selection = _evaluate_panel(model, probe, selection_images, selection_labels, device, "selection")
    selection["probe_contract_valid"] = True
    selection["interpretation_stable"] = True
    confirmation: dict[str, Any] | None = None
    if selection["selection_gate"]["passed"]:
        confirmation_images = bundle.parent.parent.data.images.index_select(0, bundle.parent.parent.data.confirmation_indices)
        confirmation_labels = bundle.parent.parent.data.labels.index_select(0, bundle.parent.parent.data.confirmation_indices)
        confirmation = _evaluate_panel(model, probe, confirmation_images, confirmation_labels, device, "confirmation")
        confirmation["confirmation_gate"] = _panel_gate(confirmation, require_bootstrap=True)
        confirmation["probe_contract_valid"] = True
        confirmation["interpretation_stable"] = True
    classification, next_action = _classify(selection, confirmation)
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    state_unchanged = _state_equal(state_before, _state_cpu(model))
    ols_unchanged = _state_equal(ols_before, _state_cpu(model.ols))
    probe_payload = _probe_artifact(probe, selection, confirmation)
    estimated_bytes = probe.weights.numel() * probe.weights.element_size() + fit_indices.numel() * fit_indices.element_size()
    if estimated_bytes > MAX_STORAGE_BYTES:
        raise RuntimeError("RMFA probe artifact exceeds storage gate")
    atomic_torch_save(probe_path, probe_payload)
    probe_hash = file_sha256(probe_path)
    compact = {
        "schema": "scaled_chen_reconstruction_manifold_failure_compact_report.v1",
        "status": "passed" if classification == "reconstruction_mapping_misalignment_supported" else "valid_failed",
        "classification": classification,
        "next_action": next_action,
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "selection": _json_report(selection),
        "confirmation": _json_report(confirmation) if confirmation is not None else {"executed": False},
        "probe": {"input_dim": probe.input_dim, "output_dim": probe.output_dim, "rank": probe.rank, "effective_rank": probe.effective_rank, "condition_number": probe.condition_number, "cutoff": probe.cutoff, "probe_hash": probe.probe_hash, "artifact_sha256": probe_hash, "frozen": probe.frozen},
        "reconstruction_manifold_supported": classification == "reconstruction_mapping_misalignment_supported",
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "generation_started": False,
        "physical_forward_modified": False,
        "mechanism_evidence_only": True,
    }
    _write_new(compact_path, compact, artifact_kind="RMFA compact report")
    compact_hash = file_sha256(compact_path)
    audit_contract = {
        "probe": {"input_dim": probe.input_dim, "output_dim": probe.output_dim, "fit_count": FIT_COUNT, "rank": probe.rank, "effective_rank": probe.effective_rank, "condition_number": probe.condition_number, "cutoff": probe.cutoff, "coefficient_count": int(probe.weights.numel()), "feature_contract_hash": probe.feature_contract_hash, "fit_panel_hash": probe.fit_panel_hash, "probe_hash": probe.probe_hash, "frozen": probe.frozen},
        "preflight": preflight,
        "selection": _json_report(selection),
        "confirmation": _json_report(confirmation) if confirmation is not None else {"executed": False},
        "classification": classification,
        "next_action": next_action,
        "physical_forward_modified": False,
        "probe_applied_to_physical_forward": False,
        "native_baseline_anchors": PHYSICAL_ANCHORS,
        "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only", "output_bytes": int(sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()))},
        "probe_path": str(probe_path.resolve()),
        "probe_sha256": probe_hash,
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": compact_hash,
        "model_state_before": _state_digest(state_before),
        "model_state_after": _state_digest(_state_cpu(model)),
        "ols_state_before": _state_digest(ols_before),
        "ols_state_after": _state_digest(_state_cpu(model.ols)),
        "state_unchanged": state_unchanged,
        "ols_state_unchanged": ols_unchanged,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "generation_started": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
    }
    gates = (
        GateResult("finite_measurements", selection.get("finite") is True and (confirmation is None or confirmation.get("finite") is True), {}),
        GateResult("frozen_model_and_OLS", state_unchanged and ols_unchanged and selection.get("state_unchanged") is True and selection.get("ols_state_unchanged") is True, {}),
        GateResult("native_physical_replay", all(selection.get("native_replay", {}).values()) and (confirmation is None or all(confirmation.get("native_replay", {}).values())), {}),
        GateResult("probe_frozen_and_disjoint", probe.frozen and probe.fit_panel_hash == tensor_sha256(fit_indices), {}),
        GateResult("no_update_isolation", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False, "generation_started": False}),
        GateResult("resource_limits", peak <= MAX_PEAK_CUDA_BYTES and sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()) <= MAX_STORAGE_BYTES, {"peak_bytes": peak}),
    )
    status = "passed" if classification == "reconstruction_mapping_misalignment_supported" else "valid_failed"
    hashes = _input_hashes(spec, review_path, bundle, source, fit_indices)
    hashes.update({"probe": probe_hash, "compact_report": compact_hash})
    payload = _record(AUDIT_SCHEMA, status, classification, spec, state, bundle, hashes, gates, (next_action,), audit_contract, "RMFA is diagnostic-only; no downstream training or promotion")
    payload.update({"reconstruction_manifold_supported": compact["reconstruction_manifold_supported"], "generative_manifold_supported": False, "generation_readiness": "not_ready", "probe_sha256": probe_hash, "compact_report_sha256": compact_hash})
    _write_new(output_path, payload, artifact_kind="RMFA audit evidence")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-reconstruction-manifold-failure-validate":
        return validate(args)
    if args.research_action == "scaled-chen-reconstruction-manifold-failure-run":
        return run(args)
    raise ValueError(f"unsupported RMFA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA", "AUTHORIZATION", "CONFIG_SCHEMA", "EVALUATION_PANELS", "FIT_PANEL", "GATES",
    "HYPOTHESIS", "MEASUREMENT", "NEXT_ACTION_R1", "NEXT_ACTION_R2", "NEXT_ACTION_R3", "PARENT_TRBA",
    "PROBE", "REFERENCES", "REPRESENTATION", "REVIEW_SCHEMA", "TARGET", "VALIDATION_SCHEMA", "Probe", "Spec",
    "_apply_probe", "_assert_contract", "_classify", "_fit_indices", "_load_review", "_load_spec", "_panel_gate", "_solve_minimum_norm",
    "_stratified_bootstrap_ci", "dispatch", "run", "validate",
]
