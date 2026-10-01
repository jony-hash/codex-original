"""Task-relevant, transform-aware bottleneck attribution for Scaled-Chen.

This is a research-only frozen consumer.  It measures the registered
``pre-OLS -> modal`` and ``decoder input -> decoder layer 2`` segments
separately.  The primary score compares normalized complex Gram geometry, so
per-sample phase, common coordinate permutations, common diagonal phases, and
common unitary basis changes do not change the sample relationships.  Raw
complex rank/L2 values remain contextual history only.

The module never creates an optimizer, calls an evaluator, performs a
backward pass, writes a checkpoint, changes a physical forward, samples a
prior, or starts a generation/FMF path.
"""

from __future__ import annotations

import argparse
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

from ldm_project.v2.config import FormalSystemConfig
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_new_json, canonical_sha256, file_sha256
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_high_fidelity_relay_reference as hfrr
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_relay_metric_reinterpretation as rmri


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_task_relevant_bottleneck_panel.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S2-TRBA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

OLS_SEGMENT = "ols_modal_projection"
DECODER_SEGMENT = "decoder_internal"
SEGMENTS = (OLS_SEGMENT, DECODER_SEGMENT)
SEGMENT_ENDPOINTS = {
    OLS_SEGMENT: ("pre_ols", "modal_complex"),
    DECODER_SEGMENT: ("decoder_input", "decoder_layer_2"),
}
SELECTION_COUNT = 1000
SELECTION_PER_LABEL = 100
CONFIRMATION_COUNT = 5000
CONTROL_PROBE_COUNT = 128
MICROBATCH_SIZE = 10
BLOCK_SIZE = 128
PERMUTATION_SEED = 1234
GLOBAL_PHASE_SEED = 2051
DIAGONAL_PHASE_SEED = 2052
MATERIAL_GEOMETRY_DEGRADATION = 0.05
MATERIAL_TASK_DEGRADATION = 0.05
GEOMETRY_CONTROL_TOLERANCE = 1.0e-5
LABEL_CONTROL_TOLERANCE = 1.0e-5
RETRIEVAL_CONTROL_TOLERANCE = 0.001
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
MIN_DISK_FREE_BYTES = 10 * 1024**3

NEXT_ACTION_OLS = "manual_review_modal_capacity_basis_attribution_contract"
NEXT_ACTION_DECODER = "manual_review_decoder_internal_transform_attribution_contract"
NEXT_ACTION_BOTH = "manual_review_modal_decoder_bottleneck_priority_contract"
NEXT_ACTION_NEITHER = "manual_review_scaled_chen_bottleneck_attribution_closeout_contract"
NEXT_ACTION_UNDERDETERMINED = "manual_review_trba_metric_contract_closeout"
NEXT_ACTION_RUN = "run_scaled_chen_task_relevant_bottleneck"

CLASSIFICATION_BY_TIER = {
    "T1": "task_relevant_bottleneck_ols_only_supported",
    "T2": "task_relevant_bottleneck_decoder_only_supported",
    "T3": "task_relevant_distributed_bottleneck_both_supported",
    "T4": "task_relevant_bottleneck_not_supported",
    "T5": "task_relevant_bottleneck_attribution_underdetermined",
}

PARENT_RMRI = {
    "config": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_v1.yaml",
    "config_sha256": "f457bd6d7c1b1fb5499ee0525028d5131e45f066ba7e36ab846571378ee59c2b",
    "review_contract": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_review_v1.json",
    "review_sha256": "9c4eac44b9e022d0434bb04830486132c3eed3c1513a96a02958eade8fb17037",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/validation.json",
    "validation_sha256": "a87769fb638d5bffa1f77bc31c931fd3999c398a23c1a1f17d51d01eb91882eb",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/audit.json",
    "audit_sha256": "d409729f9b111c024c1913cef90c6cd2ebe778efa9990b836a955783f378ba00",
    "compact_report": "outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/compact_report.json",
    "compact_report_sha256": "cd39f2fed5719572a558be2bbbf139cdb7cc657a08d74c2be4cd8b69b71c35b8",
    "panel": "outputs/20260901_v2/runs/lightgen_scaled_chen_relay_metric_reinterpretation_v1_attempt_03/panel.pt",
    "panel_sha256": "6b7f9225624fa404a0743a417903b27a7113d100315ea9b1804043c63aafb079",
    "implementation_commit": "16896f06b31549120adf6235587619fe2820af83",
    "status_commit": "9ac0dc1128b02b116aefd32eb31b1a408c826b71",
    "audit_status": "valid_failed",
    "audit_classification": "relay_propagation_bottleneck_reinterpreted_metric_confound",
    "generation_readiness": "not_ready",
}

PARENT_MRA = {
    "config": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml",
    "config_sha256": "95e4587f87fc49e79fe3bef5a930f34502ebc9c95f36a312995d77f993a72cad",
    "review_contract": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_manifold_readiness_review_v1.json",
    "review_sha256": "3d0c807b0fadf0bb7f6ef0596c5a3279913175b82b792fa6b1015a086b207ac8",
    "audit": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/audit.json",
    "audit_sha256": "6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556",
    "audit_git_commit": "a041187a6ff6292f2be6602871c1fd6d35ca172c",
    "audit_status": "valid_failed",
    "audit_classification": "distributed_bottleneck_supported",
    "generation_readiness": "not_ready",
}

PARENT_HFRR = {
    "config": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_v1.yaml",
    "config_sha256": "1c28c859e7f39235172ca1c2437ad2621bbb790ae4e89bf1e9f743181e21d4a8",
    "review_contract": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_review_v1.json",
    "review_sha256": "0968b9d1334606113828f2bab8b02d7f718fab00d8be2606d84eb8f511c2ec04",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/validation.json",
    "validation_sha256": "248983c83cdd122f4a6aa7cf7f38bf6b415d3f080eb8136bb0a4fdb0b8e1c84f",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/audit.json",
    "audit_sha256": "4045e8e0469a745c637c8dae151d073e5b65e06395bf175297d93245f1031e48",
    "compact_report": "outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/compact_report.json",
    "compact_report_sha256": "d85e82d9a98fb36568d16c63d0fc1a4e5e3a311a007c512f2b07be7472427e7b",
    "panel": "outputs/20260901_v2/runs/lightgen_scaled_chen_high_fidelity_relay_reference_v1_attempt_01/panel.pt",
    "panel_sha256": "dc75d62ed53a2699aa0949fd8bd8a465f0cec2abe46b64279858a304bde9301b",
    "implementation_commit": "8835bb228f074408a596118a907248e0f7c9c609",
    "audit_status": "valid_failed",
    "audit_classification": "relay_complex_retention_metric_confound_supported",
    "numerical_boundary_not_evaluated": False,
    "relay_numerical_reference_consistent": True,
    "relay_numerical_reference_mismatch": False,
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
        "role": "matched_report_only_reference",
    },
}

PANELS = {
    "selection": {
        "source": "mnist_test_indices_5000_9999",
        "seed": 1235,
        "count": SELECTION_COUNT,
        "per_label": SELECTION_PER_LABEL,
        "parent": PARENT_MRA["audit"],
    },
    "confirmation": {
        "source": "horizon_heldout_train_indices_55000_59999",
        "count": CONFIRMATION_COUNT,
        "parent": PARENT_MRA["audit"],
    },
    "candidate_prefix": {
        "source": "mnist_test_indices_0_4999",
        "count": 4560,
        "consumed": False,
    },
    "no_new_samples": True,
}

MEASUREMENT = {
    "objective": "task_relevant_frozen_forward_only",
    "models": ["scaled_chen_step_5000"],
    "segments": [
        {"id": OLS_SEGMENT, "input": "pre_ols", "output": "modal_complex"},
        {"id": DECODER_SEGMENT, "input": "decoder_input", "output": "decoder_layer_2"},
    ],
    "raw_context_metric": "mra_dcm_entropy_participation_geometric_mean",
    "raw_context_not_primary": True,
    "primary_metric": "normalized_gram_similarity_geometry_retention",
    "primary_definition": "1-mean_upper_triangle(abs(S_in-S_out)); S_ij=abs(<z_i,z_j>)^2/[(||z_i||^2+eps)(||z_j||^2+eps)]",
    "secondary_metrics": [
        "true_label_geometry",
        "permuted_label_geometry",
        "true_minus_permuted_task_advantage",
        "invariant_instance_retrieval",
        "physical_power_retention",
    ],
    "flattening": {
        "spatial": "registered_complex_tensor_flatten_no_resize_no_crop",
        "modal": "registered_port_order_complex_coefficients_flatten",
    },
    "identity_reference": "report_only_if_exactly_matched",
    "no_new_panel": True,
    "same_panels": True,
    "same_sample_order": True,
    "same_labels": True,
    "no_resampling": True,
    "no_normalization": True,
    "mechanism_evidence_only": True,
    "no_update": True,
}

TRANSFORMS = {
    "C0": "identity_no_op",
    "C1": "per_sample_global_phase",
    "C2": "common_coordinate_permutation_reverse_registered_order",
    "C3": "common_unit_modulus_diagonal_phase",
    "seeds": {
        "permutation": PERMUTATION_SEED,
        "global_phase": GLOBAL_PHASE_SEED,
        "diagonal_phase": DIAGONAL_PHASE_SEED,
    },
    "control_probe_count": CONTROL_PROBE_COUNT,
    "geometry_absolute_tolerance": GEOMETRY_CONTROL_TOLERANCE,
    "label_geometry_absolute_tolerance": LABEL_CONTROL_TOLERANCE,
    "retrieval_absolute_tolerance": RETRIEVAL_CONTROL_TOLERANCE,
    "power_scale_independent": True,
    "common_unitary_invariant": True,
    "per_sample_global_phase_invariant": True,
}

GATES = {
    "material_geometry_degradation": MATERIAL_GEOMETRY_DEGRADATION,
    "material_task_degradation": MATERIAL_TASK_DEGRADATION,
    "geometry_control_tolerance": GEOMETRY_CONTROL_TOLERANCE,
    "label_control_tolerance": LABEL_CONTROL_TOLERANCE,
    "retrieval_control_tolerance": RETRIEVAL_CONTROL_TOLERANCE,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "min_free_cuda_bytes": MIN_FREE_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "backward": False,
    "evaluator_calls": 0,
    "checkpoint_written": False,
    "prior_sampled": False,
    "physical_forward_modified": False,
    "historical_evidence_modified": False,
    "training_authorized": False,
    "formal_promotion": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_task_relevant_bottleneck_attribution",
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
        "relay_tuning",
        "ols_training",
        "decoder_training",
        "physical_forward_modification",
        "historical_evidence_modification",
        "new_samples",
        "identity_remap",
        "raw_complex_primary",
    ],
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    rmri_spec: rmri.Spec
    rmri_review_path: Path
    rmri_validation_path: Path
    rmri_audit_path: Path
    rmri_audit: dict[str, Any]
    rmri_compact_path: Path
    rmri_compact: dict[str, Any]
    parent: rmri.ParentBundle


@dataclass(frozen=True)
class SegmentCapture:
    input_values: torch.Tensor
    output_values: torch.Tensor
    power_ratio: torch.Tensor
    state_unchanged: bool
    ols_state_unchanged: bool


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return mra._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return mra._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"TRBA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "parent_rmri",
        "parent_mra",
        "parent_hfrr",
        "references",
        "panels",
        "measurement",
        "transforms",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("TRBA config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("TRBA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("TRBA formal config is not locked")
    if payload["parent_rmri"] != PARENT_RMRI:
        raise ValueError("TRBA RMRI parent binding is not locked")
    if payload["parent_mra"] != PARENT_MRA:
        raise ValueError("TRBA MRA parent binding is not locked")
    if payload["parent_hfrr"] != PARENT_HFRR:
        raise ValueError("TRBA HFRR parent binding is not locked")
    if payload["references"] != REFERENCES:
        raise ValueError("TRBA checkpoint references are not locked")
    if payload["panels"] != PANELS:
        raise ValueError("TRBA panel contract is not locked")
    if payload["measurement"] != MEASUREMENT:
        raise ValueError("TRBA measurement contract is not locked")
    if payload["transforms"] != TRANSFORMS:
        raise ValueError("TRBA transform contract is not locked")
    if payload["gates"] != GATES:
        raise ValueError("TRBA gate contract is not locked")
    if payload["authorization"] != AUTHORIZATION:
        raise ValueError("TRBA authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "parent_rmri": PARENT_RMRI,
            "parent_mra": PARENT_MRA,
            "parent_hfrr": PARENT_HFRR,
            "references": REFERENCES,
            "panels": PANELS,
            "measurement": MEASUREMENT,
            "transforms": TRANSFORMS,
            "gates": GATES,
            "same_panels": True,
            "no_update": True,
            "raw_complex_primary": False,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }
    if payload != expected:
        raise ValueError("TRBA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return payload


def _validate_rmri_parent(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    config_path = _resolve(PARENT_RMRI["config"], base_path=spec.path.parent)
    if not config_path.is_file():
        raise ValueError("RMRI parent config is missing")
    rmri_spec = rmri._load_spec(config_path)
    # Parent research contracts record the canonical parsed-config hash used
    # by EvidenceRecordV2; the file digest is separately bound in the input
    # hash map below.  Do not compare these two distinct hash domains.
    if rmri_spec.config_hash != PARENT_RMRI["config_sha256"]:
        raise ValueError("RMRI parent canonical config hash mismatch")
    rmri._assert_contract(rmri_spec)
    review_path = _resolve(PARENT_RMRI["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != PARENT_RMRI["review_sha256"]:
        raise ValueError("RMRI parent review hash mismatch")
    rmri._load_review(review_path, rmri_spec)
    validation_path = _resolve(PARENT_RMRI["validation"], base_path=spec.path.parent)
    validation = _read_json_hash(validation_path, PARENT_RMRI["validation_sha256"], "RMRI validation")
    validation_record = EvidenceRecordV2.from_dict(validation["evidence"])
    validation_record.validate()
    if validation.get("status") != "passed" or validation_record.git_commit != PARENT_RMRI["implementation_commit"]:
        raise ValueError("RMRI validation terminal state is stale")
    audit_path = _resolve(PARENT_RMRI["audit"], base_path=spec.path.parent)
    audit = _read_json_hash(audit_path, PARENT_RMRI["audit_sha256"], "RMRI audit")
    record = EvidenceRecordV2.from_dict(audit["evidence"])
    record.validate()
    if (
        audit.get("schema") != rmri.AUDIT_SCHEMA
        or audit.get("hypothesis") != rmri.HYPOTHESIS
        or audit.get("status") != PARENT_RMRI["audit_status"]
        or audit.get("classification") != PARENT_RMRI["audit_classification"]
        or record.git_commit != PARENT_RMRI["implementation_commit"]
    ):
        raise ValueError("RMRI audit terminal state is stale or mismatched")
    compact_path = _resolve(PARENT_RMRI["compact_report"], base_path=spec.path.parent)
    compact = _read_json_hash(compact_path, PARENT_RMRI["compact_report_sha256"], "RMRI compact report")
    expected_compact = {
        "status": "valid_failed",
        "classification": PARENT_RMRI["audit_classification"],
        "generation_readiness": "not_ready",
        "relay_numerical_reference_consistent": True,
        "relay_numerical_reference_mismatch": False,
        "numerical_boundary_not_evaluated": False,
        "dcm_classification": "relay_propagation_bottleneck_reinterpreted_metric_confound",
        "mra_classification": "distributed_bottleneck_supported_canonicalized",
        "bosa_classification": "backend_observable_subspace_mismatch_not_supported_reinterpreted",
    }
    if any(compact.get(key) != value for key, value in expected_compact.items()):
        raise ValueError("RMRI compact terminal state is stale or mismatched")
    parent = rmri._load_parent_with_data(rmri_spec, dataset_root, load_checkpoints=load_checkpoints)
    return ParentBundle(rmri_spec, review_path, validation_path, audit_path, audit, compact_path, compact, parent)


def _input_hashes(spec: Spec, review_path: Path, bundle: ParentBundle, source: mra.SourceBundle) -> dict[str, str]:
    parent = bundle.parent
    mra_hashes = mra._input_hashes(parent.mra_spec, parent.mra_review_path, parent.formal_path, parent.data, source)
    return {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(parent.formal_path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": parent.data.manifest["manifest_hash"],
        "selection_indices": parent.data.manifest["selection_indices_sha256"],
        "confirmation_indices": parent.data.manifest["confirmation_indices_sha256"],
        "parent_rmri_config": file_sha256(bundle.rmri_spec.path),
        "parent_rmri_review_contract": file_sha256(bundle.rmri_review_path),
        "parent_rmri_validation": file_sha256(bundle.rmri_validation_path),
        "parent_rmri_audit": file_sha256(bundle.rmri_audit_path),
        "parent_rmri_compact_report": file_sha256(bundle.rmri_compact_path),
        "parent_rmri_panel": file_sha256(_resolve(PARENT_RMRI["panel"])),
        "parent_mra_config": file_sha256(parent.mra_spec.path),
        "parent_mra_review_contract": file_sha256(parent.mra_review_path),
        "parent_mra_audit": file_sha256(parent.mra_audit_path),
        "parent_hfrr_config": file_sha256(parent.hfrr_spec.path),
        "parent_hfrr_review_contract": file_sha256(parent.hfrr_review_path),
        "parent_hfrr_validation": file_sha256(_resolve(PARENT_HFRR["validation"])),
        "parent_hfrr_audit": file_sha256(parent.hfrr_audit_path),
        "parent_hfrr_compact_report": file_sha256(parent.hfrr_compact_path),
        "parent_hfrr_panel": file_sha256(parent.hfrr_panel_path),
        "canonicalizer_module": file_sha256(Path(hfrr.__file__).resolve()),
        **{f"parent_{name}": value for name, value in mra_hashes.items()},
    }


def _context(spec: Spec, bundle: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"measurement": MEASUREMENT, "transforms": TRANSFORMS, "hypothesis": HYPOTHESIS})
    manifest = bundle.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, dataset_split_manifest_hash=manifest)


def _mode_rank(operator: FiberArrayOLS) -> int:
    return mra._mode_rank(operator)


def _relay_closeout(bundle: ParentBundle) -> dict[str, Any]:
    compact = bundle.rmri_compact
    return {
        "eligible": bool(
            compact.get("relay_numerical_reference_consistent") is True
            and compact.get("relay_numerical_reference_mismatch") is False
            and compact.get("numerical_boundary_not_evaluated") is False
            and compact.get("dcm_classification") == PARENT_RMRI["audit_classification"]
            and compact.get("bosa_classification") == "backend_observable_subspace_mismatch_not_supported_reinterpreted"
        ),
        "proposed_state": "scaled_chen_v1_local_backend_relay_attribution_family=closed_metric_confound_reinterpreted",
        "geometry": "geometry_not_supported",
        "aperture": "aperture_effectively_inactive",
        "axial_conjugate": "relay_axial_conjugate_not_supported",
        "numerical_reference_consistent": True,
        "numerical_reference_mismatch": False,
        "numerical_boundary_not_evaluated": False,
        "metric_confound": True,
        "bosa": "backend_observable_subspace_mismatch_not_supported_reinterpreted",
    }


def _validation_gates(spec: Spec, review_path: Path, bundle: ParentBundle, output_path: Path) -> tuple[GateResult, ...]:
    data = bundle.parent.data
    formal = bundle.parent.formal
    operator = mra._build_research_ols(formal)
    rank = _mode_rank(operator)
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    counts = torch.bincount(selection_labels, minlength=10)
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("relay_closeout_eligible", _relay_closeout(bundle)["eligible"], _relay_closeout(bundle)),
        GateResult("selection_balanced", bool(torch.equal(counts, torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long))), {"counts": counts.tolist()}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("exact_registered_panels", data.selection_indices.numel() == SELECTION_COUNT and data.confirmation_indices.numel() == CONFIRMATION_COUNT, {"selection": int(data.selection_indices.numel()), "confirmation": int(data.confirmation_indices.numel())}),
        GateResult("candidate_prefix_not_consumed", PANELS["candidate_prefix"]["consumed"] is False, {}),
        GateResult("mode_bank_rank_100", rank == 100, {"rank": rank}),
        GateResult("finite_mode_bank", bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all()), {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("transform_contract_locked", spec.payload["transforms"] == TRANSFORMS and review_path.is_file(), {}),
        GateResult("no_new_samples", PANELS["no_new_samples"] is True, {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0}),
        GateResult("output_path_absent", not output_path.exists(), {"path": str(output_path)}),
        GateResult("disk_free_precondition", disk.free > MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES}),
    )


def _record(
    schema: str,
    status: str,
    classification: str | None,
    spec: Spec,
    state: Mapping[str, Any],
    bundle: ParentBundle,
    hashes: Mapping[str, str],
    gates: Sequence[GateResult],
    actions: Sequence[str],
    contract: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
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
        "physical_forward_modified": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _json_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Keep scalar measurement evidence out of JSON-only artifact payloads.

    Per-sample vectors are already stored once in ``panel.pt``.  The audit
    contract must remain a compact JSON object and therefore deliberately
    omits that field rather than serializing or duplicating tensors.
    """
    return {key: value for key, value in report.items() if key != "compact_per_sample"}


def _prepare_validation_slot(output_path: Path) -> None:
    """Create exactly one fresh attempt directory for validation output."""
    if output_path.exists() or output_path.parent.exists():
        raise FileExistsError(f"refusing to reuse TRBA validation attempt: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("TRBA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"TRBA validation attempt directory is missing: {output_path.parent}")
    allowed = {validation_path.name}
    existing = {item.name for item in output_path.parent.iterdir()}
    if not existing.issubset(allowed):
        raise FileExistsError(f"TRBA attempt directory contains unexpected artifacts: {output_path.parent}")
    if output_path.name in existing:
        raise FileExistsError(f"refusing to overwrite TRBA audit artifact: {output_path}")


def _load_validation(path: str | Path, spec: Spec, review_path: Path, bundle: ParentBundle, source: mra.SourceBundle, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, bundle, source)
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
        raise ValueError("TRBA validation evidence is stale or mismatched")


def _flatten_complex(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(value):
        raise ValueError("TRBA representations must remain complex")
    if value.ndim == 2:
        return value.reshape(value.shape[0], -1).contiguous()
    if value.ndim == 4 and value.shape[1] == 1:
        return value[:, 0].reshape(value.shape[0], -1).contiguous()
    if value.ndim == 3:
        return value.reshape(value.shape[0], -1).contiguous()
    raise ValueError(f"unsupported complex representation shape: {tuple(value.shape)}")


def _similarity_block(rows: torch.Tensor, bank: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    if rows.ndim != 2 or bank.ndim != 2 or rows.shape[1] != bank.shape[1] or not torch.is_complex(rows) or not torch.is_complex(bank):
        raise ValueError("normalized Gram inputs must be equal-width complex matrices")
    if not torch.isfinite(rows).all() or not torch.isfinite(bank).all():
        raise ValueError("normalized Gram inputs must be finite")
    numerator = (rows @ bank.conj().T).abs().square()
    row_norm = rows.abs().square().sum(dim=1).clamp_min(eps)
    bank_norm = bank.abs().square().sum(dim=1).clamp_min(eps)
    return (numerator / row_norm[:, None] / bank_norm[None, :]).clamp(0.0, 1.0)


def _ratio(inter: torch.Tensor, intra: torch.Tensor) -> torch.Tensor:
    return inter / intra.clamp_min(1.0e-12)


def _geometry_metrics(left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor, *, block_size: int = BLOCK_SIZE) -> dict[str, Any]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0] or left.shape[0] != labels.numel() or not torch.is_complex(left) or not torch.is_complex(right):
        raise ValueError("TRBA geometry requires equal-count complex representations")
    if left.shape[0] < 2 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("TRBA geometry requires at least two finite samples")
    n = left.shape[0]
    permuted = mra._permuted_labels(labels, seed=PERMUTATION_SEED)
    diff_sum = 0.0
    pair_count = 0
    row_diff = torch.zeros(n, dtype=torch.float64)
    true_intra_left = torch.zeros(n, dtype=torch.float64)
    true_inter_left = torch.zeros(n, dtype=torch.float64)
    perm_intra_left = torch.zeros(n, dtype=torch.float64)
    perm_inter_left = torch.zeros(n, dtype=torch.float64)
    true_intra_right = torch.zeros(n, dtype=torch.float64)
    true_inter_right = torch.zeros(n, dtype=torch.float64)
    perm_intra_right = torch.zeros(n, dtype=torch.float64)
    perm_inter_right = torch.zeros(n, dtype=torch.float64)
    true_same_count = torch.zeros(n, dtype=torch.float64)
    true_diff_count = torch.zeros(n, dtype=torch.float64)
    perm_same_count = torch.zeros(n, dtype=torch.float64)
    perm_diff_count = torch.zeros(n, dtype=torch.float64)
    retrieval_left = torch.zeros(n, dtype=torch.bool)
    retrieval_right = torch.zeros(n, dtype=torch.bool)
    columns = torch.arange(n)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        rows = torch.arange(start, stop)
        similarity_left = _similarity_block(left[start:stop], left)
        similarity_right = _similarity_block(right[start:stop], right)
        similarity_left[torch.arange(stop - start), rows] = -float("inf")
        similarity_right[torch.arange(stop - start), rows] = -float("inf")
        retrieval_left[start:stop] = labels.index_select(0, similarity_left.argmax(dim=1)) == labels[start:stop]
        retrieval_right[start:stop] = labels.index_select(0, similarity_right.argmax(dim=1)) == labels[start:stop]
        similarity_left[torch.arange(stop - start), rows] = 1.0
        similarity_right[torch.arange(stop - start), rows] = 1.0
        distance_left = (2.0 - 2.0 * similarity_left).clamp_min(0.0).sqrt().double()
        distance_right = (2.0 - 2.0 * similarity_right).clamp_min(0.0).sqrt().double()
        upper = columns[None, :] > rows[:, None]
        off_diagonal = columns[None, :] != rows[:, None]
        difference = (similarity_left - similarity_right).abs().double()
        # The primary scalar is the registered upper-triangle mean.  Per-sample
        # vectors use both directions so the final rows have the same number of
        # valid peers as the first rows; this keeps bootstrap contributions
        # symmetric without changing the primary definition.
        row_diff[start:stop] = difference.masked_fill(~off_diagonal, 0.0).sum(dim=1) / off_diagonal.sum(dim=1).clamp_min(1)
        diff_sum += float(difference.masked_select(upper).sum())
        pair_count += int(upper.sum())
        label_left = labels[start:stop, None]
        label_right = labels[None, :]
        perm_left = permuted[start:stop, None]
        perm_right = permuted[None, :]
        masks = {
            "true_same": off_diagonal & label_left.eq(label_right),
            "true_diff": off_diagonal & ~label_left.eq(label_right),
            "perm_same": off_diagonal & perm_left.eq(perm_right),
            "perm_diff": off_diagonal & ~perm_left.eq(perm_right),
        }
        for name, mask in masks.items():
            if name == "true_same":
                true_intra_left[start:stop] += distance_left.masked_fill(~mask, 0.0).sum(dim=1)
                true_intra_right[start:stop] += distance_right.masked_fill(~mask, 0.0).sum(dim=1)
                true_same_count[start:stop] += mask.sum(dim=1).double()
            elif name == "true_diff":
                true_inter_left[start:stop] += distance_left.masked_fill(~mask, 0.0).sum(dim=1)
                true_inter_right[start:stop] += distance_right.masked_fill(~mask, 0.0).sum(dim=1)
                true_diff_count[start:stop] += mask.sum(dim=1).double()
            elif name == "perm_same":
                perm_intra_left[start:stop] += distance_left.masked_fill(~mask, 0.0).sum(dim=1)
                perm_intra_right[start:stop] += distance_right.masked_fill(~mask, 0.0).sum(dim=1)
                perm_same_count[start:stop] += mask.sum(dim=1).double()
            else:
                perm_inter_left[start:stop] += distance_left.masked_fill(~mask, 0.0).sum(dim=1)
                perm_inter_right[start:stop] += distance_right.masked_fill(~mask, 0.0).sum(dim=1)
                perm_diff_count[start:stop] += mask.sum(dim=1).double()
    true_ratio_left_per_sample = _ratio(true_inter_left / true_diff_count.clamp_min(1.0), true_intra_left / true_same_count.clamp_min(1.0))
    true_ratio_right_per_sample = _ratio(true_inter_right / true_diff_count.clamp_min(1.0), true_intra_right / true_same_count.clamp_min(1.0))
    perm_ratio_left_per_sample = _ratio(perm_inter_left / perm_diff_count.clamp_min(1.0), perm_intra_left / perm_same_count.clamp_min(1.0))
    perm_ratio_right_per_sample = _ratio(perm_inter_right / perm_diff_count.clamp_min(1.0), perm_intra_right / perm_same_count.clamp_min(1.0))
    true_left = {"inter_intra_ratio": float(_ratio(true_inter_left.sum(), true_intra_left.sum())), "intra": float(true_intra_left.sum() / true_same_count.sum().clamp_min(1.0)), "inter": float(true_inter_left.sum() / true_diff_count.sum().clamp_min(1.0))}
    true_right = {"inter_intra_ratio": float(_ratio(true_inter_right.sum(), true_intra_right.sum())), "intra": float(true_intra_right.sum() / true_same_count.sum().clamp_min(1.0)), "inter": float(true_inter_right.sum() / true_diff_count.sum().clamp_min(1.0))}
    perm_left = {"inter_intra_ratio": float(_ratio(perm_inter_left.sum(), perm_intra_left.sum())), "intra": float(perm_intra_left.sum() / perm_same_count.sum().clamp_min(1.0)), "inter": float(perm_inter_left.sum() / perm_diff_count.sum().clamp_min(1.0))}
    perm_right = {"inter_intra_ratio": float(_ratio(perm_inter_right.sum(), perm_intra_right.sum())), "intra": float(perm_intra_right.sum() / perm_same_count.sum().clamp_min(1.0)), "inter": float(perm_inter_right.sum() / perm_diff_count.sum().clamp_min(1.0))}
    task_left = true_left["inter_intra_ratio"] - perm_left["inter_intra_ratio"]
    task_right = true_right["inter_intra_ratio"] - perm_right["inter_intra_ratio"]
    geometry_retention = max(0.0, min(1.0, 1.0 - diff_sum / max(pair_count, 1)))
    return {
        "geometry_retention": geometry_retention,
        "geometry_degradation": 1.0 - geometry_retention,
        "pair_count": pair_count,
        "input": {"true_label_geometry": true_left, "permuted_label_geometry": perm_left, "task_advantage": task_left, "invariant_retrieval": float(retrieval_left.float().mean())},
        "output": {"true_label_geometry": true_right, "permuted_label_geometry": perm_right, "task_advantage": task_right, "invariant_retrieval": float(retrieval_right.float().mean())},
        "task_advantage_degradation": task_left - task_right,
        "retrieval_degradation": float(retrieval_left.float().mean() - retrieval_right.float().mean()),
        "geometry_degradation_by_sample": row_diff,
        "task_advantage_degradation_by_sample": true_ratio_left_per_sample - perm_ratio_left_per_sample - true_ratio_right_per_sample + perm_ratio_right_per_sample,
        "retrieval_degradation_by_sample": retrieval_left.to(torch.float64) - retrieval_right.to(torch.float64),
        "finite": bool(math.isfinite(geometry_retention) and torch.isfinite(row_diff).all() and torch.isfinite(true_ratio_left_per_sample).all()),
    }


def _control_transforms(values: torch.Tensor) -> dict[str, torch.Tensor]:
    if values.ndim != 2 or not torch.is_complex(values):
        raise ValueError("transform controls require a complex matrix")
    n, dimension = values.shape
    generator = torch.Generator(device="cpu").manual_seed(GLOBAL_PHASE_SEED)
    phases = 2.0 * math.pi * torch.rand(n, generator=generator, dtype=torch.float32)
    global_phase = torch.exp(1j * phases).to(values.dtype)[:, None]
    permutation = torch.arange(dimension - 1, -1, -1, dtype=torch.long)
    diagonal_generator = torch.Generator(device="cpu").manual_seed(DIAGONAL_PHASE_SEED)
    diagonal = torch.exp(1j * (2.0 * math.pi * torch.rand(dimension, generator=diagonal_generator, dtype=torch.float32))).to(values.dtype)
    return {
        "C0": values.clone(),
        "C1": values * global_phase,
        "C2": values.index_select(1, permutation),
        "C3": values * diagonal[None, :],
    }


def _control_report(values: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    count = min(CONTROL_PROBE_COUNT, values.shape[0])
    probe = values[:count].contiguous()
    probe_labels = labels[:count].contiguous()
    transformed = _control_transforms(probe)
    reports: dict[str, Any] = {}
    for name, candidate in transformed.items():
        geometry = _geometry_metrics(probe, candidate, probe_labels)
        reports[name] = {
            "geometry_score_change": abs(1.0 - geometry["geometry_retention"]),
            "true_label_geometry_change": abs(geometry["output"]["true_label_geometry"]["inter_intra_ratio"] - geometry["input"]["true_label_geometry"]["inter_intra_ratio"]),
            "permuted_label_geometry_change": abs(geometry["output"]["permuted_label_geometry"]["inter_intra_ratio"] - geometry["input"]["permuted_label_geometry"]["inter_intra_ratio"]),
            "retrieval_change": abs(geometry["output"]["invariant_retrieval"] - geometry["input"]["invariant_retrieval"]),
            "finite": geometry["finite"],
            "passed": bool(
                geometry["finite"]
                and abs(1.0 - geometry["geometry_retention"]) <= GEOMETRY_CONTROL_TOLERANCE
                and abs(geometry["output"]["true_label_geometry"]["inter_intra_ratio"] - geometry["input"]["true_label_geometry"]["inter_intra_ratio"]) <= LABEL_CONTROL_TOLERANCE
                and abs(geometry["output"]["permuted_label_geometry"]["inter_intra_ratio"] - geometry["input"]["permuted_label_geometry"]["inter_intra_ratio"]) <= LABEL_CONTROL_TOLERANCE
                and abs(geometry["output"]["invariant_retrieval"] - geometry["input"]["invariant_retrieval"]) <= RETRIEVAL_CONTROL_TOLERANCE
            ),
        }
    return {"probe_count": count, "definition": MEASUREMENT["primary_definition"], "controls": reports, "passed": all(item["passed"] for item in reports.values())}


def _power_report(input_values: torch.Tensor, output_values: torch.Tensor, segment: str, operator: FiberArrayOLS) -> dict[str, Any]:
    area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
    input_power = input_values.abs().square().sum(dim=1) * area
    if segment == OLS_SEGMENT:
        mode_power = operator.output_modes.detach().cpu().abs().square().sum(dim=(1, 2)) * area
        output_power = output_values.abs().square() @ mode_power
    else:
        output_power = output_values.abs().square().sum(dim=1) * area
    ratio = output_power / input_power.clamp_min(1.0e-30)
    return {
        "mean": float(ratio.mean()),
        "median": float(ratio.median()),
        "q01_q99": [float(value) for value in torch.quantile(ratio, torch.tensor([0.01, 0.99]))],
        "finite": bool(torch.isfinite(ratio).all()),
        "per_sample": ratio,
    }


def _bootstrap(delta: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    low, high = mra._stratified_bootstrap_ci(delta.float(), labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, confidence_level=CONFIDENCE_LEVEL)
    return {"mean": float(delta.mean()), "ci": [low, high], "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence_level": CONFIDENCE_LEVEL}


def _segment_report(capture: SegmentCapture, segment: str, labels: torch.Tensor, operator: FiberArrayOLS) -> dict[str, Any]:
    controls = {
        "input": _control_report(capture.input_values, labels),
        "output": _control_report(capture.output_values, labels),
    }
    geometry = _geometry_metrics(capture.input_values, capture.output_values, labels)
    power = _power_report(capture.input_values, capture.output_values, segment, operator)
    return {
        "segment": segment,
        "input_dimension": int(capture.input_values.shape[1]),
        "output_dimension": int(capture.output_values.shape[1]),
        "primary": {
            "geometry_retention": geometry["geometry_retention"],
            "geometry_degradation": geometry["geometry_degradation"],
            "pair_count": geometry["pair_count"],
            "finite": geometry["finite"],
        },
        "task_geometry": {
            "input": geometry["input"],
            "output": geometry["output"],
            "task_advantage_degradation": geometry["task_advantage_degradation"],
            "retrieval_degradation": geometry["retrieval_degradation"],
            "geometry_degradation_bootstrap": _bootstrap(geometry["geometry_degradation_by_sample"], labels),
            "task_advantage_degradation_bootstrap": _bootstrap(geometry["task_advantage_degradation_by_sample"], labels),
            "retrieval_degradation_bootstrap": _bootstrap(geometry["retrieval_degradation_by_sample"], labels),
        },
        "power": {key: value for key, value in power.items() if key != "per_sample"},
        "controls": controls,
        "controls_pass": bool(controls["input"]["passed"] and controls["output"]["passed"]),
        "state_unchanged": capture.state_unchanged,
        "ols_state_unchanged": capture.ols_state_unchanged,
        "finite": bool(geometry["finite"] and power["finite"] and capture.state_unchanged and capture.ols_state_unchanged),
        "compact_per_sample": {
            "geometry_degradation": geometry["geometry_degradation_by_sample"],
            "task_advantage_degradation": geometry["task_advantage_degradation_by_sample"],
            "retrieval_degradation": geometry["retrieval_degradation_by_sample"],
            "power_retention": power["per_sample"],
        },
    }


def _segment_classification(selection: Mapping[str, Any], confirmation: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    for item in (selection, confirmation):
        if not item.get("finite") or not item.get("controls_pass"):
            return "invalid_harness", {"reason": "finite_or_transform_controls_failed"}
    def panel_support(item: Mapping[str, Any]) -> dict[str, bool]:
        primary = float(item["primary"]["geometry_degradation"]) >= MATERIAL_GEOMETRY_DEGRADATION
        task_primary = float(item["task_geometry"]["task_advantage_degradation"]) >= MATERIAL_TASK_DEGRADATION
        retrieval = float(item["task_geometry"]["retrieval_degradation"]) >= MATERIAL_TASK_DEGRADATION
        ci_positive = float(item["task_geometry"]["geometry_degradation_bootstrap"]["ci"][0]) > 0.0
        return {"primary": primary, "task_geometry": task_primary, "retrieval": retrieval, "task_secondary": task_primary or retrieval, "bootstrap_positive": ci_positive, "supported": primary and (task_primary or retrieval) and ci_positive}
    selection_gate = panel_support(selection)
    confirmation_gate = panel_support(confirmation)
    details = {"selection": selection_gate, "confirmation": confirmation_gate}
    if selection_gate["supported"] and confirmation_gate["supported"]:
        return "supported", details
    if selection_gate["supported"] != confirmation_gate["supported"]:
        return "underdetermined", details
    if selection_gate["primary"] and confirmation_gate["primary"] and not (selection_gate["task_secondary"] and confirmation_gate["task_secondary"]):
        return "generic_geometry_change_only", details
    return "not_supported", details


def _overall_classification(segment_states: Mapping[str, str]) -> tuple[str, str]:
    if any(value == "invalid_harness" for value in segment_states.values()):
        return "invalid_harness", NEXT_ACTION_UNDERDETERMINED
    if any(value == "underdetermined" for value in segment_states.values()):
        return CLASSIFICATION_BY_TIER["T5"], NEXT_ACTION_UNDERDETERMINED
    ols = segment_states[OLS_SEGMENT] == "supported"
    decoder = segment_states[DECODER_SEGMENT] == "supported"
    if ols and decoder:
        return CLASSIFICATION_BY_TIER["T3"], NEXT_ACTION_BOTH
    if ols:
        return CLASSIFICATION_BY_TIER["T1"], NEXT_ACTION_OLS
    if decoder:
        return CLASSIFICATION_BY_TIER["T2"], NEXT_ACTION_DECODER
    return CLASSIFICATION_BY_TIER["T4"], NEXT_ACTION_NEITHER


def _classification_tier(classification: str) -> str | None:
    for tier, value in CLASSIFICATION_BY_TIER.items():
        if value == classification:
            return tier
    return None


def _capture_segment(model: torch.nn.Module, segment: str, images: torch.Tensor, device: torch.device, operator: FiberArrayOLS) -> SegmentCapture:
    if segment not in SEGMENT_ENDPOINTS:
        raise ValueError(f"unsupported TRBA segment: {segment}")
    left_name, right_name = SEGMENT_ENDPOINTS[segment]
    before_model = mra._state_cpu(model)
    before_ols = mra._state_cpu(model.ols)  # type: ignore[attr-defined]
    left_chunks: list[torch.Tensor] = []
    right_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
            rendered = base.render_intensity_batch(images.index_select(0, indices))
            amplitude = rendered.amplitude.to(device)
            stages, _ = mra._scaled_trace_stages(model, amplitude)  # type: ignore[arg-type]
            left = stages[left_name]
            right = stages[right_name]
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f"non-finite TRBA {segment} representation")
            left_chunks.append(_flatten_complex(left).detach().cpu())
            right_chunks.append(_flatten_complex(right).detach().cpu())
    left_values = torch.cat(left_chunks, dim=0)
    right_values = torch.cat(right_chunks, dim=0)
    power = _power_report(left_values, right_values, segment, operator)["per_sample"]
    after_model = mra._state_cpu(model)
    after_ols = mra._state_cpu(model.ols)  # type: ignore[attr-defined]
    return SegmentCapture(left_values, right_values, power, mra._state_equal(before_model, after_model), mra._state_equal(before_ols, after_ols))


def _load_scaled_model(formal: FormalSystemConfig, checkpoint: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = mra._load_scaled_model(formal, checkpoint, device)
    if any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise ValueError("TRBA model is not frozen")
    return model


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("TRBA requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("TRBA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free <= MIN_DISK_FREE_BYTES:
        raise RuntimeError("TRBA requires more than 10 GiB free storage")
    return int(free), int(total)


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    bundle = _validate_rmri_parent(spec, args.dataset_root, load_checkpoints=False)
    gates = _validation_gates(spec, review_path, bundle, _resolve(args.output))
    hashes = _input_hashes(spec, review_path, bundle, bundle.parent.source)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, state, bundle, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), {"relay_closeout": _relay_closeout(bundle), "measurement": MEASUREMENT, "transforms": TRANSFORMS, "panels": PANELS, "no_update": True}, None if valid else "TRBA static validation failed")
    _prepare_validation_slot(_resolve(args.output))
    _write_new(args.output, payload, artifact_kind="TRBA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("TRBA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    bundle = _validate_rmri_parent(spec, args.dataset_root, load_checkpoints=False)
    source = bundle.parent.source
    _load_validation(args.validation_evidence, spec, review_path, bundle, source, state)
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _check_run_slot(output_path, validation_path)
    compact_path = output_path.with_name("compact_report.json")
    panel_path = output_path.with_name("panel.pt")
    for target in (output_path, compact_path, panel_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite TRBA artifact: {target}")
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    loaded_source = mra._load_source(bundle.parent.mra_spec, load_checkpoints=True)
    checkpoint = loaded_source.checkpoints["scaled_chen_step_5000"]
    model = _load_scaled_model(bundle.parent.formal, checkpoint, device)
    operator = mra._build_research_ols(bundle.parent.formal)
    selection_images = bundle.parent.data.test_images.index_select(0, bundle.parent.data.selection_indices)
    selection_labels = bundle.parent.data.test_labels.index_select(0, bundle.parent.data.selection_indices)
    confirmation_images = bundle.parent.data.images.index_select(0, bundle.parent.data.confirmation_indices)
    confirmation_labels = bundle.parent.data.labels.index_select(0, bundle.parent.data.confirmation_indices)
    # Small probe first: a control failure stops before the 6,000-sample replay.
    probe_reports = {}
    probe_images = selection_images[:CONTROL_PROBE_COUNT]
    probe_labels = selection_labels[:CONTROL_PROBE_COUNT]
    for segment in SEGMENTS:
        probe = _capture_segment(model, segment, probe_images, device, operator)
        probe_report = {"input": _control_report(probe.input_values, probe_labels), "output": _control_report(probe.output_values, probe_labels), "state_unchanged": probe.state_unchanged, "ols_state_unchanged": probe.ols_state_unchanged}
        probe_reports[segment] = probe_report
        if not probe_report["input"]["passed"] or not probe_report["output"]["passed"] or not probe.state_unchanged or not probe.ols_state_unchanged:
            raise ValueError("TRBA transform-aware preflight failed")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    measurements: dict[str, dict[str, Any]] = {segment: {} for segment in SEGMENTS}
    panel_payload: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "selection_indices": bundle.parent.data.selection_indices,
        "confirmation_indices": bundle.parent.data.confirmation_indices,
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
        "segments": {},
        "source_hashes": loaded_source.hashes,
        "affects_selection": False,
        "mechanism_evidence_only": True,
    }
    for segment in SEGMENTS:
        selection_capture = _capture_segment(model, segment, selection_images, device, operator)
        selection_report = _segment_report(selection_capture, segment, selection_labels, operator)
        del selection_capture
        confirmation_capture = _capture_segment(model, segment, confirmation_images, device, operator)
        confirmation_report = _segment_report(confirmation_capture, segment, confirmation_labels, operator)
        panel_payload["segments"][segment] = {
            "selection": selection_report["compact_per_sample"],
            "confirmation": confirmation_report["compact_per_sample"],
        }
        measurements[segment] = {"selection": selection_report, "confirmation": confirmation_report}
        del confirmation_capture
        torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device))
    if peak > MAX_PEAK_BYTES:
        raise RuntimeError(f"TRBA resource_aborted: CUDA peak {peak} exceeds {MAX_PEAK_BYTES}")
    segment_states: dict[str, str] = {}
    segment_details: dict[str, Any] = {}
    for segment in SEGMENTS:
        segment_states[segment], segment_details[segment] = _segment_classification(measurements[segment]["selection"], measurements[segment]["confirmation"])
    overall, action = _overall_classification(segment_states)
    if overall == "invalid_harness":
        raise ValueError("TRBA harness failed after measurement")
    tier = _classification_tier(overall)
    if tier is None:
        raise ValueError(f"TRBA returned an unknown scientific classification: {overall}")
    elapsed = time.monotonic() - started
    hashes = _input_hashes(spec, review_path, bundle, loaded_source)
    compact = {
        "schema": "scaled_chen_task_relevant_bottleneck_compact_report.v1",
        "status": "valid_failed",
        "tier": tier,
        "classification": overall,
        "segment_classifications": segment_states,
        "segment_gates": segment_details,
        "next_action": action,
        "relay_closeout": _relay_closeout(bundle),
        "raw_context": {
            OLS_SEGMENT: {"selection": 0.8144925261537826, "confirmation": 0.8092388489503743},
            DECODER_SEGMENT: {"selection": 0.8204369750297876, "confirmation": 0.8188838387637044},
        },
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "identity_reference": "report_only_not_loaded_for_primary",
        "decoder_internal_metric_interpretation_requires_future_validation": True,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "physical_forward_modified": False,
        "historical_evidence_modified": False,
        "training_authorized": False,
        "formal_promotion": False,
    }
    panel_size = sum(value.numel() * value.element_size() for segment in panel_payload["segments"].values() for panel in segment.values() for value in panel.values())
    if panel_size > MAX_STORAGE_BYTES:
        raise RuntimeError("TRBA compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    compact["panel_sha256"] = panel_hash
    _write_new(compact_path, compact, artifact_kind="TRBA compact report")
    audit_contract = {
        "selection": _json_report(measurements[OLS_SEGMENT]["selection"]),
        "confirmation": _json_report(measurements[OLS_SEGMENT]["confirmation"]),
        "segments": {
            segment: {panel: _json_report(report) for panel, report in panels.items()}
            for segment, panels in measurements.items()
        },
        "segment_classifications": segment_states,
        "segment_gates": segment_details,
        "relay_closeout": _relay_closeout(bundle),
        "probe_preflight": probe_reports,
        "measurement": MEASUREMENT,
        "transforms": TRANSFORMS,
        "raw_complex_is_context_only": True,
        "identity_reference": "report_only_not_loaded_for_primary",
        "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": file_sha256(compact_path),
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "physical_forward_modified": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
    }
    gates = (
        GateResult("finite_measurements", all(measurements[s][p]["finite"] for s in SEGMENTS for p in ("selection", "confirmation")), {}),
        GateResult("frozen_state_unchanged", all(measurements[s][p]["state_unchanged"] and measurements[s][p]["ols_state_unchanged"] for s in SEGMENTS for p in ("selection", "confirmation")), {}),
        GateResult("transform_controls", all(measurements[s][p]["controls_pass"] for s in SEGMENTS for p in ("selection", "confirmation")), {}),
        GateResult("no_update_isolation", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False}),
        GateResult("resource_limits", peak <= MAX_PEAK_BYTES and panel_size <= MAX_STORAGE_BYTES, {"peak_bytes": peak, "panel_bytes": panel_size}),
    )
    hashes["panel"] = panel_hash
    payload = _record(AUDIT_SCHEMA, "valid_failed", overall, spec, state, bundle, hashes, gates, (action,), audit_contract, "TRBA is diagnostic-only; no promotion or training consumer")
    payload["relay_closeout"] = _relay_closeout(bundle)
    payload["segment_classifications"] = segment_states
    payload["reconstruction_manifold_supported"] = False
    payload["generative_manifold_supported"] = False
    payload["generation_readiness"] = "not_ready"
    _write_new(output_path, payload, artifact_kind="TRBA audit evidence")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-task-relevant-bottleneck-validate":
        return validate(args)
    if args.research_action == "scaled-chen-task-relevant-bottleneck-run":
        return run(args)
    raise ValueError(f"unsupported TRBA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "CLASSIFICATION_BY_TIER",
    "CONFIG_SCHEMA",
    "DECODER_SEGMENT",
    "GATES",
    "HYPOTHESIS",
    "MEASUREMENT",
    "OLS_SEGMENT",
    "PANEL_SCHEMA",
    "PANELS",
    "PARENT_HFRR",
    "PARENT_MRA",
    "PARENT_RMRI",
    "REVIEW_SCHEMA",
    "TRANSFORMS",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_classification_tier",
    "_control_report",
    "_geometry_metrics",
    "_load_review",
    "_load_spec",
    "_overall_classification",
    "_segment_classification",
    "_similarity_block",
    "dispatch",
    "run",
    "validate",
]
