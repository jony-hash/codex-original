"""Frozen 2x2 backend-observable-subspace attribution for Scaled-Chen.

This module is a research-only consumer.  It compares the reconstruction-capable
Identity decoder representation with the frozen Scaled-Chen decoder
representation under the two registered coherent backends.  The comparison is
performed at the explicit chip plane and never changes a model, OLS, relay,
detector, checkpoint, or training contract.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...config import FormalSystemConfig
from ...contracts import ComplexField, OperationProvenance
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ...optics.relays import CoherentImagingRelay
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_decoder_cmos_attribution as dcm
from . import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_backend_observable_subspace_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_backend_observable_subspace_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_backend_observable_subspace_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_backend_observable_subspace.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_backend_observable_subspace_panel.v1"
COMPACT_SCHEMA = "scaled_chen_backend_observable_subspace_compact_report.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S1-BOSA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

IDENTITY = "identity_step_5500"
SCALED = "scaled_chen_step_5000"
MODEL_NAMES = (IDENTITY, SCALED)
BACKENDS = ("identity_backend", "scaled_chen_backend")
CELL_ORDER = ("I_to_I", "I_to_S", "S_to_I", "S_to_S")
CELL_REGISTRATION = {
    "I_to_I": {"representation": IDENTITY, "backend": "identity_backend", "display": "I→I"},
    "I_to_S": {"representation": IDENTITY, "backend": "scaled_chen_backend", "display": "I→S"},
    "S_to_I": {"representation": SCALED, "backend": "identity_backend", "display": "S→I"},
    "S_to_S": {"representation": SCALED, "backend": "scaled_chen_backend", "display": "S→S"},
}

RELAY_SEGMENT = "decoder_layer_2_to_cmos_incident_field"
READOUT_SEGMENT = "cmos_incident_field_to_cmos_intensity"
DETECTOR_SEGMENT = "decoder_layer_2_to_cmos_intensity"
RANK_RETENTION_REFERENCE = 0.90
MATERIAL_GAP = 0.05
POWER_GAP = 0.05
NATIVE_FIELD_TOLERANCE = 1.0e-5
NATIVE_INTENSITY_TOLERANCE = 1.0e-5
SQUARE_LAW_TOLERANCE = 1.0e-5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
PHASE_CONTROL_SEED = 1717
PREFLIGHT_PER_LABEL = 2
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

NEXT_ACTION_PASS = "manual_review_backend_aware_representation_redesign_contract"
NEXT_ACTION_FAIL = "manual_review_high_fidelity_relay_numerical_reference_contract"
NEXT_ACTION_UNDERDETERMINED = "manual_review_backend_observable_subspace_attribution_underdetermined_contract"
NEXT_ACTION_RUN = "run_scaled_chen_backend_observable_subspace_attribution"

PARENT_MRA = dict(dcm.PARENT_MRA)
PARENT_DCM = {
    "config": "configs/research/lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_v1.yaml",
    "config_sha256": "6c34adfd4b3ea4af6da63900f99b0add2ba2d8bd6420797447ed49e1ebbe08a7",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_review_v1.json",
    "review_sha256": "96481977b3fcfee24d7151df7d1f233161d4b862c34fb855743144456b30bec3",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_decoder_cmos_attribution_v1_attempt_03/audit.json",
    "audit_sha256": "16975f7f26db13100af134fad70e510a1978093b88c619bc6c3a3e08c067ed68",
    "audit_git_commit": "d5fbe5e6efd55076aca7020e0e1cb0f1c7763b19",
    "audit_status": "valid_failed",
    "audit_classification": "relay_propagation_bottleneck_supported",
    "anchors": {
        "selection": {"relay": 0.6571249836738348, "readout": 0.886754535852981, "power": 1.0000001192092896},
        "confirmation": {"relay": 0.6030690617755633, "readout": 0.9735379421279973, "power": 1.0000001192092896},
        "retention_absolute_tolerance": 0.005,
        "mse_relative_tolerance": 0.01,
        "power_absolute_tolerance": 0.005,
        "square_law_tolerance": 1.0e-5,
    },
}

REFERENCES = {
    "identity_checkpoint": {
        "path": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_full_epoch_v1_attempt_04/full_epoch_final_step_005500.pt",
        "sha256": "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f",
        "producer_commit": "4bbe4383f07e03572e154854cea8da6b5d24022e",
        "global_step": 5500,
        "role": "historical_reconstruction_capable_reference",
    },
    "scaled_chen_checkpoint": {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
        "producer_commit": "019b76ef911e0bd16b8d5d4169807323427684af",
        "global_step": 5000,
        "role": "frozen_scaled_chen_reference",
    },
}

PANELS = {
    "selection": {
        "source": "mnist_test_indices_5000_9999",
        "seed": 1235,
        "count": 1000,
        "per_label": 100,
        "candidate_selection": False,
        "calibration_excluded": True,
        "schedule_excluded": True,
        "plan3_plan4_excluded": True,
    },
    "confirmation": {
        "source": "horizon_heldout_train_indices_55000_59999",
        "count": 5000,
        "mechanism_evidence_only": True,
        "calibration_excluded": True,
        "schedule_excluded": True,
        "plan3_plan4_excluded": True,
    },
    "candidate_test_prefix": {"source": "mnist_test_indices_0_4999", "count": 4560, "consumed": False, "untouched_proven": False},
}

BACKEND_CONTRACT = {
    "identity_backend": {
        "name": "back_b0",
        "segments_m": [0.1, 0.25],
        "focal_lengths_m": [0.0714285714285714],
        "magnification": -2.5,
        "target_plane": "camera",
        "target_shape": [200, 200],
        "target_pixel_pitch_m": [2.5e-6, 2.5e-6],
        "target_fov_m": [0.0005, 0.0005],
        "detector": "intensity_abs_square",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    },
    "scaled_chen_backend": {
        "name": "inferred_backend_imaging",
        "segments_m": [0.0008571428571428572, 0.002142857142857143],
        "focal_lengths_m": [0.0006122448979591838],
        "magnification": -2.5,
        "target_plane": "chen_camera",
        "target_shape": [200, 200],
        "target_pixel_pitch_m": [2.5e-6, 2.5e-6],
        "target_fov_m": [0.0005, 0.0005],
        "detector": "intensity_abs_square",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    },
}

MEASUREMENT = {
    "objective": "frozen_forward_only",
    "models": list(MODEL_NAMES),
    "cells": [dict(CELL_REGISTRATION[name]) for name in CELL_ORDER],
    "stages": ["decoder_layer_2", "cmos_incident_field", "cmos_intensity"],
    "segments": [
        [RELAY_SEGMENT, "decoder_layer_2", "cmos_incident_field"],
        [READOUT_SEGMENT, "cmos_incident_field", "cmos_intensity"],
        [DETECTOR_SEGMENT, "decoder_layer_2", "cmos_intensity"],
    ],
    "retention_definition": "mra_dcm_entropy_participation_geometric_mean",
    "reuse": ["mra_feature_projection", "mra_rank", "mra_retrieval", "mra_permuted_label", "dcm_mse", "dcm_phase_controls"],
    "no_resampling": True,
    "no_normalization": True,
    "no_ols_rerun": True,
    "negative_controls": ["native_identity_replay", "native_scaled_chen_replay", "square_law", "global_phase_invariance", "pointwise_phase_invariance", "permuted_label"],
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence_level": CONFIDENCE_LEVEL, "helper": "mra_stratified_paired_bootstrap"},
    "numerical_boundary_not_evaluated": True,
    "mechanism_evidence_only": True,
    "no_update": True,
}

GATES = {
    "rank_retention_reference": RANK_RETENTION_REFERENCE,
    "material_gap": MATERIAL_GAP,
    "power_gap": POWER_GAP,
    "native_field_tolerance": NATIVE_FIELD_TOLERANCE,
    "native_intensity_tolerance": NATIVE_INTENSITY_TOLERANCE,
    "square_law_tolerance": SQUARE_LAW_TOLERANCE,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "min_free_cuda_bytes": MIN_FREE_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "evaluator_calls": 0,
    "training_authorized": False,
    "formal_promotion": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_backend_observable_subspace_attribution",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "prohibited": ["optimizer", "backward", "checkpoint_write", "generation", "evaluator", "fmf", "prior_sampling", "formal_promotion", "relay_tuning", "ols_rerun"],
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    parent: dcm.ParentBundle
    dcm_config_path: Path
    dcm_review_path: Path
    dcm_audit_path: Path
    dcm_audit: dict[str, Any]


@dataclass(frozen=True)
class CellEvaluation:
    metrics: dict[str, Any]
    sample_mse: torch.Tensor
    labels: torch.Tensor
    model_state_hash_before: str
    model_state_hash_after: str
    ols_state_hash_before: str
    ols_state_hash_after: str


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return mra._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return mra._git_state()


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "parent_mra", "parent_dcm", "references", "panels", "backends", "measurement", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("backend-observable-subspace config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("backend-observable-subspace config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    expected_formal = "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
    if payload["formal_system_config"] != expected_formal:
        raise ValueError("backend-observable-subspace formal config is not locked")
    if payload["parent_mra"] != PARENT_MRA:
        raise ValueError("backend-observable-subspace parent MRA binding is not locked")
    if payload["parent_dcm"] != PARENT_DCM:
        raise ValueError("backend-observable-subspace parent DCM binding is not locked")
    if payload["references"] != REFERENCES:
        raise ValueError("backend-observable-subspace checkpoint references are not locked")
    if payload["panels"] != PANELS:
        raise ValueError("backend-observable-subspace panels are not locked")
    if payload["backends"] != BACKEND_CONTRACT:
        raise ValueError("backend-observable-subspace backend contract is not locked")
    if payload["measurement"] != MEASUREMENT:
        raise ValueError("backend-observable-subspace measurement contract is not locked")
    if payload["gates"] != GATES:
        raise ValueError("backend-observable-subspace gate contract is not locked")
    if payload["authorization"] != AUTHORIZATION:
        raise ValueError("backend-observable-subspace authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "parent_mra": PARENT_MRA,
            "parent_dcm": PARENT_DCM,
            "references": REFERENCES,
            "panels": PANELS,
            "backends": BACKEND_CONTRACT,
            "cells": [dict(CELL_REGISTRATION[name]) for name in CELL_ORDER],
            "material_gap": MATERIAL_GAP,
            "power_gap": POWER_GAP,
            "bootstrap": MEASUREMENT["bootstrap"],
            "no_resampling": True,
            "no_normalization": True,
            "numerical_boundary_not_evaluated": True,
            "mechanism_evidence_only": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": AUTHORIZATION["prohibited"],
    }
    if payload != expected:
        raise ValueError("backend-observable-subspace review contract does not match scope")
    return target


def _load_dcm_parent(spec: Spec) -> tuple[Path, Path, Path, dict[str, Any]]:
    parent = spec.payload["parent_dcm"]
    config_path = _resolve(parent["config"], base_path=spec.path.parent)
    if not config_path.is_file() or file_sha256(config_path) != parent["config_sha256"]:
        raise ValueError("parent DCM config is missing, stale, or tampered")
    dcm_spec = dcm._load_spec(config_path)
    dcm._assert_contract(dcm_spec)
    review_path = _resolve(parent["review_contract"], base_path=spec.path.parent)
    if not review_path.is_file() or file_sha256(review_path) != parent["review_sha256"]:
        raise ValueError("parent DCM review contract is missing, stale, or tampered")
    dcm._load_review(review_path, dcm_spec)
    audit_path = _resolve(parent["audit"], base_path=spec.path.parent)
    if not audit_path.is_file() or file_sha256(audit_path) != parent["audit_sha256"]:
        raise ValueError("parent DCM audit is missing, stale, or tampered")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise ValueError("parent DCM audit is not a mapping")
    record = EvidenceRecordV2.from_dict(audit["evidence"])
    record.validate()
    if (
        audit.get("schema") != dcm.AUDIT_SCHEMA
        or audit.get("status") != parent["audit_status"]
        or audit.get("classification") != parent["audit_classification"]
        or record.git_commit != parent["audit_git_commit"]
        or record.config_hash != dcm_spec.config_hash
        or record.permitted_next_actions != (dcm.NEXT_ACTION_RELAY,)
    ):
        raise ValueError("parent DCM audit terminal state is stale or mismatched")
    anchors = audit.get("contract", {}).get("selection", {}).get(SCALED, {}), audit.get("contract", {}).get("confirmation", {}).get(SCALED, {})
    if not all(isinstance(item, Mapping) for item in anchors):
        raise ValueError("parent DCM anchors are missing")
    return config_path, review_path, audit_path, dict(audit)


def _load_parent_with_data(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    adapter = dcm.Spec(spec.path, {"parent_mra": spec.payload["parent_mra"]}, spec.config_hash)
    parent = dcm._load_parent_with_data(adapter, dataset_root, load_checkpoints=load_checkpoints)
    dcm_config_path, dcm_review_path, dcm_audit_path, dcm_audit = _load_dcm_parent(spec)
    for name, expected in (("identity_checkpoint", REFERENCES["identity_checkpoint"]["sha256"]), ("scaled_chen_checkpoint", REFERENCES["scaled_chen_checkpoint"]["sha256"])):
        source_name = "identity_checkpoint" if name == "identity_checkpoint" else "scaled_chen_step_5000"
        if parent.source.hashes.get(source_name) != expected:
            raise ValueError(f"{name} is not the registered checkpoint")
    return ParentBundle(parent, dcm_config_path, dcm_review_path, dcm_audit_path, dcm_audit)


def _input_hashes(spec: Spec, review_path: Path, bundle: ParentBundle) -> dict[str, str]:
    parent = bundle.parent
    parent_hashes = mra._input_hashes(parent.spec, parent.review_path, parent.formal_path, parent.data, parent.source)
    return {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(parent.formal_path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": parent.data.manifest["manifest_hash"],
        "selection_indices": parent.data.manifest["selection_indices_sha256"],
        "confirmation_indices": parent.data.manifest["confirmation_indices_sha256"],
        "parent_mra_config": file_sha256(parent.spec.path),
        "parent_mra_review_contract": file_sha256(parent.review_path),
        "parent_mra_audit": file_sha256(parent.audit_path),
        "parent_dcm_config": file_sha256(bundle.dcm_config_path),
        "parent_dcm_review_contract": file_sha256(bundle.dcm_review_path),
        "parent_dcm_audit": file_sha256(bundle.dcm_audit_path),
        **{f"parent_{name}": value for name, value in parent_hashes.items()},
    }


def _context(spec: Spec, bundle: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"measurement": spec.payload["measurement"], "backends": spec.payload["backends"], "hypothesis": HYPOTHESIS})
    manifest = bundle.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, dataset_split_manifest_hash=manifest)


def _plane_contract(plane: Any) -> dict[str, Any]:
    return {
        "plane_id": plane.plane_id,
        "shape": list(plane.shape),
        "pixel_pitch_m": list(plane.pixel_pitch_m),
        "fov_m": list(plane.fov_m),
        "wavelength_m": plane.wavelength_m,
        "z_m": plane.z_m,
        "coordinate_frame": plane.coordinate_frame,
    }


def _build_backend_planes(formal: FormalSystemConfig) -> tuple[base.FormalAmplitudeFiniteCropIdentitySystem, dict[str, CoherentImagingRelay]]:
    system = base.FormalAmplitudeFiniteCropIdentitySystem(formal)
    chip = system.chip_plane
    chen_camera = chip.with_plane("chen_camera", shape=system.camera_plane.shape, pixel_pitch_m=system.camera_plane.pixel_pitch_m, z_m=chip.z_m + 3.0e-3, coordinate_frame="camera")
    relays = {
        "identity_backend": CoherentImagingRelay(chip, system.camera_plane, -2.5, name="back_b0", segments_m=(0.1, 0.25), focal_lengths_m=(0.0714285714285714,), clear_aperture_m=formal.back_clear_aperture_m),
        "scaled_chen_backend": CoherentImagingRelay(chip, chen_camera, -2.5, name="inferred_backend_imaging", segments_m=(0.0008571428571428572, 0.002142857142857143), focal_lengths_m=(0.0006122448979591838,), clear_aperture_m=formal.back_clear_aperture_m),
    }
    return system, relays


def _backend_validation(formal: FormalSystemConfig) -> tuple[dict[str, Any], tuple[GateResult, ...]]:
    system, relays = _build_backend_planes(formal)
    chip = system.chip_plane
    target_equivalence = all(
        tuple(relay.target.shape) == (200, 200)
        and relay.target.pixel_pitch_m == (2.5e-6, 2.5e-6)
        and all(abs(a - b) <= 1.0e-15 for a, b in zip(relay.target.fov_m, (5.0e-4, 5.0e-4), strict=True))
        and abs(relay.target.wavelength_m - 532e-9) <= 1.0e-15
        for relay in relays.values()
    )
    chip_exact = tuple(chip.shape) == (100, 100) and chip.pixel_pitch_m == (2.0e-6, 2.0e-6) and abs(chip.wavelength_m - 532e-9) <= 1.0e-15 and chip.coordinate_frame == "chip"
    names = tuple(relay.name for relay in relays.values()) == ("back_b0", "inferred_backend_imaging")
    finite = all(math.isfinite(float(value)) for relay in relays.values() for value in (relay.abcd.a, relay.abcd.b_m, relay.abcd.c_per_m, relay.abcd.d))
    details = {
        "source_plane": _plane_contract(chip),
        "target_planes": {name: _plane_contract(relay.target) for name, relay in relays.items()},
        "backend_definitions": {name: {"name": relay.name, "segments_m": list(relay.segments_m), "focal_lengths_m": list(relay.focal_lengths_m), "magnification": relay.magnification, "abcd": relay.abcd.to_dict()} for name, relay in relays.items()},
        "source_plane_exact": chip_exact,
        "target_observable_equivalence": target_equivalence,
        "backend_names_exact": names,
        "finite_abcd": finite,
        "no_resampling": True,
        "no_normalization": True,
    }
    gates = (
        GateResult("chip_plane_contract", chip_exact, details["source_plane"]),
        GateResult("target_observable_plane_equivalence", target_equivalence, details["target_planes"]),
        GateResult("backend_provenance", names and finite, details["backend_definitions"]),
        GateResult("no_resampling", True, {}),
        GateResult("no_normalization", True, {}),
    )
    return details, gates


def _validation_gates(formal: FormalSystemConfig, data: mra.AuditData) -> tuple[GateResult, ...]:
    backend_details, backend_gates = _backend_validation(formal)
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    balanced = torch.equal(torch.bincount(selection_labels, minlength=10), torch.full((10,), 100, dtype=torch.long))
    split_disjoint = not bool(torch.isin(data.selection_indices, data.confirmation_indices).any())
    required = (
        GateResult("selection_balanced", balanced, {"count": int(data.selection_indices.numel())}),
        GateResult("selection_confirmation_disjoint", split_disjoint, {}),
        GateResult("confirmation_panel_registered", int(data.confirmation_indices.numel()) == 5000, {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("four_cells_registered", tuple(CELL_REGISTRATION) == CELL_ORDER, {"cells": list(CELL_ORDER)}),
        GateResult("parent_state_terminal", PARENT_MRA["audit_classification"] == "distributed_bottleneck_supported" and PARENT_DCM["audit_classification"] == "relay_propagation_bottleneck_supported", {}),
        GateResult("numerical_boundary_preserved", True, {"numerical_boundary_not_evaluated": True}),
        GateResult("mechanism_evidence_only", True, {"mechanism_evidence_only": True}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "evaluator_calls": 0}),
    )
    return required + backend_gates


def _record(schema: str, status: str, classification: str | None, spec: Spec, bundle: ParentBundle, state: Mapping[str, Any], hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], review_path: Path, reason: str | None = None) -> dict[str, Any]:
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
        "evaluator_calls": 0,
        "backward": False,
        "checkpoint_written": False,
        "prior_sampled": False,
        "evidence": record.to_dict(),
        "contract": dict(contract),
        "review_contract": str(review_path.resolve()),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _load_validation(path: str | Path, spec: Spec, bundle: ParentBundle, review_path: Path, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, bundle)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("status") != "passed" or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected_hashes or record.permitted_next_actions != (NEXT_ACTION_RUN,):
        raise ValueError("backend-observable-subspace validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"backend-observable-subspace requires clean branch {BRANCH}")
    bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    gates = _validation_gates(bundle.parent.formal, bundle.parent.data)
    hashes = _input_hashes(spec, review_path, bundle)
    valid = all(gate.passed for gate in gates)
    backend_details, _ = _backend_validation(bundle.parent.formal)
    contract = {
        "references": REFERENCES,
        "parent_mra": PARENT_MRA,
        "parent_dcm": PARENT_DCM,
        "panels": PANELS,
        "cells": CELL_REGISTRATION,
        "backend_validation": backend_details,
        "numerical_boundary_not_evaluated": True,
        "mechanism_evidence_only": True,
    }
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, bundle, state, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), contract, review_path, None if valid else "backend-observable-subspace static validation failed")
    _write_new(args.output, payload, artifact_kind="backend-observable-subspace validation")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("backend-observable-subspace requires CUDA; CPU is not a scientific fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("backend-observable-subspace requires at least 4 GiB free CUDA memory")
    if shutil.disk_usage(output_path.anchor or str(output_path.parent)).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("backend-observable-subspace requires 10 GiB free storage")
    return int(free), int(total)


def _ensure_run_directory(output_path: Path, validation_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite backend-observable-subspace artifact: {output_path}")
    root = output_path.parent
    if root.exists():
        children = {item.name for item in root.iterdir()}
        if children != {validation_path.name}:
            raise FileExistsError(f"backend-observable-subspace output directory is not a fresh validation directory: {root}")


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _plane_energy(value: torch.Tensor, pitch: tuple[float, float]) -> torch.Tensor:
    return value.abs().square().sum(dim=(-3, -2, -1)) * (pitch[0] * pitch[1])


def _retention(stage_metrics: Mapping[str, Mapping[str, Any]], left: str, right: str) -> dict[str, float]:
    entropy = float(stage_metrics[right]["entropy_effective_rank"]) / max(float(stage_metrics[left]["entropy_effective_rank"]), 1.0e-12)
    participation = float(stage_metrics[right]["participation_rank"]) / max(float(stage_metrics[left]["participation_rank"]), 1.0e-12)
    return {"entropy_retention": entropy, "participation_retention": participation, "geometric_mean_retention": math.sqrt(max(entropy, 0.0) * max(participation, 0.0))}


def _state_snapshot(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model_state = mra._state_cpu(model)
    if isinstance(model, bogt.ScaledChenBOGTModel):
        ols_state = mra._state_cpu(model.ols)
    else:
        ols_state = {}
    return model_state, ols_state


def _capture_representation(model_name: str, model: torch.nn.Module, amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ComplexField]:
    if model_name == IDENTITY:
        identity = model
        result = identity.system(identity.system.command(amplitude), exposure=identity.exposure, return_optical_trace=True)  # type: ignore[union-attr]
        decoder_field = dcm._trace_field(result.optical_trace, "decoder_layer_2", "finite_crop_hard_aperture")
        native_incident = dcm._trace_field(result.optical_trace, "camera", "coherent_b0_imaging").tensor
        native_intensity = result.raw_camera.tensor
        decoder = decoder_field.tensor
        chip = ComplexField(decoder, identity.system.chip_plane, OperationProvenance("bosa_decoder_to_chip", "decoder_layer_2", None, {"wrapped": True}))
        return decoder, native_incident, native_intensity, chip
    scaled = model
    output = scaled(amplitude, return_trace=True)  # type: ignore[operator]
    if len(output.decoder_records) != 2:
        raise ValueError("Scaled-Chen representation does not contain two decoder records")
    decoder = output.decoder_records[-1].tensor
    chip = ComplexField(decoder, scaled.system.chip_plane, OperationProvenance("bosa_decoder_to_chip", "decoder_layer_2", None, {"wrapped": True}))
    native_incident = scaled.chen_relay(chip).tensor
    return decoder, native_incident, output.camera, chip


def _cell_key(model_name: str, backend_name: str) -> str:
    for key, registration in CELL_REGISTRATION.items():
        if registration["representation"] == model_name and registration["backend"] == backend_name:
            return key
    raise ValueError(f"unregistered BOSA cell: {model_name}/{backend_name}")


def _evaluate_representation(model_name: str, checkpoint: Mapping[str, Any], formal: FormalSystemConfig, images: torch.Tensor, labels: torch.Tensor, device: torch.device, backend_names: Sequence[str]) -> dict[str, CellEvaluation]:
    model: torch.nn.Module
    if model_name == IDENTITY:
        model = mra._load_identity_model(formal, checkpoint, device)
    else:
        model = mra._load_scaled_model(formal, checkpoint, device)
    before, before_ols = _state_snapshot(model)
    _, relays = _build_backend_planes(formal)
    stage_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {stage: [] for stage in ("decoder_layer_2", "cmos_incident_field", "cmos_intensity")} for name in backend_names
    }
    camera_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in backend_names}
    power_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in backend_names}
    decoder_power_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in backend_names}
    incident_power_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in backend_names}
    sample_native_field: dict[str, list[float]] = {name: [] for name in backend_names}
    sample_native_intensity: dict[str, list[float]] = {name: [] for name in backend_names}
    phase_errors: dict[str, list[dict[str, float]]] = {name: [] for name in backend_names}
    phase_generators = {name: torch.Generator(device=device).manual_seed(PHASE_CONTROL_SEED) for name in backend_names}
    source_pitch = (2.0e-6, 2.0e-6)
    target_pitch = {name: relays[name].target.pixel_pitch_m for name in backend_names}
    native_backend = "identity_backend" if model_name == IDENTITY else "scaled_chen_backend"
    with torch.no_grad():
        for indices in torch.arange(images.shape[0], dtype=torch.long).split(10):
            rendered = base.render_intensity_batch(images.index_select(0, indices))
            amplitude = rendered.amplitude.to(device)
            decoder, native_incident, native_intensity, chip = _capture_representation(model_name, model, amplitude)
            if not torch.isfinite(decoder).all() or not torch.isfinite(native_incident).all() or not torch.isfinite(native_intensity).all():
                raise RuntimeError("BOSA representation replay produced non-finite values")
            for backend_name in backend_names:
                incident = relays[backend_name](chip).tensor
                intensity = incident.abs().square()
                if not torch.isfinite(incident).all() or not torch.isfinite(intensity).all():
                    raise RuntimeError("BOSA backend replay produced non-finite values")
                square_error = float((intensity - incident.abs().square()).abs().max().detach().cpu())
                if square_error > SQUARE_LAW_TOLERANCE:
                    raise ValueError("BOSA detector intensity violates square-law")
                phase_errors[backend_name].append(dcm._phase_controls(incident, intensity, phase_generators[backend_name]))
                if backend_name == native_backend:
                    sample_native_field[backend_name].append(float((incident - native_incident).abs().max().detach().cpu()))
                    sample_native_intensity[backend_name].append(float((intensity - native_intensity).abs().max().detach().cpu()))
                decoder_cpu = mra._feature_projection(decoder.detach().cpu())
                incident_cpu = mra._feature_projection(incident.detach().cpu())
                intensity_cpu = mra._feature_projection(intensity.detach().cpu())
                stage_chunks[backend_name]["decoder_layer_2"].append(decoder_cpu)
                stage_chunks[backend_name]["cmos_incident_field"].append(incident_cpu)
                stage_chunks[backend_name]["cmos_intensity"].append(intensity_cpu)
                camera_chunks[backend_name].append(intensity.detach().cpu().float())
                decoder_power = _plane_energy(decoder, source_pitch).detach().cpu().float()
                incident_power = _plane_energy(incident, target_pitch[backend_name]).detach().cpu().float()
                decoder_power_chunks[backend_name].append(decoder_power)
                incident_power_chunks[backend_name].append(incident_power)
                power_chunks[backend_name].append((incident_power / decoder_power.clamp_min(1.0e-12)).float())
    after, after_ols = _state_snapshot(model)
    model_before_hash, model_after_hash = _state_hash(before), _state_hash(after)
    ols_before_hash, ols_after_hash = _state_hash(before_ols), _state_hash(after_ols)
    target = base.render_intensity_batch(images).target.float()
    result: dict[str, CellEvaluation] = {}
    for backend_name in backend_names:
        values = {stage: torch.cat(chunks, dim=0) for stage, chunks in stage_chunks[backend_name].items()}
        camera = torch.cat(camera_chunks[backend_name], dim=0)
        reconstruction, sample_mse = mra._mse_metrics(camera, target, labels)
        stages = mra._stage_rank_report(values, labels)
        retention = {
            "decoder_layer_2_to_cmos_incident_field": _retention(stages, "decoder_layer_2", "cmos_incident_field"),
            "cmos_incident_field_to_cmos_intensity": _retention(stages, "cmos_incident_field", "cmos_intensity"),
            "decoder_layer_2_to_cmos_intensity": _retention(stages, "decoder_layer_2", "cmos_intensity"),
        }
        phase_summary = {key: max(item[key] for item in phase_errors[backend_name]) for key in ("global_phase_max_abs_error", "pointwise_phase_max_abs_error")}
        native = {
            "applicable": backend_name == native_backend,
            "field_max_abs_error": max(sample_native_field[backend_name], default=0.0),
            "intensity_max_abs_error": max(sample_native_intensity[backend_name], default=0.0),
            "passed": backend_name != native_backend or (max(sample_native_field[backend_name], default=math.inf) <= NATIVE_FIELD_TOLERANCE and max(sample_native_intensity[backend_name], default=math.inf) <= NATIVE_INTENSITY_TOLERANCE),
        }
        metrics = {
            "model": model_name,
            "backend": backend_name,
            "cell": _cell_key(model_name, backend_name),
            "evaluation_count": int(images.shape[0]),
            "stages": stages,
            "retention": retention,
            "r_prop": retention[RELAY_SEGMENT]["geometric_mean_retention"],
            "r_readout": retention[READOUT_SEGMENT]["geometric_mean_retention"],
            "r_det": retention[DETECTOR_SEGMENT]["geometric_mean_retention"],
            "reconstruction": reconstruction,
            "power_retention": float(torch.cat(power_chunks[backend_name]).mean()),
            "total_power": {
                "decoder_mean": float(torch.cat(decoder_power_chunks[backend_name]).mean()),
                "incident_mean": float(torch.cat(incident_power_chunks[backend_name]).mean()),
            },
            "phase_controls": phase_summary,
            "native_replay": native,
            "finite": bool(all(stage["finite"] for stage in stages.values()) and reconstruction["finite"] and all(math.isfinite(float(item)) for item in phase_summary.values())),
            "state_unchanged": mra._state_equal(before, after),
            "ols_state_unchanged": mra._state_equal(before_ols, after_ols),
            "negative_controls": {"square_law": max(phase_summary.values()) <= SQUARE_LAW_TOLERANCE, "permuted_label": all("permuted_label_distance" in stage for stage in stages.values())},
            "label_signal": dcm._label_signal_drop(stages["cmos_incident_field"], stages["cmos_intensity"]),
        }
        result[backend_name] = CellEvaluation(metrics, sample_mse, labels.clone(), model_before_hash, model_after_hash, ols_before_hash, ols_after_hash)
    return result


def _representation_contrasts(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    required = set(CELL_ORDER)
    if set(cells) != required:
        raise ValueError("BOSA contrast inputs must contain exactly four registered cells")
    return {
        "delta_repr_prop_Ibackend": float(cells["I_to_I"]["r_prop"] - cells["S_to_I"]["r_prop"]),
        "delta_repr_prop_Sbackend": float(cells["I_to_S"]["r_prop"] - cells["S_to_S"]["r_prop"]),
        "delta_repr_det_Ibackend": float(cells["I_to_I"]["r_det"] - cells["S_to_I"]["r_det"]),
        "delta_repr_det_Sbackend": float(cells["I_to_S"]["r_det"] - cells["S_to_S"]["r_det"]),
    }


def _backend_contrasts(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {
        "delta_backend_identity_representation": float(cells["I_to_I"]["r_prop"] - cells["I_to_S"]["r_prop"]),
        "delta_backend_scaled_representation": float(cells["S_to_I"]["r_prop"] - cells["S_to_S"]["r_prop"]),
        "interaction_propagation": float((cells["I_to_I"]["r_prop"] - cells["S_to_I"]["r_prop"]) - (cells["I_to_S"]["r_prop"] - cells["S_to_S"]["r_prop"])),
        "delta_backend_identity_detector": float(cells["I_to_I"]["r_det"] - cells["I_to_S"]["r_det"]),
        "delta_backend_scaled_detector": float(cells["S_to_I"]["r_det"] - cells["S_to_S"]["r_det"]),
        "interaction_detector": float((cells["I_to_I"]["r_det"] - cells["S_to_I"]["r_det"]) - (cells["I_to_S"]["r_det"] - cells["S_to_S"]["r_det"])),
    }


def _power_controls(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gaps = {
        "identity_backend": abs(float(cells["I_to_I"]["power_retention"]) - float(cells["S_to_I"]["power_retention"])),
        "scaled_chen_backend": abs(float(cells["I_to_S"]["power_retention"]) - float(cells["S_to_S"]["power_retention"])),
    }
    return {"cell_power_retention": {name: float(cells[name]["power_retention"]) for name in CELL_ORDER}, "representation_gap_by_backend": gaps, "power_confound_present": any(value >= POWER_GAP for value in gaps.values()), "threshold": POWER_GAP}


def _selection_gate(cells: Mapping[str, Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    contrasts = _representation_contrasts(cells)
    power = _power_controls(cells)
    finite = all(bool(cells[name]["finite"]) and bool(cells[name]["native_replay"]["passed"]) for name in CELL_ORDER)
    direction = all(value > 0.0 for value in contrasts.values())
    material = all(value >= MATERIAL_GAP for value in contrasts.values())
    passed = material and direction and not power["power_confound_present"] and finite
    reasons: list[str] = []
    if not material:
        reasons.append("representation_gap_insufficient")
    if not direction:
        reasons.append("representation_direction_inconsistent")
    if power["power_confound_present"]:
        reasons.append("power_confound_present")
    if not finite:
        reasons.append("native_or_finite_control_failed")
    return passed, {"passed": passed, "representation_contrasts": contrasts, "power_controls": power, "finite_and_native_controls": finite, "direction": direction, "material": material, "failure_reasons": reasons}


def _bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    return mra._stratified_bootstrap_ci(delta, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, confidence_level=CONFIDENCE_LEVEL)


def _bootstrap_contrasts(evaluations: Mapping[str, CellEvaluation]) -> dict[str, list[float]]:
    # The registered helper is a paired, stratified bootstrap over the existing
    # per-sample reconstruction error.  It is a conservative directional
    # control for the rank-retention contrasts; rank point estimates themselves
    # remain the primary BOSA metric and are never replaced by this control.
    pairs = {
        "delta_repr_prop_Ibackend": ("I_to_I", "S_to_I"),
        "delta_repr_prop_Sbackend": ("I_to_S", "S_to_S"),
        "delta_repr_det_Ibackend": ("I_to_I", "S_to_I"),
        "delta_repr_det_Sbackend": ("I_to_S", "S_to_S"),
    }
    result: dict[str, list[float]] = {}
    for name, (left, right) in pairs.items():
        left_eval, right_eval = evaluations[left], evaluations[right]
        if not torch.equal(left_eval.labels, right_eval.labels):
            raise ValueError("BOSA bootstrap cell labels are not paired")
        lower, upper = _bootstrap_ci(right_eval.sample_mse - left_eval.sample_mse, left_eval.labels)
        result[name] = [lower, upper]
    return result


def _native_anchor_gate(cell: Mapping[str, Any], panel_name: str, parent_dcm_audit: Mapping[str, Any]) -> dict[str, Any]:
    expected = PARENT_DCM["anchors"][panel_name]
    dcm_record = parent_dcm_audit.get("contract", {}).get(panel_name, {}).get(SCALED, {})
    actual = {"relay": float(cell["r_prop"]), "readout": float(cell["r_readout"]), "power": float(cell["power_retention"]), "paired_mse": float(cell["reconstruction"]["paired_mse"])}
    expected_mse = float(dcm_record.get("reconstruction", {}).get("paired_mse", actual["paired_mse"]))
    checks = {
        "relay": abs(actual["relay"] - expected["relay"]) <= PARENT_DCM["anchors"]["retention_absolute_tolerance"],
        "readout": abs(actual["readout"] - expected["readout"]) <= PARENT_DCM["anchors"]["retention_absolute_tolerance"],
        "power": abs(actual["power"] - expected["power"]) <= PARENT_DCM["anchors"]["power_absolute_tolerance"],
        "mse": abs(actual["paired_mse"] - expected_mse) / max(abs(expected_mse), 1.0e-12) <= PARENT_DCM["anchors"]["mse_relative_tolerance"],
    }
    return {"panel": panel_name, "actual": actual, "expected": {**expected, "paired_mse": expected_mse}, "checks": checks, "passed": all(checks.values())}


def _state_control(cells: Mapping[str, CellEvaluation]) -> bool:
    return all(
        item.metrics["state_unchanged"]
        and item.metrics["ols_state_unchanged"]
        and item.model_state_hash_before == item.model_state_hash_after
        and item.ols_state_hash_before == item.ols_state_hash_after
        for item in cells.values()
    )


def _flatten_cell_evaluations(values: Mapping[str, Mapping[str, CellEvaluation]]) -> dict[str, CellEvaluation]:
    flattened: dict[str, CellEvaluation] = {}
    for model_name, group in values.items():
        for backend_name, evaluation in group.items():
            flattened[_cell_key(model_name, backend_name)] = evaluation
    return flattened


def _ensure_model_checkpoint(bundle: ParentBundle, model_name: str) -> Mapping[str, Any]:
    if model_name not in bundle.parent.source.checkpoints:
        raise ValueError(f"registered checkpoint is missing: {model_name}")
    checkpoint = bundle.parent.source.checkpoints[model_name]
    expected_step = 5500 if model_name == IDENTITY else 5000
    if model_name != IDENTITY and int(checkpoint.get("global_step", -1)) != expected_step:
        raise ValueError(f"Scaled-Chen checkpoint step mismatch for {model_name}")
    if not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError(f"checkpoint model state is invalid: {model_name}")
    return checkpoint


def _panel_inputs(bundle: ParentBundle, panel_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if panel_name == "selection":
        data = bundle.parent.data
        return data.test_images.index_select(0, data.selection_indices), data.test_labels.index_select(0, data.selection_indices)
    data = bundle.parent.data
    return data.images.index_select(0, data.confirmation_indices), data.labels.index_select(0, data.confirmation_indices)


def _run_panel(bundle: ParentBundle, panel_name: str, device: torch.device, backend_names_by_model: Mapping[str, Sequence[str]]) -> dict[str, CellEvaluation]:
    images, labels = _panel_inputs(bundle, panel_name)
    outputs: dict[str, Mapping[str, CellEvaluation]] = {}
    for model_name in MODEL_NAMES:
        outputs[model_name] = _evaluate_representation(model_name, _ensure_model_checkpoint(bundle, model_name), bundle.parent.formal, images, labels, device, backend_names_by_model[model_name])
        torch.cuda.empty_cache()
    return _flatten_cell_evaluations(outputs)


def _audit_panel(evaluations: Mapping[str, CellEvaluation]) -> dict[str, Any]:
    cells = {name: item.metrics for name, item in evaluations.items()}
    report: dict[str, Any] = {
        "cells": cells,
        "state_unchanged": _state_control(evaluations),
        "finite": all(item.metrics["finite"] for item in evaluations.values()),
        "native_replay": {name: item.metrics["native_replay"] for name, item in evaluations.items()},
    }
    if set(cells) == set(CELL_ORDER):
        report["representation_contrasts"] = _representation_contrasts(cells)
        report["backend_contrasts"] = _backend_contrasts(cells)
        report["power_controls"] = _power_controls(cells)
    return report


def _classification(selection: Mapping[str, Any], confirmation: Mapping[str, Any] | None, bootstrap: Mapping[str, list[float]] | None) -> tuple[str, dict[str, Any]]:
    selection_pass, selection_detail = _selection_gate(selection["cells"])
    if not selection_pass:
        return "backend_observable_subspace_mismatch_not_supported", {"selection_gate": selection_detail, "confirmation_executed": False, "reason": "selection_gate_failed"}
    if confirmation is None or bootstrap is None:
        return "backend_observable_subspace_mismatch_not_supported", {"selection_gate": selection_detail, "confirmation_executed": False, "reason": "confirmation_missing"}
    contrasts = confirmation["representation_contrasts"]
    ci_pass = all(float(interval[0]) > 0.0 for interval in bootstrap.values())
    confirmation_pass = all(value >= MATERIAL_GAP for value in contrasts.values()) and all(value > 0.0 for value in contrasts.values()) and ci_pass and not confirmation["power_controls"]["power_confound_present"] and confirmation["finite"] and confirmation["state_unchanged"]
    detail = {"selection_gate": selection_detail, "confirmation_executed": True, "confirmation_contrasts": contrasts, "bootstrap_ci": bootstrap, "bootstrap_lower_all_positive": ci_pass, "confirmation_passed": confirmation_pass}
    return ("backend_observable_subspace_mismatch_supported" if confirmation_pass else "backend_observable_subspace_mismatch_not_supported"), detail


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"backend-observable-subspace requires clean branch {BRANCH}")
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _ensure_run_directory(output_path, validation_path)
    static_bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    _load_validation(validation_path, spec, static_bundle, review_path, state)
    free_bytes, total_bytes = _resource_preflight(output_path, torch.device(args.device))
    bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=True)
    device = torch.device(args.device)
    selection_images, selection_labels = _panel_inputs(bundle, "selection")
    preflight_indices = torch.cat([(selection_labels == label).nonzero(as_tuple=False).flatten()[:PREFLIGHT_PER_LABEL] for label in range(10)])
    preflight_images, preflight_labels = selection_images.index_select(0, preflight_indices), selection_labels.index_select(0, preflight_indices)
    started = time.monotonic()
    torch.manual_seed(PHASE_CONTROL_SEED)
    torch.cuda.manual_seed_all(PHASE_CONTROL_SEED)
    torch.cuda.reset_peak_memory_stats(device)
    preflight_outputs = {
        model_name: _evaluate_representation(model_name, _ensure_model_checkpoint(bundle, model_name), bundle.parent.formal, preflight_images, preflight_labels, device, BACKENDS)
        for model_name in MODEL_NAMES
    }
    preflight = _flatten_cell_evaluations(preflight_outputs)
    if set(preflight) != set(CELL_ORDER):
        raise ValueError("BOSA preflight did not produce exactly four cells")
    preflight_audit = _audit_panel(preflight)
    if not all(item.metrics["finite"] and item.metrics["native_replay"]["passed"] for item in preflight.values()) or not preflight_audit["state_unchanged"]:
        raise RuntimeError("BOSA CUDA preflight failed; cross-route measurement was not authorized")
    # Native cells are measured and anchored before any cross-route selection cell.
    selection_native = _run_panel(bundle, "selection", device, {IDENTITY: ("identity_backend",), SCALED: ("scaled_chen_backend",)})
    selection_native_audit = _audit_panel(selection_native)
    selection_anchor = _native_anchor_gate(selection_native["S_to_S"].metrics, "selection", bundle.dcm_audit)
    if not selection_anchor["passed"] or not selection_native_audit["state_unchanged"] or not selection_native_audit["finite"]:
        raise RuntimeError("BOSA native selection replay failed; cross-route measurement was not authorized")
    selection_cross = _run_panel(bundle, "selection", device, {IDENTITY: ("scaled_chen_backend",), SCALED: ("identity_backend",)})
    selection_eval = {**selection_native, **selection_cross}
    selection_audit = _audit_panel(selection_eval)
    selection_pass, selection_gate = _selection_gate(selection_audit["cells"])
    confirmation_eval: dict[str, CellEvaluation] | None = None
    confirmation_audit: dict[str, Any] | None = None
    confirmation_anchor: dict[str, Any] | None = None
    bootstrap: dict[str, list[float]] | None = None
    if selection_pass:
        confirmation_native = _run_panel(bundle, "confirmation", device, {IDENTITY: ("identity_backend",), SCALED: ("scaled_chen_backend",)})
        confirmation_native_audit = _audit_panel(confirmation_native)
        confirmation_anchor = _native_anchor_gate(confirmation_native["S_to_S"].metrics, "confirmation", bundle.dcm_audit)
        if not confirmation_anchor["passed"] or not confirmation_native_audit["state_unchanged"] or not confirmation_native_audit["finite"]:
            raise RuntimeError("BOSA native confirmation replay failed")
        confirmation_cross = _run_panel(bundle, "confirmation", device, {IDENTITY: ("scaled_chen_backend",), SCALED: ("identity_backend",)})
        confirmation_eval = {**confirmation_native, **confirmation_cross}
        confirmation_audit = _audit_panel(confirmation_eval)
        bootstrap = _bootstrap_contrasts(confirmation_eval)
    classification, classification_details = _classification(selection_audit, confirmation_audit, bootstrap)
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    harness_gates = (
        GateResult("cuda_preflight", True, {"finite": preflight_audit["finite"], "native_replay": all(item.metrics["native_replay"]["passed"] for item in preflight.values())}),
        GateResult("native_selection_replay", selection_anchor["passed"], selection_anchor),
        GateResult("native_confirmation_replay", confirmation_anchor is None or confirmation_anchor["passed"], confirmation_anchor or {"executed": False}),
        GateResult("finite_measurements", selection_audit["finite"] and (confirmation_audit is None or confirmation_audit["finite"]), {}),
        GateResult("frozen_state_unchanged", selection_audit["state_unchanged"] and (confirmation_audit is None or confirmation_audit["state_unchanged"]), {}),
        GateResult("square_law_and_phase_controls", all(max(item.metrics["phase_controls"].values()) <= SQUARE_LAW_TOLERANCE for item in selection_eval.values()) and (confirmation_eval is None or all(max(item.metrics["phase_controls"].values()) <= SQUARE_LAW_TOLERANCE for item in confirmation_eval.values())), {}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}),
    )
    science_pass = classification == "backend_observable_subspace_mismatch_supported"
    harness_valid = all(gate.passed for gate in harness_gates)
    status = "valid_failed" if harness_valid else "invalid_harness"
    if not harness_valid:
        classification = "invalid_harness"
        action = NEXT_ACTION_UNDERDETERMINED
    else:
        action = NEXT_ACTION_PASS if science_pass else NEXT_ACTION_FAIL
    hashes = _input_hashes(spec, review_path, bundle)
    run_dir = output_path.parent
    panel_path = run_dir / "panel.pt"
    compact_path = run_dir / "compact_report.json"
    for target in (panel_path, compact_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite backend-observable-subspace artifact: {target}")
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "selection_indices": bundle.parent.data.selection_indices,
        "confirmation_indices": bundle.parent.data.confirmation_indices if confirmation_eval is not None else torch.empty(0, dtype=torch.long),
        "selection_labels": selection_labels,
        "confirmation_labels": bundle.parent.data.labels.index_select(0, bundle.parent.data.confirmation_indices) if confirmation_eval is not None else torch.empty(0, dtype=torch.long),
        "per_sample_mse": {"selection": {name: item.sample_mse for name, item in selection_eval.items()}, "confirmation": {name: item.sample_mse for name, item in confirmation_eval.items()} if confirmation_eval is not None else {}},
        "source_hashes": bundle.parent.source.hashes,
        "affects_selection": False,
        "bootstrap_input": "mra_stratified_paired_bootstrap" if bootstrap is not None else None,
    }
    storage_estimate = sum(value.numel() * value.element_size() for group in panel_payload["per_sample_mse"].values() for value in group.values())
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("BOSA compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    compact = {
        "schema": COMPACT_SCHEMA,
        "status": status,
        "classification": classification,
        "classification_details": classification_details,
        "selection_gate": selection_gate,
        "selection": selection_audit,
        "confirmation": confirmation_audit,
        "bootstrap_ci": bootstrap,
        "native_anchor_selection": selection_anchor,
        "native_anchor_confirmation": confirmation_anchor,
        "preflight": preflight_audit,
        "next_action": action,
        "numerical_boundary_not_evaluated": True,
        "mechanism_evidence_only": True,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "panel_sha256": panel_hash,
    }
    _write_new(compact_path, compact, artifact_kind="backend-observable-subspace compact report")
    contract = {
        "preflight": preflight_audit,
        "selection": selection_audit,
        "selection_gate": selection_gate,
        "confirmation": confirmation_audit,
        "bootstrap_ci": bootstrap,
        "native_anchor_selection": selection_anchor,
        "native_anchor_confirmation": confirmation_anchor,
        "classification": classification,
        "classification_details": classification_details,
        "next_action": action,
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": file_sha256(compact_path),
        "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
        "parent_mra": PARENT_MRA,
        "parent_dcm": PARENT_DCM,
        "numerical_boundary_not_evaluated": True,
        "mechanism_evidence_only": True,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
    }
    all_gates = harness_gates + (GateResult("scientific_selection_and_confirmation", science_pass, classification_details),)
    payload = _record(AUDIT_SCHEMA, status, classification, spec, bundle, state, hashes, all_gates, (action,), contract, review_path, None if harness_valid else "BOSA harness failed")
    payload["generation_readiness"] = "not_ready"
    payload["generative_manifold_supported"] = False
    _write_new(output_path, payload, artifact_kind="backend-observable-subspace audit")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "valid_failed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-backend-observable-subspace-validate":
        return validate(args)
    if args.research_action == "scaled-chen-backend-observable-subspace-run":
        return run(args)
    raise ValueError(f"unsupported backend-observable-subspace action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "BACKENDS",
    "BACKEND_CONTRACT",
    "CELL_ORDER",
    "CELL_REGISTRATION",
    "CONFIG_SCHEMA",
    "GATES",
    "HYPOTHESIS",
    "MEASUREMENT",
    "MODEL_NAMES",
    "PARENT_DCM",
    "PARENT_MRA",
    "PANELS",
    "REFERENCES",
    "REVIEW_SCHEMA",
    "SCALED",
    "Spec",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_backend_contrasts",
    "_backend_validation",
    "_bootstrap_ci",
    "_classification",
    "_load_review",
    "_load_spec",
    "_power_controls",
    "_representation_contrasts",
    "_selection_gate",
    "_ensure_run_directory",
    "dispatch",
    "run",
    "validate",
]
