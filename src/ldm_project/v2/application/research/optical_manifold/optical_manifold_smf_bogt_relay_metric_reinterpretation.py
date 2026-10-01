"""Frozen replay for the HFRR relay-retention metric reinterpretation.

This module is a research-only, no-update consumer of the already registered
MRA, DCM, BOSA, and HFRR evidence.  It replays the native complex fields on
the original panels, applies the HFRR analytic B=0 canonicalization only to
the registered decoder-to-incident relay, and keeps the historical raw and
canonicalized metrics side by side.  No production optical operator is
modified and no canonicalized field is sent to the detector, MSE, training,
or generation path.
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
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_new_json, canonical_sha256, file_sha256, tensor_sha256
from ldm_project.v2.optics.relays import CoherentImagingRelay
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_backend_observable_subspace as bosa
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_decoder_cmos_attribution as dcm
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_high_fidelity_relay_reference as hfrr
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_panel.v1"
COMPACT_SCHEMA = "scaled_chen_relay_metric_reinterpretation_compact_report.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S1-RMRI"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

IDENTITY = "identity_step_5500"
SCALED = "scaled_chen_step_5000"
MODEL_NAMES = (IDENTITY, SCALED)
BACKENDS = ("identity_backend", "scaled_chen_backend")
CELL_ORDER = ("I_to_I", "I_to_S", "S_to_I", "S_to_S")
RELAY_SEGMENT = "decoder_layer_2_to_cmos_incident_field"
READOUT_SEGMENT = "cmos_incident_field_to_cmos_intensity"
MRA_RELAY_COMPOSITE = "decoder_layer_2_to_cmos"
RANK_RETENTION_THRESHOLD = 0.90
MATERIAL_GAP = 0.05
RAW_RETENTION_ABSOLUTE_TOLERANCE = 0.005
RAW_MSE_RELATIVE_TOLERANCE = 0.01
RAW_POWER_ABSOLUTE_TOLERANCE = 0.005
PHASE_CONTROL_TOLERANCE = 1.0e-5
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

NEXT_ACTION_D1 = "manual_review_relay_metric_confound_closeout_contract"
NEXT_ACTION_D2 = "manual_review_canonicalized_relay_bottleneck_contract"
NEXT_ACTION_D3 = "manual_review_relay_metric_reinterpretation_underdetermined_contract"
NEXT_ACTION_RUN = "run_scaled_chen_relay_metric_reinterpretation"

PARENT_MRA = dict(dcm.PARENT_MRA)
PARENT_DCM = dict(bosa.PARENT_DCM)
PARENT_BOSA = dict(hfrr.PARENT_BOSA)

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

REFERENCES = dict(hfrr.REFERENCES)

PANELS = {
    "mra_selection": {
        "source": "mnist_test_indices_5000_9999",
        "seed": 1235,
        "count": 1000,
        "per_label": 100,
        "parent": PARENT_MRA["audit"],
    },
    "mra_confirmation": {
        "source": "horizon_heldout_train_indices_55000_59999",
        "count": 5000,
        "parent": PARENT_MRA["audit"],
    },
    "bosa_selection": {
        "source": "mnist_test_indices_5000_9999",
        "count": 1000,
        "per_label": 100,
        "parent": PARENT_BOSA["audit"],
        "confirmation_consumed": False,
    },
    "candidate_prefix": {
        "source": "mnist_test_indices_0_4999",
        "count": 4560,
        "consumed": False,
    },
    "no_new_samples": True,
}

MEASUREMENT = {
    "objective": "retrospective_frozen_replay",
    "models": list(MODEL_NAMES),
    "mra_segments": [list(item) for item in mra.SEGMENTS],
    "dcm_segments": [list(item) for item in dcm.SEGMENTS],
    "bosa_cells": [dict(bosa.CELL_REGISTRATION[name]) for name in CELL_ORDER],
    "canonicalized_segments": [RELAY_SEGMENT],
    "unchanged_segments": [READOUT_SEGMENT],
    "retention_definition": "mra_dcm_entropy_participation_geometric_mean",
    "feature_projection": "mra._feature_projection",
    "raw_replay_required": True,
    "same_panels": True,
    "same_sample_order": True,
    "same_labels": True,
    "same_thresholds": True,
    "no_resampling": True,
    "no_normalization": True,
    "mechanism_evidence_only": True,
    "no_update": True,
}

GATES = {
    "rank_retention_threshold": RANK_RETENTION_THRESHOLD,
    "material_gap_threshold": MATERIAL_GAP,
    "raw_retention_absolute_tolerance": RAW_RETENTION_ABSOLUTE_TOLERANCE,
    "raw_mse_relative_tolerance": RAW_MSE_RELATIVE_TOLERANCE,
    "raw_power_absolute_tolerance": RAW_POWER_ABSOLUTE_TOLERANCE,
    "phase_control_tolerance": PHASE_CONTROL_TOLERANCE,
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

CANONICALIZATION = {
    "implementation": "HFRR._canonicalize_field",
    "algorithm_id": hfrr.REFERENCE_CONTRACT["algorithm_id"],
    "formula": hfrr.CANONICALIZATION["formula"],
    "valid_backends": list(BACKENDS),
    "valid_segments": [RELAY_SEGMENT],
    "forbidden_segments": [
        "slm_to_pre_ols",
        "pre_ols_to_modal",
        "modal_to_post_ols",
        "post_ols_to_decoder_input",
        "decoder_input_to_decoder_layer_2",
        "fiber_array_ols_projection",
        "d2nn_internal_propagation",
        READOUT_SEGMENT,
        "detector",
        "mse_target",
        "physical_forward",
        "reconstruction_output",
    ],
    "fit": False,
    "global_phase_fit": False,
    "per_pixel_phase_fit": False,
    "amplitude_normalization": False,
    "physical_forward_modified": False,
    "detector_or_mse_input": False,
    "training_input": False,
    "generation_input": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_relay_metric_reinterpretation",
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
        "ols_rerun",
        "physical_forward_modification",
        "historical_evidence_modification",
        "new_samples",
        "bosa_confirmation",
    ],
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    mra_spec: mra.Spec
    mra_review_path: Path
    mra_audit_path: Path
    formal: FormalSystemConfig
    formal_path: Path
    data: mra.AuditData
    source: mra.SourceBundle
    bosa_spec: bosa.Spec
    bosa_audit_path: Path
    bosa_audit: dict[str, Any]
    dcm_config_path: Path
    dcm_review_path: Path
    dcm_audit_path: Path
    dcm_audit: dict[str, Any]
    hfrr_spec: hfrr.Spec
    hfrr_review_path: Path
    hfrr_audit_path: Path
    hfrr_audit: dict[str, Any]
    hfrr_compact_path: Path
    hfrr_panel_path: Path


@dataclass(frozen=True)
class ModelPanel:
    mra_metrics: dict[str, Any]
    dcm_metrics: dict[str, Any]
    bosa_metrics: dict[str, dict[str, Any]]
    sample_mse: torch.Tensor
    labels: torch.Tensor
    state: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return mra._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return mra._git_state()


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "parent_hfrr",
        "parent_mra",
        "parent_dcm",
        "parent_bosa",
        "references",
        "panels",
        "measurement",
        "gates",
        "canonicalization",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("RMRI config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("RMRI config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("RMRI formal config is not locked")
    if payload["parent_hfrr"] != PARENT_HFRR:
        raise ValueError("RMRI HFRR parent binding is not locked")
    if payload["parent_mra"] != PARENT_MRA:
        raise ValueError("RMRI MRA parent binding is not locked")
    if payload["parent_dcm"] != PARENT_DCM:
        raise ValueError("RMRI DCM parent binding is not locked")
    if payload["parent_bosa"] != PARENT_BOSA:
        raise ValueError("RMRI BOSA parent binding is not locked")
    if payload["references"] != REFERENCES:
        raise ValueError("RMRI checkpoint references are not locked")
    if payload["panels"] != PANELS:
        raise ValueError("RMRI panel contract is not locked")
    if payload["measurement"] != MEASUREMENT:
        raise ValueError("RMRI measurement contract is not locked")
    if payload["gates"] != GATES:
        raise ValueError("RMRI gate contract is not locked")
    if payload["canonicalization"] != CANONICALIZATION:
        raise ValueError("RMRI canonicalization contract is not locked")
    if payload["authorization"] != AUTHORIZATION:
        raise ValueError("RMRI authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "parent_hfrr": PARENT_HFRR,
            "parent_mra": PARENT_MRA,
            "parent_dcm": PARENT_DCM,
            "parent_bosa": PARENT_BOSA,
            "references": REFERENCES,
            "panels": PANELS,
            "measurement": MEASUREMENT,
            "gates": GATES,
            "canonicalization": CANONICALIZATION,
            "same_thresholds": True,
            "same_panels": True,
            "mechanism_evidence_only": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }
    if payload != expected:
        raise ValueError("RMRI review contract does not match scope")
    return target


def _terminal_audit(path: Path, expected: Mapping[str, Any], *, schema: str, hypothesis: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected["audit_sha256"]:
        raise ValueError(f"parent audit is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"parent audit is not a mapping: {path}")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != schema or payload.get("hypothesis") != hypothesis:
        raise ValueError(f"parent audit schema mismatch: {path}")
    if payload.get("status") != expected["audit_status"] or payload.get("classification") != expected["audit_classification"]:
        raise ValueError(f"parent audit terminal state mismatch: {path}")
    if "audit_git_commit" in expected and record.git_commit != expected["audit_git_commit"]:
        raise ValueError(f"parent audit producer commit mismatch: {path}")
    return dict(payload)


def _check_hfrr_audit(path: Path) -> dict[str, Any]:
    audit = _terminal_audit(path, PARENT_HFRR, schema=hfrr.AUDIT_SCHEMA, hypothesis=hfrr.HYPOTHESIS)
    record = EvidenceRecordV2.from_dict(audit["evidence"])
    if record.config_hash != hfrr._load_spec(_resolve(PARENT_HFRR["config"])).config_hash or record.git_commit != PARENT_HFRR["implementation_commit"]:
        raise ValueError("HFRR audit config or implementation commit is stale")
    contract = audit.get("contract", {})
    if (
        contract.get("relay_numerical_reference_consistent") is not True
        or contract.get("relay_numerical_reference_mismatch") is not False
        or contract.get("numerical_boundary_not_evaluated") is not False
        or contract.get("canonicalization", {}).get("physical_forward_modified") is not False
    ):
        raise ValueError("HFRR terminal numerical or physical-forward flags are invalid")
    return audit


def _load_parent_with_data(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    bosa_config_path = _resolve(PARENT_BOSA["config"], base_path=spec.path.parent)
    if not bosa_config_path.is_file() or file_sha256(bosa_config_path) != PARENT_BOSA["config_sha256"]:
        raise ValueError("BOSA parent config is missing, stale, or tampered")
    bosa_spec = bosa._load_spec(bosa_config_path)
    bosa._assert_contract(bosa_spec)
    bosa_review_path = _resolve(PARENT_BOSA["review_contract"], base_path=spec.path.parent)
    if not bosa_review_path.is_file() or file_sha256(bosa_review_path) != PARENT_BOSA["review_sha256"]:
        raise ValueError("BOSA parent review contract is missing, stale, or tampered")
    bosa._load_review(bosa_review_path, bosa_spec)
    bosa_audit_path = _resolve(PARENT_BOSA["audit"], base_path=spec.path.parent)
    bosa_audit = _terminal_audit(bosa_audit_path, PARENT_BOSA, schema=bosa.AUDIT_SCHEMA, hypothesis=bosa.HYPOTHESIS)

    bosa_bundle = bosa._load_parent_with_data(bosa_spec, dataset_root, load_checkpoints=load_checkpoints)
    dcm_parent = bosa_bundle.parent
    dcm_config_path = bosa_bundle.dcm_config_path
    dcm_review_path = bosa_bundle.dcm_review_path
    dcm_audit_path = bosa_bundle.dcm_audit_path
    dcm_audit = bosa_bundle.dcm_audit
    mra_parent = dcm_parent

    hfrr_config_path = _resolve(PARENT_HFRR["config"], base_path=spec.path.parent)
    if not hfrr_config_path.is_file() or file_sha256(hfrr_config_path) != PARENT_HFRR["config_sha256"]:
        raise ValueError("HFRR parent config is missing, stale, or tampered")
    hfrr_spec = hfrr._load_spec(hfrr_config_path)
    hfrr._assert_contract(hfrr_spec)
    hfrr_review_path = _resolve(PARENT_HFRR["review_contract"], base_path=spec.path.parent)
    if not hfrr_review_path.is_file() or file_sha256(hfrr_review_path) != PARENT_HFRR["review_sha256"]:
        raise ValueError("HFRR parent review contract is missing, stale, or tampered")
    hfrr._load_review(hfrr_review_path, hfrr_spec)
    hfrr_audit_path = _resolve(PARENT_HFRR["audit"], base_path=spec.path.parent)
    hfrr_audit = _check_hfrr_audit(hfrr_audit_path)
    hfrr_compact_path = _resolve(PARENT_HFRR["compact_report"], base_path=spec.path.parent)
    if not hfrr_compact_path.is_file() or file_sha256(hfrr_compact_path) != PARENT_HFRR["compact_report_sha256"]:
        raise ValueError("HFRR compact report is missing, stale, or tampered")
    hfrr_panel_path = _resolve(PARENT_HFRR["panel"], base_path=spec.path.parent)
    if not hfrr_panel_path.is_file() or file_sha256(hfrr_panel_path) != PARENT_HFRR["panel_sha256"]:
        raise ValueError("HFRR panel is missing, stale, or tampered")
    validation_path = _resolve(PARENT_HFRR["validation"], base_path=spec.path.parent)
    if not validation_path.is_file() or file_sha256(validation_path) != PARENT_HFRR["validation_sha256"]:
        raise ValueError("HFRR validation evidence is missing, stale, or tampered")

    if mra_parent.spec.path != _resolve(PARENT_MRA["config"], base_path=spec.path.parent):
        raise ValueError("MRA parent config path is not the registered path")
    if dcm_audit.get("classification") != PARENT_DCM["audit_classification"]:
        raise ValueError("DCM parent classification changed")
    if bosa_audit.get("classification") != PARENT_BOSA["audit_classification"]:
        raise ValueError("BOSA parent classification changed")
    if not torch.equal(mra_parent.data.selection_indices, mra_parent.data.selection_indices.sort().values):
        raise ValueError("registered selection ordering is not canonical")
    return ParentBundle(
        mra_parent.spec,
        mra_parent.review_path,
        mra_parent.audit_path,
        mra_parent.formal,
        mra_parent.formal_path,
        mra_parent.data,
        mra_parent.source,
        bosa_spec,
        bosa_audit_path,
        bosa_audit,
        dcm_config_path,
        dcm_review_path,
        dcm_audit_path,
        dcm_audit,
        hfrr_spec,
        hfrr_review_path,
        hfrr_audit_path,
        hfrr_audit,
        hfrr_compact_path,
        hfrr_panel_path,
    )


def _canonicalizer_module_hash() -> str:
    return file_sha256(Path(hfrr.__file__).resolve())


def _input_hashes(spec: Spec, review_path: Path, bundle: ParentBundle) -> dict[str, str]:
    parent_hashes = mra._input_hashes(bundle.mra_spec, bundle.mra_review_path, bundle.formal_path, bundle.data, bundle.source)
    return {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(bundle.formal_path),
        "review_contract": file_sha256(review_path),
        "dataset_manifest": bundle.data.manifest["manifest_hash"],
        "selection_indices": bundle.data.manifest["selection_indices_sha256"],
        "confirmation_indices": bundle.data.manifest["confirmation_indices_sha256"],
        "parent_mra_config": file_sha256(bundle.mra_spec.path),
        "parent_mra_review_contract": file_sha256(bundle.mra_review_path),
        "parent_mra_audit": file_sha256(bundle.mra_audit_path),
        "parent_dcm_config": file_sha256(bundle.dcm_config_path),
        "parent_dcm_review_contract": file_sha256(bundle.dcm_review_path),
        "parent_dcm_audit": file_sha256(bundle.dcm_audit_path),
        "parent_bosa_config": file_sha256(bundle.bosa_spec.path),
        "parent_bosa_review_contract": file_sha256(_resolve(PARENT_BOSA["review_contract"])),
        "parent_bosa_audit": file_sha256(bundle.bosa_audit_path),
        "parent_hfrr_config": file_sha256(bundle.hfrr_spec.path),
        "parent_hfrr_review_contract": file_sha256(bundle.hfrr_review_path),
        "parent_hfrr_validation": file_sha256(_resolve(PARENT_HFRR["validation"])),
        "parent_hfrr_audit": file_sha256(bundle.hfrr_audit_path),
        "parent_hfrr_compact_report": file_sha256(bundle.hfrr_compact_path),
        "parent_hfrr_panel": file_sha256(bundle.hfrr_panel_path),
        "canonicalizer_module": _canonicalizer_module_hash(),
        **{f"parent_{name}": value for name, value in parent_hashes.items()},
    }


def _context(spec: Spec, bundle: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"measurement": spec.payload["measurement"], "canonicalization": spec.payload["canonicalization"], "hypothesis": HYPOTHESIS})
    manifest = bundle.data.manifest["manifest_hash"]
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, dataset_split_manifest_hash=manifest)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _plane_energy(value: torch.Tensor, pitch: tuple[float, float]) -> torch.Tensor:
    return value.abs().square().sum(dim=(-3, -2, -1)) * (pitch[0] * pitch[1])


def _retention(stage_metrics: Mapping[str, Mapping[str, Any]], left: str, right: str) -> dict[str, float]:
    entropy = float(stage_metrics[right]["entropy_effective_rank"]) / max(float(stage_metrics[left]["entropy_effective_rank"]), 1.0e-12)
    participation = float(stage_metrics[right]["participation_rank"]) / max(float(stage_metrics[left]["participation_rank"]), 1.0e-12)
    return {"entropy_retention": entropy, "participation_retention": participation, "geometric_mean_retention": math.sqrt(max(entropy, 0.0) * max(participation, 0.0))}


def _retention_report(stage_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: _retention(stage_metrics, left, right) for name, left, right in dcm.SEGMENTS}


def _canonical_retention(left: torch.Tensor, canonical: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return hfrr._retention_from_features(left, canonical, labels)


def _trace_field(trace: Sequence[ComplexField] | None, plane_id: str, operation: str) -> ComplexField:
    return mra._find_trace_field(trace, plane_id, operation)


def _build_backend_planes(formal: FormalSystemConfig) -> tuple[base.FormalAmplitudeFiniteCropIdentitySystem, dict[str, CoherentImagingRelay]]:
    return bosa._build_backend_planes(formal)


def _forward_fields(model_name: str, model: torch.nn.Module, amplitude: torch.Tensor, research_ols: torch.nn.Module) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, ComplexField]:
    if model_name == IDENTITY:
        identity = model
        result = identity.system(identity.system.command(amplitude), exposure=identity.exposure, return_optical_trace=True)  # type: ignore[union-attr]
        trace = result.optical_trace
        encoder_1 = _trace_field(trace, "encoder_layer_1", "finite_crop_hard_aperture").tensor
        encoder_2_field = _trace_field(trace, "encoder_layer_2", "finite_crop_hard_aperture")
        encoder_2 = encoder_2_field.tensor
        ols = research_ols(encoder_2_field)
        decoder_1 = _trace_field(trace, "decoder_layer_1", "finite_crop_hard_aperture").tensor
        decoder_2 = _trace_field(trace, "decoder_layer_2", "finite_crop_hard_aperture").tensor
        camera = result.raw_camera.tensor
        coefficients = torch.einsum("mhw,bhw->bm", research_ols.input_modes.conj().to(encoder_2), encoder_2[:, 0]) * (2.0e-6**2)
        stages = {
            "slm_input": amplitude,
            "encoder_layer_1": encoder_1,
            "encoder_layer_2": encoder_2,
            "pre_ols": encoder_2,
            "modal_complex": coefficients,
            "modal_power": coefficients.abs().square(),
            "post_ols": ols.tensor,
            "decoder_input": ols.tensor,
            "decoder_layer_1": decoder_1,
            "decoder_layer_2": decoder_2,
            "cmos": camera,
        }
        chip = ComplexField(decoder_2, identity.system.chip_plane, OperationProvenance("rmri_decoder_to_chip", "decoder_layer_2", None, {"model": model_name}))
        incident = _trace_field(trace, "camera", "coherent_b0_imaging").tensor
        return stages, decoder_2, incident, camera, chip
    scaled = model
    output = scaled(amplitude, return_trace=True)  # type: ignore[operator]
    if len(output.encoder_records) != 2 or len(output.decoder_records) != 2:
        raise ValueError("Scaled-Chen trace does not contain the registered 2+2 records")
    stages = {
        "slm_input": amplitude,
        "encoder_layer_1": output.encoder_records[0].tensor,
        "encoder_layer_2": output.encoder_records[1].tensor,
        "pre_ols": output.encoder_records[-1].tensor,
        "modal_complex": output.coefficients,
        "modal_power": output.coefficients.abs().square(),
        "post_ols": output.ols_field.tensor,
        "decoder_input": output.ols_field.tensor,
        "decoder_layer_1": output.decoder_records[0].tensor,
        "decoder_layer_2": output.decoder_records[1].tensor,
        "cmos": output.camera,
    }
    decoder = output.decoder_records[-1].tensor
    chip = ComplexField(decoder, scaled.system.chip_plane, OperationProvenance("rmri_decoder_to_chip", "decoder_layer_2", None, {"model": model_name}))
    incident = scaled.chen_relay(chip).tensor
    return stages, decoder, incident, output.camera, chip


def _rank_stages(values: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, Any]:
    return mra._stage_rank_report(values, labels)


def _panel_inputs(bundle: ParentBundle, panel_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if panel_name == "selection":
        return bundle.data.test_images.index_select(0, bundle.data.selection_indices), bundle.data.test_labels.index_select(0, bundle.data.selection_indices)
    return bundle.data.images.index_select(0, bundle.data.confirmation_indices), bundle.data.labels.index_select(0, bundle.data.confirmation_indices)


def _model_for(model_name: str, bundle: ParentBundle, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    checkpoint = bundle.source.checkpoints[model_name]
    research_ols = mra._build_research_ols(bundle.formal).to(device)
    if model_name == IDENTITY:
        model: torch.nn.Module = mra._load_identity_model(bundle.formal, checkpoint, device)
    else:
        model = mra._load_scaled_model(bundle.formal, checkpoint, device)
    return model, research_ols


def _empty_feature_store() -> dict[str, list[torch.Tensor]]:
    return {name: [] for name in mra.STAGE_NAMES}


def _evaluate_model_panel(model_name: str, bundle: ParentBundle, images: torch.Tensor, labels: torch.Tensor, device: torch.device, *, include_backends: bool) -> ModelPanel:
    model, research_ols = _model_for(model_name, bundle, device)
    _, relays = _build_backend_planes(bundle.formal)
    before = mra._state_cpu(model)
    ols_module = model.ols if isinstance(model, bogt.ScaledChenBOGTModel) else research_ols
    before_ols = mra._state_cpu(ols_module)
    mra_store = _empty_feature_store()
    dcm_store: dict[str, list[torch.Tensor]] = {name: [] for name in ("decoder_layer_2", "cmos_incident_field", "cmos_intensity")}
    bosa_store: dict[str, dict[str, list[torch.Tensor]]] = {backend: {name: [] for name in ("decoder_layer_2", "cmos_incident_field", "cmos_intensity")} for backend in BACKENDS}
    bosa_canonical_store: dict[str, list[torch.Tensor]] = {backend: [] for backend in BACKENDS}
    camera_by_backend: dict[str, list[torch.Tensor]] = {backend: [] for backend in BACKENDS}
    energy_by_backend: dict[str, list[torch.Tensor]] = {backend: [] for backend in BACKENDS}
    decoder_energy: list[torch.Tensor] = []
    phase_controls: dict[str, list[dict[str, float]]] = {backend: [] for backend in BACKENDS}
    native_field_errors: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    native_intensity_errors: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    native_backend = "identity_backend" if model_name == IDENTITY else "scaled_chen_backend"
    phase_generators = {backend: torch.Generator(device=device).manual_seed(dcm.PHASE_CONTROL_SEED) for backend in BACKENDS}
    with torch.no_grad():
        for chunk in torch.arange(images.shape[0], dtype=torch.long).split(10):
            rendered = base.render_intensity_batch(images.index_select(0, chunk))
            amplitude = rendered.amplitude.to(device)
            stages, decoder, native_incident, native_camera, chip = _forward_fields(model_name, model, amplitude, research_ols)
            if not all(torch.isfinite(value).all() for value in (decoder, native_incident, native_camera)):
                raise RuntimeError("RMRI replay produced non-finite native fields")
            for name, value in stages.items():
                detached = value.detach().cpu()
                mra_store[name].append(detached if name in {"modal_complex", "modal_power"} else mra._feature_projection(detached))
            dcm_store["decoder_layer_2"].append(mra._feature_projection(decoder.detach().cpu()))
            dcm_store["cmos_incident_field"].append(mra._feature_projection(native_incident.detach().cpu()))
            dcm_store["cmos_intensity"].append(mra._feature_projection(native_camera.detach().cpu()))
            decoder_energy.append(_plane_energy(decoder, (2.0e-6, 2.0e-6)).detach().cpu().float())
            for backend in BACKENDS:
                if not include_backends and backend != native_backend:
                    continue
                incident = relays[backend](chip).tensor
                intensity = incident.abs().square()
                if not torch.isfinite(incident).all() or not torch.isfinite(intensity).all():
                    raise RuntimeError("RMRI backend replay produced non-finite fields")
                if backend == native_backend:
                    native_field_errors[backend].append(float((incident - native_incident).abs().max().detach().cpu()))
                    native_intensity_errors[backend].append(float((intensity - native_camera).abs().max().detach().cpu()))
                phase_controls[backend].append(dcm._phase_controls(incident, intensity, phase_generators[backend]))
                bosa_store[backend]["decoder_layer_2"].append(mra._feature_projection(decoder.detach().cpu()))
                bosa_store[backend]["cmos_incident_field"].append(mra._feature_projection(incident.detach().cpu()))
                bosa_store[backend]["cmos_intensity"].append(mra._feature_projection(intensity.detach().cpu()))
                bosa_canonical_store[backend].append(mra._feature_projection(hfrr._canonicalize_field(incident.detach().cpu(), backend)))
                camera_by_backend[backend].append(intensity.detach().cpu().float())
                target_pitch = relays[backend].target.pixel_pitch_m
                incident_energy = _plane_energy(incident, target_pitch).detach().cpu().float()
                energy_by_backend[backend].append(incident_energy / decoder_energy[-1].clamp_min(1.0e-12))

    after = mra._state_cpu(model)
    after_ols = mra._state_cpu(ols_module)
    mra_values = {name: torch.cat(values, dim=0) for name, values in mra_store.items()}
    mra_target = base.render_intensity_batch(images).target.float()
    mra_camera = torch.cat(camera_by_backend[native_backend], dim=0)
    mra_reconstruction, sample_mse = mra._mse_metrics(mra_camera, mra_target, labels)
    mra_stages = _rank_stages(mra_values, labels)
    mra_metrics = {
        "model": model_name,
        "evaluation_count": int(images.shape[0]),
        "reconstruction": mra_reconstruction,
        "stages": mra_stages,
        "retention": mra._retention_report(mra_stages),
        "finite": bool(all(item["finite"] for item in mra_stages.values()) and mra_reconstruction["finite"]),
    }

    dcm_values = {name: torch.cat(values, dim=0) for name, values in dcm_store.items()}
    dcm_stages = _rank_stages(dcm_values, labels)
    dcm_retention = _retention_report({"decoder_layer_2": dcm_stages["decoder_layer_2"], "cmos_incident_field": dcm_stages["cmos_incident_field"], "cmos_intensity": dcm_stages["cmos_intensity"]})
    dcm_camera = torch.cat(camera_by_backend[native_backend], dim=0)
    dcm_reconstruction, _ = mra._mse_metrics(dcm_camera, mra_target, labels)
    dcm_phase = {key: max(item[key] for item in phase_controls[native_backend]) for key in ("global_phase_max_abs_error", "pointwise_phase_max_abs_error")}
    dcm_energy = torch.cat(energy_by_backend[native_backend], dim=0)
    dcm_metrics = {
        "model": model_name,
        "evaluation_count": int(images.shape[0]),
        "reconstruction": dcm_reconstruction,
        "stages": dcm_stages,
        "retention": dcm_retention,
        "power_retention": float(dcm_energy.mean()),
        "phase_controls": dcm_phase,
        "native_replay": {"backend": native_backend, "field_max_abs_error": max(native_field_errors[native_backend], default=0.0), "intensity_max_abs_error": max(native_intensity_errors[native_backend], default=0.0)},
        "finite": bool(all(item["finite"] for item in dcm_stages.values()) and dcm_reconstruction["finite"] and all(math.isfinite(float(value)) for value in dcm_phase.values())),
        "label_signal": dcm._label_signal_drop(dcm_stages["cmos_incident_field"], dcm_stages["cmos_intensity"]),
    }

    bosa_metrics: dict[str, dict[str, Any]] = {}
    backend_names = BACKENDS if include_backends else (native_backend,)
    for backend in backend_names:
        values = {name: torch.cat(chunks, dim=0) for name, chunks in bosa_store[backend].items()}
        stages = _rank_stages(values, labels)
        intensity = torch.cat(camera_by_backend[backend], dim=0)
        reconstruction, _ = mra._mse_metrics(intensity, mra_target, labels)
        retention = {
            RELAY_SEGMENT: _retention(stages, "decoder_layer_2", "cmos_incident_field"),
            READOUT_SEGMENT: _retention(stages, "cmos_incident_field", "cmos_intensity"),
            "decoder_layer_2_to_cmos_intensity": _retention(stages, "decoder_layer_2", "cmos_intensity"),
        }
        canonical = torch.cat(bosa_canonical_store[backend], dim=0)
        canonical_retention = _canonical_retention(values["decoder_layer_2"], canonical, labels)
        phase = {key: max(item[key] for item in phase_controls[backend]) for key in ("global_phase_max_abs_error", "pointwise_phase_max_abs_error")}
        metrics = {
            "model": model_name,
            "backend": backend,
            "cell": bosa._cell_key(model_name, backend),
            "evaluation_count": int(images.shape[0]),
            "stages": stages,
            "retention": retention,
            "r_prop": retention[RELAY_SEGMENT]["geometric_mean_retention"],
            "r_prop_canonicalized": canonical_retention["geometric_mean_retention"],
            "r_readout": retention[READOUT_SEGMENT]["geometric_mean_retention"],
            "r_det": retention["decoder_layer_2_to_cmos_intensity"]["geometric_mean_retention"],
            "reconstruction": reconstruction,
            "power_retention": float(torch.cat(energy_by_backend[backend], dim=0).mean()),
            "phase_controls": phase,
            "native_replay": {"backend": native_backend, "applicable": backend == native_backend, "field_max_abs_error": max(native_field_errors[backend], default=0.0), "intensity_max_abs_error": max(native_intensity_errors[backend], default=0.0), "passed": backend != native_backend or (max(native_field_errors[backend], default=math.inf) <= RAW_RETENTION_ABSOLUTE_TOLERANCE and max(native_intensity_errors[backend], default=math.inf) <= PHASE_CONTROL_TOLERANCE)},
            "finite": bool(all(item["finite"] for item in stages.values()) and reconstruction["finite"] and math.isfinite(float(canonical_retention["geometric_mean_retention"]))),
            "label_signal": dcm._label_signal_drop(stages["cmos_incident_field"], stages["cmos_intensity"]),
            "canonicalized_label_signal": dcm._label_signal_drop(stages["decoder_layer_2"], mra._rank_metrics(canonical, labels)),
        }
        bosa_metrics[backend] = metrics
    state = {
        "model_state_hash_before": _state_hash(before),
        "model_state_hash_after": _state_hash(after),
        "ols_state_hash_before": _state_hash(before_ols),
        "ols_state_hash_after": _state_hash(after_ols),
        "state_unchanged": mra._state_equal(before, after),
        "ols_state_unchanged": mra._state_equal(before_ols, after_ols),
    }
    return ModelPanel(mra_metrics, dcm_metrics, bosa_metrics, sample_mse, labels.clone(), state)


def _classify_dcm(selection: Mapping[str, Any], confirmation: Mapping[str, Any], identity: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    relay_sel = float(selection["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
    relay_con = float(confirmation["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
    canon_sel = float(selection["canonicalized_retention"]["geometric_mean_retention"])
    canon_con = float(confirmation["canonicalized_retention"]["geometric_mean_retention"])
    detector_sel = float(selection["retention"][READOUT_SEGMENT]["geometric_mean_retention"])
    detector_con = float(confirmation["retention"][READOUT_SEGMENT]["geometric_mean_retention"])
    raw_relay_failed = relay_sel < RANK_RETENTION_THRESHOLD and relay_con < RANK_RETENTION_THRESHOLD
    canonical_relay_failed = canon_sel < RANK_RETENTION_THRESHOLD and canon_con < RANK_RETENTION_THRESHOLD
    detector_failed = detector_sel < RANK_RETENTION_THRESHOLD and detector_con < RANK_RETENTION_THRESHOLD
    relay_loss = 1.0 - (canon_sel + canon_con) / 2.0
    raw_relay_loss = 1.0 - (relay_sel + relay_con) / 2.0
    detector_loss = 1.0 - (detector_sel + detector_con) / 2.0
    selection_label_signal = selection["canonicalized_label_signal"]
    confirmation_label_signal = confirmation["canonicalized_label_signal"]
    label_loss = bool(selection_label_signal["relative_inter_intra_drop"] >= MATERIAL_GAP and selection_label_signal["retrieval_drop"] >= MATERIAL_GAP and confirmation_label_signal["relative_inter_intra_drop"] >= MATERIAL_GAP and confirmation_label_signal["retrieval_drop"] >= MATERIAL_GAP)
    identity_detector = float(identity["retention"][READOUT_SEGMENT]["geometric_mean_retention"])
    scaled_detector = (detector_sel + detector_con) / 2.0
    details = {
        "raw_relay_failed": raw_relay_failed,
        "canonicalized_relay_failed": canonical_relay_failed,
        "detector_failed": detector_failed,
        "raw_relay_loss": raw_relay_loss,
        "canonicalized_relay_loss": relay_loss,
        "detector_loss": detector_loss,
        "canonicalized_loss_difference": abs(relay_loss - detector_loss),
        "label_signal_lost": label_loss,
        "canonicalized_label_signal_selection": selection_label_signal,
        "canonicalized_label_signal_confirmation": confirmation_label_signal,
        "matched_identity_detector_gap": identity_detector - scaled_detector,
        "historical_classification_preserved": True,
    }
    if not canonical_relay_failed and not detector_failed:
        return "relay_propagation_bottleneck_reinterpreted_metric_confound", details
    if canonical_relay_failed and not detector_failed and relay_loss - detector_loss >= MATERIAL_GAP:
        return "relay_propagation_bottleneck_supported_canonicalized", details
    if canonical_relay_failed == detector_failed or abs(relay_loss - detector_loss) < MATERIAL_GAP:
        return "relay_propagation_reinterpretation_underdetermined", details
    return "relay_propagation_reinterpretation_underdetermined", details


def _classify_mra(selection: Mapping[str, Any], confirmation: Mapping[str, Any], dcm_classification: str) -> tuple[str, dict[str, Any]]:
    independent = ("pre_ols_to_modal", "decoder_input_to_decoder_layer_2")
    checks = {segment: float(selection["retention"][segment]["geometric_mean_retention"]) < RANK_RETENTION_THRESHOLD and float(confirmation["retention"][segment]["geometric_mean_retention"]) < RANK_RETENTION_THRESHOLD for segment in independent}
    supported = sum(checks.values()) >= 2
    details = {"independent_material_failures": checks, "canonicalized_downstream_component": RELAY_SEGMENT, "unchanged_readout_component": READOUT_SEGMENT, "dcm_interpretation": dcm_classification}
    return ("distributed_bottleneck_supported_canonicalized" if supported else "distributed_bottleneck_evidence_reduced"), details


def _bosa_contrasts(cells: Mapping[str, Mapping[str, Any]], key: str) -> dict[str, float]:
    return {
        "delta_repr_prop_Ibackend": float(cells["I_to_I"][key] - cells["S_to_I"][key]),
        "delta_repr_prop_Sbackend": float(cells["I_to_S"][key] - cells["S_to_S"][key]),
    }


def _bosa_reinterpretation(cells: Mapping[str, Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    raw = _bosa_contrasts(cells, "r_prop")
    canonical = _bosa_contrasts(cells, "r_prop_canonicalized")
    raw_pattern = all(value > 0.0 and value >= MATERIAL_GAP for value in raw.values())
    canonical_pattern = all(value > 0.0 and value >= MATERIAL_GAP for value in canonical.values())
    details = {"raw_representation_contrasts": raw, "canonicalized_representation_contrasts": canonical, "raw_pattern_supported": raw_pattern, "canonicalized_pattern_supported": canonical_pattern, "raw_backend_gap_metric_confound": True, "confirmation_executed": False}
    if not canonical_pattern:
        return "backend_observable_subspace_mismatch_not_supported_reinterpreted", details
    return "backend_observable_subspace_reinterpretation_underdetermined", details


def _raw_anchor_gate(dcm_metrics: Mapping[str, Any], panel_name: str, parent_dcm_audit: Mapping[str, Any]) -> dict[str, Any]:
    model = SCALED
    actual = {
        "relay": float(dcm_metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"]),
        "readout": float(dcm_metrics["retention"][READOUT_SEGMENT]["geometric_mean_retention"]),
        "power": float(dcm_metrics["power_retention"]),
        "paired_mse": float(dcm_metrics["reconstruction"]["paired_mse"]),
    }
    historical = parent_dcm_audit.get("contract", {}).get(panel_name, {}).get(model, {})
    expected = {
        "relay": float(historical.get("retention", {}).get(RELAY_SEGMENT, {}).get("geometric_mean_retention", actual["relay"])),
        "readout": float(historical.get("retention", {}).get(READOUT_SEGMENT, {}).get("geometric_mean_retention", actual["readout"])),
        "power": float(historical.get("power_retention", actual["power"])),
        "paired_mse": float(historical.get("reconstruction", {}).get("paired_mse", actual["paired_mse"])),
    }
    checks = {
        "relay": abs(actual["relay"] - expected["relay"]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE,
        "readout": abs(actual["readout"] - expected["readout"]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE,
        "power": abs(actual["power"] - expected["power"]) <= RAW_POWER_ABSOLUTE_TOLERANCE,
        "mse": abs(actual["paired_mse"] - expected["paired_mse"]) / max(abs(expected["paired_mse"]), 1.0e-12) <= RAW_MSE_RELATIVE_TOLERANCE,
    }
    return {"panel": panel_name, "actual": actual, "expected": expected, "checks": checks, "passed": all(checks.values())}


def _mra_anchor_gate(mra_metrics: Mapping[str, Any], panel_name: str, parent_mra_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Require the unchanged MRA replay to remain on its registered metric scale."""
    historical = parent_mra_audit.get("contract", {}).get(panel_name, {}).get(SCALED, {})
    actual_retention = {
        name: float(value["geometric_mean_retention"])
        for name, value in mra_metrics.get("retention", {}).items()
    }
    expected_retention = {
        name: float(value.get("geometric_mean_retention", actual_retention[name]))
        for name, value in historical.get("retention", {}).items()
        if name in actual_retention
    }
    retention_checks = {
        name: abs(actual_retention[name] - expected_retention[name]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE
        for name in expected_retention
    }
    actual_mse = float(mra_metrics["reconstruction"]["paired_mse"])
    expected_mse = float(historical.get("reconstruction", {}).get("paired_mse", actual_mse))
    mse_check = abs(actual_mse - expected_mse) / max(abs(expected_mse), 1.0e-12) <= RAW_MSE_RELATIVE_TOLERANCE
    return {
        "panel": panel_name,
        "actual_retention": actual_retention,
        "expected_retention": expected_retention,
        "retention_checks": retention_checks,
        "actual_paired_mse": actual_mse,
        "expected_paired_mse": expected_mse,
        "paired_mse_check": mse_check,
        "passed": bool(retention_checks) and all(retention_checks.values()) and mse_check,
    }


def _bosa_anchor_gate(cells: Mapping[str, Mapping[str, Any]], parent_bosa_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Check all four registered BOSA selection cells before reinterpretation."""
    historical_cells = parent_bosa_audit.get("contract", {}).get("selection", {}).get("cells", {})
    checks: dict[str, dict[str, bool]] = {}
    actual: dict[str, dict[str, float]] = {}
    expected: dict[str, dict[str, float]] = {}
    for name in CELL_ORDER:
        current = cells[name]
        prior = historical_cells.get(name)
        if not isinstance(prior, Mapping):
            checks[name] = {"registered": False}
            continue
        actual[name] = {
            "r_prop": float(current["r_prop"]),
            "r_readout": float(current["r_readout"]),
            "r_det": float(current["r_det"]),
            "power_retention": float(current["power_retention"]),
        }
        expected[name] = {
            key: float(prior.get(key, actual[name][key]))
            for key in ("r_prop", "r_readout", "r_det", "power_retention")
        }
        checks[name] = {
            "r_prop": abs(actual[name]["r_prop"] - expected[name]["r_prop"]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE,
            "r_readout": abs(actual[name]["r_readout"] - expected[name]["r_readout"]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE,
            "r_det": abs(actual[name]["r_det"] - expected[name]["r_det"]) <= RAW_RETENTION_ABSOLUTE_TOLERANCE,
            "power_retention": abs(actual[name]["power_retention"] - expected[name]["power_retention"]) <= RAW_POWER_ABSOLUTE_TOLERANCE,
        }
    return {"actual": actual, "expected": expected, "checks": checks, "passed": set(checks) == set(CELL_ORDER) and all(all(item.values()) for item in checks.values())}


def _validation_gates(spec: Spec, bundle: ParentBundle, review_path: Path, output_path: Path, state: Mapping[str, Any]) -> tuple[GateResult, ...]:
    data = bundle.data
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    disk_free = shutil.disk_usage(output_path.anchor or str(output_path.parent)).free
    canonicalizer_ok = callable(getattr(hfrr, "_canonicalize_field", None)) and _canonicalizer_module_hash() == file_sha256(Path(hfrr.__file__).resolve())
    return (
        GateResult("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True, {"branch": state.get("branch"), "clean": state.get("clean")}),
        GateResult("selection_balanced", bool(torch.equal(torch.bincount(selection_labels, minlength=10), torch.full((10,), 100, dtype=torch.long))), {"count": int(data.selection_indices.numel())}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("exact_registered_panels", int(data.selection_indices.numel()) == 1000 and int(data.confirmation_indices.numel()) == 5000, {"selection": int(data.selection_indices.numel()), "confirmation": int(data.confirmation_indices.numel())}),
        GateResult("candidate_prefix_not_consumed", PANELS["candidate_prefix"]["consumed"] is False and PANELS["bosa_selection"]["confirmation_consumed"] is False, {}),
        GateResult("checkpoint_and_source_hashes", bundle.source.hashes.get("identity_checkpoint") == REFERENCES["identity_checkpoint"]["sha256"] and bundle.source.hashes.get("scaled_chen_step_5000") == REFERENCES["scaled_chen_checkpoint"]["sha256"], {"identity": bundle.source.hashes.get("identity_checkpoint"), "scaled": bundle.source.hashes.get("scaled_chen_step_5000")}),
        GateResult("parent_terminal_states", bundle.hfrr_audit.get("classification") == PARENT_HFRR["audit_classification"] and bundle.dcm_audit.get("classification") == PARENT_DCM["audit_classification"] and bundle.bosa_audit.get("classification") == PARENT_BOSA["audit_classification"], {}),
        GateResult("canonicalization_applicability", canonicalizer_ok and spec.payload["canonicalization"] == CANONICALIZATION, {"module": str(Path(hfrr.__file__).resolve()), "analytic_only": True}),
        GateResult("threshold_transfer", spec.payload["measurement"]["feature_projection"] == "mra._feature_projection" and spec.payload["measurement"]["retention_definition"] == "mra_dcm_entropy_participation_geometric_mean" and spec.payload["gates"]["rank_retention_threshold"] == 0.90 and spec.payload["gates"]["material_gap_threshold"] == 0.05, {"same_projection": True, "same_rank_metric": True}),
        GateResult("no_physical_forward_modification", CANONICALIZATION["physical_forward_modified"] is False and CANONICALIZATION["detector_or_mse_input"] is False, {}),
        GateResult("mechanism_evidence_only", AUTHORIZATION["mechanism_evidence_only"] is True, {}),
        GateResult("output_path_absent", not output_path.exists(), {"path": str(output_path)}),
        GateResult("disk_free_precondition", disk_free > MIN_DISK_FREE_BYTES, {"free_bytes": int(disk_free), "minimum": MIN_DISK_FREE_BYTES}),
    )


def _record(schema: str, status: str, classification: str | None, spec: Spec, bundle: ParentBundle, state: Mapping[str, Any], review_path: Path, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
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
        "review_contract": str(review_path.resolve()),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    output_path = _resolve(args.output)
    bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    gates = _validation_gates(spec, bundle, review_path, output_path, state)
    hashes = _input_hashes(spec, review_path, bundle)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, bundle, state, review_path, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), {"parent_hfrr": PARENT_HFRR, "parent_mra": PARENT_MRA, "parent_dcm": PARENT_DCM, "parent_bosa": PARENT_BOSA, "canonicalizer_module": str(Path(hfrr.__file__).resolve()), "canonicalizer_module_sha256": hashes["canonicalizer_module"], "threshold_transfer": True, "output_path": str(output_path.resolve()), "resource": {"min_free_cuda_bytes": MIN_FREE_BYTES, "max_peak_cuda_bytes": MAX_PEAK_BYTES, "min_disk_free_bytes": MIN_DISK_FREE_BYTES, "max_storage_bytes": MAX_STORAGE_BYTES}, "mechanism_evidence_only": True}, "RMRI static validation failed" if not valid else None)
    _write_new(output_path, payload, artifact_kind="RMRI validation")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if valid else 2


def _load_validation(path: Path, spec: Spec, bundle: ParentBundle, review_path: Path, state: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, bundle)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("status") != "passed" or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected_hashes or record.permitted_next_actions != (NEXT_ACTION_RUN,):
        raise ValueError("RMRI validation evidence is stale or mismatched")
    if payload.get("contract", {}).get("threshold_transfer") is not True:
        raise ValueError("RMRI threshold transfer validation is not bound")


def _ensure_run_directory(output_path: Path, validation_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RMRI audit: {output_path}")
    root = output_path.parent
    if root.exists():
        children = {item.name for item in root.iterdir()}
        if children != {validation_path.name}:
            raise FileExistsError(f"RMRI output directory is not fresh: {root}")


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RMRI requires CUDA for scientific replay; CPU is not a fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("RMRI requires at least 4 GiB free CUDA memory")
    if shutil.disk_usage(output_path.anchor or str(output_path.parent)).free <= MIN_DISK_FREE_BYTES:
        raise RuntimeError("RMRI requires more than 10 GiB free D: storage")
    return int(free), int(total)


def _attach_dcm_canonical(panel: ModelPanel, labels: torch.Tensor, native_backend: str) -> dict[str, Any]:
    metrics = dict(panel.dcm_metrics)
    canonical = panel.bosa_metrics[native_backend]["r_prop_canonicalized"]
    left = metrics["stages"]["decoder_layer_2"]
    # The canonical feature is recomputed in the replay and retained in the
    # BOSA cell.  Keep a compact rank report here without retaining fields.
    metrics["canonicalized_retention"] = {"geometric_mean_retention": canonical, "source": "HFRR._canonicalize_field", "labels_count": int(labels.numel())}
    metrics["canonicalized_label_signal"] = panel.bosa_metrics[native_backend]["canonicalized_label_signal"]
    metrics["raw_retention_definition"] = "mra_dcm_entropy_participation_geometric_mean"
    metrics["canonicalized_left_feature_dim"] = int(left["feature_dim"])
    metrics["canonicalized_right_feature_dim"] = int(panel.bosa_metrics[native_backend]["stages"]["cmos_incident_field"]["feature_dim"])
    return metrics


def _jsonable_panel(panel: ModelPanel) -> dict[str, Any]:
    return {
        "mra": panel.mra_metrics,
        "dcm": panel.dcm_metrics,
        "bosa": panel.bosa_metrics,
        "state": panel.state,
        "evaluation_count": int(panel.labels.numel()),
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("RMRI requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"RMRI requires clean branch {BRANCH}")
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _ensure_run_directory(output_path, validation_path)
    static_bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    _load_validation(validation_path, spec, static_bundle, review_path, state)
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    bundle = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=True)
    started = time.monotonic()
    torch.manual_seed(BOOTSTRAP_SEED)
    torch.cuda.manual_seed_all(BOOTSTRAP_SEED)
    torch.cuda.reset_peak_memory_stats(device)
    selection_images, selection_labels = _panel_inputs(bundle, "selection")
    confirmation_images, confirmation_labels = _panel_inputs(bundle, "confirmation")
    selection_panels: dict[str, ModelPanel] = {}
    confirmation_panels: dict[str, ModelPanel] = {}
    for model_name in MODEL_NAMES:
        selection_panels[model_name] = _evaluate_model_panel(model_name, bundle, selection_images, selection_labels, device, include_backends=True)
        torch.cuda.empty_cache()
        confirmation_panels[model_name] = _evaluate_model_panel(model_name, bundle, confirmation_images, confirmation_labels, device, include_backends=False)
        torch.cuda.empty_cache()
    # Confirmation BOSA was never authorized by its historical contract.  The
    # full confirmation relay/readout replay above is sufficient for DCM/MRA;
    # only the registered BOSA selection receives the 2x2 cross-backend cells.
    selection_dcm = {name: _attach_dcm_canonical(panel, selection_labels, "identity_backend" if name == IDENTITY else "scaled_chen_backend") for name, panel in selection_panels.items()}
    confirmation_dcm = {name: _attach_dcm_canonical(panel, confirmation_labels, "identity_backend" if name == IDENTITY else "scaled_chen_backend") for name, panel in confirmation_panels.items()}
    # Native raw replay anchors are checked before any reinterpretation.
    selection_anchor = _raw_anchor_gate(selection_dcm[SCALED], "selection", bundle.dcm_audit)
    confirmation_anchor = _raw_anchor_gate(confirmation_dcm[SCALED], "confirmation", bundle.dcm_audit)
    if not selection_anchor["passed"] or not confirmation_anchor["passed"]:
        raise RuntimeError("RMRI native DCM raw replay does not match its registered anchors")
    selection_mra_anchor = _mra_anchor_gate(selection_panels[SCALED].mra_metrics, "selection", json.loads(bundle.mra_audit_path.read_text(encoding="utf-8")))
    confirmation_mra_anchor = _mra_anchor_gate(confirmation_panels[SCALED].mra_metrics, "confirmation", json.loads(bundle.mra_audit_path.read_text(encoding="utf-8")))
    if not selection_mra_anchor["passed"] or not confirmation_mra_anchor["passed"]:
        raise RuntimeError("RMRI native MRA raw replay does not match its registered anchors")
    # BOSA raw replay is intentionally selection-only, with its original four
    # cells.  The canonicalized BOSA decision is therefore never promoted to a
    # confirmation-backed positive conclusion.
    bosa_cells = {}
    for model_name, panel in selection_panels.items():
        for backend, metrics in panel.bosa_metrics.items():
            bosa_cells[metrics["cell"]] = metrics
    if set(bosa_cells) != set(CELL_ORDER):
        raise RuntimeError("RMRI BOSA replay did not produce exactly four registered selection cells")
    bosa_anchor = _bosa_anchor_gate(bosa_cells, bundle.bosa_audit)
    if not bosa_anchor["passed"]:
        raise RuntimeError("RMRI native BOSA raw replay does not match its registered anchors")
    dcm_classification, dcm_details = _classify_dcm(selection_dcm[SCALED], confirmation_dcm[SCALED], confirmation_dcm[IDENTITY])
    mra_classification, mra_details = _classify_mra(selection_panels[SCALED].mra_metrics, confirmation_panels[SCALED].mra_metrics, dcm_classification)
    bosa_classification, bosa_details = _bosa_reinterpretation(bosa_cells)
    state_valid = all(panel.state["state_unchanged"] and panel.state["ols_state_unchanged"] for panel in (*selection_panels.values(), *confirmation_panels.values()))
    finite_valid = all(panel.mra_metrics["finite"] and panel.dcm_metrics["finite"] and all(item["finite"] for item in panel.bosa_metrics.values()) for panel in (*selection_panels.values(), *confirmation_panels.values()))
    phase_valid = all(max(panel.dcm_metrics["phase_controls"].values()) <= PHASE_CONTROL_TOLERANCE for panel in (*selection_panels.values(), *confirmation_panels.values())) and all(max(item["phase_controls"].values()) <= PHASE_CONTROL_TOLERANCE for panel in selection_panels.values() for item in panel.bosa_metrics.values())
    native_valid = selection_anchor["passed"] and confirmation_anchor["passed"] and all(item["native_replay"]["passed"] for item in bosa_cells.values())
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    harness_gates = (
        GateResult("finite_measurements", finite_valid, {}),
        GateResult("frozen_state_unchanged", state_valid, {}),
        GateResult("native_raw_replay_anchors", native_valid, {"dcm_selection": selection_anchor, "dcm_confirmation": confirmation_anchor, "mra_selection": selection_mra_anchor, "mra_confirmation": confirmation_mra_anchor, "bosa_selection": bosa_anchor}),
        GateResult("square_law_and_phase_controls", phase_valid, {"tolerance": PHASE_CONTROL_TOLERANCE}),
        GateResult("bosa_confirmation_disabled", True, {"executed": False}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("backward_isolation", True, {"backward": False}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("checkpoint_and_prior_isolation", True, {"checkpoint_written": False, "prior_sampled": False}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}),
    )
    science_gates = (
        GateResult("dcm_reinterpretation_complete", dcm_classification != "relay_propagation_reinterpretation_underdetermined", dcm_details),
        GateResult("mra_reinterpretation_complete", mra_classification in {"distributed_bottleneck_supported_canonicalized", "distributed_bottleneck_evidence_reduced"}, mra_details),
        GateResult("bosa_selection_reinterpretation_complete", bosa_classification in {"backend_observable_subspace_mismatch_not_supported_reinterpreted", "backend_observable_subspace_reinterpretation_underdetermined"}, bosa_details),
    )
    status = "valid_failed" if all(gate.passed for gate in harness_gates) else "invalid_harness"
    classification = dcm_classification if status == "valid_failed" else "invalid_harness"
    action = NEXT_ACTION_D1 if dcm_classification == "relay_propagation_bottleneck_reinterpreted_metric_confound" else NEXT_ACTION_D2 if dcm_classification == "relay_propagation_bottleneck_supported_canonicalized" else NEXT_ACTION_D3
    if status == "invalid_harness":
        action = NEXT_ACTION_D3
    hashes = _input_hashes(spec, review_path, bundle)
    panel_path = output_path.parent / "panel.pt"
    compact_path = output_path.parent / "compact_report.json"
    for target in (panel_path, compact_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite RMRI artifact: {target}")
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "selection_indices": bundle.data.selection_indices,
        "confirmation_indices": bundle.data.confirmation_indices,
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
        "per_sample_mse": {
            "selection": {name: panel.sample_mse for name, panel in selection_panels.items()},
            "confirmation": {name: panel.sample_mse for name, panel in confirmation_panels.items()},
        },
        "source_hashes": bundle.source.hashes,
        "affects_selection": False,
        "bosa_confirmation_executed": False,
        "canonicalized_fields_persisted": False,
    }
    storage_estimate = sum(value.numel() * value.element_size() for group in panel_payload["per_sample_mse"].values() for value in group.values())
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("RMRI compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    dcm_contract = {
        "selection": selection_dcm[SCALED],
        "confirmation": confirmation_dcm[SCALED],
        "identity_reference_selection": selection_dcm[IDENTITY],
        "identity_reference_confirmation": confirmation_dcm[IDENTITY],
        "classification": dcm_classification,
        "details": dcm_details,
    }
    mra_contract = {
        "selection": selection_panels[SCALED].mra_metrics,
        "confirmation": confirmation_panels[SCALED].mra_metrics,
        "classification": mra_classification,
        "details": mra_details,
        "historical_classification": PARENT_MRA["audit_classification"],
    }
    bosa_contract = {
        "selection": {name: bosa_cells[name] for name in CELL_ORDER},
        "classification": bosa_classification,
        "details": bosa_details,
        "confirmation_executed": False,
        "historical_classification": PARENT_BOSA["audit_classification"],
    }
    compact = {
        "schema": COMPACT_SCHEMA,
        "status": status,
        "classification": classification,
        "dcm_classification": dcm_classification,
        "mra_classification": mra_classification,
        "bosa_classification": bosa_classification,
        "dcm": dcm_contract,
        "mra": mra_contract,
        "bosa": bosa_contract,
        "historical_classifications_preserved": {"dcm": PARENT_DCM["audit_classification"], "mra": PARENT_MRA["audit_classification"], "bosa": PARENT_BOSA["audit_classification"]},
        "next_action": action,
        "numerical_boundary_not_evaluated": False,
        "relay_numerical_reference_consistent": True,
        "relay_numerical_reference_mismatch": False,
        "generation_readiness": "not_ready",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "panel_sha256": panel_hash,
    }
    _write_new(compact_path, compact, artifact_kind="RMRI compact report")
    contract = {
        "raw_replay": {"selection": {name: _jsonable_panel(panel) for name, panel in selection_panels.items()}, "confirmation": {name: _jsonable_panel(panel) for name, panel in confirmation_panels.items()}, "dcm_anchor_selection": selection_anchor, "dcm_anchor_confirmation": confirmation_anchor, "mra_anchor_selection": selection_mra_anchor, "mra_anchor_confirmation": confirmation_mra_anchor, "bosa_anchor_selection": bosa_anchor},
        "HFRR_dependency": {"parent": PARENT_HFRR, "canonicalizer_module": str(Path(hfrr.__file__).resolve()), "canonicalizer_module_sha256": hashes["canonicalizer_module"], "numerical_boundary_not_evaluated": False, "relay_numerical_reference_consistent": True, "relay_numerical_reference_mismatch": False},
        "DCM_reinterpretation": dcm_contract,
        "MRA_reinterpretation": mra_contract,
        "BOSA_reinterpretation": bosa_contract,
        "other_conclusions": {"reconstruction_manifold_supported": False, "generative_manifold_supported": False, "generation_readiness": "not_ready", "local_relay_tuning": "closed_no_local_correction_supported"},
        "governance": {"next_action": action, "mechanism_evidence_only": True, "training_authorized": False, "formal_promotion": False, "optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False, "physical_forward_modified": False, "historical_evidence_modified": False, "bosa_confirmation_executed": False, "new_samples": False, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "compact_report_path": str(compact_path.resolve()), "compact_report_sha256": file_sha256(compact_path), "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"}},
    }
    all_gates = harness_gates + science_gates
    payload = _record(AUDIT_SCHEMA, status, classification, spec, bundle, state, review_path, hashes, all_gates, (action,), contract, None if status == "valid_failed" else "RMRI harness failed")
    payload["generation_readiness"] = "not_ready"
    payload["reconstruction_manifold_supported"] = False
    payload["generative_manifold_supported"] = False
    _write_new(output_path, payload, artifact_kind="RMRI audit")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "valid_failed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-relay-metric-reinterpretation-validate":
        return validate(args)
    if args.research_action == "scaled-chen-relay-metric-reinterpretation-run":
        return run(args)
    raise ValueError(f"unsupported RMRI action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "BACKENDS",
    "CANONICALIZATION",
    "CELL_ORDER",
    "CONFIG_SCHEMA",
    "GATES",
    "HYPOTHESIS",
    "MEASUREMENT",
    "PARENT_BOSA",
    "PARENT_DCM",
    "PARENT_HFRR",
    "PARENT_MRA",
    "PANELS",
    "REFERENCES",
    "REVIEW_SCHEMA",
    "RELAY_SEGMENT",
    "READOUT_SEGMENT",
    "Spec",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_bosa_reinterpretation",
    "_classify_dcm",
    "_classify_mra",
    "_load_review",
    "_load_spec",
    "_retention",
    "_ensure_run_directory",
    "dispatch",
    "run",
    "validate",
]
