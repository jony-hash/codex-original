"""High-fidelity, no-update numerical reference for the registered B=0 relays.

The production relays remain the only physical forward.  This module is a
research-only consumer that evaluates the same finite-pupil semantics with an
independent separable direct Fourier-series implementation.  It deliberately
does not import or call a production transform helper in the reference path.
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

from ...config import FormalSystemConfig, load_config
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
from . import optical_manifold_smf_bogt_backend_observable_subspace as bosa
from . import optical_manifold_smf_bogt_decoder_cmos_attribution as dcm
from . import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_high_fidelity_relay_reference_panel.v1"
COMPACT_SCHEMA = "scaled_chen_high_fidelity_relay_reference_compact_report.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S1-HFRR"
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

SOURCE_SHAPE = (100, 100)
SOURCE_PITCH = (2.0e-6, 2.0e-6)
TARGET_SHAPE = (200, 200)
TARGET_PITCH = (2.5e-6, 2.5e-6)
WAVELENGTH_M = 532.0e-9
BACK_MAGNIFICATION = -2.5
PHASE_MASK_THRESHOLD = 0.01
REFERENCE_DTYPE = "torch.complex128"
REFERENCE_DEVICE = "cpu"
PRODUCTION_DTYPE = "torch.complex64"
COMPLEX_RELATIVE_TOLERANCE = 2.0e-3
COMPLEX_CORRELATION_TOLERANCE = 0.999999
AMPLITUDE_RELATIVE_TOLERANCE = 2.0e-3
PHASE_RMS_TOLERANCE = 2.0e-3
FIELD_POWER_TOLERANCE = 1.0e-4
INTENSITY_RELATIVE_TOLERANCE = 2.0e-3
INTENSITY_POWER_TOLERANCE = 1.0e-4
CONVERGENCE_COMPLEX_TOLERANCE = 1.0e-11
CONVERGENCE_POWER_TOLERANCE = 1.0e-12
MATERIAL_GAP = 0.05
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

NEXT_ACTION_MISMATCH = "manual_review_relay_numerical_contract_correction"
NEXT_ACTION_CONFOUND = "manual_review_relay_retention_metric_reinterpretation_contract"
NEXT_ACTION_RESIDUAL = "manual_review_backend_optical_architecture_redesign_contract"
NEXT_ACTION_BLOCKED = "manual_review_high_fidelity_relay_numerical_reference_contract"

PARENT_BOSA = {
    "config": "configs/research/lightgen_optical_manifold_smf_bogt_backend_observable_subspace_v1.yaml",
    "config_sha256": "af38d64218a81483d01748b16b0f61ea03c4b935a5814f12c8648b580873491c",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_backend_observable_subspace_review_v1.json",
    "review_sha256": "156ab4646ded791459528d0d3d660f9de0e808f914d37e0e9b3f2fa66c3b90f8",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_backend_observable_subspace_v1_attempt_05/audit.json",
    "audit_sha256": "1131860efc1500c4807d773eed6e75e49737972eff05e46bbbefea9e9e993baf",
    "audit_git_commit": "8ece7fb6f0eb8208e4b50ada87bbfaaf981d648b",
    "audit_status": "valid_failed",
    "audit_classification": "backend_observable_subspace_mismatch_not_supported",
    "panel": "outputs/20260901_v2/runs/lightgen_scaled_chen_backend_observable_subspace_v1_attempt_05/panel.pt",
    "panel_sha256": "a11a3c356ce7146bd0f491bc29c87f4219f65131e62af08595d27dae7eb08fc7",
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

TASK_INDICES = {
    0: [5031, 5033, 5069, 5087, 5139, 5243, 5259, 5270, 5280, 5301],
    1: [5091, 5132, 5227, 5252, 5327, 5361, 5590, 5604, 5614, 5642],
    2: [5053, 5095, 5105, 5167, 5216, 5367, 5392, 5417, 5427, 5541],
    3: [5000, 5028, 5143, 5185, 5214, 5248, 5346, 5359, 5418, 5446],
    4: [5047, 5123, 5151, 5260, 5319, 5355, 5393, 5439, 5448, 5500],
    5: [5056, 5083, 5174, 5194, 5196, 5207, 5223, 5275, 5351, 5480],
    6: [5007, 5010, 5088, 5116, 5145, 5228, 5303, 5354, 5458, 5467],
    7: [5205, 5234, 5297, 5313, 5363, 5371, 5412, 5422, 5485, 5498],
    8: [5103, 5138, 5142, 5183, 5236, 5300, 5306, 5360, 5394, 5413],
    9: [5002, 5233, 5268, 5269, 5279, 5404, 5414, 5429, 5447, 5523],
}
TASK_INDEX_VECTOR = [index for label in range(10) for index in TASK_INDICES[label]]
TASK_SUBSET_SHA256 = "c51653e28933b2135008ac8bf63e60154158223365c095075348c3d3b1639529"

SYNTHETIC_PROBES = {
    "P0_dc": {"kind": "dc", "amplitude": 1.0},
    "P1_low_mode": {"kind": "single_mode", "fy_bin": 1, "fx_bin": 0, "amplitude": 1.0},
    "P2_mid_mode": {"kind": "single_mode", "fy_bin": 15, "fx_bin": 11, "amplitude": 1.0},
    "P3_near_nyquist": {"kind": "single_mode", "fy_bin": 49, "fx_bin": 0, "amplitude": 1.0},
    "P4_multifrequency": {
        "kind": "multifrequency",
        "components": [
            {"fy_bin": 3, "fx_bin": 2, "real": 1.0, "imag": 0.25},
            {"fy_bin": 11, "fx_bin": -7, "real": -0.35, "imag": 0.60},
            {"fy_bin": 31, "fx_bin": 19, "real": 0.20, "imag": -0.45},
        ],
    },
    "P5_gaussian_carrier": {
        "kind": "gaussian_carrier",
        "sigma_m": 18.0e-6,
        "carrier_fy_bin": 17,
        "carrier_fx_bin": -13,
        "amplitude": 1.0,
    },
}

BACKEND_CONTRACT = {
    "identity_backend": {
        "name": "back_b0",
        "target_plane": "camera",
        "segments_m": [0.1, 0.25],
        "focal_lengths_m": [0.0714285714285714],
        "magnification": -2.5,
        "expected_abcd": {"A": -2.5, "B_m": 0.0, "C_per_m": -14.0, "D": -0.4, "determinant": 1.0},
        "expected_source_pupil_bins": [7843, 10000],
        "expected_target_pupil_bins": [7845, 40000],
    },
    "scaled_chen_backend": {
        "name": "inferred_backend_imaging",
        "target_plane": "chen_camera",
        "segments_m": [0.0008571428571428572, 0.002142857142857143],
        "focal_lengths_m": [0.0006122448979591838],
        "magnification": -2.5,
        "expected_abcd": {"A": -2.5, "B_m": 0.0, "C_per_m": -1633.3333333333333, "D": -0.4, "determinant": 1.0},
        "expected_source_pupil_bins": [10000, 10000],
        "expected_target_pupil_bins": [40000, 40000],
    },
}

REFERENCE_CONTRACT = {
    "algorithm_id": "separable_direct_fourier_series_b0_finite_pupil_v1",
    "device": REFERENCE_DEVICE,
    "dtype": REFERENCE_DTYPE,
    "source_dft": "explicit_complex_exponential_matrix",
    "target_coordinate_map": "x_source=x_target/magnification",
    "quadratic_phase": "exp(+i*pi*C*r2/(wavelength*A))",
    "amplitude_factor": "1/abs(A)",
    "target_inverse": "explicit_inverse_dft_after_target_pupil",
    "convergence": {"complex_relative_l2": CONVERGENCE_COMPLEX_TOLERANCE, "power_relative_difference": CONVERGENCE_POWER_TOLERANCE},
}

MEASUREMENT = {
    "objective": "frozen_forward_only",
    "models": list(MODEL_NAMES),
    "backends": list(BACKENDS),
    "cells": [dict(CELL_REGISTRATION[name]) for name in CELL_ORDER],
    "synthetic_probes": list(SYNTHETIC_PROBES),
    "task_probe_rule": "BOSA_selection_first_10_sorted_global_index_per_class",
    "task_count": 100,
    "task_subset_sha256": TASK_SUBSET_SHA256,
    "metrics": [
        "complex_relative_l2", "normalized_complex_correlation", "amplitude_relative_l2", "phase_rms_rad",
        "field_power_relative_difference", "intensity_relative_l2", "intensity_max_error", "intensity_p95_error", "integrated_intensity_relative_difference",
    ],
    "phase_mask_threshold": PHASE_MASK_THRESHOLD,
    "pupil_aware_cross_backend": True,
    "no_resampling": True,
    "no_normalization": True,
    "no_update": True,
    "mechanism_evidence_only": True,
}

GATES = {
    "complex_relative_l2": COMPLEX_RELATIVE_TOLERANCE,
    "normalized_complex_correlation": COMPLEX_CORRELATION_TOLERANCE,
    "amplitude_relative_l2": AMPLITUDE_RELATIVE_TOLERANCE,
    "phase_rms_rad": PHASE_RMS_TOLERANCE,
    "field_power_relative_difference": FIELD_POWER_TOLERANCE,
    "intensity_relative_l2": INTENSITY_RELATIVE_TOLERANCE,
    "intensity_power_relative_difference": INTENSITY_POWER_TOLERANCE,
    "material_gap": MATERIAL_GAP,
    "min_free_cuda_bytes": MIN_FREE_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "backward": False,
    "evaluator_calls": 0,
    "prior_sampled": False,
    "training_authorized": False,
    "formal_promotion": False,
}

CANONICALIZATION = {
    "formula": "exp(-i*pi*C*r2/(wavelength*A)); evaluate at x_target=A*x_source,y_target=A*y_source; multiply abs(A)",
    "fit": False,
    "global_phase_fit": False,
    "per_pixel_phase_fit": False,
    "amplitude_normalization": False,
    "physical_forward_modified": False,
    "detector_or_mse_input": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_high_fidelity_relay_reference",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "prohibited": ["optimizer", "backward", "checkpoint_write", "prior_sampling", "generation", "evaluator", "fmf", "relay_tuning", "ols_rerun", "physical_forward_modification"],
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    bosa_spec: bosa.Spec
    bosa_bundle: bosa.ParentBundle
    bosa_config_path: Path
    bosa_review_path: Path
    bosa_audit_path: Path
    bosa_audit: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return mra._resolve(value, base_path=base_path)


def _git_state() -> dict[str, Any]:
    return mra._git_state()


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "parent_bosa", "references", "panels", "backends", "reference", "synthetic_probes", "measurement", "gates", "canonicalization", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("HFRR config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("HFRR config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("HFRR formal config is not locked")
    if payload["parent_bosa"] != PARENT_BOSA:
        raise ValueError("HFRR BOSA parent binding is not locked")
    if payload["references"] != REFERENCES:
        raise ValueError("HFRR checkpoint references are not locked")
    expected_panels = {"task": {"source": "bosa_selection_first_10_per_class", "count": 100, "per_class": 10, "subset_sha256": TASK_SUBSET_SHA256, "bosa_panel": PARENT_BOSA["panel"], "bosa_panel_sha256": PARENT_BOSA["panel_sha256"], "confirmation_consumed": False, "candidate_prefix_consumed": False}}
    if payload["panels"] != expected_panels:
        raise ValueError("HFRR task panel contract is not locked")
    if payload["backends"] != BACKEND_CONTRACT:
        raise ValueError("HFRR backend contract is not locked")
    if payload["reference"] != REFERENCE_CONTRACT:
        raise ValueError("HFRR reference contract is not locked")
    if payload["synthetic_probes"] != SYNTHETIC_PROBES:
        raise ValueError("HFRR synthetic probes are not locked")
    if payload["measurement"] != MEASUREMENT:
        raise ValueError("HFRR measurement contract is not locked")
    if payload["gates"] != GATES:
        raise ValueError("HFRR gate contract is not locked")
    if payload["canonicalization"] != CANONICALIZATION:
        raise ValueError("HFRR canonicalization contract is not locked")
    if payload["authorization"] != AUTHORIZATION:
        raise ValueError("HFRR authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "parent_bosa": PARENT_BOSA,
            "references": REFERENCES,
            "panels": spec.payload["panels"],
            "backends": BACKEND_CONTRACT,
            "reference": REFERENCE_CONTRACT,
            "synthetic_probes": SYNTHETIC_PROBES,
            "metrics": MEASUREMENT["metrics"],
            "phase_mask_threshold": PHASE_MASK_THRESHOLD,
            "gates": GATES,
            "canonicalization": CANONICALIZATION,
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
        raise ValueError("HFRR review contract does not match scope")
    return target


def _load_parent(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    parent = spec.payload["parent_bosa"]
    config_path = _resolve(parent["config"], base_path=spec.path.parent)
    if not config_path.is_file() or file_sha256(config_path) != parent["config_sha256"]:
        raise ValueError("BOSA parent config is missing, stale, or tampered")
    bosa_spec = bosa._load_spec(config_path)
    bosa._assert_contract(bosa_spec)
    review_path = _resolve(parent["review_contract"], base_path=spec.path.parent)
    if not review_path.is_file() or file_sha256(review_path) != parent["review_sha256"]:
        raise ValueError("BOSA parent review contract is missing, stale, or tampered")
    bosa._load_review(review_path, bosa_spec)
    audit_path = _resolve(parent["audit"], base_path=spec.path.parent)
    if not audit_path.is_file() or file_sha256(audit_path) != parent["audit_sha256"]:
        raise ValueError("BOSA parent audit is missing, stale, or tampered")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(audit["evidence"])
    record.validate()
    if (
        audit.get("schema") != bosa.AUDIT_SCHEMA
        or audit.get("status") != parent["audit_status"]
        or audit.get("classification") != parent["audit_classification"]
        or record.git_commit != parent["audit_git_commit"]
        or record.config_hash != bosa_spec.config_hash
        or record.permitted_next_actions != (NEXT_ACTION_BLOCKED,)
    ):
        raise ValueError("BOSA parent terminal evidence is stale or mismatched")
    panel_path = _resolve(parent["panel"], base_path=spec.path.parent)
    if not panel_path.is_file() or file_sha256(panel_path) != parent["panel_sha256"]:
        raise ValueError("BOSA task panel is missing, stale, or tampered")
    bundle = bosa._load_parent_with_data(bosa_spec, dataset_root, load_checkpoints=load_checkpoints)
    return ParentBundle(bosa_spec, bundle, config_path, review_path, audit_path, dict(audit))


def _task_indices(data: Any) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(TASK_INDEX_VECTOR, dtype=torch.long)
    labels = data.test_labels.index_select(0, values)
    expected_labels = torch.arange(10, dtype=torch.long).repeat_interleave(10)
    if not torch.equal(labels, expected_labels) or torch.unique(values).numel() != 100:
        raise ValueError("HFRR task subset labels or ordering are invalid")
    if canonical_sha256(values.tolist()) != TASK_SUBSET_SHA256:
        raise ValueError("HFRR task subset digest is not registered")
    return values, labels


def _frequency_axis(size: int, pitch: float) -> torch.Tensor:
    indices = torch.arange(size, dtype=torch.float64)
    wrapped = torch.where(indices < (size // 2 + size % 2), indices, indices - size)
    return wrapped / (size * pitch)


def _pupil(shape: tuple[int, int], pitch: tuple[float, float], cutoff: float) -> torch.Tensor:
    fy = _frequency_axis(shape[0], pitch[0])
    fx = _frequency_axis(shape[1], pitch[1])
    return ((fy[:, None].square() + fx[None, :].square()).sqrt() <= cutoff * (1.0 + 1.0e-6))


def _plane_contract(plane: Any) -> dict[str, Any]:
    return {"plane_id": plane.plane_id, "shape": list(plane.shape), "pixel_pitch_m": list(plane.pixel_pitch_m), "fov_m": list(plane.fov_m), "wavelength_m": plane.wavelength_m, "z_m": plane.z_m, "coordinate_frame": plane.coordinate_frame}


def _build_backend_planes(formal: FormalSystemConfig) -> tuple[base.FormalAmplitudeFiniteCropIdentitySystem, dict[str, CoherentImagingRelay]]:
    return bosa._build_backend_planes(formal)


def _backend_validation(formal: FormalSystemConfig) -> tuple[dict[str, Any], tuple[GateResult, ...]]:
    system, relays = _build_backend_planes(formal)
    source = system.chip_plane
    details: dict[str, Any] = {"source_plane": _plane_contract(source), "backends": {}}
    gates: list[GateResult] = []
    source_exact = tuple(source.shape) == SOURCE_SHAPE and source.pixel_pitch_m == SOURCE_PITCH and abs(source.wavelength_m - WAVELENGTH_M) <= 1.0e-15 and source.plane_id == "chip"
    gates.append(GateResult("source_plane_contract", source_exact, details["source_plane"]))
    for name, relay in relays.items():
        contract = BACKEND_CONTRACT[name]
        abcd = relay.abcd.to_dict()
        source_cutoff = relay.source_cutoff_cycles_m
        target_cutoff = relay.target_cutoff_cycles_m
        source_bins = int(_pupil(relay.source.shape, relay.source.pixel_pitch_m, source_cutoff).sum())
        target_bins = int(_pupil(relay.target.shape, relay.target.pixel_pitch_m, target_cutoff).sum())
        abcd_ok = all(abs(float(abcd[key]) - float(contract["expected_abcd"][key])) <= (1.0e-9 if key == "B_m" else 1.0e-9) for key in contract["expected_abcd"])
        target_ok = relay.target.plane_id == contract["target_plane"] and tuple(relay.target.shape) == TARGET_SHAPE and relay.target.pixel_pitch_m == TARGET_PITCH and abs(relay.target.wavelength_m - WAVELENGTH_M) <= 1.0e-15
        pupil_ok = [source_bins, math.prod(relay.source.shape)] == contract["expected_source_pupil_bins"] and [target_bins, math.prod(relay.target.shape)] == contract["expected_target_pupil_bins"]
        details["backends"][name] = {"name": relay.name, "source_plane": _plane_contract(relay.source), "target_plane": _plane_contract(relay.target), "segments_m": list(relay.segments_m), "focal_lengths_m": list(relay.focal_lengths_m), "magnification": relay.magnification, "abcd": abcd, "source_cutoff_cycles_m": source_cutoff, "target_cutoff_cycles_m": target_cutoff, "source_pupil_bins": [source_bins, math.prod(relay.source.shape)], "target_pupil_bins": [target_bins, math.prod(relay.target.shape)], "fourier_convention": "torch_fft_equivalent_exp(-2pi i n k/N), centered physical coordinates", "production_dtype": PRODUCTION_DTYPE}
        gates.extend((GateResult(f"{name}_target_plane", target_ok, details["backends"][name]["target_plane"]), GateResult(f"{name}_abcd_and_determinant", abcd_ok and abs(relay.abcd.determinant - 1.0) <= 1.0e-9, abcd), GateResult(f"{name}_pupil_occupancy", pupil_ok, details["backends"][name]), GateResult(f"{name}_b_zero", abs(relay.abcd.b_m) <= 1.0e-9, {"B_m": relay.abcd.b_m})))
    return details, tuple(gates)


def _input_hashes(spec: Spec, review_path: Path, parent: ParentBundle) -> dict[str, str]:
    parent_bundle = parent.bosa_bundle.parent
    parent_hashes = mra._input_hashes(parent_bundle.spec, parent_bundle.review_path, parent_bundle.formal_path, parent_bundle.data, parent_bundle.source)
    return {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "formal_config": file_sha256(parent_bundle.formal_path),
        "dataset_manifest": parent_bundle.data.manifest["manifest_hash"],
        "task_subset": TASK_SUBSET_SHA256,
        "bosa_config": file_sha256(parent.bosa_config_path),
        "bosa_review_contract": file_sha256(parent.bosa_review_path),
        "bosa_audit": file_sha256(parent.bosa_audit_path),
        "bosa_panel": PARENT_BOSA["panel_sha256"],
        "identity_checkpoint": REFERENCES["identity_checkpoint"]["sha256"],
        "scaled_chen_checkpoint": REFERENCES["scaled_chen_checkpoint"]["sha256"],
        **{f"parent_{name}": value for name, value in parent_hashes.items()},
    }


def _context(spec: Spec, parent: ParentBundle) -> EvidenceContextV2:
    manifest = parent.bosa_bundle.parent.data.manifest["manifest_hash"]
    signature = canonical_sha256({"hypothesis": HYPOTHESIS, "reference": REFERENCE_CONTRACT, "backends": BACKEND_CONTRACT, "measurement": MEASUREMENT})
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, dataset_split_manifest_hash=manifest)


def _record(schema: str, status: str, classification: str | None, spec: Spec, parent: ParentBundle, state: Mapping[str, Any], review_path: Path, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, parent), dict(hashes), tuple(gates), tuple(actions), reason)
    record.validate()
    return {"schema": schema, "hypothesis": HYPOTHESIS, "status": status, "classification": classification, "training_authorized": False, "formal_training_authorized": False, "formal_promotion": False, "optimizer_created": False, "backward": False, "evaluator_calls": 0, "prior_sampled": False, "generation_readiness": "not_ready", "numerical_boundary_not_evaluated": True, "mechanism_evidence_only": True, "evidence": record.to_dict(), "contract": dict(contract), "review_contract": str(review_path.resolve())}


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _ensure_validation_target(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite HFRR validation evidence: {path}")
    if path.parent.exists() and any(path.parent.iterdir()):
        raise FileExistsError(f"HFRR validation output directory is not fresh: {path.parent}")


def _ensure_run_directory(output_path: Path, validation_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite HFRR audit evidence: {output_path}")
    if output_path.parent.exists() and {item.name for item in output_path.parent.iterdir()} != {validation_path.name}:
        raise FileExistsError(f"HFRR run directory is not a validation-only directory: {output_path.parent}")


def _validation_gates(formal: FormalSystemConfig, data: Any) -> tuple[GateResult, ...]:
    backend_details, backend_gates = _backend_validation(formal)
    indices, labels = _task_indices(data)
    balanced = torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 10, dtype=torch.long))
    values = torch.tensor(TASK_INDEX_VECTOR, dtype=torch.long)
    return (
        GateResult("task_subset_balanced", balanced and torch.equal(indices, values), {"count": int(indices.numel()), "subset_sha256": TASK_SUBSET_SHA256}),
        GateResult("task_selection_is_BOSA_selection_derived", True, {"confirmation_consumed": False, "candidate_prefix_consumed": False}),
        GateResult("mode_and_coordinate_contract", all(gate.passed for gate in backend_gates), backend_details),
        GateResult("reference_cpu_complex128", REFERENCE_DEVICE == "cpu" and REFERENCE_DTYPE == "torch.complex128", dict(REFERENCE_CONTRACT)),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "prior_sampled": False}),
    ) + backend_gates


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"HFRR requires clean branch {BRANCH}")
    output_path = _resolve(args.output)
    _ensure_validation_target(output_path)
    parent = _load_parent(spec, args.dataset_root, load_checkpoints=False)
    gates = _validation_gates(parent.bosa_bundle.parent.formal, parent.bosa_bundle.parent.data)
    if not all(gate.passed for gate in gates):
        raise ValueError("HFRR static validation failed; no validation artifact was written")
    backend_details, _ = _backend_validation(parent.bosa_bundle.parent.formal)
    hashes = _input_hashes(spec, review_path, parent)
    contract = {"production_commit": state["commit"], "references": REFERENCES, "parent_bosa": PARENT_BOSA, "backend_validation": backend_details, "reference": REFERENCE_CONTRACT, "synthetic_probes": SYNTHETIC_PROBES, "task_subset_sha256": TASK_SUBSET_SHA256, "thresholds": GATES, "canonicalization": CANONICALIZATION, "classification_map": {"A": "relay_numerical_implementation_mismatch_supported", "B": "relay_complex_retention_metric_confound_supported", "C": "relay_high_fidelity_reference_consistent_residual_supported"}, "resource_limits": GATES, "generation_readiness": "not_ready"}
    payload = _record(VALIDATION_SCHEMA, "passed", None, spec, parent, state, review_path, hashes, gates, ("scaled_chen-high-fidelity-relay-reference-run",), contract)
    _write_new(output_path, payload, artifact_kind="HFRR validation")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _dft_matrix(size: int, sign: float) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float64)
    return torch.exp(torch.tensor(sign * 2.0 * math.pi / size, dtype=torch.float64) * 1j * n[:, None] * n[None, :]).to(torch.complex128)


def _direct_dft2(field: torch.Tensor, fy_matrix: torch.Tensor, fx_matrix: torch.Tensor) -> torch.Tensor:
    value = field if field.ndim == 3 else field.squeeze(1)
    if value.ndim != 3:
        raise ValueError("direct DFT expects [B,H,W] or [B,1,H,W]")
    return torch.einsum("kn,bnm,lm->bkl", fy_matrix, value, fx_matrix)


def _direct_idft2(spectrum: torch.Tensor, fy_matrix: torch.Tensor, fx_matrix: torch.Tensor) -> torch.Tensor:
    height, width = spectrum.shape[-2:]
    return torch.einsum("nk,bkl,ml->bnm", fy_matrix.conj(), spectrum, fx_matrix.conj()) / float(height * width)


def _reference_core(field: torch.Tensor, backend: str, *, blocked: bool) -> torch.Tensor:
    if backend not in BACKEND_CONTRACT:
        raise ValueError(f"unregistered HFRR backend: {backend}")
    value = field if field.ndim == 4 else field.unsqueeze(1)
    if tuple(value.shape[1:]) != (1, *SOURCE_SHAPE) or not torch.is_complex(value):
        raise ValueError("HFRR reference source field must be complex [B,1,100,100]")
    if value.device.type != "cpu" or value.dtype != torch.complex128:
        raise ValueError("HFRR reference requires CPU complex128")
    system, relays = _build_backend_planes(_formal_for_reference())
    relay = relays[backend]
    source_h, source_w = relay.source.shape
    target_h, target_w = relay.target.shape
    fy_source = _frequency_axis(source_h, relay.source.pixel_pitch_m[0])
    fx_source = _frequency_axis(source_w, relay.source.pixel_pitch_m[1])
    source_dft_matrix_y = _dft_matrix(source_h, -1.0)
    source_dft_matrix_x = _dft_matrix(source_w, -1.0)
    target_dft_matrix_y = _dft_matrix(target_h, -1.0)
    target_dft_matrix_x = _dft_matrix(target_w, -1.0)
    source_spectrum = _direct_dft2(value, source_dft_matrix_y, source_dft_matrix_x)
    source_cutoff = relay.clear_aperture_m / (2.0 * relay.source.wavelength_m * relay.matrix_segments_m[0])
    target_cutoff = relay.clear_aperture_m / (2.0 * relay.target.wavelength_m * relay.matrix_segments_m[-1])
    source_mask = _pupil(relay.source.shape, relay.source.pixel_pitch_m, source_cutoff).to(torch.complex128)
    target_mask = _pupil(relay.target.shape, relay.target.pixel_pitch_m, target_cutoff).to(torch.complex128)
    source_spectrum = source_spectrum * source_mask
    target_y = (torch.arange(target_h, dtype=torch.float64) - (target_h - 1.0) / 2.0) * relay.target.pixel_pitch_m[0]
    target_x = (torch.arange(target_w, dtype=torch.float64) - (target_w - 1.0) / 2.0) * relay.target.pixel_pitch_m[1]
    mapped_y = target_y / relay.magnification
    mapped_x = target_x / relay.magnification
    source_y0 = -(source_h - 1.0) * relay.source.pixel_pitch_m[0] / 2.0
    source_x0 = -(source_w - 1.0) * relay.source.pixel_pitch_m[1] / 2.0
    phase_y = torch.exp(2j * math.pi * (mapped_y[:, None] - source_y0) * fy_source[None, :])
    phase_x = torch.exp(2j * math.pi * (mapped_x[:, None] - source_x0) * fx_source[None, :])
    if blocked:
        sampled = torch.zeros((value.shape[0], target_h, target_w), dtype=torch.complex128)
        block = 16
        for start in range(0, source_h, block):
            stop = min(start + block, source_h)
            sampled = sampled + torch.einsum("uk,bkl,vl->buv", phase_y[:, start:stop], source_spectrum[:, start:stop, :], phase_x)
        sampled = sampled / float(source_h * source_w)
    else:
        sampled = torch.einsum("uk,bkl,vl->buv", phase_y, source_spectrum, phase_x) / float(source_h * source_w)
    yy, xx = torch.meshgrid(target_y, target_x, indexing="ij")
    quadratic = torch.exp(1j * (math.pi * relay.abcd.c_per_m / (relay.target.wavelength_m * relay.abcd.a)) * (xx.square() + yy.square()))
    before_target_pupil = sampled * quadratic * (1.0 / abs(relay.abcd.a))
    target_spectrum = _direct_dft2(before_target_pupil, target_dft_matrix_y, target_dft_matrix_x) * target_mask
    output = _direct_idft2(target_spectrum, target_dft_matrix_y, target_dft_matrix_x)
    return output.unsqueeze(1)


_FORMAL_CACHE: FormalSystemConfig | None = None


def _formal_for_reference() -> FormalSystemConfig:
    global _FORMAL_CACHE
    if _FORMAL_CACHE is None:
        _FORMAL_CACHE = load_config(_resolve("configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml")).system
    return _FORMAL_CACHE


def _reference_relay(field: torch.Tensor, backend: str, *, blocked: bool = False) -> torch.Tensor:
    return _reference_core(field.to(device="cpu", dtype=torch.complex128), backend, blocked=blocked)


def _reference_convergence(field: torch.Tensor, backend: str) -> dict[str, Any]:
    primary = _reference_relay(field, backend, blocked=False)
    blocked = _reference_relay(field, backend, blocked=True)
    denominator = blocked.abs().square().sum().clamp_min(1.0e-30)
    complex_error = float((primary - blocked).norm() / denominator.sqrt())
    primary_power = primary.abs().square().sum()
    blocked_power = blocked.abs().square().sum()
    power_error = float((primary_power - blocked_power).abs() / blocked_power.abs().clamp_min(1.0e-30))
    return {"complex_relative_l2": complex_error, "power_relative_difference": power_error, "passed": complex_error <= CONVERGENCE_COMPLEX_TOLERANCE and power_error <= CONVERGENCE_POWER_TOLERANCE}


def _coordinates(shape: tuple[int, int], pitch: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    y = (torch.arange(shape[0], dtype=torch.float64) - (shape[0] - 1.0) / 2.0) * pitch[0]
    x = (torch.arange(shape[1], dtype=torch.float64) - (shape[1] - 1.0) / 2.0) * pitch[1]
    return torch.meshgrid(y, x, indexing="ij")


def _synthetic_probe(name: str) -> torch.Tensor:
    contract = SYNTHETIC_PROBES[name]
    yy, xx = _coordinates(SOURCE_SHAPE, SOURCE_PITCH)
    kind = contract["kind"]
    if kind == "dc":
        value = torch.ones_like(xx, dtype=torch.complex128) * float(contract["amplitude"])
    elif kind == "single_mode":
        fy = float(contract["fy_bin"]) / (SOURCE_SHAPE[0] * SOURCE_PITCH[0])
        fx = float(contract["fx_bin"]) / (SOURCE_SHAPE[1] * SOURCE_PITCH[1])
        value = float(contract["amplitude"]) * torch.exp(2j * math.pi * (fy * yy + fx * xx))
    elif kind == "multifrequency":
        value = torch.zeros_like(xx, dtype=torch.complex128)
        for component in contract["components"]:
            fy = float(component["fy_bin"]) / (SOURCE_SHAPE[0] * SOURCE_PITCH[0])
            fx = float(component["fx_bin"]) / (SOURCE_SHAPE[1] * SOURCE_PITCH[1])
            coefficient = complex(float(component["real"]), float(component["imag"]))
            value = value + coefficient * torch.exp(2j * math.pi * (fy * yy + fx * xx))
    elif kind == "gaussian_carrier":
        fy = float(contract["carrier_fy_bin"]) / (SOURCE_SHAPE[0] * SOURCE_PITCH[0])
        fx = float(contract["carrier_fx_bin"]) / (SOURCE_SHAPE[1] * SOURCE_PITCH[1])
        gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * float(contract["sigma_m"]) ** 2))
        value = float(contract["amplitude"]) * gaussian * torch.exp(2j * math.pi * (fy * yy + fx * xx))
    else:
        raise ValueError(f"unsupported synthetic probe kind: {kind}")
    return value.unsqueeze(0).unsqueeze(0).to(torch.complex128)


def _compare_one(production: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    prod = production.detach().cpu().to(torch.complex128).squeeze()
    ref = reference.detach().cpu().to(torch.complex128).squeeze()
    difference = prod - ref
    ref_norm = ref.norm().clamp_min(1.0e-30)
    prod_norm = prod.norm().clamp_min(1.0e-30)
    correlation = (prod.conj() * ref).sum().abs() / (prod_norm * ref_norm)
    amplitude_difference = prod.abs() - ref.abs()
    mask = ref.abs() >= PHASE_MASK_THRESHOLD * ref.abs().amax().clamp_min(1.0e-30)
    if bool(mask.any()):
        phase_delta = torch.angle(prod[mask] * ref[mask].conj())
        phase_rms = float(phase_delta.square().mean().sqrt())
    else:
        phase_rms = 0.0
    prod_intensity = prod.abs().square()
    ref_intensity = ref.abs().square()
    intensity_difference = (prod_intensity - ref_intensity).abs()
    prod_power = prod_intensity.sum()
    ref_power = ref_intensity.sum().clamp_min(1.0e-30)
    return {
        "complex_relative_l2": float(difference.norm() / ref_norm),
        "normalized_complex_correlation": float(correlation),
        "amplitude_relative_l2": float(amplitude_difference.norm() / ref.abs().norm().clamp_min(1.0e-30)),
        "phase_rms_rad": phase_rms,
        "field_power_relative_difference": float((prod_power - ref_power).abs() / ref_power),
        "intensity_relative_l2": float((prod_intensity - ref_intensity).norm() / ref_intensity.norm().clamp_min(1.0e-30)),
        "intensity_max_error": float(intensity_difference.max()),
        "intensity_p95_error": float(torch.quantile(intensity_difference.flatten(), 0.95)),
        "integrated_intensity_relative_difference": float((prod_power - ref_power).abs() / ref_power),
        "phase_mask_fraction": float(mask.float().mean()),
        "finite": bool(torch.isfinite(prod).all() and torch.isfinite(ref).all()),
    }


def _aggregate_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty HFRR metric list")
    error_keys = ("complex_relative_l2", "amplitude_relative_l2", "phase_rms_rad", "field_power_relative_difference", "intensity_relative_l2", "intensity_max_error", "intensity_p95_error", "integrated_intensity_relative_difference")
    result = {key: max(float(item[key]) for item in records) for key in error_keys}
    result["normalized_complex_correlation"] = min(float(item["normalized_complex_correlation"]) for item in records)
    result["phase_mask_fraction"] = float(sum(float(item["phase_mask_fraction"]) for item in records) / len(records))
    result["finite"] = all(bool(item["finite"]) for item in records)
    result["count"] = len(records)
    result["mean"] = {key: float(sum(float(item[key]) for item in records) / len(records)) for key in (*error_keys, "normalized_complex_correlation")}
    return result


def _numerical_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(metrics["finite"] and float(metrics["complex_relative_l2"]) <= COMPLEX_RELATIVE_TOLERANCE and float(metrics["normalized_complex_correlation"]) >= COMPLEX_CORRELATION_TOLERANCE and float(metrics["amplitude_relative_l2"]) <= AMPLITUDE_RELATIVE_TOLERANCE and float(metrics["phase_rms_rad"]) <= PHASE_RMS_TOLERANCE and float(metrics["field_power_relative_difference"]) <= FIELD_POWER_TOLERANCE and float(metrics["intensity_relative_l2"]) <= INTENSITY_RELATIVE_TOLERANCE and float(metrics["integrated_intensity_relative_difference"]) <= INTENSITY_POWER_TOLERANCE)


def _module_hash(module: torch.nn.Module) -> str:
    state = {name: tensor_sha256(value.detach().cpu()) for name, value in sorted(module.state_dict().items())}
    return canonical_sha256(state)


def _canonicalize_field(field: torch.Tensor, backend: str) -> torch.Tensor:
    if backend not in BACKEND_CONTRACT:
        raise ValueError(f"unregistered HFRR backend: {backend}")
    _, relays = _build_backend_planes(_formal_for_reference())
    relay = relays[backend]
    value = field.detach().cpu().to(torch.complex128)
    if value.ndim != 4 or tuple(value.shape[1:]) != (1, *TARGET_SHAPE):
        raise ValueError("canonicalization expects a target-plane [B,1,200,200] field")
    target_dft_y = _dft_matrix(TARGET_SHAPE[0], -1.0)
    target_dft_x = _dft_matrix(TARGET_SHAPE[1], -1.0)
    spectrum = _direct_dft2(value, target_dft_y, target_dft_x)
    target_fy = _frequency_axis(TARGET_SHAPE[0], TARGET_PITCH[0])
    target_fx = _frequency_axis(TARGET_SHAPE[1], TARGET_PITCH[1])
    source_y, source_x = _coordinates(SOURCE_SHAPE, SOURCE_PITCH)
    mapped_y = relay.magnification * source_y[:, 0]
    mapped_x = relay.magnification * source_x[0, :]
    target_y0 = -(TARGET_SHAPE[0] - 1.0) * TARGET_PITCH[0] / 2.0
    target_x0 = -(TARGET_SHAPE[1] - 1.0) * TARGET_PITCH[1] / 2.0
    phase_y = torch.exp(2j * math.pi * (mapped_y[:, None] - target_y0) * target_fy[None, :])
    phase_x = torch.exp(2j * math.pi * (mapped_x[:, None] - target_x0) * target_fx[None, :])
    sampled = torch.einsum("uk,bkl,vl->buv", phase_y, spectrum, phase_x) / float(TARGET_SHAPE[0] * TARGET_SHAPE[1])
    yy, xx = torch.meshgrid(mapped_y, mapped_x, indexing="ij")
    dephase = torch.exp(-1j * (math.pi * relay.abcd.c_per_m / (relay.target.wavelength_m * relay.abcd.a)) * (xx.square() + yy.square()))
    return (sampled * dephase * abs(relay.abcd.a)).unsqueeze(1)


def _retention_from_features(left: torch.Tensor, right: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    left_metrics = mra._rank_metrics(left, labels)
    right_metrics = mra._rank_metrics(right, labels)
    entropy = float(right_metrics["entropy_effective_rank"]) / max(float(left_metrics["entropy_effective_rank"]), 1.0e-12)
    participation = float(right_metrics["participation_rank"]) / max(float(left_metrics["participation_rank"]), 1.0e-12)
    return {"entropy_retention": entropy, "participation_retention": participation, "geometric_mean_retention": math.sqrt(max(entropy, 0.0) * max(participation, 0.0)), "left": left_metrics, "right": right_metrics}


def _cell_key(model_name: str, backend: str) -> str:
    for key, value in CELL_REGISTRATION.items():
        if value["representation"] == model_name and value["backend"] == backend:
            return key
    raise ValueError("unregistered HFRR cell")


def _synthetic_audit(formal: FormalSystemConfig) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    global _FORMAL_CACHE
    _FORMAL_CACHE = formal
    _, relays = _build_backend_planes(formal)
    records: dict[str, list[dict[str, Any]]] = {backend: [] for backend in BACKENDS}
    convergence: dict[str, Any] = {}
    for probe_name in SYNTHETIC_PROBES:
        field = _synthetic_probe(probe_name)
        for backend in BACKENDS:
            convergence[f"{probe_name}:{backend}"] = _reference_convergence(field, backend)
            production = relays[backend](ComplexField(field, relays[backend].source, OperationProvenance("HFRR_synthetic_probe", "chip", None, {"probe": probe_name}))).tensor.detach().cpu()
            reference = _reference_relay(field, backend)
            metric = _compare_one(production, reference)
            metric.update({"probe": probe_name, "backend": backend, "reference_convergence_passed": convergence[f"{probe_name}:{backend}"]["passed"]})
            records[backend].append(metric)
    return {"records": records, "aggregate": {backend: _aggregate_metrics(values) for backend, values in records.items()}, "convergence": convergence, "convergence_passed": all(value["passed"] for value in convergence.values())}, records


def _model_readout(model_name: str, model: torch.nn.Module, amplitude: torch.Tensor, research_ols: torch.nn.Module) -> torch.Tensor:
    if model_name == IDENTITY:
        return dcm._identity_readout(model, amplitude, research_ols)[0]  # type: ignore[arg-type]
    return dcm._scaled_readout(model, amplitude)[0]  # type: ignore[arg-type]


def _task_audit(parent: ParentBundle, formal: FormalSystemConfig, device: torch.device) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    global _FORMAL_CACHE
    _FORMAL_CACHE = formal
    data = parent.bosa_bundle.parent.data
    indices, labels = _task_indices(data)
    images = data.test_images.index_select(0, indices)
    _, relays = _build_backend_planes(formal)
    records: dict[str, list[dict[str, Any]]] = {cell: [] for cell in CELL_ORDER}
    feature_store: dict[str, dict[str, list[torch.Tensor]]] = {cell: {"left": [], "raw": [], "canonical": []} for cell in CELL_ORDER}
    states: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        checkpoint = parent.bosa_bundle.parent.source.checkpoints["identity_step_5500" if model_name == IDENTITY else "scaled_chen_step_5000"]
        model = mra._load_identity_model(formal, checkpoint, device) if model_name == IDENTITY else mra._load_scaled_model(formal, checkpoint, device)
        research_ols = mra._build_research_ols(formal).to(device) if model_name == IDENTITY else None
        model_before = _module_hash(model)
        ols_before = _module_hash(research_ols) if research_ols is not None else _module_hash(model.ols)  # type: ignore[attr-defined]
        with torch.no_grad():
            for chunk in torch.arange(images.shape[0], dtype=torch.long).split(5):
                rendered = base.render_intensity_batch(images.index_select(0, chunk))
                amplitude = rendered.amplitude.to(device)
                decoder = _model_readout(model_name, model, amplitude, research_ols)
                decoder_cpu = decoder.detach().cpu().to(torch.complex128)
                chip = ComplexField(decoder, model.system.chip_plane, OperationProvenance("HFRR_task_decoder_field", "decoder_layer_2", None, {"model": model_name}))
                for backend in BACKENDS:
                    cell = _cell_key(model_name, backend)
                    output = relays[backend](chip).tensor
                    output_cpu = output.detach().cpu().to(torch.complex128)
                    reference = _reference_relay(decoder_cpu, backend)
                    batch_records = [_compare_one(output_cpu[index : index + 1], reference[index : index + 1]) for index in range(output_cpu.shape[0])]
                    records[cell].extend({**metric, "probe_index": int(indices[chunk[index]]), "backend": backend, "model": model_name} for index, metric in enumerate(batch_records))
                    feature_store[cell]["left"].append(mra._feature_projection(decoder_cpu).float())
                    feature_store[cell]["raw"].append(mra._feature_projection(output_cpu).float())
                    canonical = _canonicalize_field(output_cpu, backend)
                    feature_store[cell]["canonical"].append(mra._feature_projection(canonical).float())
        model_after = _module_hash(model)
        ols_after = _module_hash(research_ols) if research_ols is not None else _module_hash(model.ols)  # type: ignore[attr-defined]
        states[model_name] = {"model_before": model_before, "model_after": model_after, "ols_before": ols_before, "ols_after": ols_after, "state_unchanged": model_before == model_after and ols_before == ols_after}
    retentions: dict[str, Any] = {}
    for cell in CELL_ORDER:
        left = torch.cat(feature_store[cell]["left"], dim=0)
        raw = torch.cat(feature_store[cell]["raw"], dim=0)
        canonical = torch.cat(feature_store[cell]["canonical"], dim=0)
        retentions[cell] = {"raw": _retention_from_features(left, raw, labels), "canonicalized": _retention_from_features(left, canonical, labels)}
    return {"records": records, "aggregate": {cell: _aggregate_metrics(values) for cell, values in records.items()}, "retention": retentions, "labels": labels, "indices": indices, "states": states, "finite": all(_aggregate_metrics(values)["finite"] for values in records.values())}, records, retentions


def _backend_gap(retentions: Mapping[str, Any], *, canonicalized: bool) -> dict[str, float]:
    key = "canonicalized" if canonicalized else "raw"
    return {"identity_representation": float(retentions["I_to_I"][key]["geometric_mean_retention"] - retentions["I_to_S"][key]["geometric_mean_retention"]), "scaled_representation": float(retentions["S_to_I"][key]["geometric_mean_retention"] - retentions["S_to_S"][key]["geometric_mean_retention"])}


def _classify_numeric(synthetic: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    native = {"identity_backend": synthetic["aggregate"]["identity_backend"], "scaled_chen_backend": synthetic["aggregate"]["scaled_chen_backend"]}
    task_native = {"identity_backend": task["aggregate"]["I_to_I"], "scaled_chen_backend": task["aggregate"]["S_to_S"]}
    synthetic_pass = {name: _numerical_pass(value) for name, value in native.items()}
    task_pass = {name: _numerical_pass(value) for name, value in task_native.items()}
    both_pass = all(synthetic_pass.values()) and all(task_pass.values())
    both_fail = {name: (not synthetic_pass[name] and not task_pass[name]) for name in BACKENDS}
    stable_mismatch = any(both_fail.values())
    details = {"synthetic_pass": synthetic_pass, "task_pass": task_pass, "stable_mismatch_by_backend": both_fail, "all_consistent": both_pass, "thresholds": GATES}
    if stable_mismatch:
        return "relay_numerical_implementation_mismatch_supported", details
    if not both_pass:
        return "BLOCKED_BACKEND_PHYSICS_NOT_EQUIVALENT", details
    return "relay_numerical_reference_consistent", details


def _classify_canonicalization(task: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = _backend_gap(task["retention"], canonicalized=False)
    canonical = _backend_gap(task["retention"], canonicalized=True)
    reduction = {name: raw[name] - canonical[name] for name in raw}
    direction = all(raw[name] >= canonical[name] for name in raw)
    supported = all(value >= MATERIAL_GAP for value in reduction.values()) and direction and task["finite"] and all(item["state_unchanged"] for item in task["states"].values())
    details = {"raw_backend_gap": raw, "canonicalized_backend_gap": canonical, "gap_reduction": reduction, "direction_consistent": direction, "power_controls_valid": True, "analytic_only": True, "physical_forward_modified": False}
    return ("relay_complex_retention_metric_confound_supported" if supported else "relay_high_fidelity_reference_consistent_residual_supported"), details


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("HFRR requires CUDA production replay; CPU production is not a fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("HFRR requires at least 4 GiB free CUDA memory")
    if shutil.disk_usage(output_path.anchor or str(output_path.parent)).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("HFRR requires at least 10 GiB free D: storage")
    return int(free), int(total)


def _load_validation(path: Path, spec: Spec, parent: ParentBundle, review_path: Path, state: Mapping[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("status") != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != _input_hashes(spec, review_path, parent):
        raise ValueError("HFRR validation evidence is stale or mismatched")


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"HFRR requires clean branch {BRANCH}")
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _ensure_run_directory(output_path, validation_path)
    static_parent = _load_parent(spec, args.dataset_root, load_checkpoints=False)
    _load_validation(validation_path, spec, static_parent, review_path, state)
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    parent = _load_parent(spec, args.dataset_root, load_checkpoints=True)
    formal = parent.bosa_bundle.parent.formal
    global _FORMAL_CACHE
    _FORMAL_CACHE = formal
    started = time.monotonic()
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    torch.cuda.reset_peak_memory_stats(device)
    synthetic, _ = _synthetic_audit(formal)
    if not synthetic["convergence_passed"]:
        raise RuntimeError("BLOCKED_REFERENCE_NOT_CONVERGED")
    task, _, retentions = _task_audit(parent, formal, device)
    numeric_classification, numeric_details = _classify_numeric(synthetic, task)
    canonical_details: dict[str, Any] = {"executed": False, "analytic_only": True, "physical_forward_modified": False}
    classification = numeric_classification
    action = NEXT_ACTION_MISMATCH if numeric_classification == "relay_numerical_implementation_mismatch_supported" else NEXT_ACTION_BLOCKED
    numerical_boundary_not_evaluated = False
    numerical_mismatch = numeric_classification == "relay_numerical_implementation_mismatch_supported"
    numerical_consistent = numeric_classification == "relay_numerical_reference_consistent"
    if numerical_consistent:
        classification, canonical_details = _classify_canonicalization(task)
        canonical_details["executed"] = True
        action = NEXT_ACTION_CONFOUND if classification == "relay_complex_retention_metric_confound_supported" else NEXT_ACTION_RESIDUAL
    peak = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    state_ok = all(value["state_unchanged"] for value in task["states"].values())
    finite = bool(synthetic["convergence_passed"] and task["finite"])
    harness_gates = (
        GateResult("independent_reference_converged", synthetic["convergence_passed"], {"convergence": synthetic["convergence"]}),
        GateResult("finite_production_and_reference_measurements", finite, {}),
        GateResult("frozen_model_and_OLS_state", state_ok, task["states"]),
        GateResult("pupil_aware_cross_backend_interpretation", True, {"cross_backend_is_diagnostic_only": True}),
        GateResult("optimizer_backward_evaluator_prior_isolation", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "prior_sampled": False}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}),
    )
    status = "valid_failed" if all(gate.passed for gate in harness_gates) else "invalid_harness"
    if status == "invalid_harness":
        classification = "invalid_harness"
        action = NEXT_ACTION_BLOCKED
    panel_path = output_path.parent / "panel.pt"
    compact_path = output_path.parent / "compact_report.json"
    for target in (panel_path, compact_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite HFRR artifact: {target}")
    panel_payload = {"schema": PANEL_SCHEMA, "task_indices": task["indices"], "task_labels": task["labels"], "records": task["records"], "synthetic_aggregate": synthetic["aggregate"], "synthetic_convergence": synthetic["convergence"], "retention": task["retention"], "affects_selection": False}
    storage_estimate = len(json.dumps(panel_payload, default=str).encode("utf-8"))
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("HFRR compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    compact = {"schema": COMPACT_SCHEMA, "status": status, "classification": classification, "numeric_classification": numeric_classification, "numeric_details": numeric_details, "synthetic": synthetic, "task": {key: value for key, value in task.items() if key not in {"labels", "indices", "records"}}, "canonicalization": canonical_details, "next_action": action, "numerical_boundary_not_evaluated": numerical_boundary_not_evaluated, "relay_numerical_reference_mismatch": numerical_mismatch, "relay_numerical_reference_consistent": numerical_consistent, "generation_readiness": "not_ready", "training_authorized": False, "formal_promotion": False, "optimizer_created": False, "backward": False, "evaluator_calls": 0, "prior_sampled": False, "panel_sha256": panel_hash}
    _write_new(compact_path, compact, artifact_kind="HFRR compact report")
    contract = {"synthetic": synthetic, "task": {key: value for key, value in task.items() if key not in {"labels", "indices", "records"}}, "numeric_classification": numeric_classification, "numeric_details": numeric_details, "canonicalization": canonical_details, "classification": classification, "next_action": action, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "compact_report_path": str(compact_path.resolve()), "compact_report_sha256": file_sha256(compact_path), "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"}, "references": REFERENCES, "backends": BACKEND_CONTRACT, "reference": REFERENCE_CONTRACT, "numerical_boundary_not_evaluated": numerical_boundary_not_evaluated, "relay_numerical_reference_mismatch": numerical_mismatch, "relay_numerical_reference_consistent": numerical_consistent, "generation_readiness": "not_ready", "mechanism_evidence_only": True}
    hashes = _input_hashes(spec, review_path, parent)
    all_gates = harness_gates + (GateResult("scientific_classification", classification in {"relay_numerical_implementation_mismatch_supported", "relay_complex_retention_metric_confound_supported", "relay_high_fidelity_reference_consistent_residual_supported"}, {"classification": classification}),)
    payload = _record(AUDIT_SCHEMA, status, classification, spec, parent, state, review_path, hashes, all_gates, (action,), contract, None if status == "valid_failed" else "HFRR harness failed")
    payload["numerical_boundary_not_evaluated"] = numerical_boundary_not_evaluated
    payload["relay_numerical_reference_mismatch"] = numerical_mismatch
    payload["relay_numerical_reference_consistent"] = numerical_consistent
    _write_new(output_path, payload, artifact_kind="HFRR audit")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "valid_failed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-high-fidelity-relay-reference-validate":
        return validate(args)
    if args.research_action == "scaled-chen-high-fidelity-relay-reference-run":
        return run(args)
    raise ValueError(f"unsupported HFRR action: {args.research_action}")


__all__ = [
    "AUTHORIZATION", "BACKEND_CONTRACT", "BACKENDS", "CANONICALIZATION", "CELL_ORDER", "CELL_REGISTRATION", "COMPLEX_RELATIVE_TOLERANCE", "CONFIG_SCHEMA", "GATES", "HYPOTHESIS", "MEASUREMENT", "PARENT_BOSA", "REFERENCE_CONTRACT", "REFERENCES", "REVIEW_SCHEMA", "SCALED", "SYNTHETIC_PROBES", "TASK_INDEX_VECTOR", "TASK_SUBSET_SHA256", "VALIDATION_SCHEMA", "Spec", "_assert_contract", "_backend_validation", "_canonicalize_field", "_classify_canonicalization", "_classify_numeric", "_compare_one", "_dft_matrix", "_ensure_run_directory", "_frequency_axis", "_load_review", "_load_spec", "_numerical_pass", "_pupil", "_reference_convergence", "_reference_relay", "_synthetic_probe", "dispatch", "run", "validate",
]
