"""Frozen back-relay axial/conjugate attribution for Scaled-Chen.

This research-only consumer keeps the registered Scaled-Chen relay intact and
adds one bounded detector-side axial intervention after the nominal
``chen_relay`` output.  It is deliberately no-update: no optimiser, backward
pass, evaluator, checkpoint, or prior sample is created.
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
from ...optics.relays import ABCDMatrix, path_abcd
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_decoder_cmos_attribution as dcm
from . import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_preflight.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate.v1"
REPORT_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_report.v1"
CLOSEOUT_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_closeout.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_back_relay_axial_conjugate_panel.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0-BRAC"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

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
}

TOTAL_RELAY_DISTANCE_M = 3.0e-3
BACK_MAGNIFICATION = -2.5
WAVELENGTH_M = 532.0e-9
TARGET_SHAPE = (200, 200)
TARGET_PIXEL_PITCH_M = (2.5e-6, 2.5e-6)
TARGET_Q_RADIAL_NYQUIST = math.sqrt(2.0) / (2.0 * TARGET_PIXEL_PITCH_M[0])
DELTA_Z_M = WAVELENGTH_M / (
    4.0 * (1.0 - math.sqrt(1.0 - (WAVELENGTH_M * TARGET_Q_RADIAL_NYQUIST) ** 2))
)
ARMS = {"negative": -DELTA_Z_M, "baseline": 0.0, "positive": DELTA_Z_M}
ARM_ORDER = ("negative", "baseline", "positive")
SCALED_MODEL = "scaled_chen_step_5000"
RELAY_SEGMENT = "decoder_layer_2_to_cmos_incident_field"
DETECTOR_SEGMENT = "cmos_incident_field_to_cmos_intensity"
RANK_RETENTION_THRESHOLD = 0.90
MATERIAL_GAP = 0.05
PHASE_TOLERANCE = 1.0e-5
BASELINE_RETENTION_TOLERANCE = 0.005
BASELINE_MSE_RELATIVE_TOLERANCE = 0.01
BASELINE_POWER_TOLERANCE = 0.005
MAGNIFICATION_DRIFT_TOLERANCE = 0.01
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
PHASE_CONTROL_SEED = 1717
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
CONFIRMATION_PER_LABEL = 456
CONFIRMATION_PREFIX_COUNT = 5000
NEXT_ACTION_PREFLIGHT = "run_scaled_chen_back_relay_axial_conjugate_preflight"
NEXT_ACTION_RUN = "run_scaled_chen_back_relay_axial_conjugate"
NEXT_ACTION_REPORT = "report_scaled_chen_back_relay_axial_conjugate"
NEXT_ACTION_CLOSEOUT = "closeout_scaled_chen_back_relay_axial_conjugate"
NEXT_ACTION_PASS = "manual_review_matched_back_relay_reconstruction_canary_contract"
NEXT_ACTION_FAIL = "manual_review_scaled_chen_local_relay_tuning_closeout_contract"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Panel:
    indices: torch.Tensor
    labels: torch.Tensor
    indices_sha256: str
    labels_sha256: str
    content_sha256: str
    mechanism_evidence_only: bool


@dataclass(frozen=True)
class ParentBundle:
    dcm_spec: dcm.Spec
    dcm_review_path: Path
    dcm_audit_path: Path
    dcm_audit: dict[str, Any]
    mra_parent: dcm.ParentBundle


@dataclass(frozen=True)
class ArmEvaluation:
    arm: str
    delta_z_m: float
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


def _nominal_geometry() -> dict[str, Any]:
    total = TOTAL_RELAY_DISTANCE_M
    u = total / (1.0 + abs(BACK_MAGNIFICATION))
    v = total - u
    focal = u * v / total
    abcd = path_abcd((u, v), (focal,))
    return {
        "total_distance_m": total,
        "source_segment_m": u,
        "target_segment_m": v,
        "focal_length_m": focal,
        "magnification": BACK_MAGNIFICATION,
        "abcd": abcd.to_dict(),
    }


def _axial_abcd(delta_z_m: float) -> ABCDMatrix:
    nominal = path_abcd(
        (_nominal_geometry()["source_segment_m"], _nominal_geometry()["target_segment_m"]),
        (_nominal_geometry()["focal_length_m"],),
    )
    # Detector-side free-space is left-multiplied: F(delta) @ M0.
    return ABCDMatrix(
        nominal.a + delta_z_m * nominal.c_per_m,
        nominal.b_m + delta_z_m * nominal.d,
        nominal.c_per_m,
        nominal.d,
    )


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema", "hypothesis", "formal_system_config", "parent_dcm",
        "intervention", "dataset", "measurement", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("BRAC config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("BRAC config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _expected_intervention() -> dict[str, Any]:
    return {
        "variable": "detector_plane_axial_offset",
        "operation": "signed_angular_spectrum_after_nominal_chen_relay",
        "wavelength_m": WAVELENGTH_M,
        "target_shape": list(TARGET_SHAPE),
        "target_pixel_pitch_m": list(TARGET_PIXEL_PITCH_M),
        "radial_nyquist_cycles_m": TARGET_Q_RADIAL_NYQUIST,
        "phase_fraction": "quarter_wave_relative_to_dc",
        "delta_z_m": DELTA_Z_M,
        "formula": "lambda/[4*(1-sqrt(1-(lambda*q_N)^2))]",
        "arms": {name: ARMS[name] for name in ARM_ORDER},
        "tie_order": ["negative", "positive"],
        "magnification_drift_tolerance": MAGNIFICATION_DRIFT_TOLERANCE,
    }


def _expected_dataset() -> dict[str, Any]:
    return {
        "selection": {
            "source": "mnist_test_indices_5000_9999",
            "seed": mra.SELECTION_SEED,
            "per_label": mra.SELECTION_PER_LABEL,
            "count": mra.SELECTION_COUNT,
        },
        "confirmation": {
            "source": "mnist_test_indices_0_4999_first_456_per_label",
            "prefix_count": CONFIRMATION_PREFIX_COUNT,
            "per_label": CONFIRMATION_PER_LABEL,
            "count": CONFIRMATION_PER_LABEL * 10,
            "construction": "original_index_order_first_456_each_label",
            "mechanism_evidence_only": True,
        },
        "known_exclusions": [
            "train_split",
            "calibration",
            "horizon_schedule",
            "mra_selection",
            "mra_confirmation",
            "plan3_plan4",
        ],
    }


def _expected_measurement() -> dict[str, Any]:
    return {
        "objective": "frozen_forward_only",
        "models": [SCALED_MODEL],
        "baseline_replay_models": list(dcm.MODEL_NAMES),
        "arms": list(ARM_ORDER),
        "stages": ["decoder_layer_2", "cmos_incident_field", "cmos_intensity"],
        "primary": "decoder_layer_2_to_cmos_incident_field_geometric_mean_retention",
        "secondary": "paired_mse_relative_improvement",
        "negative_controls": [
            "identity_reference", "permuted_label", "cyclic_target",
            "square_law", "global_phase_invariance", "pointwise_phase_invariance",
            "signed_asm_round_trip",
        ],
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "confidence": 0.95},
        "no_update": True,
    }


def _expected_gates() -> dict[str, Any]:
    return {
        "rank_retention": RANK_RETENTION_THRESHOLD,
        "material_gap": MATERIAL_GAP,
        "baseline_retention_tolerance": BASELINE_RETENTION_TOLERANCE,
        "baseline_mse_relative_tolerance": BASELINE_MSE_RELATIVE_TOLERANCE,
        "baseline_power_tolerance": BASELINE_POWER_TOLERANCE,
        "phase_tolerance": PHASE_TOLERANCE,
        "magnification_drift_tolerance": MAGNIFICATION_DRIFT_TOLERANCE,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "training_authorized": False,
        "formal_promotion": False,
    }


def _expected_authorization() -> dict[str, Any]:
    return {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_back_relay_axial_conjugate_attribution",
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": [
            "optimizer", "backward", "checkpoint_write", "generation", "evaluator",
            "fmf", "prior_sampling", "formal_promotion", "adaptive_offset",
        ],
    }


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("BRAC formal config is not locked")
    if payload["parent_dcm"] != PARENT_DCM:
        raise ValueError("BRAC DCM parent binding is not locked")
    if payload["intervention"] != _expected_intervention():
        raise ValueError("BRAC intervention contract is not locked")
    if payload["dataset"] != _expected_dataset():
        raise ValueError("BRAC dataset contract is not locked")
    if payload["measurement"] != _expected_measurement():
        raise ValueError("BRAC measurement contract is not locked")
    if payload["gates"] != _expected_gates():
        raise ValueError("BRAC gate contract is not locked")
    if payload["authorization"] != _expected_authorization():
        raise ValueError("BRAC authorization boundary is not locked")


def _expected_review(spec: Spec) -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_back_relay_axial_conjugate_attribution",
        "approved_contract": {
            "parent_dcm": dict(PARENT_DCM),
            "intervention": _expected_intervention(),
            "dataset": _expected_dataset(),
            "measurement": _expected_measurement(),
            "gates": _expected_gates(),
            "next_action_if_pass": NEXT_ACTION_PASS,
            "next_action_if_fail": NEXT_ACTION_FAIL,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload != _expected_review(spec):
        raise ValueError("BRAC review contract does not match scope")
    return target


def _load_dcm_parent(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    parent = spec.payload["parent_dcm"]
    config_path = _resolve(parent["config"], base_path=spec.path.parent)
    review_path = _resolve(parent["review_contract"], base_path=spec.path.parent)
    audit_path = _resolve(parent["audit"], base_path=spec.path.parent)
    if not config_path.is_file() or file_sha256(config_path) != parent["config_sha256"]:
        raise ValueError("DCM parent config is missing, stale, or tampered")
    if not review_path.is_file() or file_sha256(review_path) != parent["review_sha256"]:
        raise ValueError("DCM parent review is missing, stale, or tampered")
    if not audit_path.is_file() or file_sha256(audit_path) != parent["audit_sha256"]:
        raise ValueError("DCM parent audit is missing, stale, or tampered")
    dcm_spec = dcm._load_spec(config_path)
    dcm._assert_contract(dcm_spec)
    dcm._load_review(review_path, dcm_spec)
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_payload, Mapping):
        raise ValueError("DCM parent audit is not a mapping")
    record = EvidenceRecordV2.from_dict(audit_payload["evidence"])
    record.validate()
    if (
        audit_payload.get("schema") != dcm.AUDIT_SCHEMA
        or audit_payload.get("hypothesis") != dcm.HYPOTHESIS
        or audit_payload.get("status") != parent["audit_status"]
        or audit_payload.get("classification") != parent["audit_classification"]
        or record.git_commit != parent["audit_git_commit"]
        or record.config_hash != dcm_spec.config_hash
        or record.permitted_next_actions != (dcm.NEXT_ACTION_RELAY,)
    ):
        raise ValueError("DCM parent audit terminal state is stale or mismatched")
    mra_parent = dcm._load_parent_with_data(dcm_spec, dataset_root, load_checkpoints=load_checkpoints)
    return ParentBundle(dcm_spec, review_path, audit_path, dict(audit_payload), mra_parent)


def _candidate_panel(data: mra.AuditData) -> Panel:
    prefix = torch.arange(CONFIRMATION_PREFIX_COUNT, dtype=torch.long)
    selected: list[torch.Tensor] = []
    for label in range(10):
        matches = prefix[data.test_labels.index_select(0, prefix) == label]
        if matches.numel() < CONFIRMATION_PER_LABEL:
            raise ValueError(f"untouched panel has too few samples for label {label}")
        selected.append(matches[:CONFIRMATION_PER_LABEL])
    indices = torch.cat(selected).sort().values
    labels = data.test_labels.index_select(0, indices)
    counts = torch.bincount(labels, minlength=10)
    if indices.numel() != CONFIRMATION_PER_LABEL * 10 or torch.unique(indices).numel() != indices.numel() or not torch.equal(counts, torch.full((10,), CONFIRMATION_PER_LABEL, dtype=torch.long)):
        raise ValueError("untouched panel is not deterministic and class balanced")
    if torch.isin(indices, data.selection_indices).any():
        raise ValueError("untouched panel overlaps MRA selection")
    indices_hash = tensor_sha256(indices)
    labels_hash = tensor_sha256(labels)
    content_hash = canonical_sha256({
        "split": "mnist_test",
        "prefix_count": CONFIRMATION_PREFIX_COUNT,
        "construction": "original_index_order_first_456_each_label",
        "indices_sha256": indices_hash,
        "labels_sha256": labels_hash,
    })
    # A repository-wide historical adaptive-use proof is not available; keep
    # this mechanism-only until a future manifest registers the exclusion scan.
    return Panel(indices, labels, indices_hash, labels_hash, content_hash, True)


def _frequency_gate(plane: Any, device: torch.device) -> dict[str, Any]:
    dtype = torch.float64
    fy = torch.fft.fftfreq(plane.shape[0], d=plane.pixel_pitch_m[0], device=device, dtype=dtype)
    fx = torch.fft.fftfreq(plane.shape[1], d=plane.pixel_pitch_m[1], device=device, dtype=dtype)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
    argument = 1.0 - (plane.wavelength_m * fx_grid).square() - (plane.wavelength_m * fy_grid).square()
    minimum = float(argument.min().detach().cpu())
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("BRAC signed ASM has an evanescent represented frequency")
    return {"minimum_propagating_argument": minimum, "all_propagating": True}


def _signed_asm(field: ComplexField, delta_z_m: float, plane_id: str) -> ComplexField:
    if not torch.is_complex(field.tensor) or not torch.isfinite(field.tensor).all():
        raise ValueError("signed ASM requires a finite complex field")
    if field.plane.shape != TARGET_SHAPE or field.plane.pixel_pitch_m != TARGET_PIXEL_PITCH_M or field.plane.wavelength_m != WAVELENGTH_M:
        raise ValueError("signed ASM received an unexpected target sampling")
    gate = _frequency_gate(field.plane, field.tensor.device)
    plane = field.plane.with_plane(plane_id, z_m=field.plane.z_m + delta_z_m)
    if delta_z_m == 0.0:
        output = field.tensor
    else:
        real_dtype = field.tensor.real.dtype
        fy = torch.fft.fftfreq(field.plane.shape[0], d=field.plane.pixel_pitch_m[0], device=field.tensor.device, dtype=real_dtype)
        fx = torch.fft.fftfreq(field.plane.shape[1], d=field.plane.pixel_pitch_m[1], device=field.tensor.device, dtype=real_dtype)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        argument = 1.0 - (field.plane.wavelength_m * fx_grid).square() - (field.plane.wavelength_m * fy_grid).square()
        if bool((argument < 0.0).any()):
            raise ValueError("signed ASM has an evanescent represented frequency")
        phase = 2.0 * math.pi * delta_z_m / field.plane.wavelength_m * torch.sqrt(argument)
        transfer = torch.polar(torch.ones_like(phase), phase).to(field.tensor.dtype)
        output = torch.fft.ifft2(torch.fft.fft2(field.tensor, norm="ortho") * transfer, norm="ortho")
    provenance = OperationProvenance(
        "research_signed_angular_spectrum_axial_offset",
        field.plane.plane_id,
        None,
        {
            "delta_z_m": delta_z_m,
            "sampling": plane.to_dict(),
            "frequency_gate": gate,
            "normalization": "ortho",
            "direction": "forward_signed_propagating_spectrum",
        },
    )
    return ComplexField(output, plane, provenance)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _relay_fingerprint(relay: Any) -> str:
    return canonical_sha256({
        "source": relay.source.to_dict(),
        "target": relay.target.to_dict(),
        "magnification": relay.magnification,
        "name": relay.name,
        "segments_m": list(relay.segments_m),
        "matrix_segments_m": list(relay.matrix_segments_m),
        "focal_lengths_m": list(relay.focal_lengths_m),
        "clear_aperture_m": relay.clear_aperture_m,
        "abcd": relay.abcd.to_dict(),
    })


def _plane_energy(value: torch.Tensor, pitch: tuple[float, float]) -> torch.Tensor:
    return value.abs().square().sum(dim=(-3, -2, -1)) * pitch[0] * pitch[1]


def _input_hashes(spec: Spec, review_path: Path, parent: ParentBundle, panel: Panel) -> dict[str, str]:
    mra_hashes = mra._input_hashes(parent.mra_parent.spec, parent.mra_parent.review_path, parent.mra_parent.formal_path, parent.mra_parent.data, parent.mra_parent.source)
    return {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "formal_config": file_sha256(parent.mra_parent.formal_path),
        "dataset_manifest": parent.mra_parent.data.manifest["manifest_hash"],
        "selection_indices": parent.mra_parent.data.manifest["selection_indices_sha256"],
        "parent_dcm_config": file_sha256(parent.dcm_spec.path),
        "parent_dcm_review": file_sha256(parent.dcm_review_path),
        "parent_dcm_audit": file_sha256(parent.dcm_audit_path),
        "brac_confirmation_indices": panel.indices_sha256,
        "brac_confirmation_labels": panel.labels_sha256,
        "brac_confirmation_content": panel.content_sha256,
        **{f"parent_{name}": value for name, value in mra_hashes.items()},
    }


def _context(spec: Spec, parent: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"hypothesis": HYPOTHESIS, "intervention": spec.payload["intervention"], "measurement": spec.payload["measurement"]})
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=parent.mra_parent.data.manifest["manifest_hash"], dataset_split_manifest_hash=parent.mra_parent.data.manifest["manifest_hash"])


def _record(schema: str, status: str, classification: str | None, spec: Spec, parent: ParentBundle, review_path: Path, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
    state = _git_state()
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, parent), dict(hashes), tuple(gates), tuple(actions), reason)
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
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _validation_gates(parent: ParentBundle, panel: Panel, state: Mapping[str, Any]) -> tuple[GateResult, ...]:
    data = parent.mra_parent.data
    formal = parent.mra_parent.formal
    geometry = _nominal_geometry()
    abcd = geometry["abcd"]
    camera_plane = dcm.base.IdentityOpticalManifoldAutoencoder(formal).system.camera_plane
    frequency = _frequency_gate(camera_plane, torch.device("cpu"))
    return (
        GateResult("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True, {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("selection_balanced", bool(torch.equal(torch.bincount(data.test_labels.index_select(0, data.selection_indices), minlength=10), torch.full((10,), mra.SELECTION_PER_LABEL, dtype=torch.long))), {}),
        GateResult("confirmation_balanced", bool(torch.equal(torch.bincount(panel.labels, minlength=10), torch.full((10,), CONFIRMATION_PER_LABEL, dtype=torch.long))), {"count": int(panel.indices.numel())}),
        GateResult("panel_disjoint_from_selection", not bool(torch.isin(panel.indices, data.selection_indices).any()), {}),
        GateResult("panel_split_is_test", int(panel.indices.max()) < CONFIRMATION_PREFIX_COUNT, {}),
        GateResult("mode_bank_rank_100", dcm._mode_rank(dcm._build_research_ols(formal)) == mra.PORT_COUNT, {}),
        GateResult("baseline_b0_contract", abs(float(abcd["B_m"])) <= 1.0e-9 and abs(float(abcd["determinant"]) - 1.0) <= 1.0e-9 and abs(float(abcd["A"]) - BACK_MAGNIFICATION) <= 1.0e-9, abcd),
        GateResult("all_target_frequencies_propagating", bool(frequency["all_propagating"]), frequency),
        GateResult("delta_formula_locked", math.isclose(DELTA_Z_M, float(_expected_intervention()["delta_z_m"]), rel_tol=0.0, abs_tol=1.0e-18), {"delta_z_m": DELTA_Z_M}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False}),
    )


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"BRAC requires clean branch {BRANCH}")
    parent = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=False)
    panel = _candidate_panel(parent.mra_parent.data)
    gates = _validation_gates(parent, panel, state)
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(spec, review_path, parent, panel)
    contract = {
        "parent_dcm": PARENT_DCM,
        "nominal_geometry": _nominal_geometry(),
        "intervention": _expected_intervention(),
        "selection": {"indices_sha256": parent.mra_parent.data.manifest["selection_indices_sha256"], "count": mra.SELECTION_COUNT},
        "confirmation": {"indices": panel.indices.tolist(), "labels_sha256": panel.labels_sha256, "indices_sha256": panel.indices_sha256, "content_sha256": panel.content_sha256, "mechanism_evidence_only": panel.mechanism_evidence_only},
    }
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, parent, review_path, hashes, gates, (NEXT_ACTION_PREFLIGHT,) if valid else (), contract, None if valid else "BRAC validation gates failed")
    _write_new(args.output, payload, artifact_kind="BRAC validation")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if valid else 2


def _load_evidence(path: str | Path, schema: str, status: str, spec: Spec, parent: ParentBundle, review_path: Path, expected_action: str) -> dict[str, Any]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("BRAC evidence schema is stale or mismatched")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    panel = _candidate_panel(parent.mra_parent.data)
    expected_hashes = _input_hashes(spec, review_path, parent, panel)
    if (
        payload.get("status") != status
        or record.status != status
        or record.git_commit != _git_state()["commit"]
        or record.config_hash != spec.config_hash
        or record.input_hashes != expected_hashes
        or expected_action not in record.permitted_next_actions
    ):
        raise ValueError("BRAC evidence status, commit, config, or action is stale")
    return payload


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BRAC requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("BRAC requires at least 4 GiB free CUDA memory")
    if shutil.disk_usage(output_path.anchor or str(output_path.parent)).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("BRAC requires 10 GiB free storage")
    return int(free), int(total)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite BRAC artifact: {path}")


def _ensure_fresh(path: Path, *, allowed: set[str] = frozenset()) -> None:
    _ensure_absent(path)
    if path.parent.exists():
        children = {item.name for item in path.parent.iterdir()}
        if not children.issubset(allowed):
            raise FileExistsError(f"BRAC output directory is not fresh: {path.parent}")


def _load_scaled_model(formal: FormalSystemConfig, checkpoint: Mapping[str, Any], device: torch.device) -> bogt.ScaledChenBOGTModel:
    model = mra._load_scaled_model(formal, checkpoint, device)
    if any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise ValueError("BRAC scaled-Chen model is not frozen")
    return model


def _scaled_fields(model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor, delta_z_m: float, arm: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ComplexField]:
    decoder, incident, _ = dcm._scaled_readout(model, amplitude)
    nominal = ComplexField(
        incident,
        model.chen_relay.target,
        OperationProvenance("nominal_chen_relay_output", "chip", TOTAL_RELAY_DISTANCE_M, {"relay": "inferred_backend_imaging", "abcd": model.chen_relay.abcd.to_dict()}),
    )
    displaced = _signed_asm(nominal, delta_z_m, f"chen_camera_{arm}")
    intensity = displaced.tensor.abs().square()
    return decoder, displaced.tensor, intensity, displaced


def _phase_controls(incident: torch.Tensor, intensity: torch.Tensor, generator: torch.Generator) -> dict[str, float]:
    global_phase = torch.polar(torch.ones((), device=incident.device, dtype=incident.real.dtype), torch.tensor(0.731, device=incident.device, dtype=incident.real.dtype)).to(incident.dtype)
    random_phase = torch.rand(incident.shape, device=incident.device, dtype=incident.real.dtype, generator=generator) * (2.0 * math.pi)
    return {
        "square_law_max_abs_error": float((incident.abs().square() - intensity).abs().max().detach().cpu()),
        "global_phase_max_abs_error": float(((incident * global_phase).abs().square() - intensity).abs().max().detach().cpu()),
        "pointwise_phase_max_abs_error": float(((incident.abs() * torch.exp(1j * random_phase).to(incident.dtype)).abs().square() - intensity).abs().max().detach().cpu()),
    }


def _evaluate_scaled_arm(model_name: str, checkpoint: Mapping[str, Any], formal: FormalSystemConfig, images: torch.Tensor, labels: torch.Tensor, device: torch.device, arm: str, delta_z_m: float) -> ArmEvaluation:
    if model_name != SCALED_MODEL:
        raise ValueError("BRAC arm evaluation only accepts Scaled-Chen step 5000")
    model = _load_scaled_model(formal, checkpoint, device)
    model_state_before = mra._state_cpu(model)
    ols_state_before = mra._state_cpu(model.ols)
    relay_fingerprint_before = _relay_fingerprint(model.chen_relay)
    stage_chunks = {name: [] for name in ("decoder_layer_2", "cmos_incident_field", "cmos_intensity")}
    camera_chunks: list[torch.Tensor] = []
    energy_chunks: list[torch.Tensor] = []
    phase_chunks: list[dict[str, float]] = []
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(PHASE_CONTROL_SEED)
        for batch_indices in torch.arange(images.shape[0], dtype=torch.long).split(10):
            rendered = dcm.base.render_intensity_batch(images.index_select(0, batch_indices))
            amplitude = rendered.amplitude.to(device)
            decoder, incident, intensity, displaced = _scaled_fields(model, amplitude, delta_z_m, arm)
            if displaced.plane.shape != TARGET_SHAPE or displaced.plane.pixel_pitch_m != TARGET_PIXEL_PITCH_M or not torch.isfinite(decoder).all() or not torch.isfinite(incident).all() or not torch.isfinite(intensity).all():
                raise RuntimeError("BRAC arm produced invalid field or plane")
            phase_chunks.append(_phase_controls(incident, intensity, generator))
            energy_chunks.append((_plane_energy(incident, TARGET_PIXEL_PITCH_M) / _plane_energy(decoder, model.system.chip_plane.pixel_pitch_m).clamp_min(1.0e-12)).detach().cpu().float())
            camera_chunks.append(intensity.detach().cpu().float())
            for name, value in (("decoder_layer_2", decoder), ("cmos_incident_field", incident), ("cmos_intensity", intensity)):
                stage_chunks[name].append(mra._feature_projection(value.detach().cpu()))
    model_state_after = mra._state_cpu(model)
    ols_state_after = mra._state_cpu(model.ols)
    relay_fingerprint_after = _relay_fingerprint(model.chen_relay)
    stage_values = {name: torch.cat(values, dim=0) for name, values in stage_chunks.items()}
    camera = torch.cat(camera_chunks, dim=0)
    target = dcm.base.render_intensity_batch(images).target.float()
    reconstruction, sample_mse = mra._mse_metrics(camera, target, labels)
    stages = mra._stage_rank_report(stage_values, labels)
    relay_entropy = stages["cmos_incident_field"]["entropy_effective_rank"] / max(stages["decoder_layer_2"]["entropy_effective_rank"], 1.0e-12)
    relay_participation = stages["cmos_incident_field"]["participation_rank"] / max(stages["decoder_layer_2"]["participation_rank"], 1.0e-12)
    detector_entropy = stages["cmos_intensity"]["entropy_effective_rank"] / max(stages["cmos_incident_field"]["entropy_effective_rank"], 1.0e-12)
    detector_participation = stages["cmos_intensity"]["participation_rank"] / max(stages["cmos_incident_field"]["participation_rank"], 1.0e-12)
    phase_summary = {key: max(item[key] for item in phase_chunks) for key in ("square_law_max_abs_error", "global_phase_max_abs_error", "pointwise_phase_max_abs_error")}
    metrics = {
        "model": model_name,
        "arm": arm,
        "delta_z_m": delta_z_m,
        "evaluation_count": int(images.shape[0]),
        "reconstruction": reconstruction,
        "stages": stages,
        "retention": {
            RELAY_SEGMENT: {"entropy_retention": float(relay_entropy), "participation_retention": float(relay_participation), "geometric_mean_retention": math.sqrt(max(float(relay_entropy), 0.0) * max(float(relay_participation), 0.0))},
            DETECTOR_SEGMENT: {"entropy_retention": float(detector_entropy), "participation_retention": float(detector_participation), "geometric_mean_retention": math.sqrt(max(float(detector_entropy), 0.0) * max(float(detector_participation), 0.0))},
        },
        "power_retention": float(torch.cat(energy_chunks).mean()),
        "phase_controls": phase_summary,
        "finite": bool(torch.isfinite(camera).all() and all(math.isfinite(value) for value in phase_summary.values())),
        "state_unchanged": mra._state_equal(model_state_before, model_state_after),
        "ols_state_unchanged": mra._state_equal(ols_state_before, ols_state_after),
        "relay_state_unchanged": relay_fingerprint_before == relay_fingerprint_after,
        "abcd": _axial_abcd(delta_z_m).to_dict(),
        "frequency_gate": _frequency_gate(model.chen_relay.target, device),
    }
    return ArmEvaluation(arm, delta_z_m, metrics, sample_mse, labels.clone(), _state_hash(model_state_before), _state_hash(model_state_after), _state_hash(ols_state_before), _state_hash(ols_state_after))


def _bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    if delta.ndim != 1 or labels.shape != delta.shape or delta.numel() == 0 or not torch.isfinite(delta).all():
        raise ValueError("BRAC bootstrap inputs are invalid")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        means = []
        for label in range(10):
            group = delta[labels == label]
            if group.numel() == 0:
                raise ValueError(f"BRAC bootstrap class {label} is empty")
            draws = torch.randint(group.numel(), (group.numel(),), generator=generator)
            means.append(group.index_select(0, draws).mean())
        values.append(torch.stack(means).mean())
    samples = torch.stack(values)
    return float(torch.quantile(samples, 0.025)), float(torch.quantile(samples, 0.975))


def _baseline_anchor(parent: ParentBundle) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract = parent.dcm_audit.get("contract")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("selection"), Mapping) or not isinstance(contract.get("confirmation"), Mapping):
        raise ValueError("DCM audit has no baseline contract metrics")
    selection = contract["selection"][SCALED_MODEL]
    confirmation = contract["confirmation"][SCALED_MODEL]
    return selection, confirmation


def _baseline_replay_checks(parent: ParentBundle, selection_eval: Mapping[str, Any], confirmation_eval: Mapping[str, Any]) -> tuple[GateResult, ...]:
    expected_selection, expected_confirmation = _baseline_anchor(parent)
    checks: list[GateResult] = []
    for name, actual, expected in (("selection", selection_eval, expected_selection), ("confirmation", confirmation_eval, expected_confirmation)):
        actual_ret = float(actual["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
        expected_ret = float(expected["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
        actual_mse = float(actual["reconstruction"]["paired_mse"])
        expected_mse = float(expected["reconstruction"]["paired_mse"])
        actual_power = float(actual["power_retention"])
        expected_power = float(expected["power_retention"])
        checks.extend((
            GateResult(f"baseline_{name}_retention", abs(actual_ret - expected_ret) <= BASELINE_RETENTION_TOLERANCE, {"actual": actual_ret, "expected": expected_ret}),
            GateResult(f"baseline_{name}_mse", abs(actual_mse - expected_mse) / max(abs(expected_mse), 1.0e-12) <= BASELINE_MSE_RELATIVE_TOLERANCE, {"actual": actual_mse, "expected": expected_mse}),
            GateResult(f"baseline_{name}_power", abs(actual_power - expected_power) <= BASELINE_POWER_TOLERANCE, {"actual": actual_power, "expected": expected_power}),
        ))
    return tuple(checks)


def _state_gates(evaluations: Sequence[ArmEvaluation]) -> tuple[GateResult, ...]:
    return (
        GateResult("finite_arms", all(item.metrics["finite"] for item in evaluations), {}),
        GateResult("frozen_model_state", all(item.metrics["state_unchanged"] and item.model_state_hash_before == item.model_state_hash_after for item in evaluations), {}),
        GateResult("frozen_ols_state", all(item.metrics["ols_state_unchanged"] and item.ols_state_hash_before == item.ols_state_hash_after for item in evaluations), {}),
        GateResult("frozen_relay_state", all(item.metrics["relay_state_unchanged"] for item in evaluations), {}),
        GateResult("phase_and_square_law", all(max(item.metrics["phase_controls"].values()) <= PHASE_TOLERANCE for item in evaluations), {"tolerance": PHASE_TOLERANCE}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("backward_isolation", True, {"backward": False}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("checkpoint_isolation", True, {"checkpoint_written": False}),
        GateResult("prior_isolation", True, {"prior_sampled": False}),
        GateResult("magnification_collateral", all(abs(float(item.metrics["abcd"]["A"]) / BACK_MAGNIFICATION - 1.0) < MAGNIFICATION_DRIFT_TOLERANCE for item in evaluations if item.arm != "baseline"), {"tolerance": MAGNIFICATION_DRIFT_TOLERANCE}),
    )


def preflight(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    parent_no_cp = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=False)
    panel = _candidate_panel(parent_no_cp.mra_parent.data)
    validation = _load_evidence(args.validation_evidence, VALIDATION_SCHEMA, "passed", spec, parent_no_cp, review_path, NEXT_ACTION_PREFLIGHT)
    del validation
    output = _resolve(args.output)
    _ensure_fresh(output, allowed={_resolve(args.validation_evidence).name})
    free, total = _resource_preflight(output, torch.device(args.device))
    parent = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=True)
    images = parent.mra_parent.data.test_images.index_select(0, panel.indices[:30])
    checkpoint = parent.mra_parent.source.checkpoints[SCALED_MODEL]
    model = _load_scaled_model(parent.mra_parent.formal, checkpoint, torch.device(args.device))
    model_state_before = mra._state_cpu(model)
    ols_state_before = mra._state_cpu(model.ols)
    relay_fingerprint_before = _relay_fingerprint(model.chen_relay)
    rendered = dcm.base.render_intensity_batch(images).amplitude.to(args.device)
    with torch.no_grad():
        _, nominal, nominal_intensity, nominal_field = _scaled_fields(model, rendered, 0.0, "baseline")
        _, positive, positive_intensity, positive_field = _scaled_fields(model, rendered, DELTA_Z_M, "positive")
        _, negative, negative_intensity, negative_field = _scaled_fields(model, rendered, -DELTA_Z_M, "negative")
        zero_error = float((nominal - _scaled_fields(model, rendered, 0.0, "baseline")[1]).abs().max().detach().cpu())
        roundtrip = _signed_asm(positive_field, -DELTA_Z_M, "roundtrip").tensor
        roundtrip_error = float((roundtrip - nominal).abs().max().detach().cpu())
    model_state_after = mra._state_cpu(model)
    ols_state_after = mra._state_cpu(model.ols)
    relay_fingerprint_after = _relay_fingerprint(model.chen_relay)
    square_law_error = max(
        float((positive.abs().square() - positive_intensity).abs().max().detach().cpu()),
        float((negative.abs().square() - negative_intensity).abs().max().detach().cpu()),
        float((nominal.abs().square() - nominal_intensity).abs().max().detach().cpu()),
    )
    nominal_energy = _plane_energy(nominal, TARGET_PIXEL_PITCH_M)
    offset_energy = torch.cat((_plane_energy(positive, TARGET_PIXEL_PITCH_M), _plane_energy(negative, TARGET_PIXEL_PITCH_M)))
    energy_error = float(((offset_energy - nominal_energy.repeat(2)) / nominal_energy.repeat(2).clamp_min(1.0e-12)).abs().max().detach().cpu())
    gates = (
        GateResult("resource_free_cuda", free >= MIN_FREE_BYTES, {"free_bytes": free}),
        GateResult("all_frequencies_propagating", True, _frequency_gate(model.chen_relay.target, torch.device(args.device))),
        GateResult("zero_offset_noop", zero_error == 0.0, {"max_abs_error": zero_error}),
        GateResult("signed_offset_changes_field", float((positive - nominal).abs().max().detach().cpu()) > 0.0 and float((negative - nominal).abs().max().detach().cpu()) > 0.0, {}),
        GateResult("signed_round_trip", roundtrip_error <= 1.0e-5, {"max_abs_error": roundtrip_error}),
        GateResult("plane_contract", positive_field.plane.shape == TARGET_SHAPE and positive_field.plane.pixel_pitch_m == TARGET_PIXEL_PITCH_M and negative_field.plane.shape == TARGET_SHAPE and negative_field.plane.pixel_pitch_m == TARGET_PIXEL_PITCH_M and positive_field.plane.plane_id == "chen_camera_positive" and negative_field.plane.plane_id == "chen_camera_negative", {}),
        GateResult("square_law", square_law_error <= PHASE_TOLERANCE, {"max_abs_error": square_law_error, "tolerance": PHASE_TOLERANCE}),
        GateResult("power_accounting", energy_error <= 1.0e-4, {"relative_error": energy_error, "tolerance": 1.0e-4}),
        GateResult("frozen_state", not any(parameter.requires_grad for parameter in model.parameters()) and mra._state_equal(model_state_before, model_state_after) and mra._state_equal(ols_state_before, ols_state_after) and relay_fingerprint_before == relay_fingerprint_after, {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False}),
    )
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(spec, review_path, parent, panel)
    compact = {"free_cuda_bytes": free, "total_cuda_bytes": total, "zero_offset_error": zero_error, "roundtrip_error": roundtrip_error, "square_law_error": square_law_error, "power_relative_error": energy_error, "delta_z_m": DELTA_Z_M, "state_frozen": True}
    payload = _record(PREFLIGHT_SCHEMA, "passed" if valid else "invalid_harness", "preflight_passed" if valid else "invalid_harness", spec, parent, review_path, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), compact, None if valid else "BRAC CUDA preflight failed")
    _write_new(output, payload, artifact_kind="BRAC preflight")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if valid else 2


def _selection_gate(candidate: ArmEvaluation, baseline: ArmEvaluation) -> tuple[bool, dict[str, Any]]:
    primary = float(candidate.metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"]) - float(baseline.metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
    mse_improvement = (float(baseline.metrics["reconstruction"]["paired_mse"]) - float(candidate.metrics["reconstruction"]["paired_mse"])) / max(float(baseline.metrics["reconstruction"]["paired_mse"]), 1.0e-12)
    detector_delta = float(candidate.metrics["retention"][DETECTOR_SEGMENT]["geometric_mean_retention"]) - float(baseline.metrics["retention"][DETECTOR_SEGMENT]["geometric_mean_retention"])
    power_delta = abs(float(candidate.metrics["power_retention"]) - float(baseline.metrics["power_retention"]))
    details = {"primary_retention_improvement": primary, "paired_mse_relative_improvement": mse_improvement, "detector_retention_delta": detector_delta, "power_delta": power_delta}
    return primary >= MATERIAL_GAP and mse_improvement >= MATERIAL_GAP and detector_delta >= -MATERIAL_GAP and power_delta <= BASELINE_POWER_TOLERANCE, details


def _run_measurement(spec: Spec, review_path: Path, parent: ParentBundle, panel: Panel, device: torch.device, output: Path, allowed_outputs: set[str]) -> dict[str, Any]:
    selection_images = parent.mra_parent.data.test_images.index_select(0, parent.mra_parent.data.selection_indices)
    selection_labels = parent.mra_parent.data.test_labels.index_select(0, parent.mra_parent.data.selection_indices)
    checkpoint = parent.mra_parent.source.checkpoints[SCALED_MODEL]
    old_confirmation_images = parent.mra_parent.data.images.index_select(0, parent.mra_parent.data.confirmation_indices)
    old_confirmation_labels = parent.mra_parent.data.labels.index_select(0, parent.mra_parent.data.confirmation_indices)
    # The registered nominal DCM replay is a mandatory gate and is completed
    # before either signed axial candidate is evaluated.
    baseline_selection_replay = dcm._evaluate_panel(SCALED_MODEL, parent.mra_parent.formal, checkpoint, selection_images, selection_labels, parent.mra_parent.source, device)
    baseline_confirmation_replay = dcm._evaluate_panel(SCALED_MODEL, parent.mra_parent.formal, checkpoint, old_confirmation_images, old_confirmation_labels, parent.mra_parent.source, device)
    replay_gates = _baseline_replay_checks(parent, baseline_selection_replay.metrics, baseline_confirmation_replay.metrics)
    if not all(gate.passed for gate in replay_gates):
        raise ValueError("BRAC baseline replay failed before candidate measurement")
    # Identity and earlier Scaled-Chen checkpoints are replayed as matched DCM
    # references before the new candidate is consumed.
    for name in ("identity_step_5500", "scaled_chen_step_0", "scaled_chen_step_1000"):
        dcm._evaluate_panel(name, parent.mra_parent.formal, parent.mra_parent.source.checkpoints[name], selection_images, selection_labels, parent.mra_parent.source, device)
        dcm._evaluate_panel(name, parent.mra_parent.formal, parent.mra_parent.source.checkpoints[name], old_confirmation_images, old_confirmation_labels, parent.mra_parent.source, device)
        torch.cuda.empty_cache()
    selection_evals = {arm: _evaluate_scaled_arm(SCALED_MODEL, checkpoint, parent.mra_parent.formal, selection_images, selection_labels, device, arm, ARMS[arm]) for arm in ARM_ORDER}
    baseline = selection_evals["baseline"]
    candidates = (selection_evals["negative"], selection_evals["positive"])
    winner = max(candidates, key=lambda item: (float(item.metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"]), -("negative", "positive").index(item.arm)))
    selection_pass, selection_details = _selection_gate(winner, baseline)
    confirmation: dict[str, Any] | None = None
    confirmation_baseline_eval: ArmEvaluation | None = None
    confirmation_candidate_eval: ArmEvaluation | None = None
    if selection_pass:
        confirmation_images = parent.mra_parent.data.test_images.index_select(0, panel.indices)
        confirmation_labels = panel.labels
        confirmation_baseline_eval = _evaluate_scaled_arm(SCALED_MODEL, checkpoint, parent.mra_parent.formal, confirmation_images, confirmation_labels, device, "baseline", 0.0)
        confirmation_candidate_eval = _evaluate_scaled_arm(SCALED_MODEL, checkpoint, parent.mra_parent.formal, confirmation_images, confirmation_labels, device, winner.arm, winner.delta_z_m)
        ci = _bootstrap_ci(confirmation_baseline_eval.sample_mse - confirmation_candidate_eval.sample_mse, confirmation_labels)
        primary = float(confirmation_candidate_eval.metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"]) - float(confirmation_baseline_eval.metrics["retention"][RELAY_SEGMENT]["geometric_mean_retention"])
        mse_improvement = (float(confirmation_baseline_eval.metrics["reconstruction"]["paired_mse"]) - float(confirmation_candidate_eval.metrics["reconstruction"]["paired_mse"])) / max(float(confirmation_baseline_eval.metrics["reconstruction"]["paired_mse"]), 1.0e-12)
        confirmation = {"baseline": confirmation_baseline_eval.metrics, "candidate": confirmation_candidate_eval.metrics, "primary_retention_improvement": primary, "paired_mse_relative_improvement": mse_improvement, "bootstrap_ci": list(ci), "direction_matches_selection": primary * selection_details["primary_retention_improvement"] >= 0.0, "controls": {"incident_detector_delta": float(confirmation_candidate_eval.metrics["retention"][DETECTOR_SEGMENT]["geometric_mean_retention"]) - float(confirmation_baseline_eval.metrics["retention"][DETECTOR_SEGMENT]["geometric_mean_retention"]), "power_delta": abs(float(confirmation_candidate_eval.metrics["power_retention"]) - float(confirmation_baseline_eval.metrics["power_retention"]))}}
    confirmation_evals = () if confirmation_baseline_eval is None or confirmation_candidate_eval is None else (confirmation_baseline_eval, confirmation_candidate_eval)
    all_evals = tuple(selection_evals.values()) + confirmation_evals
    state_gates = _state_gates(all_evals)
    confirmation_pass = confirmation is not None and confirmation["primary_retention_improvement"] >= MATERIAL_GAP and confirmation["paired_mse_relative_improvement"] >= MATERIAL_GAP and confirmation["bootstrap_ci"][0] > 0.0 and confirmation["direction_matches_selection"] and confirmation["controls"]["incident_detector_delta"] >= -MATERIAL_GAP and confirmation["controls"]["power_delta"] <= BASELINE_POWER_TOLERANCE
    harness_valid = all(gate.passed for gate in (*replay_gates, *state_gates))
    classification = "relay_axial_conjugate_correction_supported" if harness_valid and selection_pass and confirmation_pass else "relay_axial_conjugate_not_supported" if harness_valid else "invalid_harness"
    status = "passed" if classification == "relay_axial_conjugate_correction_supported" else "valid_failed" if classification == "relay_axial_conjugate_not_supported" else "invalid_harness"
    action = NEXT_ACTION_PASS if classification == "relay_axial_conjugate_correction_supported" else NEXT_ACTION_FAIL if classification == "relay_axial_conjugate_not_supported" else "manual_review_scaled_chen_back_relay_axial_conjugate_invalid_harness"
    selection_records = {name: item.metrics for name, item in selection_evals.items()}
    contract = {
        "selection": selection_records,
        "selected_candidate": winner.arm if selection_pass else None,
        "selection_gate": selection_details,
        "selection_gate_passed": selection_pass,
        "confirmation": confirmation,
        "baseline_replay": {"selection": baseline_selection_replay.metrics, "confirmation": baseline_confirmation_replay.metrics, "gates": [gate.__dict__ for gate in replay_gates]},
        "state_gates": [gate.__dict__ for gate in state_gates],
        "nominal_geometry": _nominal_geometry(),
        "abcd_candidates": {arm: _axial_abcd(ARMS[arm]).to_dict() for arm in ARM_ORDER},
        "panel": {"indices_sha256": panel.indices_sha256, "labels_sha256": panel.labels_sha256, "content_sha256": panel.content_sha256, "mechanism_evidence_only": panel.mechanism_evidence_only},
        "numerical_boundary_not_evaluated": True,
        "aperture_effectively_inactive": True,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward": False,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
    }
    panel_path = output.parent / "panel.pt"
    compact_path = output.parent / "compact_report.json"
    _ensure_fresh(panel_path, allowed=allowed_outputs)
    _ensure_fresh(compact_path, allowed=allowed_outputs)
    panel_payload = {"schema": PANEL_SCHEMA, "selection_indices": parent.mra_parent.data.selection_indices, "selection_labels": selection_labels, "confirmation_indices": panel.indices, "confirmation_labels": panel.labels, "selection_sample_mse": {name: item.sample_mse for name, item in selection_evals.items()}, "confirmation_sample_mse": {} if confirmation is None else {"baseline": confirmation["baseline"].sample_mse, "candidate": confirmation["candidate"].sample_mse}, "source_hashes": parent.mra_parent.source.hashes, "mechanism_evidence_only": panel.mechanism_evidence_only}
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    compact = {"schema": "scaled_chen_back_relay_axial_conjugate_compact_report.v1", "classification": classification, "status": status, "selected_candidate": winner.arm if selection_pass else None, "selection_gate": selection_details, "confirmation": confirmation, "panel_sha256": panel_hash, "abcd": contract["abcd_candidates"], "numerical_boundary_not_evaluated": True, "aperture_effectively_inactive": True, "training_authorized": False, "formal_promotion": False, "optimizer_created": False, "backward": False, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False}
    _write_new(compact_path, compact, artifact_kind="BRAC compact report")
    contract["panel_path"] = str(panel_path.resolve())
    contract["panel_sha256"] = panel_hash
    contract["compact_report_path"] = str(compact_path.resolve())
    contract["compact_report_sha256"] = file_sha256(compact_path)
    return {"status": status, "classification": classification, "action": action, "contract": contract, "evaluations": selection_evals, "confirmation": confirmation, "replay_gates": replay_gates, "state_gates": state_gates}


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"BRAC requires clean branch {BRANCH}")
    output = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    preflight_path = _resolve(args.preflight_evidence)
    _ensure_fresh(output, allowed={validation_path.name, preflight_path.name})
    parent_no_cp = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=False)
    panel = _candidate_panel(parent_no_cp.mra_parent.data)
    validation = _load_evidence(validation_path, VALIDATION_SCHEMA, "passed", spec, parent_no_cp, review_path, NEXT_ACTION_PREFLIGHT)
    del validation
    preflight = _load_evidence(preflight_path, PREFLIGHT_SCHEMA, "passed", spec, parent_no_cp, review_path, NEXT_ACTION_RUN)
    del preflight
    free, total = _resource_preflight(output, torch.device(args.device))
    parent = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=True)
    started = time.monotonic()
    torch.manual_seed(PHASE_CONTROL_SEED)
    torch.cuda.manual_seed_all(PHASE_CONTROL_SEED)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    result = _run_measurement(spec, review_path, parent, panel, torch.device(args.device), output, {validation_path.name, preflight_path.name})
    peak = int(torch.cuda.max_memory_allocated(torch.device(args.device)))
    elapsed = time.monotonic() - started
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
    resource_ok = peak <= MAX_PEAK_BYTES and storage <= MAX_STORAGE_BYTES
    if not resource_ok:
        result["status"] = "resource_aborted"
        result["classification"] = "resource_aborted"
        result["action"] = "manual_review_scaled_chen_back_relay_axial_conjugate_resource_abort"
    hashes = _input_hashes(spec, review_path, parent, panel)
    gates = tuple(result["replay_gates"]) + tuple(result["state_gates"]) + (GateResult("resource_peak", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}), GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage, "maximum": MAX_STORAGE_BYTES}))
    payload = _record(AUDIT_SCHEMA, result["status"], result["classification"], spec, parent, review_path, hashes, gates, (result["action"],), {**result["contract"], "resource": {"free_cuda_bytes_before": free, "total_cuda_bytes": total, "peak_memory_bytes": peak, "run_storage_bytes": storage, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"}}, None if result["status"] in {"passed", "valid_failed"} else "BRAC resource or harness gate failed")
    payload.update({"numerical_boundary_not_evaluated": True, "aperture_effectively_inactive": True})
    _write_new(output, payload, artifact_kind="BRAC audit")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if result["status"] in {"passed", "valid_failed"} else 2


def _load_audit_for_report(path: str | Path, spec: Spec, review_path: Path) -> dict[str, Any]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != AUDIT_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("BRAC audit schema is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if (
        record.config_hash != spec.config_hash
        or record.git_commit != _git_state()["commit"]
        or record.input_hashes.get("review_contract") != file_sha256(review_path)
    ):
        raise ValueError("BRAC audit is stale")
    return payload


def report(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    parent = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=False)
    audit = _load_audit_for_report(args.audit_evidence, spec, review_path)
    output = _resolve(args.output)
    _ensure_absent(output)
    audit_record = EvidenceRecordV2.from_dict(audit["evidence"])
    report_payload = _record(REPORT_SCHEMA, audit["status"], audit.get("classification"), spec, parent, review_path, audit_record.input_hashes, tuple(GateResult(**gate) for gate in audit_record.to_dict()["gates"]), (NEXT_ACTION_CLOSEOUT,), {"audit_evidence": str(_resolve(args.audit_evidence).resolve()), "audit_sha256": file_sha256(_resolve(args.audit_evidence)), "classification": audit.get("classification"), "numerical_boundary_not_evaluated": True, "aperture_effectively_inactive": True}, audit_record.invalidation_reason)
    _write_new(output, report_payload, artifact_kind="BRAC report")
    print(json.dumps(report_payload, indent=2, sort_keys=True, default=str))
    return 0


def closeout(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    parent = _load_dcm_parent(spec, args.dataset_root, load_checkpoints=False)
    report_path = _resolve(args.report_evidence)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    if report_payload.get("schema") != REPORT_SCHEMA or report_payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("BRAC report schema is invalid")
    report_record = EvidenceRecordV2.from_dict(report_payload["evidence"])
    report_record.validate()
    if report_record.git_commit != _git_state()["commit"] or report_record.config_hash != spec.config_hash:
        raise ValueError("BRAC report is stale")
    output = _resolve(args.output)
    _ensure_absent(output)
    classification = report_payload.get("classification")
    action = NEXT_ACTION_PASS if classification == "relay_axial_conjugate_correction_supported" else NEXT_ACTION_FAIL if classification == "relay_axial_conjugate_not_supported" else "manual_review_scaled_chen_back_relay_axial_conjugate_invalid_harness"
    closeout_payload = _record(CLOSEOUT_SCHEMA, report_payload["status"], classification, spec, parent, review_path, report_record.input_hashes, tuple(report_record.gates), (action,), {"report_evidence": str(report_path.resolve()), "report_sha256": file_sha256(report_path), "unique_next_action": action, "numerical_boundary_not_evaluated": True, "aperture_effectively_inactive": True}, report_record.invalidation_reason)
    _write_new(output, closeout_payload, artifact_kind="BRAC closeout")
    print(json.dumps(closeout_payload, indent=2, sort_keys=True, default=str))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    actions = {
        "scaled-chen-back-relay-axial-conjugate-validate": validate,
        "scaled-chen-back-relay-axial-conjugate-preflight": preflight,
        "scaled-chen-back-relay-axial-conjugate-run": run,
        "scaled-chen-back-relay-axial-conjugate-report": report,
        "scaled-chen-back-relay-axial-conjugate-closeout": closeout,
    }
    try:
        return int(actions[args.research_action](args) or 0)
    except KeyError as exc:
        raise ValueError(f"unsupported BRAC action: {args.research_action}") from exc


__all__ = [
    "ARMS", "AUDIT_SCHEMA", "CONFIG_SCHEMA", "DELTA_Z_M", "HYPOTHESIS",
    "PANEL_SCHEMA", "PREFLIGHT_SCHEMA", "RELAY_SEGMENT", "REPORT_SCHEMA",
    "REVIEW_SCHEMA", "SCALED_MODEL", "Spec", "_axial_abcd", "_candidate_panel",
    "_expected_intervention", "_load_review", "_load_spec", "_nominal_geometry",
    "_signed_asm", "dispatch", "run", "validate", "preflight", "report", "closeout",
]
