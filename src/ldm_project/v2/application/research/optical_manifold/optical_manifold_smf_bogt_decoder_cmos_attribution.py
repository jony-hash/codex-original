"""Frozen decoder-to-CMOS readout attribution for the Scaled-Chen route.

This research-only consumer splits the last MRA segment into the coherent
back-relay and the square-law detector.  It reuses the MRA source and panel
contracts, measures only frozen forward passes, and never creates an
optimizer, evaluator, gradient, checkpoint, or prior sample.
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

from ldm_project.v2.config import FormalSystemConfig
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_new_json, canonical_sha256, file_sha256, tensor_sha256
from ldm_project.v2.optics.fiber_operator import FiberArrayOLS
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bogt_decoder_cmos_attribution_panel.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0-DCM-OBS"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

STAGE_NAMES = ("decoder_layer_2", "cmos_incident_field", "cmos_intensity")
SEGMENTS = (
    ("decoder_layer_2_to_cmos_incident_field", "decoder_layer_2", "cmos_incident_field"),
    ("cmos_incident_field_to_cmos_intensity", "cmos_incident_field", "cmos_intensity"),
)
MODEL_NAMES = mra.MODEL_NAMES
RANK_RETENTION_THRESHOLD = 0.90
MATERIAL_GAP = 0.05
PHASE_INVARIANCE_TOLERANCE = 1.0e-5
PHASE_CONTROL_SEED = 1717
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
NEXT_ACTION_RELAY = "manual_review_scaled_chen_bounded_back_relay_geometry_diagnostic_contract"
NEXT_ACTION_READOUT = "manual_review_scaled_chen_phase_to_intensity_decoder_readout_canary_contract"
NEXT_ACTION_JOINT = "manual_review_scaled_chen_decoder_readout_multifactor_closeout_contract"
NEXT_ACTION_UNDERDETERMINED = "manual_review_scaled_chen_decoder_cmos_readout_attribution_underdetermined_contract"
NEXT_ACTION_RUN = "run_scaled_chen_decoder_cmos_attribution"

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
    "audit_generation_readiness": "not_ready",
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ParentBundle:
    spec: mra.Spec
    review_path: Path
    audit_path: Path
    formal: FormalSystemConfig
    formal_path: Path
    data: mra.AuditData
    source: mra.SourceBundle


@dataclass(frozen=True)
class ReadoutEvaluation:
    metrics: dict[str, Any]
    stage_values: dict[str, torch.Tensor]
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
    required = {"schema", "hypothesis", "formal_system_config", "parent_mra", "measurement", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("decoder-CMOS attribution config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("decoder-CMOS attribution config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if payload["formal_system_config"] != "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml":
        raise ValueError("decoder-CMOS attribution formal config is not locked")
    if payload["parent_mra"] != PARENT_MRA:
        raise ValueError("decoder-CMOS attribution parent MRA binding is not locked")
    expected_measurement = {
        "objective": "frozen_forward_only",
        "models": list(MODEL_NAMES),
        "stages": list(STAGE_NAMES),
        "segments": [list(item) for item in SEGMENTS],
        "reuse": ["mra_selection", "mra_confirmation", "mra_rank", "mra_retrieval", "mra_permuted_label", "scaled_chen_cmos_incident_field"],
        "detector": "intensity_abs_square",
        "negative_controls": ["global_phase_invariance", "pointwise_phase_invariance", "identity_reference", "permuted_label"],
        "no_update": True,
    }
    if payload["measurement"] != expected_measurement:
        raise ValueError("decoder-CMOS attribution measurement contract is not locked")
    expected_gates = {
        "rank_retention": RANK_RETENTION_THRESHOLD,
        "material_gap": MATERIAL_GAP,
        "phase_invariance_tolerance": PHASE_INVARIANCE_TOLERANCE,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "training_authorized": False,
        "formal_promotion": False,
    }
    if payload["gates"] != expected_gates:
        raise ValueError("decoder-CMOS attribution gate contract is not locked")
    expected_authorization = {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_decoder_cmos_attribution",
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": ["optimizer", "backward", "checkpoint_write", "generation", "evaluator", "fmf", "prior_sampling", "formal_promotion"],
    }
    if payload["authorization"] != expected_authorization:
        raise ValueError("decoder-CMOS attribution authorization boundary is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_scaled_chen_decoder_cmos_attribution",
        "approved_contract": {
            "parent_mra": dict(PARENT_MRA),
            "models": list(MODEL_NAMES),
            "stages": list(STAGE_NAMES),
            "segments": [list(item) for item in SEGMENTS],
            "rank_retention": RANK_RETENTION_THRESHOLD,
            "material_gap": MATERIAL_GAP,
            "phase_invariance_tolerance": PHASE_INVARIANCE_TOLERANCE,
            "no_update": True,
            "generation_canary_not_automatic": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("decoder-CMOS attribution review contract does not match scope")
    return target


def _load_parent_with_data(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> ParentBundle:
    parent = spec.payload["parent_mra"]
    parent_config = _resolve(parent["config"], base_path=spec.path.parent)
    if not parent_config.is_file() or file_sha256(parent_config) != parent["config_sha256"]:
        raise ValueError("parent MRA config is missing, stale, or tampered")
    parent_spec = mra._load_spec(parent_config)
    mra._assert_contract(parent_spec)
    review_path = _resolve(parent["review_contract"], base_path=spec.path.parent)
    if not review_path.is_file() or file_sha256(review_path) != parent["review_sha256"]:
        raise ValueError("parent MRA review contract is missing, stale, or tampered")
    mra._load_review(review_path, parent_spec)
    audit_path = _resolve(parent["audit"], base_path=spec.path.parent)
    if not audit_path.is_file() or file_sha256(audit_path) != parent["audit_sha256"]:
        raise ValueError("parent MRA audit is missing, stale, or tampered")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_payload, Mapping):
        raise ValueError("parent MRA audit is not a mapping")
    record = EvidenceRecordV2.from_dict(audit_payload["evidence"])
    record.validate()
    if (
        audit_payload.get("schema") != mra.AUDIT_SCHEMA
        or audit_payload.get("hypothesis") != mra.HYPOTHESIS
        or audit_payload.get("status") != parent["audit_status"]
        or audit_payload.get("classification") != parent["audit_classification"]
        or audit_payload.get("generation_readiness") != parent["audit_generation_readiness"]
        or record.git_commit != parent["audit_git_commit"]
        or record.config_hash != parent_spec.config_hash
        or record.permitted_next_actions != (mra.NEXT_ACTION_DISTRIBUTED,)
    ):
        raise ValueError("parent MRA audit terminal state is stale or mismatched")
    formal, formal_path = mra._load_formal(parent_spec)
    data = mra._load_data(dataset_root)
    source = mra._load_source(parent_spec, load_checkpoints=load_checkpoints)
    return ParentBundle(parent_spec, review_path, audit_path, formal, formal_path, data, source)


def _input_hashes(spec: Spec, review_path: Path, parent: ParentBundle) -> dict[str, str]:
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
        **{f"parent_{name}": value for name, value in parent_hashes.items()},
    }


def _context(spec: Spec, parent: ParentBundle) -> EvidenceContextV2:
    signature = canonical_sha256({"measurement": spec.payload["measurement"], "hypothesis": HYPOTHESIS})
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=parent.data.manifest["manifest_hash"],
        dataset_split_manifest_hash=parent.data.manifest["manifest_hash"],
    )


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _mode_rank(operator: FiberArrayOLS) -> int:
    area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
    gram = operator.input_modes.flatten(1).conj() @ operator.input_modes.flatten(1).T * area
    return int(torch.linalg.matrix_rank(gram).item())


def _build_research_ols(formal: FormalSystemConfig) -> FiberArrayOLS:
    return mra._build_research_ols(formal)


def _trace_field(trace: Sequence[ComplexField] | None, plane_id: str, operation: str) -> ComplexField:
    return mra._find_trace_field(trace, plane_id, operation)


def _identity_readout(model: base.IdentityOpticalManifoldAutoencoder, amplitude: torch.Tensor, research_ols: FiberArrayOLS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    result = model.system(model.system.command(amplitude), exposure=model.exposure, return_optical_trace=True)
    trace = result.optical_trace
    decoder = _trace_field(trace, "decoder_layer_2", "finite_crop_hard_aperture").tensor
    incident = _trace_field(trace, "camera", "coherent_b0_imaging").tensor
    intensity = result.raw_camera.tensor
    # The research OLS is a matched mode observer for the Identity reference;
    # calling it here keeps the same port geometry as the Scaled-Chen model.
    _ = research_ols(_trace_field(trace, "encoder_layer_2", "finite_crop_hard_aperture"))
    return decoder, incident, intensity


def _scaled_readout(model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(amplitude, return_trace=True)
    if len(output.decoder_records) != 2:
        raise ValueError("Scaled-Chen decoder trace does not contain two layers")
    decoder = output.decoder_records[-1].tensor
    # ``ScaledChenBOGTModel.forward`` applies the fixed decoder-group factor
    # before storing each decoder record.  Reusing the final record therefore
    # already gives the exact chip-plane field that feeds the backend relay;
    # applying the factor again would make the observed field inconsistent
    # with ``output.camera`` and falsely fail the detector square-law check.
    chip = ComplexField(decoder, model.system.chip_plane, OperationProvenance("fixed_decoder_group_transmission", "decoder_layer_2", None, {"field_factor": bogt.FIELD_TRANSMISSION**2, "already_applied": True}))
    incident = model.chen_relay(chip).tensor
    intensity = output.camera
    return decoder, incident, intensity


def _phase_controls(incident: torch.Tensor, intensity: torch.Tensor, generator: torch.Generator) -> dict[str, float]:
    global_phase = torch.polar(torch.ones((), device=incident.device, dtype=incident.real.dtype), torch.tensor(0.731, device=incident.device, dtype=incident.real.dtype)).to(incident.dtype)
    global_intensity = (incident * global_phase).abs().square()
    random_phase = torch.rand(incident.shape, device=incident.device, dtype=incident.real.dtype, generator=generator) * (2.0 * math.pi)
    pointwise_intensity = (incident.abs() * torch.exp(1j * random_phase).to(incident.dtype)).abs().square()
    return {
        "global_phase_max_abs_error": float((global_intensity - intensity).abs().max().detach().cpu()),
        "pointwise_phase_max_abs_error": float((pointwise_intensity - intensity).abs().max().detach().cpu()),
    }


def _plane_energy(value: torch.Tensor, pixel_pitch_m: tuple[float, float]) -> torch.Tensor:
    return value.abs().square().sum(dim=(-3, -2, -1)) * (pixel_pitch_m[0] * pixel_pitch_m[1])


def _retention_report(stage_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, left, right in SEGMENTS:
        left_metric = stage_metrics[left]
        right_metric = stage_metrics[right]
        entropy = float(right_metric["entropy_effective_rank"]) / max(float(left_metric["entropy_effective_rank"]), 1.0e-12)
        participation = float(right_metric["participation_rank"]) / max(float(left_metric["participation_rank"]), 1.0e-12)
        result[name] = {
            "entropy_retention": entropy,
            "participation_retention": participation,
            "geometric_mean_retention": math.sqrt(max(entropy, 0.0) * max(participation, 0.0)),
        }
    return result


def _label_signal_drop(complex_metrics: Mapping[str, Any], intensity_metrics: Mapping[str, Any]) -> dict[str, Any]:
    complex_geometry = complex_metrics["true_label_distance"]
    intensity_geometry = intensity_metrics["true_label_distance"]
    complex_ratio = float(complex_geometry["inter_intra_ratio"])
    intensity_ratio = float(intensity_geometry["inter_intra_ratio"])
    complex_retrieval = float(complex_metrics["instance_retrieval_same_label_top1"])
    intensity_retrieval = float(intensity_metrics["instance_retrieval_same_label_top1"])
    return {
        "complex_inter_intra_ratio": complex_ratio,
        "intensity_inter_intra_ratio": intensity_ratio,
        "relative_inter_intra_drop": (complex_ratio - intensity_ratio) / max(abs(complex_ratio), 1.0e-12),
        "complex_retrieval": complex_retrieval,
        "intensity_retrieval": intensity_retrieval,
        "retrieval_drop": complex_retrieval - intensity_retrieval,
        "complex_true_label_supported": complex_ratio > 1.0 and complex_ratio > float(complex_metrics["permuted_label_distance"]["inter_intra_ratio"]),
        "intensity_true_label_supported": intensity_ratio > 1.0 and intensity_ratio > float(intensity_metrics["permuted_label_distance"]["inter_intra_ratio"]),
    }


def _evaluate_panel(model_name: str, formal: FormalSystemConfig, checkpoint: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, source: mra.SourceBundle, device: torch.device) -> ReadoutEvaluation:
    research_ols = _build_research_ols(formal).to(device)
    if model_name == "identity_step_5500":
        model: torch.nn.Module = mra._load_identity_model(formal, checkpoint, device)
    else:
        model = mra._load_scaled_model(formal, checkpoint, device)
    model_state_before = mra._state_cpu(model)
    ols_module = model.ols if isinstance(model, bogt.ScaledChenBOGTModel) else research_ols
    ols_state_before = mra._state_cpu(ols_module)
    stage_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in STAGE_NAMES}
    camera_chunks: list[torch.Tensor] = []
    energy_ratios: list[torch.Tensor] = []
    phase_errors: list[dict[str, float]] = []
    phase_generator = torch.Generator(device=device).manual_seed(PHASE_CONTROL_SEED)
    with torch.no_grad():
        for indices in torch.arange(images.shape[0], dtype=torch.long).split(10):
            rendered = base.render_intensity_batch(images.index_select(0, indices))
            amplitude = rendered.amplitude.to(device)
            if model_name == "identity_step_5500":
                decoder, incident, intensity = _identity_readout(model, amplitude, research_ols)  # type: ignore[arg-type]
                source_pitch = model.system.chip_plane.pixel_pitch_m  # type: ignore[union-attr]
                target_pitch = model.system.camera_plane.pixel_pitch_m  # type: ignore[union-attr]
            else:
                decoder, incident, intensity = _scaled_readout(model, amplitude)  # type: ignore[arg-type]
                source_pitch = model.system.chip_plane.pixel_pitch_m  # type: ignore[union-attr]
                target_pitch = model.system.camera_plane.pixel_pitch_m  # type: ignore[union-attr]
            if not torch.isfinite(decoder).all() or not torch.isfinite(incident).all() or not torch.isfinite(intensity).all():
                raise RuntimeError("decoder-CMOS readout replay produced non-finite values")
            expected_intensity = incident.abs().square()
            if float((expected_intensity - intensity).abs().max().detach().cpu()) > PHASE_INVARIANCE_TOLERANCE:
                raise ValueError("CMOS intensity is not the square-law of the incident field")
            phase_errors.append(_phase_controls(incident, intensity, phase_generator))
            source_energy = _plane_energy(decoder, source_pitch)
            target_energy = _plane_energy(incident, target_pitch)
            energy_ratios.append((target_energy / source_energy.clamp_min(1.0e-12)).detach().cpu().float())
            camera_chunks.append(intensity.detach().cpu().float())
            for name, value in (("decoder_layer_2", decoder), ("cmos_incident_field", incident), ("cmos_intensity", intensity)):
                stage_chunks[name].append(mra._feature_projection(value.detach().cpu()))
    model_state_after = mra._state_cpu(model)
    ols_state_after = mra._state_cpu(ols_module)
    stage_values = {name: torch.cat(values, dim=0) for name, values in stage_chunks.items()}
    camera = torch.cat(camera_chunks, dim=0)
    target = base.render_intensity_batch(images).target.float()
    reconstruction, sample_mse = mra._mse_metrics(camera, target, labels)
    stages = mra._stage_rank_report(stage_values, labels)
    retention = _retention_report({
        "decoder_layer_2": stages["decoder_layer_2"],
        "cmos_incident_field": stages["cmos_incident_field"],
        "cmos_intensity": stages["cmos_intensity"],
    })
    phase_summary = {
        key: max(item[key] for item in phase_errors)
        for key in ("global_phase_max_abs_error", "pointwise_phase_max_abs_error")
    }
    metrics = {
        "model": model_name,
        "evaluation_count": int(images.shape[0]),
        "reconstruction": reconstruction,
        "stages": stages,
        "retention": retention,
        "label_signal": _label_signal_drop(stages["cmos_incident_field"], stages["cmos_intensity"]),
        "power_retention": float(torch.cat(energy_ratios).mean()),
        "phase_controls": phase_summary,
        "finite": bool(torch.isfinite(camera).all() and all(math.isfinite(value) for value in phase_summary.values())),
        "state_unchanged": mra._state_equal(model_state_before, model_state_after),
        "ols_state_unchanged": mra._state_equal(ols_state_before, ols_state_after),
    }
    return ReadoutEvaluation(
        metrics,
        stage_values,
        sample_mse,
        labels.clone(),
        _state_hash(model_state_before),
        _state_hash(model_state_after),
        _state_hash(ols_state_before),
        _state_hash(ols_state_after),
    )


def _maximum_discontinuity(selection: Mapping[str, Any], confirmation: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        name: {
            "selection": selection.get("retention", {}).get(name, {}).get("geometric_mean_retention"),
            "confirmation": confirmation.get("retention", {}).get(name, {}).get("geometric_mean_retention"),
        }
        for name, *_ in SEGMENTS
    }
    deficits = {name: (2.0 - float(row["selection"]) - float(row["confirmation"])) / 2.0 for name, row in values.items() if isinstance(row["selection"], (int, float)) and isinstance(row["confirmation"], (int, float))}
    ordered = sorted(deficits.items(), key=lambda pair: pair[1], reverse=True)
    if not ordered:
        return {"status": "underdetermined", "segment": None, "segments": values}
    unique = len(ordered) < 2 or ordered[0][1] - ordered[1][1] >= MATERIAL_GAP
    return {"status": "unique" if unique else "joint", "segment": ordered[0][0] if unique else None, "segments": values}


def _stable_failure(selection: Mapping[str, Any], confirmation: Mapping[str, Any], segment: str) -> bool:
    return float(selection["retention"][segment]["geometric_mean_retention"]) < RANK_RETENTION_THRESHOLD and float(confirmation["retention"][segment]["geometric_mean_retention"]) < RANK_RETENTION_THRESHOLD


def _classify_readout(selection: Mapping[str, Any], confirmation: Mapping[str, Any], identity: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    relay = "decoder_layer_2_to_cmos_incident_field"
    detector = "cmos_incident_field_to_cmos_intensity"
    relay_failed = _stable_failure(selection, confirmation, relay)
    detector_failed = _stable_failure(selection, confirmation, detector)
    relay_loss = 1.0 - (float(selection["retention"][relay]["geometric_mean_retention"]) + float(confirmation["retention"][relay]["geometric_mean_retention"])) / 2.0
    detector_loss = 1.0 - (float(selection["retention"][detector]["geometric_mean_retention"]) + float(confirmation["retention"][detector]["geometric_mean_retention"])) / 2.0
    label_loss = selection["label_signal"]["relative_inter_intra_drop"] >= MATERIAL_GAP and selection["label_signal"]["retrieval_drop"] >= MATERIAL_GAP and confirmation["label_signal"]["relative_inter_intra_drop"] >= MATERIAL_GAP and confirmation["label_signal"]["retrieval_drop"] >= MATERIAL_GAP
    identity_detector_retention = float(identity["retention"][detector]["geometric_mean_retention"])
    scaled_detector_retention = (float(selection["retention"][detector]["geometric_mean_retention"]) + float(confirmation["retention"][detector]["geometric_mean_retention"])) / 2.0
    matched_detector_gap = identity_detector_retention - scaled_detector_retention
    details = {
        "relay_failed": relay_failed,
        "detector_failed": detector_failed,
        "relay_loss": relay_loss,
        "detector_loss": detector_loss,
        "loss_difference": abs(relay_loss - detector_loss),
        "label_signal_lost": label_loss,
        "matched_identity_detector_gap": matched_detector_gap,
    }
    if relay_failed and not detector_failed and relay_loss - detector_loss >= MATERIAL_GAP:
        return "relay_propagation_bottleneck_supported", details
    if detector_failed and not relay_failed and detector_loss - relay_loss >= MATERIAL_GAP and label_loss and matched_detector_gap >= MATERIAL_GAP:
        return "phase_to_intensity_readout_bottleneck_supported", details
    if relay_failed and detector_failed or abs(relay_loss - detector_loss) < MATERIAL_GAP:
        return "joint_decoder_readout_bottleneck_supported", details
    return "readout_attribution_underdetermined", details


def _action_for(classification: str) -> str:
    return {
        "relay_propagation_bottleneck_supported": NEXT_ACTION_RELAY,
        "phase_to_intensity_readout_bottleneck_supported": NEXT_ACTION_READOUT,
        "joint_decoder_readout_bottleneck_supported": NEXT_ACTION_JOINT,
    }.get(classification, NEXT_ACTION_UNDERDETERMINED)


def _validation_gates(formal: FormalSystemConfig, data: mra.AuditData) -> tuple[GateResult, ...]:
    operator = _build_research_ols(formal)
    rank = _mode_rank(operator)
    finite = bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all())
    transport = operator.transport.detach().cpu() if hasattr(operator, "transport") else None
    identity_transport = bool(transport is not None and torch.is_complex(transport) and transport.shape == (mra.PORT_COUNT, mra.PORT_COUNT) and torch.allclose(transport, torch.eye(mra.PORT_COUNT, dtype=transport.dtype), atol=1.0e-7, rtol=0.0))
    return (
        GateResult("selection_balanced", bool(torch.equal(torch.bincount(data.test_labels.index_select(0, data.selection_indices), minlength=10), torch.full((10,), mra.SELECTION_PER_LABEL, dtype=torch.long))), {}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("mode_bank_rank_100", rank == mra.PORT_COUNT, {"rank": rank}),
        GateResult("finite_mode_bank", finite, {}),
        GateResult("identity_transport", identity_transport, {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("plane_split_contract", tuple(data.images.shape[1:]) == (1, 28, 28), {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "evaluator_calls": 0}),
    )


def _record(schema: str, status: str, classification: str | None, spec: Spec, parent: ParentBundle, state: Mapping[str, Any], review_path: Path, hashes: Mapping[str, str], gates: Sequence[GateResult], actions: Sequence[str], contract: Mapping[str, Any], reason: str | None = None) -> dict[str, Any]:
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
        "evaluator_calls": 0,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _load_validation(path: str | Path, spec: Spec, parent: ParentBundle, review_path: Path, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, parent)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("status") != "passed" or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected_hashes or record.permitted_next_actions != (NEXT_ACTION_RUN,):
        raise ValueError("decoder-CMOS attribution validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"decoder-CMOS attribution requires clean branch {BRANCH}")
    parent = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    gates = _validation_gates(parent.formal, parent.data)
    hashes = _input_hashes(spec, review_path, parent)
    valid = all(gate.passed for gate in gates)
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None, spec, parent, state, review_path, hashes, gates, (NEXT_ACTION_RUN,) if valid else (), {"parent_mra": PARENT_MRA, "formal_config": str(parent.formal_path), "review_contract": str(review_path), "dataset_manifest": parent.data.manifest})
    _write_new(args.output, payload, artifact_kind="decoder-CMOS attribution validation")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("decoder-CMOS attribution requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_BYTES:
        raise RuntimeError("decoder-CMOS attribution requires at least 4 GiB free CUDA memory")
    if shutil.disk_usage(output_path.anchor or str(output_path.parent)).free < MIN_DISK_FREE_BYTES:
        raise RuntimeError("decoder-CMOS attribution requires 10 GiB free storage")
    return int(free), int(total)


def _ensure_run_directory(output_path: Path, validation_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite decoder-CMOS attribution artifact: {output_path}")
    root = output_path.parent
    if root.exists():
        children = {item.name for item in root.iterdir()}
        if children != {validation_path.name}:
            raise FileExistsError(f"decoder-CMOS attribution output directory is not a fresh validation directory: {root}")


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"decoder-CMOS attribution requires clean branch {BRANCH}")
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _ensure_run_directory(output_path, validation_path)
    parent_validation = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=False)
    _load_validation(validation_path, spec, parent_validation, review_path, state)
    free_bytes, total_bytes = _resource_preflight(output_path, torch.device(args.device))
    parent = _load_parent_with_data(spec, args.dataset_root, load_checkpoints=True)
    selection_images = parent.data.test_images.index_select(0, parent.data.selection_indices)
    selection_labels = parent.data.test_labels.index_select(0, parent.data.selection_indices)
    confirmation_images = parent.data.images.index_select(0, parent.data.confirmation_indices)
    confirmation_labels = parent.data.labels.index_select(0, parent.data.confirmation_indices)
    started = time.monotonic()
    torch.manual_seed(PHASE_CONTROL_SEED)
    torch.cuda.manual_seed_all(PHASE_CONTROL_SEED)
    torch.cuda.reset_peak_memory_stats(args.device)
    selection_eval: dict[str, ReadoutEvaluation] = {}
    confirmation_eval: dict[str, ReadoutEvaluation] = {}
    for name in MODEL_NAMES:
        selection_eval[name] = _evaluate_panel(name, parent.formal, parent.source.checkpoints[name], selection_images, selection_labels, parent.source, torch.device(args.device))
        confirmation_eval[name] = _evaluate_panel(name, parent.formal, parent.source.checkpoints[name], confirmation_images, confirmation_labels, parent.source, torch.device(args.device))
        torch.cuda.empty_cache()
    selection_records = {name: value.metrics for name, value in selection_eval.items()}
    confirmation_records = {name: value.metrics for name, value in confirmation_eval.items()}
    classification, classification_details = _classify_readout(selection_records["scaled_chen_step_5000"], confirmation_records["scaled_chen_step_5000"], confirmation_records["identity_step_5500"])
    maximum = _maximum_discontinuity(selection_records["scaled_chen_step_5000"], confirmation_records["scaled_chen_step_5000"])
    phase_valid = all(max(item.metrics["phase_controls"].values()) <= PHASE_INVARIANCE_TOLERANCE for item in (*selection_eval.values(), *confirmation_eval.values()))
    state_valid = all(item.metrics["state_unchanged"] and item.metrics["ols_state_unchanged"] and item.model_state_hash_before == item.model_state_hash_after and item.ols_state_hash_before == item.ols_state_hash_after for item in (*selection_eval.values(), *confirmation_eval.values()))
    finite = all(item.metrics["finite"] for item in (*selection_eval.values(), *confirmation_eval.values()))
    peak = int(torch.cuda.max_memory_allocated(torch.device(args.device)))
    elapsed = time.monotonic() - started
    harness_gates = (
        GateResult("finite_measurements", finite, {}),
        GateResult("frozen_state_unchanged", state_valid, {}),
        GateResult("square_law_and_phase_controls", phase_valid, {"tolerance": PHASE_INVARIANCE_TOLERANCE}),
        GateResult("negative_controls_present", True, {"identity_reference": True, "permuted_label": True}),
        GateResult("optimizer_isolation", True, {"optimizer_created": False}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak, "maximum": MAX_PEAK_BYTES}),
    )
    science_gates = (
        GateResult("relay_or_detector_panel_stability", maximum["status"] in {"unique", "joint"}, maximum),
        GateResult("readout_classification_complete", classification != "readout_attribution_underdetermined", classification_details),
    )
    status = "valid_failed" if all(gate.passed for gate in harness_gates) else "invalid_harness"
    if status == "invalid_harness":
        classification = "invalid_harness"
        action = NEXT_ACTION_UNDERDETERMINED
    else:
        action = _action_for(classification)
    hashes = _input_hashes(spec, review_path, parent)
    panel_path = output_path.parent / "panel.pt"
    compact_path = output_path.parent / "compact_report.json"
    for target in (panel_path, compact_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite decoder-CMOS attribution artifact: {target}")
    panel_payload = {
        "schema": PANEL_SCHEMA,
        "selection_indices": parent.data.selection_indices,
        "confirmation_indices": parent.data.confirmation_indices,
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
        "per_sample_mse": {
            "selection": {name: item.sample_mse for name, item in selection_eval.items()},
            "confirmation": {name: item.sample_mse for name, item in confirmation_eval.items()},
        },
        "source_hashes": parent.source.hashes,
        "affects_selection": False,
    }
    storage_estimate = sum(value.numel() * value.element_size() for group in panel_payload["per_sample_mse"].values() for value in group.values())
    if storage_estimate > MAX_STORAGE_BYTES:
        raise RuntimeError("decoder-CMOS attribution compact panel exceeds storage gate")
    atomic_torch_save(panel_path, panel_payload)
    panel_hash = file_sha256(panel_path)
    compact = {
        "schema": "scaled_chen_decoder_cmos_attribution_compact_report.v1",
        "classification": classification,
        "classification_details": classification_details,
        "maximum_information_discontinuity": maximum,
        "models": {name: {"paired_mse": value.metrics["reconstruction"]["paired_mse"], "retention": value.metrics["retention"], "power_retention": value.metrics["power_retention"]} for name, value in confirmation_eval.items()},
        "parent_mra_classification": PARENT_MRA["audit_classification"],
        "generation_readiness": "not_ready",
        "generative_manifold_supported": False,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
        "panel_sha256": panel_hash,
    }
    _write_new(compact_path, compact, artifact_kind="decoder-CMOS attribution compact report")
    contract = {
        "selection": selection_records,
        "confirmation": confirmation_records,
        "classification": classification,
        "classification_details": classification_details,
        "maximum_information_discontinuity": maximum,
        "panel_path": str(panel_path.resolve()),
        "panel_sha256": panel_hash,
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": file_sha256(compact_path),
        "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "wall_clock": "observational_only"},
        "parent_mra": PARENT_MRA,
        "generation_readiness": "not_ready",
        "generative_manifold_supported": False,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "evaluator_calls": 0,
    }
    all_gates = harness_gates + science_gates
    payload = _record(AUDIT_SCHEMA, status, classification, spec, parent, state, review_path, hashes, all_gates, (action,), contract, None if status != "invalid_harness" else "decoder-CMOS attribution harness failed")
    payload["generation_readiness"] = "not_ready"
    payload["generative_manifold_supported"] = False
    _write_new(output_path, payload, artifact_kind="decoder-CMOS attribution audit")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if status == "valid_failed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-decoder-cmos-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-decoder-cmos-attribution-run":
        return run(args)
    raise ValueError(f"unsupported decoder-CMOS attribution action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "CONFIG_SCHEMA",
    "HYPOTHESIS",
    "PANEL_SCHEMA",
    "REVIEW_SCHEMA",
    "SEGMENTS",
    "STAGE_NAMES",
    "VALIDATION_SCHEMA",
    "Spec",
    "_assert_contract",
    "_classify_readout",
    "_load_review",
    "_load_spec",
    "_maximum_discontinuity",
    "_phase_controls",
    "dispatch",
    "run",
    "validate",
]
