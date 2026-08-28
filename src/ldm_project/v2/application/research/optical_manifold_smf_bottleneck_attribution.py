"""No-update attribution audit for the fixed 100-port SMF bottleneck.

The audit consumes the authoritative fixed-OLS recovery canary and separates
projection capacity, encoder/decoder mismatch, and local coupling-dose
effects.  It never creates an optimizer, changes a parameter, or writes a
checkpoint.  The existing formal IdentityOLS and the fixed research SMF/FMF
operators remain untouched.
"""

from __future__ import annotations

import argparse
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
from ...optics.fiber_operator import FiberArrayOLS
from ...optics.few_mode_fiber_operator import FewModeFiberArrayOLS, FewModeFiberArraySpec
from . import optical_manifold as base
from . import optical_manifold_smf_integration_canary as smf_canary


BRANCH = "research/v2-lightgen-optical-manifold-smf-bottleneck-attribution-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_attribution_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_attribution_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_attribution_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_attribution.v1"
DECOMPOSITION_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_decomposition.v1"
PANEL_SCHEMA = "lightgen_optical_manifold_smf_bottleneck_panel.v1"

SOURCE_VALIDATION_HASH = "18d63ff5e9c3d18db7c669fc8c596423d8a2cc02ed50338cf8043fcc609d06f6"
SOURCE_PREFLIGHT_HASH = "abf063fa6e39511b01a661923f755e84c1eefd8285c91bdce0651a80066d8833"
SOURCE_TRAINING_HASH = "a73b3917d64049509b2c6f025e0025bd506db4d6f95271598c5dc95c43c0b65e"
SOURCE_PANEL_HASH = "c432221ba0725da3744f66459a3c513ef632d5aa202864123007137aa5798b61"
SOURCE_CHECKPOINT_HASH = "949bf6903db8a6c5c8b76fddfddeef31f9419717486f278cbf0ed9d86534dd94"
CONTROL_CHECKPOINT_HASH = "05aebcfd5c2d2aaa3867d2af789dba9158c0123b7e3085217082c061240b4266"
COUPLING_CHECKPOINT_HASH = "dc9550233f89f213ca45bd8ccee5144dd2e96255dfe6d447dd2555852a80739a"
MNIST_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_STEP = 1000
FINAL_STEP = 100
BATCH_SIZE = 10
ASSESSMENT_PER_LABEL = 100
GRADIENT_PANEL_COUNT = 3
GRADIENT_PANEL_SIZE = 100
SEED = 1234
REGISTERED_LAMBDA = 0.0071572503388103665
DOSE_MULTIPLIERS = (1.0, 2.0, 4.0, 8.0)
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1200.0
MAX_STORAGE_BYTES = 512 * 1024**2
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE
NEXT_DOSE_ACTION = "manual_review_fixed_ols_coupling_dose_response_pilot_contract"
NEXT_MODE_ACTION = "manual_review_fmf_mode_capacity_reconstruction_contract"
NEXT_DECODER_ACTION = "manual_review_passive_transport_decoder_mismatch_contract"
NEXT_OBJECTIVE_ACTION = "manual_review_smf_projection_objective_replacement_contract"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class AuditData:
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    heldout_indices: torch.Tensor
    assessment_indices: torch.Tensor
    gradient_panels: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    source_checkpoint: dict[str, Any]
    final_checkpoints: dict[str, dict[str, Any]]
    training_payload: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"] or state["branch"] != BRANCH:
        raise ValueError(f"SMF bottleneck audit requires clean branch {BRANCH}")


def _require_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "SMF bottleneck audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup"
        )


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema", "hypothesis", "formal_system_config", "source", "context",
        "dataset", "optical_contract", "audit", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("SMF bottleneck audit config has unexpected fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("SMF bottleneck audit config is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_smf_bottleneck_attribution_audit",
        "approved_contract": {
            "assessment_images": 1000,
            "gradient_panels": 3,
            "gradient_panel_images": 100,
            "dose_multipliers": [1.0, 2.0, 4.0, 8.0],
            "registered_lambda": REGISTERED_LAMBDA,
            "m2_capacity_comparison_only": True,
            "no_update": True,
            "post_dose_limited_next_action": NEXT_DOSE_ACTION,
            "post_mode_capacity_next_action": NEXT_MODE_ACTION,
            "post_decoder_mismatch_next_action": NEXT_DECODER_ACTION,
            "post_objective_next_action": NEXT_OBJECTIVE_ACTION,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("SMF bottleneck audit review contract does not match scope")
    return target


def _assert_contract(spec: Spec) -> None:
    p = spec.payload
    expected_source = {
        "validation_sha256": SOURCE_VALIDATION_HASH,
        "preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "training_sha256": SOURCE_TRAINING_HASH,
        "panel_sha256": SOURCE_PANEL_HASH,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "optical_control_checkpoint_sha256": CONTROL_CHECKPOINT_HASH,
        "optical_coupling_checkpoint_sha256": COUPLING_CHECKPOINT_HASH,
        "mnist_manifest_sha256": MNIST_MANIFEST_HASH,
        "source_global_step": SOURCE_STEP,
    }
    for key, value in expected_source.items():
        if p["source"].get(key) != value:
            raise ValueError(f"SMF bottleneck audit source binding mismatch: {key}")
    if p["context"] != {
        "source_training_outcome": "valid_failed",
        "source_next_action": "manual_review_smf_bottleneck_attribution_contract",
        "source_authoritative_terminal_metrics": True,
        "fmf_capacity_operator": "existing_plan_7a_frozen_m2_operator",
    }:
        raise ValueError("SMF bottleneck audit context contract mismatch")
    if p["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "assessment_images": 1000,
        "gradient_panels": 3,
        "gradient_panel_images": 100,
        "panel_seed": SEED,
        "schedule": "fresh_balanced_panels_without_replacement",
        "heldout_excluded": True,
        "prior_training_and_audit_indices_excluded": True,
        "label_role": "balancing_and_offline_grouping_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("SMF bottleneck audit dataset contract mismatch")
    optical = p["optical_contract"]
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
        "formal_ols": "identity",
        "research_ols": "100_port_lp01_smf",
        "capacity_control_ols": "200_port_lp01_lp11a_fmf",
        "fixed_transport": True,
        "fixed_exposure": {"scale": 1.0, "gamma": 1.0},
        "projection_loss": "one_minus_smf_power_retention",
        "no_complex_resize": True,
    }.items():
        if optical.get(key) != value:
            raise ValueError(f"SMF bottleneck audit optical contract mismatch: {key}")
    audit = p["audit"]
    if (
        audit.get("checkpoint_arms") != ["optical_mse_control", "optical_coupling"]
        or audit.get("context_arms") != ["posterior_no_kl_coupling", "posterior_kl_coupling"]
        or tuple(audit.get("dose_multipliers", ())) != DOSE_MULTIPLIERS
        or audit.get("registered_lambda") != REGISTERED_LAMBDA
        or audit.get("no_update") is not True
    ):
        raise ValueError("SMF bottleneck audit measurement contract mismatch")
    if p["authorization"].get("training_authorized") is not False or p["authorization"].get("formal_promotion") is not False:
        raise ValueError("SMF bottleneck audit authorization boundary is invalid")


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    config = load_config(path)
    if (
        config.architecture != "conditional_identity_logit_amplitude_v2"
        or config.system.kind != base.EXPECTED_FORMAL_KIND
        or not config.system.is_formal
    ):
        raise ValueError("SMF bottleneck audit requires current formal Identity optics")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(config.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch: {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch: {name}")
    return config.system, path


def _ancestor(commit: str, current: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    ).returncode != 0:
        raise ValueError("source evidence commit is not an ancestor")


def _load_evidence(path: Path, digest: str, current: str, expected_kind: str, expected_status: str) -> dict[str, Any]:
    if file_sha256(path) != digest:
        raise ValueError(f"source evidence hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != expected_kind or record.status != expected_status:
        raise ValueError(f"source evidence status/schema mismatch: {path.name}")
    if payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("source evidence authorization boundary is invalid")
    _ancestor(record.git_commit, current)
    return payload


def _load_source(spec: Spec, formal_path: Path, current: str) -> SourceBundle:
    src = spec.payload["source"]
    paths = {
        "validation": (_resolve(src["validation"]), SOURCE_VALIDATION_HASH),
        "preflight": (_resolve(src["preflight"]), SOURCE_PREFLIGHT_HASH),
        "training": (_resolve(src["training"]), SOURCE_TRAINING_HASH),
        "panel": (_resolve(src["panel"]), SOURCE_PANEL_HASH),
        "source_checkpoint": (_resolve(src["source_checkpoint"]), SOURCE_CHECKPOINT_HASH),
        "optical_control_checkpoint": (_resolve(src["optical_control_checkpoint"]), CONTROL_CHECKPOINT_HASH),
        "optical_coupling_checkpoint": (_resolve(src["optical_coupling_checkpoint"]), COUPLING_CHECKPOINT_HASH),
    }
    _load_evidence(paths["validation"][0], SOURCE_VALIDATION_HASH, current, "lightgen_optical_manifold_smf_ols_loss_recovery_validation.v1", "passed")
    _load_evidence(paths["preflight"][0], SOURCE_PREFLIGHT_HASH, current, "lightgen_optical_manifold_smf_ols_loss_recovery_preflight.v1", "passed")
    training = _load_evidence(paths["training"][0], SOURCE_TRAINING_HASH, current, "lightgen_optical_manifold_smf_ols_loss_recovery_training.v1", "valid_failed")
    if training["contract"].get("outcome") != "valid_failed":
        raise ValueError("fixed OLS source training outcome is not valid_failed")
    panel = safe_torch_load(paths["panel"][0], expected_sha256=SOURCE_PANEL_HASH, map_location="cpu")
    if panel.get("schema") != "lightgen_optical_manifold_identity_smf_coupling_panel.v1":
        raise ValueError("fixed OLS source panel schema mismatch")
    source_checkpoint = safe_torch_load(paths["source_checkpoint"][0], expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if source_checkpoint.get("step") != SOURCE_STEP or not isinstance(source_checkpoint.get("model_state"), Mapping):
        raise ValueError("SMF source checkpoint is incomplete or at the wrong step")
    final_checkpoints: dict[str, dict[str, Any]] = {}
    for arm, key, digest in (
        ("optical_mse_control", "optical_control_checkpoint", CONTROL_CHECKPOINT_HASH),
        ("optical_coupling", "optical_coupling_checkpoint", COUPLING_CHECKPOINT_HASH),
    ):
        checkpoint_payload = safe_torch_load(paths[key][0], expected_sha256=digest, map_location="cpu")
        if checkpoint_payload.get("schema") != "lightgen_optical_manifold_smf_ols_loss_recovery_resume.v1" or checkpoint_payload.get("arm") != arm or checkpoint_payload.get("step") != FINAL_STEP:
            raise ValueError(f"{arm} checkpoint is incomplete or at the wrong step")
        if not isinstance(checkpoint_payload.get("model_state"), Mapping) or not isinstance(checkpoint_payload.get("optimizer_state"), Mapping):
            raise ValueError(f"{arm} checkpoint does not contain model and Adam state")
        final_checkpoints[arm] = checkpoint_payload
    hashes = {
        "validation": SOURCE_VALIDATION_HASH,
        "preflight": SOURCE_PREFLIGHT_HASH,
        "training": SOURCE_TRAINING_HASH,
        "panel": SOURCE_PANEL_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "optical_control_checkpoint": CONTROL_CHECKPOINT_HASH,
        "optical_coupling_checkpoint": COUPLING_CHECKPOINT_HASH,
        "formal_config": file_sha256(formal_path),
        "mnist_manifest": MNIST_MANIFEST_HASH,
    }
    return SourceBundle(source_checkpoint, final_checkpoints, training, hashes, formal_path)


def _load_data(dataset_root: str | Path) -> AuditData:
    prior = smf_canary._load_data(dataset_root)
    excluded = torch.cat((prior.schedule.reshape(-1), prior.heldout_indices))
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("SMF bottleneck audit exclusion set contains duplicates")
    available_by_label: list[torch.Tensor] = []
    train_labels = prior.labels.index_select(0, prior.train_indices)
    for label in range(10):
        class_indices = prior.train_indices[train_labels == label]
        remaining = class_indices[~torch.isin(class_indices, excluded)]
        required = ASSESSMENT_PER_LABEL + GRADIENT_PANEL_COUNT * (GRADIENT_PANEL_SIZE // 10)
        if remaining.numel() < required:
            raise ValueError("SMF bottleneck audit lacks fresh per-label samples")
        generator = torch.Generator(device="cpu").manual_seed(SEED + label)
        available_by_label.append(remaining[torch.randperm(remaining.numel(), generator=generator)][:required])
    assessment = torch.cat([values[:ASSESSMENT_PER_LABEL] for values in available_by_label])
    gradient_panels = tuple(
        torch.cat([values[ASSESSMENT_PER_LABEL + panel * (GRADIENT_PANEL_SIZE // 10): ASSESSMENT_PER_LABEL + (panel + 1) * (GRADIENT_PANEL_SIZE // 10)] for values in available_by_label])
        for panel in range(GRADIENT_PANEL_COUNT)
    )
    all_selected = torch.cat((assessment, *gradient_panels))
    if torch.unique(all_selected).numel() != all_selected.numel() or torch.isin(all_selected, excluded).any():
        raise ValueError("SMF bottleneck audit panels overlap or consume excluded indices")
    for panel in (assessment, *gradient_panels):
        panel_labels = prior.labels.index_select(0, panel)
        if not torch.equal(torch.bincount(panel_labels, minlength=10), torch.full((10,), panel.numel() // 10, dtype=torch.long)):
            raise ValueError("SMF bottleneck audit panel is not balanced")
    manifest: dict[str, Any] = {
        "schema": "lightgen_optical_manifold_smf_bottleneck_audit_mnist_manifest.v1",
        "source_mnist_manifest_hash": MNIST_MANIFEST_HASH,
        "prior_exclusion_indices_sha256": tensor_sha256(torch.sort(excluded).values),
        "assessment_indices_sha256": tensor_sha256(assessment),
        "gradient_indices_sha256": [tensor_sha256(panel) for panel in gradient_panels],
        "assessment_count": int(assessment.numel()),
        "gradient_panel_count": GRADIENT_PANEL_COUNT,
        "gradient_panel_size": GRADIENT_PANEL_SIZE,
        "seed": SEED,
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "balancing_and_offline_grouping_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return AuditData(prior.images, prior.labels, prior.train_indices, prior.heldout_indices, assessment, gradient_panels, manifest)


def _context(spec: Spec, formal: FormalSystemConfig, data: AuditData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _input_hashes(spec: Spec, review: Path, data: AuditData, source: SourceBundle) -> dict[str, str]:
    return {**source.hashes, "research_config": file_sha256(spec.path), "review_contract": file_sha256(review), "dataset_manifest": data.manifest["manifest_hash"]}


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
        raise FileExistsError(f"refusing to overwrite SMF bottleneck audit artifact: {target}")
    atomic_write_json(target, payload)


def _model(formal: FormalSystemConfig, checkpoint: Mapping[str, Any], device: torch.device) -> base.IdentityOpticalManifoldAutoencoder:
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def _smf(model: base.IdentityOpticalManifoldAutoencoder, device: torch.device) -> FiberArrayOLS:
    return smf_canary._smf_operator(model.system, device)


def _fmf(model: base.IdentityOpticalManifoldAutoencoder, device: torch.device) -> FewModeFiberArrayOLS:
    return FewModeFiberArrayOLS(model.system.encoder_plane, FewModeFiberArraySpec(modes=("LP01", "LP11a"))).to(device)


def _batch(data: AuditData, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    return rendered.amplitude.to(device), rendered.target.to(device)


def _pre_ols(model: base.IdentityOpticalManifoldAutoencoder, amplitude: torch.Tensor) -> tuple[torch.Tensor, Any, Mapping[str, Any]]:
    system = model.system
    slm_field = system.slm(system.command(amplitude))
    front = system.front(slm_field)
    encoded, encoder_records, encoder_stats = system.encoder(front, "encoder")
    return encoded.tensor[:, 0], system, encoder_stats


def _forward(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    pre, system, encoder_stats = _pre_ols(model, amplitude)
    encoded = ComplexField(pre.unsqueeze(1), system.encoder_plane.with_plane("encoder_layer_2"), OperationProvenance("smf_bottleneck_encoder_boundary", "encoder_layer_2", 0.0))
    transformed = operator(encoded)
    decoded, _decoder_records, decoder_stats = system.decoder(transformed, "decoder")
    camera_field = system.back(ComplexField(decoded.tensor, system.chip_plane, decoded.provenance))
    raw = camera_field.tensor.abs().square()
    area = operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]
    input_energy = pre.abs().square().sum(dim=(-2, -1)) * area
    output_energy = transformed.tensor[:, 0].abs().square().sum(dim=(-2, -1)) * area
    diagnostics = {
        "pre_ols": pre,
        "input_energy": input_energy,
        "output_energy": output_energy,
        "coefficients": operator.last_diagnostics["port_input_coefficients"],
        "coupling_efficiency": operator.last_diagnostics["coupling_efficiency"],
        "provenance": transformed.provenance.to_dict(),
        "encoder_crop_retention": encoder_stats["crop_retention"],
        "decoder_crop_retention": decoder_stats["crop_retention"],
    }
    if isinstance(operator, FewModeFiberArrayOLS):
        diagnostics["mode_coefficients"] = operator.last_diagnostics["mode_coefficients"]
        diagnostics["mode_power"] = operator.last_diagnostics["mode_power"]
    return raw, diagnostics["pre_ols"], diagnostics


def _structure(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    work = torch.cat((features.real, features.imag), dim=1).to(torch.float64) if torch.is_complex(features) else features.to(torch.float64)
    centered = work - work.mean(0, keepdim=True)
    values = torch.linalg.svdvals(centered)
    weights = values.square() / values.square().sum().clamp_min(1.0e-30)
    distances = torch.cdist(work, work)
    within: list[torch.Tensor] = []
    between: list[torch.Tensor] = []
    for label in range(10):
        mask = labels == label
        if int(mask.sum()) > 1:
            within.append(distances[mask][:, mask].triu(diagonal=1)[distances[mask][:, mask].triu(diagonal=1) > 0])
        other = ~mask
        if mask.any() and other.any():
            between.append(distances[mask][:, other].reshape(-1))
    intra = torch.cat(within).mean() if within else torch.tensor(0.0, dtype=torch.float64)
    inter = torch.cat(between).mean() if between else torch.tensor(0.0, dtype=torch.float64)
    return {
        "effective_rank": float(1.0 / weights.square().sum().clamp_min(1.0e-30)),
        "inter_distance": float(inter),
        "intra_distance": float(intra),
        "inter_intra_ratio": float(inter / intra.clamp_min(1.0e-30)),
        "finite": bool(torch.isfinite(work).all() and torch.isfinite(distances).all()),
    }


def _evaluate(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS | FewModeFiberArrayOLS, data: AuditData, indices: torch.Tensor, device: torch.device, *, keep_visual: bool = False) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cameras: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    fields: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    input_energy: list[torch.Tensor] = []
    output_energy: list[torch.Tensor] = []
    visual_camera: list[torch.Tensor] = []
    visual_target: list[torch.Tensor] = []
    visual_field: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            amplitudes, target = _batch(data, batch_indices, device)
            camera, field, diagnostics = _forward(model, operator, amplitudes)
            cameras.append(camera.cpu())
            targets.append(target.cpu())
            fields.append(field.cpu())
            coefficients.append(diagnostics["coefficients"].cpu())
            input_energy.append(diagnostics["input_energy"].cpu())
            output_energy.append(diagnostics["output_energy"].cpu())
            if keep_visual and sum(value.shape[0] for value in visual_camera) < 100:
                visual_camera.append(camera.cpu())
                visual_target.append(target.cpu())
                visual_field.append(field.cpu())
    camera = torch.cat(cameras)
    target = torch.cat(targets)
    field = torch.cat(fields)
    coefficient = torch.cat(coefficients)
    in_energy = torch.cat(input_energy)
    out_energy = torch.cat(output_energy)
    flat_camera = camera.flatten(1)
    flat_target = target.flatten(1)
    camera_centered = flat_camera - flat_camera.mean(1, keepdim=True)
    target_centered = flat_target - flat_target.mean(1, keepdim=True)
    correlation = (camera_centered * target_centered).sum(1) / (camera_centered.norm(dim=1) * target_centered.norm(dim=1)).clamp_min(1.0e-30)
    labels = data.labels.index_select(0, indices)
    metrics = {
        "paired_mse": float(F.mse_loss(camera, target)),
        "cyclic_mse": float(F.mse_loss(camera, target.roll(1, 0))),
        "centered_correlation": float(correlation.mean()),
        "variance_retention": float(flat_camera.var(0, unbiased=False).mean() / flat_target.var(0, unbiased=False).mean().clamp_min(1.0e-30)),
        "power_retention": float((out_energy / in_energy.clamp_min(1.0e-30)).mean()),
        "projection_residual": float((1.0 - out_energy / in_energy.clamp_min(1.0e-30)).mean()),
        "effective_rank": _structure(coefficient, labels)["effective_rank"],
        "structure": _structure(coefficient, labels),
        "finite": bool(torch.isfinite(camera).all() and torch.isfinite(target).all() and torch.isfinite(field).all() and torch.isfinite(coefficient).all()),
        "evaluation_count": int(indices.numel()),
        "feature_shape": list(coefficient.shape),
        "provenance": operator.last_diagnostics.get("provenance", None),
    }
    values = {"camera": camera, "target": target, "pre_ols": field, "coefficients": coefficient}
    if keep_visual:
        values.update({"visual_camera": torch.cat(visual_camera)[:100], "visual_target": torch.cat(visual_target)[:100], "visual_pre_ols": torch.cat(visual_field)[:100]})
    return metrics, values


def _capacity(model: base.IdentityOpticalManifoldAutoencoder, operators: tuple[FiberArrayOLS, FewModeFiberArrayOLS], data: AuditData, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    fields: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            amplitudes, _target = _batch(data, indices[start : start + BATCH_SIZE], device)
            field, _system, _encoder_stats = _pre_ols(model, amplitudes)
            fields.append(field.cpu())
    pre = torch.cat(fields)
    report: dict[str, Any] = {"field_count": int(pre.shape[0]), "finite": bool(torch.isfinite(pre).all()), "operators": {}}
    for name, operator in zip(("smf_m1", "fmf_m2"), operators, strict=True):
        input_powers: list[torch.Tensor] = []
        output_powers: list[torch.Tensor] = []
        coefficients: list[torch.Tensor] = []
        output_finite = True
        output = None
        with torch.no_grad():
            for start in range(0, pre.shape[0], BATCH_SIZE):
                field = pre[start : start + BATCH_SIZE].to(device)
                output = operator(ComplexField(field.unsqueeze(1), model.system.encoder_plane.with_plane("encoder_layer_2"), OperationProvenance("bottleneck_capacity_probe", "encoder_layer_2", 0.0)))
                diagnostics = operator.last_diagnostics
                input_powers.append((field.abs().square().sum(dim=(-2, -1)) * operator.input_plane.pixel_pitch_m[0] * operator.input_plane.pixel_pitch_m[1]).cpu())
                output_powers.append(diagnostics["output_energy"].cpu())
                coefficients.append(diagnostics["port_input_coefficients"].cpu())
                output_finite = output_finite and bool(torch.isfinite(output.tensor).all())
        input_power = torch.cat(input_powers)
        output_power = torch.cat(output_powers)
        coefficient = torch.cat(coefficients)
        report["operators"][name] = {
            "power_retention_mean": float((output_power / input_power.clamp_min(1.0e-30)).mean()),
            "power_retention_p05": float((output_power / input_power.clamp_min(1.0e-30)).quantile(0.05)),
            "power_retention_p95": float((output_power / input_power.clamp_min(1.0e-30)).quantile(0.95)),
            "effective_rank": _structure(coefficient, data.labels.index_select(0, indices))["effective_rank"],
            "finite": output_finite,
            "provenance": output.provenance.to_dict() if output is not None else None,
        }
    report["m2_minus_m1_power_retention"] = report["operators"]["fmf_m2"]["power_retention_mean"] - report["operators"]["smf_m1"]["power_retention_mean"]
    return report


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _compose_model(formal: FormalSystemConfig, source_state: Mapping[str, torch.Tensor], encoder_state: Mapping[str, torch.Tensor], decoder_state: Mapping[str, torch.Tensor], device: torch.device) -> base.IdentityOpticalManifoldAutoencoder:
    state = {name: value.detach().clone() for name, value in source_state.items()}
    state["system.encoder.phase"] = encoder_state["system.encoder.phase"].detach().clone()
    state["system.decoder.phase"] = decoder_state["system.decoder.phase"].detach().clone()
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _phase_gradients(model: base.IdentityOpticalManifoldAutoencoder, operator: FiberArrayOLS, amplitudes: torch.Tensor, targets: torch.Tensor, objective: str) -> dict[str, torch.Tensor]:
    if objective not in {"mse", "projection"}:
        raise ValueError("unsupported bottleneck gradient objective")
    gradients = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in model.named_parameters()}
    parameters = [parameter for name, parameter in model.named_parameters() if name in EXPECTED_TRAINABLE]
    for index in range(amplitudes.shape[0]):
        _camera, pre, diagnostics = _forward(model, operator, amplitudes[index : index + 1])
        if objective == "mse":
            value = F.mse_loss(_camera, targets[index : index + 1])
        else:
            value = (1.0 - diagnostics["output_energy"] / diagnostics["input_energy"].clamp_min(1.0e-30)).mean()
        values = torch.autograd.grad(value / amplitudes.shape[0], parameters, allow_unused=True)
        for (name, _parameter), gradient in zip(((n, p) for n, p in model.named_parameters() if n in EXPECTED_TRAINABLE), values, strict=True):
            if gradient is not None:
                gradients[name].add_(gradient.detach().cpu())
    for parameter in model.parameters():
        parameter.grad = None
    return gradients


def _add(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {name: left[name] + scale * right[name] for name in left}


def _dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    return float(sum((left[name] * right[name]).sum() for name in EXPECTED_TRAINABLE))


def _norm_encoder(values: Mapping[str, torch.Tensor]) -> float:
    return float(torch.sqrt(sum(values[name].square().sum() for name in ("system.encoder.phase",))))


def _permuted(values: Mapping[str, torch.Tensor], generator: torch.Generator) -> dict[str, torch.Tensor]:
    flat = torch.cat([values[name].reshape(-1) for name in EXPECTED_TRAINABLE])
    permutation = torch.randperm(flat.numel(), generator=generator)
    signs = torch.where(torch.randint(0, 2, (flat.numel(),), generator=generator) == 0, -1.0, 1.0)
    shuffled = flat.index_select(0, permutation) * signs
    result: dict[str, torch.Tensor] = {}
    cursor = 0
    for name in EXPECTED_TRAINABLE:
        count = values[name].numel()
        result[name] = shuffled[cursor : cursor + count].reshape_as(values[name])
        cursor += count
    return result


def _dose_panel(model: base.IdentityOpticalManifoldAutoencoder, data: AuditData, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    amplitudes, targets = _batch(data, indices, device)
    operator = _smf(model, device)
    reconstruction = _phase_gradients(model, operator, amplitudes, targets, "mse")
    projection = _phase_gradients(model, operator, amplitudes, targets, "projection")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    permuted = _permuted(projection, generator)
    reports: dict[str, Any] = {}
    for multiplier in DOSE_MULTIPLIERS:
        combined = _add(reconstruction, projection, REGISTERED_LAMBDA * multiplier)
        rec_norm = _norm_encoder(reconstruction)
        reg_norm = _norm_encoder(projection)
        ratio = REGISTERED_LAMBDA * multiplier * reg_norm / max(rec_norm, 1.0e-30)
        reports[str(multiplier)] = {
            "multiplier": multiplier,
            "effective_lambda": REGISTERED_LAMBDA * multiplier,
            "reconstruction_encoder_norm": rec_norm,
            "projection_encoder_norm": reg_norm,
            "effective_encoder_gradient_ratio": ratio,
            "combined_dot_reconstruction": _dot(combined, reconstruction),
            "combined_dot_projection": _dot(combined, projection),
            "permuted_projection_dot": _dot(combined, permuted),
            "reconstruction_descent_retention": _dot(combined, reconstruction) / max(_dot(reconstruction, reconstruction), 1.0e-30),
            "correct_projection_beats_permuted": _dot(combined, projection) > _dot(combined, permuted),
            "encoder_gradient_nonzero": rec_norm > 0.0 and reg_norm > 0.0,
            "decoder_projection_gradient_zero": float(projection["system.decoder.phase"].norm()) == 0.0,
            "four_phase_reconstruction_nonzero": all(float(reconstruction[name][layer].norm()) > 0.0 for name in EXPECTED_TRAINABLE for layer in range(reconstruction[name].shape[0])),
        }
    return {"gradient_norms": {"reconstruction": {name: float(value.norm()) for name, value in reconstruction.items()}, "projection": {name: float(value.norm()) for name, value in projection.items()}}, "doses": reports}


def _cross_swap(formal: FormalSystemConfig, source_state: Mapping[str, torch.Tensor], control_state: Mapping[str, torch.Tensor], coupling_state: Mapping[str, torch.Tensor], data: AuditData, indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    combinations = {
        "control_encoder_control_decoder": (control_state, control_state),
        "control_encoder_coupling_decoder": (control_state, coupling_state),
        "coupling_encoder_control_decoder": (coupling_state, control_state),
        "coupling_encoder_coupling_decoder": (coupling_state, coupling_state),
    }
    result: dict[str, Any] = {}
    for name, (encoder, decoder) in combinations.items():
        model = _compose_model(formal, source_state, encoder, decoder, device)
        operator = _smf(model, device)
        metrics, _values = _evaluate(model, operator, data, indices, device)
        result[name] = metrics
    baseline = result["control_encoder_control_decoder"]
    for name, metrics in result.items():
        metrics["mse_delta_vs_control"] = metrics["paired_mse"] - baseline["paired_mse"]
        metrics["power_delta_vs_control"] = metrics["power_retention"] - baseline["power_retention"]
    return result


def _validate_record(state: Mapping[str, Any], spec: Spec, formal: FormalSystemConfig, data: AuditData, source: SourceBundle, review: Path) -> EvidenceRecordV2:
    hashes = _input_hashes(spec, review, data, source)
    gates = tuple(GateResult(name, True, metrics) for name, metrics in (
        ("clean_lineage", {"branch": BRANCH}),
        ("authoritative_fixed_ols_source", {"training_status": "valid_failed", "source_step": SOURCE_STEP}),
        ("source_checkpoint_and_final_checkpoints", {"control_step": FINAL_STEP, "coupling_step": FINAL_STEP}),
        ("fresh_assessment_and_gradient_panels", {"assessment": 1000, "gradient_panels": 3, "panel_size": 100}),
        ("formal_2_plus_2_identity_optics", {"grid": 2048, "formal_ols": "identity", "research_ols": "100_port_lp01_smf"}),
        ("frozen_smf_and_fmf_capacity_operators", {"trainable_transport": False, "m2_comparison_only": True}),
        ("phase_only_parameter_scope", {"parameters": list(EXPECTED_TRAINABLE)}),
        ("no_update_authorization_boundary", {"training_authorized": False, "formal_promotion": False}),
    ))
    record = EvidenceRecordV2(2, VALIDATION_SCHEMA, "diagnostic", "passed", str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), hashes, gates, ("run_smf_bottleneck_attribution_audit",))
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    record = _validate_record(state, spec, formal, data, source, review)
    payload = _record_payload(VALIDATION_SCHEMA, record, {"source_hashes": source.hashes, "dataset_manifest": data.manifest, "assessment_indices": data.assessment_indices.tolist(), "gradient_panels": [panel.tolist() for panel in data.gradient_panels], "optimizer_created": False, "cuda_context_created": False, "run_created": False})
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: Spec, formal: FormalSystemConfig, data: AuditData, source: SourceBundle, review: Path, state: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if payload.get("schema") != VALIDATION_SCHEMA or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data) or record.input_hashes != _input_hashes(spec, review, data, source) or record.permitted_next_actions != ("run_smf_bottleneck_attribution_audit",):
        raise ValueError("SMF bottleneck audit validation is stale or mismatched")
    return payload, digest


def _operator_snapshot(operator: nn.Module) -> dict[str, torch.Tensor]:
    return _module_state(operator)


def _gate_metrics(gates: tuple[GateResult, ...]) -> dict[str, bool]:
    return {gate.name: gate.passed for gate in gates}


def run(args: argparse.Namespace) -> int:
    _require_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source = _load_source(spec, formal_path, state["commit"])
    data = _load_data(args.dataset_root)
    validation, validation_hash = _load_validation(args.validation_evidence, spec, formal, data, source, review, state)
    del validation
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError("SMF bottleneck audit requires at least 4 GiB free CUDA memory")
    started = time.monotonic()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device = torch.device(args.device)
        model_snapshots = {
            "source": {name: value.detach().cpu().clone() for name, value in source.source_checkpoint["model_state"].items()},
            "control": {name: value.detach().cpu().clone() for name, value in source.final_checkpoints["optical_mse_control"]["model_state"].items()},
            "coupling": {name: value.detach().cpu().clone() for name, value in source.final_checkpoints["optical_coupling"]["model_state"].items()},
        }
        operator_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        after_models: dict[str, dict[str, torch.Tensor]] = {}
        after_operators: dict[str, dict[str, torch.Tensor]] = {}
        initial: dict[str, dict[str, Any]] = {}
        visual_values: dict[str, dict[str, torch.Tensor]] = {}

        source_model = _model(formal, source.source_checkpoint, device)
        source_smf = _smf(source_model, device)
        source_fmf = _fmf(source_model, device)
        operator_snapshots["source_smf"] = _operator_snapshot(source_smf)
        operator_snapshots["source_fmf"] = _operator_snapshot(source_fmf)
        initial["source"], visual_values["source"] = _evaluate(source_model, source_smf, data, data.assessment_indices, device, keep_visual=True)
        capacity = _capacity(source_model, (source_smf, source_fmf), data, data.assessment_indices, device)
        after_models["source"] = _state(source_model)
        after_operators["source_smf"] = _operator_snapshot(source_smf)
        after_operators["source_fmf"] = _operator_snapshot(source_fmf)
        del source_model, source_smf, source_fmf
        torch.cuda.empty_cache()

        control_model = _model(formal, source.final_checkpoints["optical_mse_control"], device)
        control_smf = _smf(control_model, device)
        operator_snapshots["control_smf"] = _operator_snapshot(control_smf)
        initial["optical_mse_control"], visual_values["optical_mse_control"] = _evaluate(control_model, control_smf, data, data.assessment_indices, device, keep_visual=True)
        after_models["control"] = _state(control_model)
        after_operators["control_smf"] = _operator_snapshot(control_smf)
        del control_model, control_smf
        torch.cuda.empty_cache()

        coupling_model = _model(formal, source.final_checkpoints["optical_coupling"], device)
        coupling_smf = _smf(coupling_model, device)
        operator_snapshots["coupling_smf"] = _operator_snapshot(coupling_smf)
        initial["optical_coupling"], visual_values["optical_coupling"] = _evaluate(coupling_model, coupling_smf, data, data.assessment_indices, device, keep_visual=True)
        after_models["coupling"] = _state(coupling_model)
        after_operators["coupling_smf"] = _operator_snapshot(coupling_smf)
        dose = {f"G{index + 1}": _dose_panel(coupling_model, data, panel, device) for index, panel in enumerate(data.gradient_panels)}
        del coupling_model, coupling_smf
        torch.cuda.empty_cache()

        cross_swap = _cross_swap(formal, source.source_checkpoint["model_state"], source.final_checkpoints["optical_mse_control"]["model_state"], source.final_checkpoints["optical_coupling"]["model_state"], data, data.assessment_indices, device)
        heldout_replay: dict[str, dict[str, Any]] = {}
        for arm in ("optical_mse_control", "optical_coupling"):
            replay_model = _model(formal, source.final_checkpoints[arm], device)
            replay_operator = _smf(replay_model, device)
            heldout_replay[arm] = _evaluate(replay_model, replay_operator, data, data.heldout_indices, device)[0]
            del replay_model, replay_operator
            torch.cuda.empty_cache()
        final_states_unchanged = all(_tree_equal(model_snapshots[name], after_models[name]) for name in model_snapshots)
        operators_unchanged = all(_tree_equal(operator_snapshots[name], after_operators[name]) for name in operator_snapshots)
        source_metrics = source.training_payload["contract"]["arms"]
        replay_gates = (
            GateResult("authoritative_terminal_mse_replay", all(abs(heldout_replay[arm]["paired_mse"] - source_metrics[arm]["metrics"]["paired_mse"]) <= 1.0e-9 for arm in ("optical_mse_control", "optical_coupling")), {arm: {"expected": source_metrics[arm]["metrics"]["paired_mse"], "replayed": heldout_replay[arm]["paired_mse"]} for arm in heldout_replay}),
            GateResult("mode_bank_capacity_finite", capacity["finite"] and all(value["finite"] for value in capacity["operators"].values()), capacity),
            GateResult("cross_swap_complete", len(cross_swap) == 4 and all(value["finite"] for value in cross_swap.values()), {"combinations": list(cross_swap)}),
            GateResult("dose_gradient_boundaries", all(report["doses"][str(multiplier)]["encoder_gradient_nonzero"] and report["doses"][str(multiplier)]["decoder_projection_gradient_zero"] and report["doses"][str(multiplier)]["four_phase_reconstruction_nonzero"] for report in dose.values() for multiplier in DOSE_MULTIPLIERS), {}),
            GateResult("model_state_unchanged", final_states_unchanged and operators_unchanged, {"model": final_states_unchanged, "operators": operators_unchanged}),
            GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
            GateResult("finite_measurements", all(value["finite"] for value in initial.values()) and all(value["finite"] for value in heldout_replay.values()), {}),
        )
        safe_doses: list[float] = []
        for multiplier in DOSE_MULTIPLIERS:
            reports = [panel["doses"][str(multiplier)] for panel in dose.values()]
            if all(item["combined_dot_reconstruction"] > 0.0 and item["combined_dot_projection"] > 0.0 and item["reconstruction_descent_retention"] >= 0.25 and 0.05 <= item["effective_encoder_gradient_ratio"] <= 0.4 and item["correct_projection_beats_permuted"] for item in reports):
                safe_doses.append(multiplier)
        cross_encoder_power = cross_swap["coupling_encoder_control_decoder"]["power_retention"] - cross_swap["control_encoder_control_decoder"]["power_retention"]
        cross_encoder_mse = cross_swap["coupling_encoder_control_decoder"]["paired_mse"] - cross_swap["control_encoder_control_decoder"]["paired_mse"]
        cross_interaction = abs((cross_swap["coupling_encoder_coupling_decoder"]["paired_mse"] - cross_swap["coupling_encoder_control_decoder"]["paired_mse"]) - (cross_swap["control_encoder_coupling_decoder"]["paired_mse"] - cross_swap["control_encoder_control_decoder"]["paired_mse"]))
        dose_limited = [value for value in safe_doses if value > 1.0]
        mode_capacity = not dose_limited and capacity["m2_minus_m1_power_retention"] >= 0.05
        decoder_mismatch = not dose_limited and not mode_capacity and cross_encoder_power >= 0.01 and cross_encoder_mse >= 0.0 and cross_interaction > abs(cross_encoder_mse)
        if dose_limited:
            outcome = "passed_dose_limited"
            next_action = NEXT_DOSE_ACTION
        elif mode_capacity:
            outcome = "valid_failed_mode_capacity"
            next_action = NEXT_MODE_ACTION
        elif decoder_mismatch:
            outcome = "valid_failed_decoder_mismatch"
            next_action = NEXT_DECODER_ACTION
        else:
            outcome = "valid_failed_objective"
            next_action = NEXT_OBJECTIVE_ACTION
        resource = (
            GateResult("free_cuda_memory", int(free_bytes) >= MIN_FREE_BYTES, {"free_bytes": int(free_bytes), "required": MIN_FREE_BYTES}),
            GateResult("cuda_peak_memory", int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_BYTES, {"bytes": int(torch.cuda.max_memory_allocated(device))}),
            GateResult("wall_time", time.monotonic() - started <= MAX_WALL_SECONDS, {"seconds": time.monotonic() - started}),
        )
        gates = replay_gates + resource
        harness_failed = any(not gate.passed for gate in replay_gates)
        resource_failed = any(not gate.passed for gate in resource)
        if harness_failed:
            status = "invalid_harness"
            outcome = "invalid_harness"
            next_action = ""
        elif resource_failed:
            status = "resource_aborted"
            outcome = "resource_aborted"
            next_action = ""
        else:
            status = "diagnostic" if outcome == "passed_dose_limited" else "valid_failed"
        panel_values = {
            "schema": PANEL_SCHEMA,
            "source_hashes": source.hashes,
            "validation_sha256": validation_hash,
            "indices_are_global_mnist": True,
            "assessment_indices": data.assessment_indices[:100].clone(),
            "tensors": {
                "target": visual_values["source"]["visual_target"][:100],
                "source_camera": visual_values["source"]["visual_camera"][:100],
                "control_camera": visual_values["optical_mse_control"]["visual_camera"][:100],
                "coupling_camera": visual_values["optical_coupling"]["visual_camera"][:100],
            },
            "cross_swap_metrics": cross_swap,
            "panel_role": "fixed_visual_attribution_only",
        }
        panel_path = Path(args.output).parent / "smf_bottleneck_attribution_panel.pt"
        atomic_torch_save(panel_path, panel_values)
        panel_hash = file_sha256(panel_path)
        decomposition = {"schema": DECOMPOSITION_SCHEMA, "capacity": capacity, "cross_swap": cross_swap, "dose": dose, "safe_dose_multipliers": safe_doses, "heldout_replay": heldout_replay, "panel_sha256": panel_hash, "source_hashes": source.hashes, "validation_sha256": validation_hash}
        decomposition_path = Path(args.output).parent / "smf_bottleneck_decomposition.json"
        atomic_write_json(decomposition_path, decomposition)
        contract = {"outcome": outcome, "next_action": next_action, "source_hashes": source.hashes, "validation_hash": validation_hash, "initial_metrics": initial, "heldout_replay": heldout_replay, "capacity": capacity, "cross_swap": cross_swap, "dose": dose, "safe_dose_multipliers": safe_doses, "panel_path": str(panel_path.resolve()), "panel_sha256": panel_hash, "decomposition_path": str(decomposition_path.resolve()), "decomposition_sha256": file_sha256(decomposition_path), "evaluator_calls": 0, "training_authorized": False, "formal_promotion": False, "resource": {"free_cuda_bytes": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)), "wall_seconds": time.monotonic() - started}}
        record = EvidenceRecordV2(2, AUDIT_SCHEMA, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, formal, data), _input_hashes(spec, review, data, source), gates, (next_action,) if next_action else (), None if status in {"diagnostic", "valid_failed"} else outcome)
        record.validate()
        payload = _record_payload(AUDIT_SCHEMA, record, contract)
        _write_new(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0 if status == "diagnostic" else 2
    finally:
        torch.use_deterministic_algorithms(deterministic_before)


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-bottleneck-attribution-validate":
        return validate(args)
    if args.research_action == "smf-bottleneck-attribution-run":
        return run(args)
    raise ValueError(f"unsupported SMF bottleneck attribution action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA", "BRANCH", "CONFIG_SCHEMA", "DOSE_MULTIPLIERS", "EXPECTED_TRAINABLE",
    "NEXT_DECODER_ACTION", "NEXT_DOSE_ACTION", "NEXT_MODE_ACTION", "NEXT_OBJECTIVE_ACTION",
    "REGISTERED_LAMBDA", "VALIDATION_SCHEMA", "_assert_contract", "_load_data", "_load_review",
    "_load_spec", "_permuted", "_tree_equal", "dispatch", "run", "validate",
]
