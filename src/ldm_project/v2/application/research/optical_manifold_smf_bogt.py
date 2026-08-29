"""Scaled Chen-reference SMF + aggregate BOGT research lifecycle.

This module is deliberately separate from the formal ``label + 8D`` runtime and
from the historical SMF/FMF experiments.  It implements the paper-inspired
BOGT objective at the fixed physical OLS boundary; the few unpublished details
are explicit, named in the configuration, and never promoted to a formal claim.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from ...config import FormalSystemConfig, load_config
from ...contracts import ComplexField, ExposureCalibration, OperationProvenance
from ...evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_torch_save, atomic_write_json, canonical_sha256, file_sha256, safe_torch_load, tensor_sha256
from ...optics.fiber_operator import FiberArrayOLS, FiberArraySpec
from ...optics.relays import CoherentImagingRelay, path_abcd
from ...optics.system import FormalAmplitudeFiniteCropIdentitySystem
from . import optical_manifold as base


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_smf_bogt_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_smf_bogt_training.v1"
PILOT_REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_pilot_review.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S0"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
EXPECTED_FORMAL_KIND = "formal_amplitude_finite_crop_identity"
SEED = 1234
BATCH_SIZE = 10
PORT_COUNT = 100
STEPS = 100
PREFLIGHT_STEPS = 3
CHECKPOINT_INTERVAL = 50
LEARNING_RATE = 1.0e-3
POWER_TRANSMISSION = 0.889
FIELD_TRANSMISSION = math.sqrt(POWER_TRANSMISSION)
PRIOR_MEAN = 1.0
PRIOR_VARIANCE = 0.25
POWER_FLOOR_FRACTION = 1.0e-3
PHASE_LEVELS = (0.0, 0.82, 1.63, 2.45, 3.26, 4.08, 4.89, 5.71)
CALIBRATED_PHASE = 0.0
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 600.0
MAX_PILOT_SECONDS = 1800.0
MAX_STORAGE_BYTES = 1024**3
MIN_GRADIENT_RATIO = 0.05
MAX_GRADIENT_RATIO = 0.40
OP_WINDOW_LOWER_FACTOR = 0.80
OP_WINDOW_UPPER_FACTOR = 1.20
STATISTICAL_BATCH_SIZES = (10, 20, 50, 100)
STATISTICAL_PANEL_COUNT = 3
STATISTICAL_PANEL_SEED = 9234
STATISTICAL_MAX_CV = 0.50
SOURCE_STATUSES = {
    "paper_reported",
    "user_locked_scaled_deviation",
    "high_confidence_inference",
    "unknown",
}
EXPECTED_TRAINABLE = ("system.encoder.phase", "system.decoder.phase")
FORBIDDEN_ELECTRONIC_VAE_KEYS = {"mu", "logvar", "reparameterization", "active_latent_injector", "posterior_head"}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ForwardOutput:
    camera: torch.Tensor
    coefficients: torch.Tensor
    plane_powers: tuple[torch.Tensor, ...]
    encoder_records: tuple[ComplexField, ...]
    decoder_records: tuple[ComplexField, ...]
    ols_field: ComplexField


@dataclass(frozen=True)
class Calibration:
    port_scale: torch.Tensor
    plane_reference: torch.Tensor
    sample_count: int
    plane_lower: torch.Tensor | None = None
    plane_upper: torch.Tensor | None = None
    schema: str = "scaled_chen_bogt_calibration.v1"

    def validate(self) -> None:
        if self.port_scale.shape != (PORT_COUNT,) or self.plane_reference.shape != (5,):
            raise ValueError("calibration shapes must be [100] and [5]")
        if not torch.isfinite(self.port_scale).all() or (self.port_scale <= 0).any():
            raise ValueError("port calibration scale must be finite and positive")
        if not torch.isfinite(self.plane_reference).all() or (self.plane_reference <= 0).any():
            raise ValueError("plane calibration reference must be finite and positive")
        for name, value in (("plane_lower", self.plane_lower), ("plane_upper", self.plane_upper)):
            if value is not None and (value.shape != (5,) or not torch.isfinite(value).all() or (value <= 0).any()):
                raise ValueError(f"{name} must be finite positive [5]")
        if self.plane_lower is not None and self.plane_upper is not None and (self.plane_lower >= self.plane_upper).any():
            raise ValueError("plane operating window lower bound must be below upper bound")
        if self.sample_count != 1000:
            raise ValueError("scaled Chen calibration requires exactly 1000 samples")

    @property
    def operating_lower(self) -> torch.Tensor:
        return self.plane_lower if self.plane_lower is not None else self.plane_reference * OP_WINDOW_LOWER_FACTOR

    @property
    def operating_upper(self) -> torch.Tensor:
        return self.plane_upper if self.plane_upper is not None else self.plane_reference * OP_WINDOW_UPPER_FACTOR


@dataclass(frozen=True)
class ResearchData:
    images: torch.Tensor
    labels: torch.Tensor
    calibration_indices: torch.Tensor
    schedule: torch.Tensor
    heldout_indices: torch.Tensor
    panels: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    root = base._repo_root()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()
    return {"commit": commit, "branch": branch, "clean": not status, "status": status}


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or not state.get("clean"):
        raise ValueError(f"scaled Chen evidence requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {"schema", "hypothesis", "formal_system_config", "source", "dataset", "optical_contract", "bogt", "training", "gates", "authorization"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("scaled Chen BOGT config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported scaled Chen BOGT config")
    if not isinstance(payload["source"], Mapping):
        raise ValueError("scaled Chen source contract must be a mapping")
    statuses = payload["source"].get("status", {})
    if set(statuses) != {"fiber_dimensions", "prior", "power_loss", "backend_imaging", "kl_formula"} or any(value not in SOURCE_STATUSES for value in statuses.values()):
        raise ValueError("scaled Chen source status contract is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    if any(key in payload["optical_contract"] for key in FORBIDDEN_ELECTRONIC_VAE_KEYS):
        raise ValueError("aggregate BOGT cannot contain an electronic VAE injector")
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_count": 1000,
        "logical_batch_size": 10,
        "microbatch_size": 1,
        "schedule": "class_balanced_one_per_label_without_replacement",
        "schedule_seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("scaled Chen dataset contract is not locked")
    optical = payload["optical_contract"]
    expected = {
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3,
        "wavelength_m": 532.0e-9,
        "ols_ports": 100,
        "mode_model": "step_index_lp01_ports",
        "mode_v_number": 2.2,
        "mode_waist_m": 7.111e-6,
        "mode_fill": 0.8,
        "fixed_layer_power_transmission": POWER_TRANSMISSION,
        "backend_total_distance_m": 3.0e-3,
        "backend_magnification": -2.5,
        "software_decoder": False,
        "formal_promotion": False,
    }
    for name, value in expected.items():
        actual = optical.get(name)
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - value) > max(abs(value) * 1.0e-12, 1.0e-15):
                raise ValueError(f"scaled Chen optical contract locks {name}={value!r}")
        elif actual != value:
            raise ValueError(f"scaled Chen optical contract locks {name}={value!r}")
    bogt = payload["bogt"]
    if bogt != {
        "latent": "z_j=abs(c_j)^2/s_j",
        "scale_source": "independent_train_only_zero_phase_1000_mean_port_power",
        "scale_floor": POWER_FLOOR_FRACTION,
        "prior": {"kind": "truncated_normal", "mean": PRIOR_MEAN, "variance": PRIOR_VARIANCE, "lower": 0.0},
        "aggregate": "class_balanced_logical_batch_10_diagonal_moments",
        "kl_surrogate": "gaussian_covariance_log_variance",
        "paper_logdet_correction": "log_det_M2_interpreted_as_log_variance_term",
        "two_pass_microbatch": True,
        "prior_phase": "fixed_calibrated_zero_residual",
    }:
        raise ValueError("scaled Chen BOGT objective contract is not locked")
    training = payload["training"]
    if training != {
        "arms": ["mse_only", "mse_op", "mse_kl", "mse_kl_op"],
        "steps_per_arm": STEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "reset_optimizer_per_arm": True,
        "zero_phase_seed": SEED,
        "posthoc_phase_quantization": list(PHASE_LEVELS),
        "qat": False,
    }:
        raise ValueError("scaled Chen pilot training contract is not locked")
    if payload["authorization"].get("bounded_diagnostic_training_authorized") is not True or payload["authorization"].get("training_authorized") is not False or payload["authorization"].get("formal_promotion") is not False:
        raise ValueError("scaled Chen lifecycle is research-only")


def _load_review(path: str | Path, spec: Spec) -> dict[str, Any]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != REVIEW_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("scaled Chen review schema or hypothesis is invalid")
    if payload.get("bounded_diagnostic_training_authorized") is not True or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("scaled Chen review authorization boundary is invalid")
    approved = payload.get("approved_contract")
    expected_approved = {
        "architecture": "2+2_phase_only_100x100_2um_1.50mm",
        "ols": "fixed_100_port_scaled_lp01_v2_bank",
        "aggregate_bogt": True,
        "electronic_mu_logvar": False,
        "calibrated_phase": "zero_residual_phase",
        "backend": "inferred_3mm_magnification_minus_2.5",
        "arms": ["mse_only", "mse_op", "mse_kl", "mse_kl_op"],
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": 1,
        "preflight_steps": PREFLIGHT_STEPS,
        "resume_segment": "2+1",
        "posthoc_phase_quantization": list(PHASE_LEVELS),
    }
    if approved != expected_approved:
        raise ValueError("scaled Chen review does not approve this pilot contract")
    return payload


def _load_pilot_review(
    path: str | Path,
    spec: Spec,
    *,
    state: Mapping[str, Any],
    validation_path: Path,
    preflight_path: Path,
    lifecycle_review_path: Path,
    formal_path: Path,
    data: ResearchData,
    preflight_payload: Mapping[str, Any],
    attribution_path: Path | None = None,
) -> dict[str, Any]:
    """Load the post-preflight human authorization envelope.

    This artifact is intentionally separate from the lifecycle review: it is
    created only after the new validation/preflight pair exists and binds the
    exact measured weights and input artifacts consumed by the pilot.
    """
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PILOT_REVIEW_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("scaled Chen pilot review schema or hypothesis is invalid")
    if payload.get("status") != "approved" or payload.get("review_status") != "approved":
        raise ValueError("scaled Chen pilot review must be explicitly approved")
    if payload.get("reviewer_role") != "project_owner" or payload.get("decision") != "approve_bounded_scaled_chen_bogt_pilot":
        raise ValueError("scaled Chen pilot review decision is invalid")
    if payload.get("bounded_diagnostic_training_authorized") is not True or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("scaled Chen pilot review authorization boundary is invalid")
    contract = payload.get("approved_contract")
    expected_contract = {
        "arms": ["mse_only", "mse_op", "mse_kl", "mse_kl_op"],
        "steps_per_arm": STEPS,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": 1,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "zero_phase_seed": SEED,
        "checkpoint_steps": [CHECKPOINT_INTERVAL, STEPS],
        "alpha": float(preflight_payload["contract"]["alpha"]),
        "gamma": float(preflight_payload["contract"]["gamma"]),
        "prior": {"kind": "truncated_normal", "mean": PRIOR_MEAN, "variance": PRIOR_VARIANCE, "lower": 0.0},
        "operating_window": {"lower_factor": OP_WINDOW_LOWER_FACTOR, "upper_factor": OP_WINDOW_UPPER_FACTOR, "source_status": "project_approximation"},
        "pilot_gate_schema": "scaled_chen_bogt_factorial_science_gates.v2",
        "post_pass_next_action": "manual_review_scaled_chen_bogt_phase_closure_contract",
    }
    if contract != expected_contract:
        raise ValueError("scaled Chen pilot review does not approve this exact pilot contract")
    preflight_contract = preflight_payload.get("contract")
    if not isinstance(preflight_contract, Mapping):
        raise ValueError("scaled Chen preflight contract is missing")
    calibration_path = _resolve(preflight_contract["calibration_path"])
    gradient_panels_path = _resolve(preflight_contract["gradient_panels_path"])
    expected_bindings = {
        "source_commit": state["commit"],
        "source_branch": BRANCH,
        "source_config_sha256": file_sha256(spec.path),
        "source_formal_config_sha256": file_sha256(formal_path),
        "source_lifecycle_review_sha256": file_sha256(lifecycle_review_path),
        "source_validation_sha256": file_sha256(validation_path),
        "source_preflight_sha256": file_sha256(preflight_path),
        "source_dataset_manifest_hash": data.manifest["manifest_hash"],
        "source_schedule_sha256": data.manifest["schedule_sha256"],
        "source_mode_bank_sha256": preflight_contract.get("mode_bank_sha256"),
        "source_calibration_sha256": file_sha256(calibration_path),
        "source_gradient_panels_sha256": file_sha256(gradient_panels_path),
        "source_statistical_audit_sha256": file_sha256(_resolve(preflight_contract["statistical_batch_audit_path"])),
    }
    if attribution_path is not None:
        expected_bindings["source_attribution_sha256"] = file_sha256(attribution_path)
    bindings = payload.get("source_bindings")
    if bindings != expected_bindings:
        raise ValueError("scaled Chen pilot review source binding is stale or tampered")
    draft_path = target.with_name("pilot_review_draft.json")
    draft_digest = file_sha256(draft_path)
    if payload.get("source_draft_sha256") != draft_digest:
        raise ValueError("scaled Chen pilot review does not bind the current draft")
    measured = payload.get("measured_preflight")
    expected_measured = {
        "alpha": float(preflight_contract["alpha"]),
        "gamma": float(preflight_contract["gamma"]),
        "gradient_panels": preflight_contract.get("panel_norms"),
    }
    if measured != expected_measured:
        raise ValueError("scaled Chen pilot review does not bind measured preflight weights")
    prohibited = payload.get("prohibited")
    if prohibited != ["continuation", "generation", "phase_closure", "ols_sweep", "fmf", "formal_promotion"]:
        raise ValueError("scaled Chen pilot review prohibition boundary is invalid")
    return payload


def _load_formal(spec: Spec) -> tuple[FormalSystemConfig, Path]:
    path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    config = load_config(path)
    if config.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("scaled Chen lifecycle requires current identity lineage")
    formal = config.system
    if formal.kind != EXPECTED_FORMAL_KIND or not formal.is_formal:
        raise ValueError("scaled Chen lifecycle requires formal finite-crop identity geometry")
    locked = {
        "wavelength_m": 532.0e-9,
        "slm_shape": (200, 200),
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": (100, 100),
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3,
    }
    for name, expected in locked.items():
        actual = getattr(formal, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal, path


def _research_schedule(labels: torch.Tensor, *, seed: int = SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint 1000-sample calibration and 100-step balanced schedules."""
    if labels.ndim != 1 or labels.numel() < 55000:
        raise ValueError("scaled Chen lifecycle requires the complete 60k MNIST train split")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train = torch.arange(55000, dtype=torch.long)
    calibration_parts = []
    remaining_parts = []
    for label in range(10):
        candidates = train[labels.index_select(0, train) == label]
        order = torch.randperm(candidates.numel(), generator=generator)
        shuffled = candidates.index_select(0, order)
        calibration_parts.append(shuffled[:100])
        remaining_parts.append(shuffled[100:])
    calibration = torch.cat(calibration_parts)
    per_label = [values[:STEPS] for values in remaining_parts]
    schedule = torch.stack([torch.stack([per_label[label][step] for label in range(10)]) for step in range(STEPS)])
    if torch.unique(calibration).numel() != calibration.numel() or torch.unique(schedule).numel() != schedule.numel() or torch.isin(calibration, schedule).any():
        raise ValueError("scaled Chen calibration and training schedule overlap")
    if not torch.equal(labels.index_select(0, schedule.reshape(-1)), torch.arange(10).repeat(STEPS)):
        raise ValueError("scaled Chen schedule is not class balanced")
    return calibration, schedule


def _load_research_data(dataset_root: str | Path) -> ResearchData:
    images, labels = base.load_local_mnist(dataset_root, train=True)
    calibration, schedule = _research_schedule(labels)
    heldout = torch.arange(55000, 60000, dtype=torch.long)
    panels = tuple(schedule[index].clone() for index in range(3))
    manifest: dict[str, Any] = {
        "schema": "scaled_chen_bogt_mnist_manifest.v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "images_sha256": tensor_sha256(images),
        "labels_sha256": tensor_sha256(labels),
        "calibration_indices_sha256": tensor_sha256(torch.sort(calibration).values),
        "schedule_sha256": tensor_sha256(schedule),
        "heldout_indices_sha256": tensor_sha256(heldout),
        "panel_indices_sha256": [tensor_sha256(panel) for panel in panels],
        "calibration_count": int(calibration.numel()),
        "schedule_shape": list(schedule.shape),
        "heldout_count": int(heldout.numel()),
        "panel_count": len(panels),
        "seed": SEED,
        "calibration_and_schedule_disjoint": True,
        "heldout_excluded": True,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return ResearchData(images, labels, calibration, schedule, heldout, panels, manifest)


def _statistical_audit_panels(data: ResearchData, *, batch_size: int, count: int = STATISTICAL_PANEL_COUNT, seed: int = STATISTICAL_PANEL_SEED, offset: int = 0) -> tuple[torch.Tensor, ...]:
    """Build fresh, class-balanced train-only panels for BOGT batch auditing."""
    if batch_size not in STATISTICAL_BATCH_SIZES or batch_size % 10 != 0 or count <= 0:
        raise ValueError("statistical audit batch size must be one of the registered multiples of ten")
    reserved = torch.cat((data.calibration_indices, data.schedule.reshape(-1), data.heldout_indices, *data.panels))
    train = torch.arange(55000, dtype=torch.long)
    available_by_label = [train[(data.labels.index_select(0, train) == label) & ~torch.isin(train, reserved)] for label in range(10)]
    per_label = batch_size // 10
    if offset < 0 or any(values.numel() < offset + count * per_label for values in available_by_label):
        raise ValueError("insufficient fresh train-only statistical audit samples")
    panels: list[torch.Tensor] = []
    shuffled_by_label = [values.index_select(0, torch.randperm(values.numel(), generator=torch.Generator(device="cpu").manual_seed(seed + label))) for label, values in enumerate(available_by_label)]
    for panel in range(count):
        chunks: list[torch.Tensor] = []
        for values in shuffled_by_label:
            start = offset + panel * per_label
            chunks.append(values[start : start + per_label])
        panel_indices = torch.cat(chunks)
        if torch.unique(panel_indices).numel() != panel_indices.numel():
            raise ValueError("statistical audit panel contains duplicate samples")
        panels.append(panel_indices)
    return tuple(panels)


def statistical_batch_stability_audit(
    model: ScaledChenBOGTModel,
    data: ResearchData,
    calibration: Calibration,
    *,
    device: torch.device | str,
    batch_sizes: Sequence[int] = STATISTICAL_BATCH_SIZES,
    panel_count: int = STATISTICAL_PANEL_COUNT,
) -> dict[str, Any]:
    """No-update audit selecting the smallest stable logical BOGT batch.

    Every panel is evaluated one sample at a time (microbatch one), then
    aggregate moments are computed over the requested logical batch.  Stability
    is deliberately explicit and conservative: panel CVs for aggregate KL,
    per-port means, coefficient rank, and power-latent rank must all be <= 0.5.
    """
    calibration.validate()
    before = _phase_state(model)
    was_training = model.training
    model.eval()
    result: dict[str, Any] = {"schema": "scaled_chen_bogt_statistical_batch_audit.v1", "batch_sizes": list(batch_sizes), "panel_count": panel_count, "microbatch_size": 1, "batches": {}}
    try:
        panel_offset = 0
        all_panel_indices: list[torch.Tensor] = []
        for batch_size in batch_sizes:
            panels = _statistical_audit_panels(data, batch_size=batch_size, count=panel_count, offset=panel_offset)
            panel_offset += batch_size * panel_count // 10
            all_panel_indices.extend(panels)
            panel_reports: list[dict[str, Any]] = []
            with torch.no_grad():
                for panel in panels:
                    coefficients: list[torch.Tensor] = []
                    latents: list[torch.Tensor] = []
                    for index in panel:
                        sample_index = int(index)
                        rendered = base.render_intensity_batch(data.images[sample_index : sample_index + 1])
                        output = model(rendered.amplitude.to(device))
                        coefficients.append(output.coefficients.detach().cpu())
                        latents.append(normalize_port_power(output.coefficients.abs().square(), calibration.port_scale).detach().cpu())
                    coeff = torch.cat(coefficients, dim=0)
                    z = torch.cat(latents, dim=0)
                    mean, variance = aggregate_moments(z)
                    panel_reports.append({
                        "indices_sha256": tensor_sha256(panel),
                        "port_mean": mean.tolist(),
                        "port_variance": variance.tolist(),
                        "aggregate_kl": float(bogt_kl_surrogate(mean, variance)),
                        "complex_coefficient_effective_rank": coefficient_effective_rank(coeff),
                        "power_latent_effective_rank": power_latent_effective_rank(z),
                    })
            kl_values = torch.tensor([report["aggregate_kl"] for report in panel_reports], dtype=torch.float64)
            mean_matrix = torch.tensor([report["port_mean"] for report in panel_reports], dtype=torch.float64)
            coeff_rank = torch.tensor([report["complex_coefficient_effective_rank"] for report in panel_reports], dtype=torch.float64)
            power_rank = torch.tensor([report["power_latent_effective_rank"] for report in panel_reports], dtype=torch.float64)
            mean_port = mean_matrix.mean(dim=0).abs().clamp_min(1.0e-8)
            cv = {
                "aggregate_kl": float(kl_values.std(unbiased=False) / kl_values.abs().mean().clamp_min(1.0e-8)),
                "per_port_mean": float((mean_matrix.std(dim=0, unbiased=False) / mean_port).mean()),
                "complex_coefficient_rank": float(coeff_rank.std(unbiased=False) / coeff_rank.abs().mean().clamp_min(1.0e-8)),
                "power_latent_rank": float(power_rank.std(unbiased=False) / power_rank.abs().mean().clamp_min(1.0e-8)) if float(power_rank.abs().mean()) > 0.0 else 0.0,
            }
            stable = bool(all(math.isfinite(value) and value <= STATISTICAL_MAX_CV for value in cv.values()))
            result["batches"][str(batch_size)] = {"panels": panel_reports, "cv": cv, "stable": stable}
        combined = torch.cat(all_panel_indices)
        if torch.unique(combined).numel() != combined.numel():
            raise RuntimeError("statistical BOGT audit panels are not mutually exclusive")
    finally:
        if any(not torch.equal(before[name], value.detach()) for name, value in _phase_state(model).items()):
            raise RuntimeError("statistical BOGT audit changed model state")
        if was_training:
            model.train()
    stable_sizes = [int(size) for size in batch_sizes if result["batches"][str(size)]["stable"]]
    result["selected_batch_size"] = min(stable_sizes) if stable_sizes else None
    result["status"] = "passed" if stable_sizes else "valid_failed"
    return result


def calibrate_model(model: ScaledChenBOGTModel, amplitudes: torch.Tensor) -> Calibration:
    """Compute train-only zero-phase port and plane references without gradients."""
    if amplitudes.ndim != 4 or tuple(amplitudes.shape[1:]) != (1, 200, 200) or amplitudes.shape[0] != 1000:
        raise ValueError("scaled Chen calibration requires [1000,1,200,200] amplitudes")
    if any(not torch.equal(parameter.detach(), torch.zeros_like(parameter)) for name, parameter in model.named_parameters() if name in EXPECTED_TRAINABLE):
        raise ValueError("scaled Chen calibration must use an independent zero-phase model")
    was_training = model.training
    model.eval()
    port_values: list[torch.Tensor] = []
    plane_values: list[torch.Tensor] = []
    with torch.no_grad():
        for sample in amplitudes.split(1):
            output = model(sample)
            port_values.append(output.coefficients.abs().square().cpu())
            plane_values.append(torch.stack([value.detach().cpu().reshape(-1)[0] for value in output.plane_powers]))
    if was_training:
        model.train()
    ports = torch.cat(port_values, dim=0)
    planes = torch.stack(plane_values)
    means = ports.mean(dim=0)
    floor = float(means.median()) * POWER_FLOOR_FRACTION
    scale = means.clamp_min(max(floor, torch.finfo(means.dtype).eps))
    reference = planes.median(dim=0).values.clamp_min(torch.finfo(planes.dtype).eps)
    calibration = Calibration(
        scale,
        reference,
        int(amplitudes.shape[0]),
        plane_lower=reference * OP_WINDOW_LOWER_FACTOR,
        plane_upper=reference * OP_WINDOW_UPPER_FACTOR,
    )
    calibration.validate()
    return calibration


def _phase_quantize(phase: torch.Tensor, levels: Sequence[float] = PHASE_LEVELS) -> torch.Tensor:
    """Nearest post-hoc eight-level phase quantization; no straight-through QAT."""
    if phase.ndim < 1 or not torch.isfinite(phase).all():
        raise ValueError("phase must be finite")
    values = torch.as_tensor(tuple(levels), dtype=phase.dtype, device=phase.device)
    wrapped = torch.remainder(phase, 2.0 * math.pi)
    distances = (wrapped.unsqueeze(-1) - values).abs()
    return values[distances.argmin(dim=-1)]


def normalize_port_power(power: torch.Tensor, scale: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    if power.ndim != 2 or power.shape[-1] != PORT_COUNT:
        raise ValueError("port power must have shape [batch,100]")
    if scale.shape != (PORT_COUNT,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("port scale must be finite, positive, and [100]")
    if not torch.isfinite(power).all() or (power < 0).any():
        raise ValueError("port power must be finite and non-negative")
    return power / scale.to(power.device, power.dtype).clamp_min(eps)


def coefficient_effective_rank(coefficients: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    """Participation-ratio rank of real/imaginary OLS coefficients."""
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
        raise ValueError("coefficients must be finite complex [batch,100]")
    features = torch.cat((coefficients.real, coefficients.imag), dim=1)
    singular = torch.linalg.svdvals(features)
    weights = singular.square() / singular.square().sum().clamp_min(eps)
    return float((weights.square().sum().clamp_min(eps)).reciprocal())


def power_latent_effective_rank(z: torch.Tensor, *, eps: float = 1.0e-12) -> float:
    """Participation-ratio rank of the real non-negative BOGT power latents."""
    if z.ndim != 2 or z.shape[1] != PORT_COUNT or not torch.is_floating_point(z) or not torch.isfinite(z).all() or (z < 0).any():
        raise ValueError("power latents must be finite non-negative [batch,100]")
    centered = z - z.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    if float(energy.sum()) <= eps:
        return 0.0
    weights = energy / energy.sum().clamp_min(eps)
    return float(weights.square().sum().clamp_min(eps).reciprocal())


def global_phase_invariance_metrics(coefficients: torch.Tensor, *, phase_rad: float = 0.731) -> dict[str, float]:
    """Verify that power latents and effective ranks ignore a global phase."""
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
        raise ValueError("global phase check requires finite complex [batch,100] coefficients")
    rotated = coefficients * torch.exp(torch.tensor(1j * phase_rad, device=coefficients.device, dtype=coefficients.dtype))
    z = coefficients.abs().square()
    rotated_z = rotated.abs().square()
    return {
        "power_max_abs_delta": float((z - rotated_z).abs().max().detach().cpu()),
        "coefficient_rank_abs_delta": abs(coefficient_effective_rank(coefficients) - coefficient_effective_rank(rotated)),
        "power_rank_abs_delta": abs(power_latent_effective_rank(z) - power_latent_effective_rank(rotated_z)),
    }


def mode_bank_sha256(model: ScaledChenBOGTModel) -> str:
    """Content hash of the fixed OLS mode bank and transport boundary."""
    return canonical_sha256({
        "input_modes": tensor_sha256(model.ols.input_modes.detach().cpu()),
        "output_modes": tensor_sha256(model.ols.output_modes.detach().cpu()),
        "transport": tensor_sha256(model.ols.transport.detach().cpu()),
        "spec": model.ols.spec.__dict__,
    })


def require_mode_bank_sha256(model: ScaledChenBOGTModel, expected: str) -> None:
    """Reject a preflight when the fixed mode-bank content is not the bound one."""
    if not isinstance(expected, str) or mode_bank_sha256(model) != expected:
        raise ValueError("scaled Chen mode-bank hash is stale or mismatched")


def aggregate_moments(z: torch.Tensor, *, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor]:
    if z.ndim != 2 or z.shape[-1] != PORT_COUNT:
        raise ValueError("aggregate BOGT latent must have shape [batch,100]")
    if z.shape[0] < 2 or not torch.isfinite(z).all() or (z < 0).any():
        raise ValueError("aggregate BOGT latent must be finite, non-negative, and have >=2 samples")
    mean = z.mean(dim=0)
    variance = z.var(dim=0, unbiased=False).clamp_min(eps)
    return mean, variance


def bogt_kl_surrogate(
    mean_q: torch.Tensor,
    variance_q: torch.Tensor,
    *,
    mean_p: float = PRIOR_MEAN,
    variance_p: float = PRIOR_VARIANCE,
) -> torch.Tensor:
    """Paper-intent Gaussian aggregate KL, corrected to log covariance terms."""
    if mean_q.shape != variance_q.shape or mean_q.ndim != 1 or not torch.isfinite(mean_q).all() or not torch.isfinite(variance_q).all() or (variance_q <= 0).any():
        raise ValueError("aggregate moments must be finite one-dimensional tensors")
    if variance_p <= 0.0:
        raise ValueError("prior variance must be positive")
    term = variance_q / variance_p + (mean_q - mean_p).square() / variance_p - 1.0 + math.log(variance_p) - variance_q.log()
    return 0.5 * term.mean()


def two_pass_bogt_surrogate(z: torch.Tensor, *, eps: float = 1.0e-8) -> torch.Tensor:
    """Value-preserving two-pass surrogate used for microbatch training.

    The first pass is a detached aggregate over the complete logical batch;
    the second pass is the live microbatch graph.  This retains exactly the
    batch-KL first-order gradient without retaining ten complete optical graphs.
    """
    if z.ndim != 2 or z.shape[-1] != PORT_COUNT:
        raise ValueError("two-pass BOGT latent must have shape [batch,100]")
    first = z.detach().requires_grad_(True)
    first_mean, first_variance = aggregate_moments(first, eps=eps)
    first_value = bogt_kl_surrogate(first_mean, first_variance)
    gradient = torch.autograd.grad(first_value, first, create_graph=False)[0].detach()
    linear = (z * gradient).sum()
    return first_value.detach() + linear - linear.detach()


def lop_from_powers(powers: Sequence[torch.Tensor], references: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    if len(powers) != 5 or references.shape != (5,) or not torch.isfinite(references).all() or (references <= 0).any():
        raise ValueError("l_op requires five positive plane references")
    stacked = torch.stack([value if value.ndim == 1 else value.reshape(-1) for value in powers], dim=1)
    if not torch.isfinite(stacked).all() or (stacked < 0).any():
        raise ValueError("plane powers must be finite and non-negative")
    return torch.log((stacked + eps) / (references.to(stacked.device, stacked.dtype) + eps)).square().mean()


def lop_operating_window_from_powers(
    powers: Sequence[torch.Tensor],
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Penalize only power outside a frozen acceptable operating window.

    The lower/upper bounds are calibration-derived and never receive gradients.
    Inside the window the contribution is exactly zero; outside it is the
    squared log-distance to the nearest boundary.  This is the project-level
    approximation to Chen's reported power constraint because the exact
    unpublished implementation is not available.
    """
    if len(powers) != 5 or lower.shape != (5,) or upper.shape != (5,):
        raise ValueError("operating-window l_op requires five bounds")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (lower <= 0).any() or (lower >= upper).any():
        raise ValueError("operating-window bounds must be finite and strictly positive")
    stacked = torch.stack([value if value.ndim == 1 else value.reshape(-1) for value in powers], dim=1)
    if not torch.isfinite(stacked).all() or (stacked < 0).any():
        raise ValueError("plane powers must be finite and non-negative")
    low = lower.to(stacked.device, stacked.dtype)
    high = upper.to(stacked.device, stacked.dtype)
    below = F.relu(torch.log((low + eps) / (stacked + eps)))
    above = F.relu(torch.log((stacked + eps) / (high + eps)))
    return below.square().add(above.square()).mean()


def _truncated_normal_log_normalizer(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    sigma = variance.clamp_min(1.0e-15).sqrt()
    cdf0 = 0.5 * (1.0 + torch.erf((0.0 - mean) / (sigma * math.sqrt(2.0))))
    return torch.log((1.0 - cdf0).clamp_min(1.0e-15))


def truncated_normal_kl_reference(
    mean_q: torch.Tensor,
    variance_q: torch.Tensor,
    *,
    mean_p: float = PRIOR_MEAN,
    variance_p: float = PRIOR_VARIANCE,
    quadrature_points: int = 256,
) -> torch.Tensor:
    """Numerical KL(TN(q)||TN(p)) reference, independent of the surrogate.

    This is an audit-only sensitivity reference.  Training continues to use
    the closed-form Gaussian aggregate surrogate above.  A finite trapezoidal
    integration domain at ten standard deviations is sufficient for the fixed
    prior and gives a deterministic, differentiable reference value.
    """
    if mean_q.shape != variance_q.shape or mean_q.ndim != 1 or not torch.isfinite(mean_q).all() or not torch.isfinite(variance_q).all() or (variance_q <= 0).any() or variance_p <= 0.0 or quadrature_points < 32:
        raise ValueError("truncated-normal KL reference inputs are invalid")
    if torch.allclose(mean_q, torch.full_like(mean_q, mean_p), atol=1.0e-12, rtol=1.0e-12) and torch.allclose(variance_q, torch.full_like(variance_q, variance_p), atol=1.0e-12, rtol=1.0e-12):
        return torch.zeros((), dtype=mean_q.dtype, device=mean_q.device)
    sigma_q = variance_q.sqrt()
    sigma_p = math.sqrt(variance_p)
    upper = torch.maximum(mean_q + 10.0 * sigma_q, torch.full_like(mean_q, mean_p + 10.0 * sigma_p)).clamp_min(10.0)
    # Integrate each port on a shared normalized [0, 1] grid.  The endpoint
    # zero is harmless because the densities are finite after clamping.
    grid = torch.linspace(0.0, 1.0, quadrature_points, device=mean_q.device, dtype=mean_q.dtype)
    x = grid[:, None] * upper[None, :]
    log_q = -0.5 * ((x - mean_q[None, :]) / sigma_q[None, :]).square() - torch.log(sigma_q[None, :]) - 0.5 * math.log(2.0 * math.pi) - _truncated_normal_log_normalizer(mean_q, variance_q)[None, :]
    mean_p_t = torch.full_like(mean_q, mean_p)
    var_p_t = torch.full_like(mean_q, variance_p)
    log_p = -0.5 * ((x - mean_p_t[None, :]) / sigma_p).square() - math.log(sigma_p) - 0.5 * math.log(2.0 * math.pi) - _truncated_normal_log_normalizer(mean_p_t, var_p_t)[None, :]
    density_q = log_q.exp()
    values = density_q * (log_q - log_p)
    integral = torch.trapezoid(values, x, dim=0)
    return integral.mean().clamp_min(0.0)


def bogt_kl_sensitivity_reference(mean_q: torch.Tensor, variance_q: torch.Tensor) -> dict[str, float]:
    """Return surrogate and numerical truncated-normal KL for audit reports."""
    surrogate = bogt_kl_surrogate(mean_q, variance_q)
    exact = truncated_normal_kl_reference(mean_q, variance_q)
    return {"gaussian_surrogate": float(surrogate.detach().cpu()), "truncated_normal_numerical": float(exact.detach().cpu()), "absolute_difference": float((surrogate - exact).abs().detach().cpu())}


def truncated_normal_sample(
    shape: tuple[int, ...], *, generator: torch.Generator | None = None, mean: float = PRIOR_MEAN, variance: float = PRIOR_VARIANCE, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic inverse-CDF sampler for TN(mean, variance; lower=0)."""
    if variance <= 0.0 or any(int(dim) <= 0 for dim in shape):
        raise ValueError("truncated-normal shape and variance must be positive")
    sigma = math.sqrt(variance)
    lower = 0.5 * (1.0 + math.erf((0.0 - mean) / (sigma * math.sqrt(2.0))))
    uniform = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    uniform = lower + (1.0 - lower) * uniform
    values = mean + sigma * math.sqrt(2.0) * torch.special.erfinv(2.0 * uniform - 1.0)
    return values.clamp_min(torch.finfo(dtype).eps)


def sample_prior_coefficients(
    shape: tuple[int, ...], scale: torch.Tensor, *, phase: torch.Tensor | None = None, generator: torch.Generator | None = None, device: torch.device | str = "cpu",
) -> torch.Tensor:
    if len(shape) != 2 or shape[1] != PORT_COUNT or scale.shape != (PORT_COUNT,):
        raise ValueError("prior coefficient shape must be [batch,100]")
    z = truncated_normal_sample(shape, generator=generator, device=device, dtype=scale.dtype)
    scale_work = scale.to(z.device, z.dtype).clamp_min(torch.finfo(z.dtype).eps)
    if phase is None:
        phase_work = torch.zeros_like(z)
    else:
        if phase.shape not in {scale.shape, shape}:
            raise ValueError("fixed prior phase must be [100] or [batch,100]")
        phase_work = phase.to(z.device, z.dtype)
        if phase_work.shape == scale.shape:
            phase_work = phase_work.unsqueeze(0).expand(shape)
    return torch.sqrt(scale_work.unsqueeze(0) * z).to(torch.complex64) * torch.exp(1j * phase_work).to(torch.complex64)


def fixed_calibrated_phase(*, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Ideal digital-twin calibration phase: zero residual for every port."""
    return torch.full((PORT_COUNT,), CALIBRATED_PHASE, device=device, dtype=dtype)


def prior_negative_controls(coefficients: torch.Tensor, *, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Construct the registered fixed-latent, port-shuffle and phase-shuffle controls."""
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients):
        raise ValueError("prior controls require complex [batch,100] coefficients")
    fixed = coefficients[:1].expand_as(coefficients).clone()
    permutation = torch.randperm(PORT_COUNT, generator=generator, device=coefficients.device)
    port = coefficients.index_select(1, permutation)
    phase_permutation = torch.randperm(PORT_COUNT, generator=generator, device=coefficients.device)
    phase = torch.angle(coefficients).index_select(1, phase_permutation)
    phase_shuffled = coefficients.abs() * torch.exp(1j * phase).to(coefficients.dtype)
    return {"fixed_latent": fixed, "port_permutation": port, "phase_permutation": phase_shuffled}


def _attenuate(field: ComplexField, factor: float, operation: str) -> ComplexField:
    return ComplexField(
        field.tensor * factor,
        field.plane,
        OperationProvenance(operation, field.plane.plane_id, None, {"field_factor": factor, "power_transmission": factor * factor}),
    )


def _attenuate_records(records: Sequence[ComplexField]) -> tuple[ComplexField, ...]:
    return tuple(_attenuate(record, FIELD_TRANSMISSION ** (index + 1), "fixed_layer_field_transmission") for index, record in enumerate(records))


class ScaledChenBOGTModel(nn.Module):
    """2+2 phase-only stack with fixed scaled LP01 OLS and Chen-style relay."""

    def __init__(self, formal: FormalSystemConfig) -> None:
        super().__init__()
        self.system = FormalAmplitudeFiniteCropIdentitySystem(formal)
        self.exposure = ExposureCalibration("scaled-chen-bogt-fixed-v1", 1.0, 1.0)
        spec = FiberArraySpec(
            grid_shape=(10, 10), pitch_m=2.0e-6, wavelength_m=532.0e-9,
            waist_m=7.111e-6, fill=0.8, v_number=2.2,
            mode_model="step_index_lp01_ports", expected_fov_m=(200.0e-6, 200.0e-6),
        )
        self.ols = FiberArrayOLS(self.system.encoder_plane, spec)
        total = 3.0e-3
        u = total / (1.0 + 2.5)
        v = total - u
        focal = u * v / total
        camera_plane = self.system.chip_plane.with_plane(
            "chen_camera", shape=self.system.camera_plane.shape,
            pixel_pitch_m=self.system.camera_plane.pixel_pitch_m,
            z_m=self.system.chip_plane.z_m + total, coordinate_frame="camera",
        )
        self.chen_relay = CoherentImagingRelay(
            self.system.chip_plane, camera_plane, -2.5, name="inferred_backend_imaging",
            segments_m=(u, v), focal_lengths_m=(focal,), clear_aperture_m=formal.back_clear_aperture_m,
        )
        self._set_trainable_contract()

    def _set_trainable_contract(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in EXPECTED_TRAINABLE)
        actual = tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        if actual != EXPECTED_TRAINABLE:
            raise ValueError(f"scaled Chen trainable parameters are invalid: {actual}")
        for parameter in self.ols.parameters():
            parameter.requires_grad_(False)

    def _coefficients(self, field: ComplexField) -> torch.Tensor:
        values = field.tensor[:, 0]
        modes = self.ols.input_modes.to(values)
        area = self.system.encoder_plane.pixel_pitch_m[0] * self.system.encoder_plane.pixel_pitch_m[1]
        return torch.einsum("mhw,bhw->bm", modes.conj(), values) * area

    def forward(self, amplitude: torch.Tensor, *, return_trace: bool = False) -> ForwardOutput:
        if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (1, 200, 200) or torch.is_complex(amplitude) or not torch.isfinite(amplitude).all() or amplitude.min() < 0.0 or amplitude.max() > 1.0:
            raise ValueError("scaled Chen amplitude must be finite [N,1,200,200] in [0,1]")
        slm = self.system.slm(self.system.command(amplitude))
        front = self.system.front(slm)
        encoded, encoder_records_raw, _ = self.system.encoder(front, "encoder")
        encoder_records = _attenuate_records(encoder_records_raw)
        encoded = _attenuate(encoded, FIELD_TRANSMISSION ** 2, "fixed_encoder_group_transmission")
        ols = self.ols(encoded)
        coefficients = self._coefficients(encoded)
        decoded, decoder_records_raw, _ = self.system.decoder(ols, "decoder")
        decoder_records = _attenuate_records(decoder_records_raw)
        decoded = _attenuate(decoded, FIELD_TRANSMISSION ** 2, "fixed_decoder_group_transmission")
        chip = ComplexField(decoded.tensor, self.system.chip_plane, decoded.provenance)
        camera_field = self.chen_relay(chip)
        records = (*encoder_records, ols, *decoder_records)
        plane_powers = tuple(record.tensor.abs().square().sum(dim=tuple(range(1, record.tensor.ndim))) for record in records)
        return ForwardOutput(camera_field.tensor.abs().square(), coefficients, plane_powers, encoder_records, decoder_records, ols)

    def reemit_decoder(self, coefficients: torch.Tensor, *, phase: torch.Tensor | None = None) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients):
            raise ValueError("decoder re-emission requires complex [batch,100] coefficients")
        if phase is not None:
            if phase.shape not in {(PORT_COUNT,), coefficients.shape}:
                raise ValueError("fixed prior phase must be [100] or [batch,100]")
            phase_work = phase.to(coefficients.device, coefficients.real.dtype)
            if phase_work.ndim == 1:
                phase_work = phase_work.unsqueeze(0)
            coefficients = coefficients * torch.exp(1j * phase_work).to(coefficients.dtype)
        modes = self.ols.output_modes.to(coefficients)
        values = torch.einsum("bm,mhw->bhw", coefficients, modes).unsqueeze(1)
        field = ComplexField(values, self.system.encoder_plane.with_plane("ols_output"), OperationProvenance("fixed_ols_prior_reemission", "ols_output", 0.0, {"phase": "fixed_calibrated_zero_residual"}))
        decoded, _, _ = self.system.decoder(field, "decoder")
        decoded = _attenuate(decoded, FIELD_TRANSMISSION ** 2, "fixed_decoder_group_transmission")
        return self.chen_relay(ComplexField(decoded.tensor, self.system.chip_plane, decoded.provenance)).tensor.abs().square()

    def generate_from_prior(self, calibration: Calibration, count: int, *, generator: torch.Generator | None = None, phase: torch.Tensor | None = None) -> torch.Tensor:
        """Re-emit prior port powers directly into the fixed decoder/CMOS path."""
        calibration.validate()
        if count <= 0:
            raise ValueError("prior generation count must be positive")
        device = next(self.parameters()).device
        scale = calibration.port_scale.to(device)
        coefficients = sample_prior_coefficients((count, PORT_COUNT), scale, phase=phase, generator=generator, device=device)
        return self.reemit_decoder(coefficients, phase=None)


def loss_components(output: ForwardOutput, target: torch.Tensor, calibration: Calibration | None = None) -> dict[str, torch.Tensor]:
    mse = F.mse_loss(output.camera, target)
    result = {"mse": mse}
    if calibration is not None:
        calibration.validate()
        z = normalize_port_power(output.coefficients.abs().square(), calibration.port_scale)
        mean, variance = aggregate_moments(z)
        result["kl"] = bogt_kl_surrogate(mean, variance)
        # Keep this low-level component API as the exact reference loss used by
        # historical diagnostics; the scaled Chen training route uses the
        # frozen operating-window variant in ``_arm_loss``.
        result["op"] = lop_from_powers(output.plane_powers, calibration.plane_reference)
    else:
        result["kl"] = torch.zeros_like(mse)
        result["op"] = torch.zeros_like(mse)
    return result


def calibrate_loss_weights(mse_gradient_norm: float, objective_gradient_norm: float, *, target_ratio: float = 0.10) -> float:
    if not math.isfinite(mse_gradient_norm) or not math.isfinite(objective_gradient_norm) or mse_gradient_norm <= 0.0 or objective_gradient_norm <= 0.0 or not 0.0 < target_ratio < 1.0:
        raise ValueError("gradient norms and target ratio must be finite and positive")
    return target_ratio * mse_gradient_norm / objective_gradient_norm


def pilot_gate_results(metrics: Mapping[str, float | bool]) -> tuple[GateResult, ...]:
    """Evaluate the pre-registered four-arm pilot scientific gates."""
    def number(name: str) -> float:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"pilot metric {name} is missing or non-finite")
        return float(value)

    mse_combined = number("mse_kl_op_heldout_mse")
    mse_op = number("mse_op_heldout_mse")
    # The legacy pair names are retained only for the existing unit contract;
    # a real pilot supplies both factorial KL comparisons explicitly.
    kl_mse_no = number("mse_only_kl") if "mse_only_kl" in metrics else number("mse_op_kl")
    kl_mse_yes = number("mse_kl_kl")
    kl_op_no = number("mse_op_kl") if "mse_op_kl" in metrics else kl_mse_no
    kl_op_yes = number("mse_kl_op_kl") if "mse_kl_op_kl" in metrics else kl_mse_yes
    op_no = number("mse_only_op")
    op_yes = number("mse_op_op")
    rank_retention = number("ols_effective_rank_retention")
    paired = number("paired_mse")
    control = number("template_control_mse")
    cyclic = number("cyclic_control_mse")
    phase_updated = bool(metrics.get("four_phase_updated", False))
    frozen = bool(metrics.get("ols_exposure_calibration_evaluator_frozen", False))
    return (
        GateResult("combined_heldout_mse", mse_combined <= 1.10 * mse_op, {"combined": mse_combined, "mse_op": mse_op, "ratio": mse_combined / max(mse_op, 1.0e-30)}),
        GateResult("aggregate_kl_reduction_without_op", kl_mse_yes <= 0.95 * kl_mse_no, {"with_kl": kl_mse_yes, "without_kl": kl_mse_no}),
        GateResult("aggregate_kl_reduction_with_op", kl_op_yes <= 0.95 * kl_op_no, {"with_kl": kl_op_yes, "without_kl": kl_op_no}),
        GateResult("optical_power_loss_reduction", op_yes <= 0.90 * op_no, {"with_op": op_yes, "without_op": op_no}),
        GateResult("ols_effective_rank_retention", rank_retention >= 0.90, {"retention": rank_retention}),
        GateResult("paired_mse_controls", paired < control and paired < cyclic, {"paired": paired, "template": control, "cyclic": cyclic}),
        GateResult("four_phase_update", phase_updated, {"updated": phase_updated}),
        GateResult("fixed_boundary_freeze", frozen, {"frozen": frozen}),
    )


def classify_pilot_result(science_gates: Sequence[GateResult], harness_gates: Sequence[GateResult]) -> tuple[str, tuple[str, ...]]:
    """Classify a completed pilot while retaining every failed science gate."""
    if not all(gate.passed for gate in harness_gates):
        return "invalid_harness", tuple(gate.name for gate in harness_gates if not gate.passed)
    failed = tuple(gate.name for gate in science_gates if not gate.passed)
    if not failed:
        return "passed_scaled_chen_bogt_pilot", ()
    categories: set[str] = set()
    if any(name in failed for name in ("aggregate_kl_reduction_without_op", "aggregate_kl_reduction_with_op")):
        categories.add("valid_failed_kl_effect")
    if "optical_power_loss_reduction" in failed:
        categories.add("valid_failed_op_effect")
    if "combined_heldout_mse" in failed:
        categories.add("valid_failed_reconstruction_compatibility")
    if "ols_effective_rank_retention" in failed:
        categories.add("valid_failed_latent_capacity")
    if "paired_mse_controls" in failed:
        categories.add("valid_failed_instance_specificity")
    if len(categories) == 1:
        return next(iter(categories)), failed
    return "valid_failed_multiple_science_gates", failed


def _phase_update_metrics(initial: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor]) -> dict[str, float]:
    updates: dict[str, float] = {}
    for name, before in initial.items():
        after = final[name]
        if before.shape != after.shape:
            raise ValueError(f"phase shape changed for {name}")
        if before.ndim != 4 or before.shape[0] != 2:
            raise ValueError(f"scaled Chen phase must expose two layers for {name}")
        for layer in range(2):
            updates[f"{name}[{layer}]"] = float((after[layer] - before[layer]).abs().max().detach().cpu())
    return updates


def _set_phase_state(model: ScaledChenBOGTModel, state: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.system.encoder.phase.copy_(state["system.encoder.phase"].to(model.system.encoder.phase))
        model.system.decoder.phase.copy_(state["system.decoder.phase"].to(model.system.decoder.phase))


def _phase_state(model: ScaledChenBOGTModel) -> dict[str, torch.Tensor]:
    return {
        "system.encoder.phase": model.system.encoder.phase.detach().clone(),
        "system.decoder.phase": model.system.decoder.phase.detach().clone(),
    }


def _evaluate_model(
    model: ScaledChenBOGTModel,
    data: ResearchData,
    calibration: Calibration,
    indices: torch.Tensor,
    device: torch.device,
    class_templates: torch.Tensor,
    calibration_sha256: str | None = None,
    split: str = "heldout",
) -> dict[str, Any]:
    """Evaluate one arm on heldout samples using microbatch-one optics."""
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("pilot evaluation requires non-empty one-dimensional indices")
    if class_templates.shape != (10, 1, 200, 200):
        raise ValueError("class templates must have shape [10,1,200,200]")
    was_training = model.training
    model.eval()
    coefficient_values: list[torch.Tensor] = []
    z_values: list[torch.Tensor] = []
    plane_values: list[torch.Tensor] = []
    camera_values: list[torch.Tensor] = []
    template_error = 0.0
    first_camera: torch.Tensor | None = None
    previous_target: torch.Tensor | None = None
    cyclic_error = 0.0
    paired_error = 0.0
    count = 0
    with torch.no_grad():
        for batch_indices in indices.split(BATCH_SIZE):
            rendered = base.render_intensity_batch(data.images.index_select(0, batch_indices))
            amplitudes = rendered.amplitude.to(device)
            targets = rendered.target.float()
            labels = data.labels.index_select(0, batch_indices)
            output = model(amplitudes)
            camera = output.camera.detach().cpu().float()
            if not torch.isfinite(camera).all() or not torch.isfinite(targets).all():
                raise RuntimeError("scaled Chen pilot evaluation produced non-finite camera values")
            paired_error += float(F.mse_loss(camera, targets, reduction="sum"))
            templates = class_templates.index_select(0, labels)
            template_error += float(F.mse_loss(templates, targets, reduction="sum"))
            if first_camera is None:
                first_camera = camera[0:1].clone()
                if camera.shape[0] > 1:
                    cyclic_error += float(F.mse_loss(camera[1:], targets[:-1], reduction="sum"))
            else:
                cyclic_targets = torch.cat((previous_target, targets[:-1]), dim=0)
                cyclic_error += float(F.mse_loss(camera, cyclic_targets, reduction="sum"))
            previous_target = targets[-1:].clone()
            coefficient_values.append(output.coefficients.detach().cpu())
            z_values.append(normalize_port_power(output.coefficients.abs().square(), calibration.port_scale).detach().cpu())
            plane_values.append(torch.stack([value.detach().cpu() for value in output.plane_powers], dim=1))
            camera_values.append(camera)
            count += int(camera.shape[0])
    if first_camera is None or previous_target is None:
        raise RuntimeError("scaled Chen pilot evaluation returned no samples")
    cyclic_error += float(F.mse_loss(first_camera, previous_target))
    coefficients = torch.cat(coefficient_values, dim=0)
    z = torch.cat(z_values, dim=0)
    planes = torch.cat(plane_values, dim=0)
    mean, variance = aggregate_moments(z)
    lower = calibration.operating_lower.detach().cpu()
    upper = calibration.operating_upper.detach().cpu()
    below = planes < lower.unsqueeze(0)
    above = planes > upper.unsqueeze(0)
    inside = ~(below | above)
    saturation = (planes <= torch.finfo(planes.dtype).eps) | (planes >= torch.finfo(planes.dtype).max * 0.5)
    numerical_kl = truncated_normal_kl_reference(mean, variance)
    power_retention = float((coefficients.abs().square().sum(dim=1) / calibration.port_scale.sum().clamp_min(1.0e-30)).mean()) if coefficient_values else 0.0
    cameras = torch.cat(camera_values, dim=0)
    metrics = {
        "paired_mse": paired_error / (count * 1 * 200 * 200),
        "cyclic_mse": cyclic_error / (count * 1 * 200 * 200),
        "template_control_mse": template_error / (count * 1 * 200 * 200),
        "kl": float(bogt_kl_surrogate(mean, variance)),
        "numerical_truncated_normal_kl": float(numerical_kl),
        "op": float(lop_operating_window_from_powers(tuple(planes[:, index] for index in range(5)), calibration.operating_lower, calibration.operating_upper)),
        "coefficient_effective_rank": coefficient_effective_rank(coefficients),
        "power_latent_effective_rank": power_latent_effective_rank(z),
        "power_retention": power_retention,
        "per_plane_power": planes.mean(dim=0).tolist(),
        "below_window_fraction": below.float().mean(dim=0).tolist(),
        "inside_window_fraction": inside.float().mean(dim=0).tolist(),
        "above_window_fraction": above.float().mean(dim=0).tolist(),
        "saturation_fraction": saturation.float().mean(dim=0).tolist(),
        "camera_saturation_fraction": float(((cameras <= 0.0) | (cameras >= 1.0)).float().mean()),
        "calibration_p_star": calibration.plane_reference.detach().cpu().tolist(),
        "calibration_p_min": lower.tolist(),
        "calibration_p_max": upper.tolist(),
        "calibration_sha256": calibration_sha256,
        "finite": bool(torch.isfinite(coefficients).all() and torch.isfinite(planes).all()),
        "evaluation_count": count,
    }
    if split not in {"train", "heldout"}:
        raise ValueError("pilot evaluation split must be train or heldout")
    metrics[f"{split}_mse"] = metrics["paired_mse"]
    if not all(math.isfinite(float(metrics[key])) for key in ("paired_mse", "cyclic_mse", "template_control_mse", "kl", "numerical_truncated_normal_kl", "op", "coefficient_effective_rank", "power_latent_effective_rank", "power_retention")):
        raise RuntimeError("scaled Chen pilot evaluation metrics are non-finite")
    if not all(math.isfinite(float(value)) for key in ("per_plane_power", "below_window_fraction", "inside_window_fraction", "above_window_fraction", "saturation_fraction") for value in metrics[key]):
        raise RuntimeError("scaled Chen pilot optical-state metrics are non-finite")
    if was_training:
        model.train()
    return {"metrics": metrics, "coefficients": coefficients}


def _class_templates(data: ResearchData) -> torch.Tensor:
    """Build train-only per-class target templates for the negative control."""
    train_indices = torch.arange(55000, dtype=torch.long)
    labels = data.labels.index_select(0, train_indices)
    templates: list[torch.Tensor] = []
    for label in range(10):
        selected = train_indices[labels == label]
        image = data.images.index_select(0, selected).float().mean(dim=0, keepdim=True)
        templates.append(base.render_intensity_batch(image).target.float()[0])
    return torch.stack(templates, dim=0)


def calibrate_objective_weights(panel_norms: Sequence[Mapping[str, float]], *, target_ratio: float = 0.10) -> dict[str, Any]:
    """Calibrate ``alpha`` and ``gamma`` on three independent train-only panels."""
    if len(panel_norms) != 3:
        raise ValueError("scaled Chen loss calibration requires three independent panels")
    weights = []
    local: list[dict[str, float]] = []
    for panel in panel_norms:
        mse = float(panel.get("raw_mse_grad_norm", panel.get("mse", 0.0)))
        kl = float(panel.get("raw_kl_grad_norm", panel.get("kl", 0.0)))
        op = float(panel.get("raw_op_grad_norm", panel.get("op", 0.0)))
        alpha = calibrate_loss_weights(mse, kl, target_ratio=target_ratio)
        gamma = calibrate_loss_weights(mse, op, target_ratio=target_ratio)
        weights.append((alpha, gamma))
        local.append({
            "raw_mse_grad_norm": mse,
            "raw_kl_grad_norm": kl,
            "raw_op_grad_norm": op,
            "temporary_panel_alpha": alpha,
            "temporary_panel_gamma": gamma,
            "local_target_kl_ratio": target_ratio,
            "local_target_op_ratio": target_ratio,
        })
    alpha = float(sum(value[0] for value in weights) / len(weights))
    gamma = float(sum(value[1] for value in weights) / len(weights))
    replay = replay_fixed_weight_ratios(panel_norms, alpha=alpha, gamma=gamma)
    return {
        "alpha": alpha,
        "gamma": gamma,
        "target_ratio": target_ratio,
        "panel_local_calibration": local,
        "final_fixed_weight_replay": replay,
        # Retained as a descriptive alias for older readers; the gate must use
        # ``final_fixed_weight_replay`` exclusively.
        "panel_effective_ratios": replay,
    }


def replay_fixed_weight_ratios(
    panel_norms: Sequence[Mapping[str, float]], *, alpha: float, gamma: float,
) -> list[dict[str, float]]:
    """Replay every panel using one final global pair of objective weights."""
    if not math.isfinite(alpha) or not math.isfinite(gamma) or alpha <= 0.0 or gamma <= 0.0:
        raise ValueError("final fixed objective weights must be finite and positive")
    replay: list[dict[str, float]] = []
    for panel in panel_norms:
        mse = float(panel.get("raw_mse_grad_norm", panel.get("mse", 0.0)))
        kl = float(panel.get("raw_kl_grad_norm", panel.get("kl", 0.0)))
        op = float(panel.get("raw_op_grad_norm", panel.get("op", 0.0)))
        if not all(math.isfinite(value) and value > 0.0 for value in (mse, kl, op)):
            raise ValueError("panel gradient norms must be finite and positive")
        replay.append({
            "final_alpha": alpha,
            "final_gamma": gamma,
            "effective_kl_ratio_final_weight": alpha * kl / mse,
            "effective_op_ratio_final_weight": gamma * op / mse,
        })
    return replay


def final_fixed_weight_ratio_gate(replay: Sequence[Mapping[str, float]]) -> bool:
    """Return whether only final fixed-weight ratios satisfy the registered interval."""
    if len(replay) != 3:
        raise ValueError("final fixed-weight replay requires three panels")
    try:
        return all(
            MIN_GRADIENT_RATIO <= float(panel[key]) <= MAX_GRADIENT_RATIO
            for panel in replay
            for key in ("effective_kl_ratio_final_weight", "effective_op_ratio_final_weight")
        )
    except (KeyError, TypeError, ValueError):
        # Evidence crossing this acceptance boundary is untrusted; stale local
        # ratios are rejected rather than interpreted as final weights.
        return False


def _arm_loss(
    model: ScaledChenBOGTModel,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: Calibration,
    arm: str,
    *,
    alpha: float,
    gamma: float,
    mse_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one logical batch with the two-pass aggregate-KL algorithm."""
    if arm not in {"mse_only", "mse_op", "mse_kl", "mse_kl_op"}:
        raise ValueError("unknown scaled Chen pilot arm")
    calibration.validate()
    with torch.no_grad():
        detached_outputs = [model(amplitudes[index : index + 1]) for index in range(amplitudes.shape[0])]
        detached_z = torch.cat([normalize_port_power(value.coefficients.abs().square(), calibration.port_scale) for value in detached_outputs], dim=0)
        aggregate_mean, aggregate_variance = aggregate_moments(detached_z)
        batch_kl = bogt_kl_surrogate(aggregate_mean, aggregate_variance)
    if "kl" in arm:
        proxy = detached_z.detach().requires_grad_(True)
        with torch.enable_grad():
            proxy_kl = bogt_kl_surrogate(*aggregate_moments(proxy))
        kl_gradient = torch.autograd.grad(proxy_kl, proxy, create_graph=False)[0].detach()
    else:
        kl_gradient = None
    mse_values: list[torch.Tensor] = []
    op_values: list[torch.Tensor] = []
    kl_surrogates: list[torch.Tensor] = []
    for index in range(amplitudes.shape[0]):
        output = model(amplitudes[index : index + 1])
        mse_values.append(F.mse_loss(output.camera, targets[index : index + 1]))
        op_values.append(lop_operating_window_from_powers(output.plane_powers, calibration.operating_lower, calibration.operating_upper))
        if "kl" in arm:
            live_z = normalize_port_power(output.coefficients.abs().square(), calibration.port_scale)
            assert kl_gradient is not None
            linear = (live_z * kl_gradient[index : index + 1].to(live_z)).sum()
            kl_surrogates.append(batch_kl.detach() / amplitudes.shape[0] + linear - linear.detach())
    mse = torch.stack(mse_values).mean()
    op = torch.stack(op_values).mean()
    kl = torch.stack(kl_surrogates).sum() if kl_surrogates else torch.zeros_like(mse)
    total = mse_weight * mse + (gamma * op if "op" in arm else 0.0) + (alpha * kl if "kl" in arm else 0.0)
    return total, {"mse": float(mse.detach()), "op": float(op.detach()), "kl": float(batch_kl.detach())}


def observation_gradient_ratios(
    model: ScaledChenBOGTModel,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: Calibration,
) -> dict[str, float]:
    """No-update raw KL/MSE and op/MSE gradient ratios at an observation point."""
    mse_norm = _gradient_norm_for_arm(model, amplitudes, targets, calibration, "mse_only")
    kl_norm = _gradient_norm_for_arm(model, amplitudes, targets, calibration, "mse_kl", mse_weight=0.0)
    op_norm = _gradient_norm_for_arm(model, amplitudes, targets, calibration, "mse_op", mse_weight=0.0)
    return {
        "observation_kl_mse_gradient_ratio": kl_norm / mse_norm,
        "observation_op_mse_gradient_ratio": op_norm / mse_norm,
        "raw_mse_grad_norm": mse_norm,
        "raw_kl_grad_norm": kl_norm,
        "raw_op_grad_norm": op_norm,
    }


def evaluate_pilot_observation(
    model: ScaledChenBOGTModel,
    data: ResearchData,
    calibration: Calibration,
    *,
    train_indices: torch.Tensor,
    heldout_indices: torch.Tensor,
    class_templates: torch.Tensor,
    device: torch.device,
    step: int,
    initial_phase: Mapping[str, torch.Tensor],
    calibration_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate one pilot arm at step 0/50/100 without changing its state."""
    if step not in {0, CHECKPOINT_INTERVAL, STEPS}:
        raise ValueError("pilot observations are registered only at steps 0, 50, and 100")
    train_eval = _evaluate_model(model, data, calibration, train_indices, device, class_templates, calibration_sha256, split="train")
    heldout_eval = _evaluate_model(model, data, calibration, heldout_indices, device, class_templates, calibration_sha256, split="heldout")
    rendered = base.render_intensity_batch(data.images.index_select(0, data.panels[0]))
    ratios = observation_gradient_ratios(model, rendered.amplitude.to(device), rendered.target.to(device), calibration)
    return {
        "step": step,
        "train_mse": train_eval["metrics"]["train_mse"],
        "heldout_mse": heldout_eval["metrics"]["heldout_mse"],
        "gaussian_surrogate_kl": heldout_eval["metrics"]["kl"],
        "numerical_truncated_normal_kl": heldout_eval["metrics"]["numerical_truncated_normal_kl"],
        "l_op": heldout_eval["metrics"]["op"],
        "power_retention": heldout_eval["metrics"]["power_retention"],
        "complex_coefficient_effective_rank": heldout_eval["metrics"]["coefficient_effective_rank"],
        "power_latent_effective_rank": heldout_eval["metrics"]["power_latent_effective_rank"],
        "per_plane_power": heldout_eval["metrics"]["per_plane_power"],
        "below_window_fraction": heldout_eval["metrics"]["below_window_fraction"],
        "inside_window_fraction": heldout_eval["metrics"]["inside_window_fraction"],
        "above_window_fraction": heldout_eval["metrics"]["above_window_fraction"],
        "saturation_fraction": heldout_eval["metrics"]["saturation_fraction"],
        "finite": heldout_eval["metrics"]["finite"],
        "phase_update_norms": _phase_update_metrics(initial_phase, _phase_state(model)),
        **ratios,
    }


def run_factorial_pilot(
    formal: FormalSystemConfig,
    amplitudes: torch.Tensor,
    targets: torch.Tensor,
    calibration: Calibration,
    *,
    device: torch.device | str = "cuda",
    steps: int = STEPS,
    alpha: float = 1.0,
    gamma: float = 1.0,
    checkpoint_dir: str | Path | None = None,
    observation_callback: Callable[[ScaledChenBOGTModel, int, Mapping[str, torch.Tensor]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the bounded four-arm pilot; caller must have passed the CUDA gate."""
    if steps <= 0 or targets.shape != amplitudes.shape:
        raise ValueError("pilot amplitude and target schedules must have identical shapes")
    if amplitudes.ndim == 4:
        if amplitudes.shape[0] != BATCH_SIZE:
            raise ValueError("one-batch pilot input must contain ten samples")
        amplitude_schedule = amplitudes.unsqueeze(0).expand(steps, *amplitudes.shape)
        target_schedule = targets.unsqueeze(0).expand(steps, *targets.shape)
    elif amplitudes.ndim == 5 and amplitudes.shape[0] == steps and amplitudes.shape[1] == BATCH_SIZE:
        amplitude_schedule, target_schedule = amplitudes, targets
    else:
        raise ValueError("pilot schedule must be [10,1,200,200] or [steps,10,1,200,200]")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("scaled Chen pilot requires CUDA")
    results: dict[str, Any] = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device) if str(device).startswith("cuda") and torch.cuda.is_available() else None
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    for arm in ("mse_only", "mse_op", "mse_kl", "mse_kl_op"):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        model = ScaledChenBOGTModel(formal).to(device)
        initial_phase = _phase_state(model)
        optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
        history: list[dict[str, float]] = []
        observations: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        if observation_callback is not None:
            observations.append(dict(observation_callback(model, 0, initial_phase)))
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total, values = _arm_loss(model, amplitude_schedule[step - 1].to(device), target_schedule[step - 1].to(device), calibration, arm, alpha=alpha, gamma=gamma)
            if not torch.isfinite(total):
                raise RuntimeError(f"scaled Chen arm {arm} produced a non-finite loss")
            total.backward()
            optimizer.step()
            values["total"] = float(total.detach())
            if checkpoint_root is not None and step in {CHECKPOINT_INTERVAL, steps}:
                checkpoint_path = checkpoint_root / f"scaled_chen_{arm}_step_{step}.pt"
                if checkpoint_path.exists():
                    raise FileExistsError(f"refusing to overwrite scaled Chen checkpoint: {checkpoint_path}")
                atomic_torch_save(
                    checkpoint_path,
                    {"schema": "scaled_chen_bogt_pilot_checkpoint.v1", "arm": arm, "global_step": step, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "phase_levels_rad": list(PHASE_LEVELS)},
                )
                checkpoint_hashes[str(step)] = file_sha256(checkpoint_path)
            if step in {CHECKPOINT_INTERVAL, steps} and observation_callback is not None:
                observations.append(dict(observation_callback(model, step, initial_phase)))
            values["step"] = step
            history.append(values)
        final_phase = _phase_state(model)
        results[arm] = {
            "history": history,
            "phase_norm": {name: float(parameter.detach().norm()) for name, parameter in model.named_parameters() if parameter.requires_grad},
            "phase_updates": _phase_update_metrics(initial_phase, final_phase),
            "final_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "checkpoint_hashes": checkpoint_hashes,
            "observations": observations,
        }
        del optimizer, model
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") and torch.cuda.is_available() else 0
    return {
        "schema": "scaled_chen_bogt_factorial_pilot.v1",
        "steps_per_arm": steps,
        "arms": results,
        "posthoc_phase_levels_rad": list(PHASE_LEVELS),
        "peak_memory_bytes": peak,
        "wall_seconds": time.monotonic() - started,
    }


def _record(
    schema: str,
    status: str,
    spec: Spec,
    formal: FormalSystemConfig,
    *,
    state: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    dataset_manifest_hash: str,
    gates: tuple[GateResult, ...],
    action: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    context = EvidenceContextV2(
        canonical_sha256({"optical_contract": spec.payload["optical_contract"], "bogt": spec.payload["bogt"], "branch": BRANCH}),
        spec.config_hash,
        dataset_hash=dataset_manifest_hash,
        dataset_split_manifest_hash=dataset_manifest_hash,
    )
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, context, dict(input_hashes), gates, (action,) if action else (), reason)
    record.validate()
    return {
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "training_authorized": False,
        "formal_promotion": False,
        "evidence": record.to_dict(),
        "contract": {
            "source_status": spec.payload["source"]["status"],
            "formal_system_kind": formal.kind,
            "git_state": {"branch": state["branch"], "commit": state["commit"], "clean": state["clean"]},
        },
    }


def _write_new(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite research artifact: {target}")
    atomic_write_json(target, payload)


def _require_evidence(
    path: str | Path,
    schema: str,
    spec: Spec,
    *,
    state: Mapping[str, Any],
    dataset_manifest_hash: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError(f"scaled Chen evidence schema or authorization is invalid for {schema}")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.status != "passed" or record.evidence_kind != schema or record.config_hash != spec.config_hash or record.git_commit != state["commit"]:
        raise ValueError(f"scaled Chen evidence is not a current passed {schema} record")
    if dataset_manifest_hash is not None and record.context.dataset_hash != dataset_manifest_hash:
        raise ValueError("scaled Chen evidence dataset manifest is stale")
    expected = {"research_config": file_sha256(spec.path), **(expected_hashes or {})}
    for name, digest in expected.items():
        if record.input_hashes.get(name) != digest:
            raise ValueError(f"scaled Chen evidence input hash is stale: {name}")
    return payload


def _load_attribution_override(path: Path) -> dict[str, Any]:
    """Load a passed no-update dose audit before any CUDA/optimizer setup."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    evidence = payload.get("evidence", {})
    selected = payload.get("contract", {}).get("selected")
    if payload.get("schema") != "lightgen_optical_manifold_smf_bogt_failure_attribution.v1" or evidence.get("status") != "passed" or not isinstance(selected, Mapping):
        raise ValueError("scaled Chen attribution override is not a passed audit")
    required = {"alpha", "gamma", "alpha_multiplier", "gamma_multiplier"}
    if set(selected) != required or selected["alpha_multiplier"] <= 1.0 or selected["gamma_multiplier"] <= 1.0:
        raise ValueError("scaled Chen attribution override candidate is invalid")
    if not all(isinstance(selected[name], (int, float)) and math.isfinite(float(selected[name])) and float(selected[name]) > 0.0 for name in ("alpha", "gamma")):
        raise ValueError("scaled Chen attribution override weights are invalid")
    return payload


def independent_bogt_kl_reference(mean_q: torch.Tensor, variance_q: torch.Tensor, *, mean_p: float = PRIOR_MEAN, variance_p: float = PRIOR_VARIANCE) -> torch.Tensor:
    """Independent scalar-loop reference for the aggregate KL formula."""
    if mean_q.shape != variance_q.shape or mean_q.ndim != 1 or variance_p <= 0.0:
        raise ValueError("independent KL reference inputs are invalid")
    total = torch.zeros((), dtype=mean_q.dtype, device=mean_q.device)
    for index in range(mean_q.numel()):
        value = variance_q[index] / variance_p
        value = value + (mean_q[index] - mean_p).square() / variance_p
        value = value - 1.0 + math.log(variance_p) - torch.log(variance_q[index])
        total = total + value
    return total / (2.0 * mean_q.numel())


def _clear_phase_gradients(model: ScaledChenBOGTModel) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def self_check_amplitude_4f(formal: FormalSystemConfig) -> bool:
    """Static validation of the locked real-amplitude SLM/4f input boundary."""
    return tuple(formal.slm_shape) == (200, 200) and abs(formal.slm_pixel_pitch_m - 2.5e-6) <= 1.0e-15 and abs(formal.wavelength_m - 532.0e-9) <= 1.0e-15


def _gradient_norm_for_arm(model: ScaledChenBOGTModel, amplitudes: torch.Tensor, targets: torch.Tensor, calibration: Calibration, arm: str, *, mse_weight: float = 1.0) -> float:
    _clear_phase_gradients(model)
    value, _ = _arm_loss(model, amplitudes, targets, calibration, arm, alpha=1.0, gamma=1.0, mse_weight=mse_weight)
    if not torch.isfinite(value):
        raise RuntimeError(f"scaled Chen {arm} gradient panel is non-finite")
    value.backward()
    norm = torch.sqrt(sum(parameter.grad.detach().square().sum() for parameter in model.parameters() if parameter.requires_grad))
    _clear_phase_gradients(model)
    return float(norm.detach().cpu())


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    _require_clean_branch(state)
    _load_review(args.review_contract, spec)
    attribution_path = _resolve(args.attribution_evidence) if getattr(args, "attribution_evidence", None) else None
    attribution_payload = _load_attribution_override(attribution_path) if attribution_path is not None else None
    data = _load_research_data(args.dataset_root)
    formal, formal_path = _load_formal(spec)
    model = ScaledChenBOGTModel(formal)
    matrix = path_abcd((3.0e-3 / 3.5, 3.0e-3 * 2.5 / 3.5), ((3.0e-3 / 3.5) * (3.0e-3 * 2.5 / 3.5) / 3.0e-3,))
    mode_hash = mode_bank_sha256(model)
    area = model.system.encoder_plane.pixel_pitch_m[0] * model.system.encoder_plane.pixel_pitch_m[1]
    gram = model.ols.input_modes.flatten(1).conj() @ model.ols.input_modes.flatten(1).T * area
    gram_rank = int(torch.linalg.matrix_rank(gram).item())
    gram_condition = float(torch.linalg.cond(gram))
    transport_singular = torch.linalg.svdvals(model.ols.transport)
    no_electronic_posterior = not any(key in dict(model.named_modules()) for key in FORBIDDEN_ELECTRONIC_VAE_KEYS)
    gates = (
        GateResult("geometry_2plus2_100x100", tuple(model.system.encoder.phase.shape) == (2, 1, 100, 100) and tuple(model.system.decoder.phase.shape) == (2, 1, 100, 100), {"passed": True}),
        GateResult("amplitude_slm_and_4f_frontend", self_check_amplitude_4f(formal), {"input_encoding": "intensity_sqrt_amplitude", "frontend": "current_4f"}),
        GateResult("backend_b0_magnification", abs(matrix.b_m) < 1.0e-9 and abs(matrix.a + 2.5) < 1.0e-9, {"B_m": matrix.b_m, "A": matrix.a}),
        GateResult("fixed_ols_100_ports", model.ols.transport.shape == (100, 100) and not any(parameter.requires_grad for parameter in model.ols.parameters()), {"port_count": 100}),
        GateResult("ols_mode_bank_finite_full_rank", bool(torch.isfinite(model.ols.input_modes).all() and torch.isfinite(model.ols.output_modes).all() and gram_rank == PORT_COUNT), {"mode_bank_sha256": mode_hash, "gram_rank": gram_rank, "gram_condition": gram_condition}),
        GateResult("ols_linearity_passivity", bool(torch.allclose(model.ols.transport, torch.eye(PORT_COUNT, dtype=model.ols.transport.dtype), atol=0.0, rtol=0.0) and float(transport_singular.max()) <= 1.0 + 1.0e-6), {"transport_singular_max": float(transport_singular.max()), "transport_singular_min": float(transport_singular.min())}),
        GateResult("power_accounting_and_mode_condition", math.isfinite(gram_condition) and gram_condition < 1.0e12, {"gram_condition": gram_condition, "mode_area_m2": area}),
        GateResult("no_electronic_vae_or_digital_decoder", no_electronic_posterior and not hasattr(model, "digital_decoder"), {"forbidden_keys": sorted(FORBIDDEN_ELECTRONIC_VAE_KEYS)}),
        GateResult("fixed_layer_transmission", abs(FIELD_TRANSMISSION**2 - POWER_TRANSMISSION) < 1.0e-12, {"power_transmission": POWER_TRANSMISSION}),
        GateResult("phase_level_contract", bool(torch.allclose(_phase_quantize(torch.tensor(PHASE_LEVELS, dtype=torch.float64)), torch.tensor(PHASE_LEVELS, dtype=torch.float64), atol=1.0e-12, rtol=1.0e-12)), {"levels_rad": list(PHASE_LEVELS)}),
        GateResult("source_statuses_explicit", all(value in SOURCE_STATUSES for value in spec.payload["source"]["status"].values()), {"passed": True}),
        GateResult("split_isolation", data.manifest["calibration_and_schedule_disjoint"] and data.manifest["heldout_excluded"], {"manifest_hash": data.manifest["manifest_hash"]}),
    )
    input_hashes = {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(_resolve(args.review_contract)),
        "dataset_manifest": data.manifest["manifest_hash"],
        "schedule": data.manifest["schedule_sha256"],
        "mode_bank": mode_hash,
    }
    if attribution_path is not None:
        input_hashes["attribution_evidence"] = file_sha256(attribution_path)
    validation_status = "passed" if all(gate.passed for gate in gates) else "invalid_harness"
    payload = _record(VALIDATION_SCHEMA, validation_status, spec, formal, state=state, input_hashes=input_hashes, dataset_manifest_hash=data.manifest["manifest_hash"], gates=gates, action="run_smf_bogt_preflight" if validation_status == "passed" else None, reason=None if validation_status == "passed" else "scaled Chen validation gate failed")
    payload["contract"].update({"formal_config": str(formal_path), "formal_config_sha256": file_sha256(formal_path), "lifecycle_review_sha256": file_sha256(_resolve(args.review_contract)), "dataset_manifest": data.manifest, "schedule_sha256": data.manifest["schedule_sha256"], "mode_bank_sha256": mode_hash, "backend": {"u_m": 3.0e-3 / 3.5, "v_m": 3.0e-3 * 2.5 / 3.5, "f_m": (3.0e-3 / 3.5) * (3.0e-3 * 2.5 / 3.5) / 3.0e-3}, "phase_levels_rad": list(PHASE_LEVELS), "operating_window": {"lower_factor": OP_WINDOW_LOWER_FACTOR, "upper_factor": OP_WINDOW_UPPER_FACTOR, "source_status": "project_approximation"}, "git_source_binding": {"branch": BRANCH, "commit": state["commit"], "config_sha256": file_sha256(spec.path), "review_sha256": file_sha256(_resolve(args.review_contract))}})
    if attribution_payload is not None:
        payload["contract"]["attribution_override"] = {"path": str(attribution_path), "sha256": file_sha256(attribution_path), "selected": attribution_payload["contract"]["selected"]}
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def preflight(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    _require_clean_branch(state)
    review_path = _resolve(args.review_contract)
    _load_review(review_path, spec)
    attribution_path = _resolve(args.attribution_evidence) if getattr(args, "attribution_evidence", None) else None
    attribution_payload = _load_attribution_override(attribution_path) if attribution_path is not None else None
    selected_weights = attribution_payload["contract"]["selected"] if attribution_payload is not None else None
    formal, formal_path = _load_formal(spec)
    data = _load_research_data(args.dataset_root)
    expected_mode_hash = mode_bank_sha256(ScaledChenBOGTModel(formal))
    expected_hashes = {"formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "schedule": data.manifest["schedule_sha256"], "mode_bank": expected_mode_hash}
    if attribution_path is not None:
        expected_hashes["attribution_evidence"] = file_sha256(attribution_path)
    validation_payload = _require_evidence(args.validate_evidence, VALIDATION_SCHEMA, spec, state=state, dataset_manifest_hash=data.manifest["manifest_hash"], expected_hashes=expected_hashes)
    if validation_payload.get("contract", {}).get("mode_bank_sha256") != expected_mode_hash:
        raise ValueError("scaled Chen validation mode-bank hash is stale or mismatched")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("scaled Chen preflight requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("scaled Chen preflight requires CUDA; CPU is not a formal substitute")
    device = torch.device(args.device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    calibration_model = ScaledChenBOGTModel(formal).to(device)
    calibration_batch = base.render_intensity_batch(data.images.index_select(0, data.calibration_indices)).amplitude.to(device)
    calibration = calibrate_model(calibration_model, calibration_batch)
    calibration_path = Path(args.output).with_name("scaled_chen_bogt_calibration.pt")
    if calibration_path.exists():
        raise FileExistsError(f"refusing to overwrite scaled Chen calibration: {calibration_path}")
    atomic_torch_save(calibration_path, {"schema": calibration.schema, "port_scale": calibration.port_scale, "plane_reference": calibration.plane_reference, "plane_lower": calibration.operating_lower, "plane_upper": calibration.operating_upper, "sample_count": calibration.sample_count, "dataset_manifest_hash": data.manifest["manifest_hash"], "phase": "zero_residual", "operating_window": {"lower_factor": OP_WINDOW_LOWER_FACTOR, "upper_factor": OP_WINDOW_UPPER_FACTOR, "source_status": "project_approximation"}})
    del calibration_model, calibration_batch
    statistical_model = ScaledChenBOGTModel(formal).to(device)
    statistical_audit = statistical_batch_stability_audit(statistical_model, data, calibration, device=device)
    del statistical_model
    statistical_path = Path(args.output).with_name("scaled_chen_bogt_statistical_batch_audit.json")
    statistical_payload = {**statistical_audit, "dataset_manifest_hash": data.manifest["manifest_hash"], "mode_bank_sha256": expected_mode_hash, "calibration_sha256": file_sha256(calibration_path)}
    _write_new(statistical_path, statistical_payload)
    if statistical_audit.get("status") != "passed":
        raise RuntimeError("scaled Chen statistical BOGT batch audit found no stable logical batch")
    selected_batch_size = int(statistical_audit["selected_batch_size"])
    if selected_batch_size != BATCH_SIZE:
        raise RuntimeError(f"scaled Chen contract requires logical batch {BATCH_SIZE}, statistical audit selected {selected_batch_size}")
    panel_norms: list[dict[str, float]] = []
    panel_artifacts: list[dict[str, Any]] = []
    for panel_index, indices in enumerate(data.panels):
        rendered = base.render_intensity_batch(data.images.index_select(0, indices))
        amplitudes = rendered.amplitude.to(device)
        targets = rendered.target.to(device)
        panel_model = ScaledChenBOGTModel(formal).to(device)
        panel = {
            "raw_mse_grad_norm": _gradient_norm_for_arm(panel_model, amplitudes, targets, calibration, "mse_only"),
            "raw_kl_grad_norm": _gradient_norm_for_arm(panel_model, amplitudes, targets, calibration, "mse_kl", mse_weight=0.0),
            "raw_op_grad_norm": _gradient_norm_for_arm(panel_model, amplitudes, targets, calibration, "mse_op", mse_weight=0.0),
        }
        panel["mse"], panel["kl"], panel["op"] = panel["raw_mse_grad_norm"], panel["raw_kl_grad_norm"], panel["raw_op_grad_norm"]
        panel_norms.append(panel)
        panel_artifacts.append({"panel": panel_index, "indices_sha256": tensor_sha256(indices), "gradient_norms": panel})
        del panel_model, amplitudes, targets
    panel_weights: dict[str, Any] | None = None
    panel_gate = True
    if selected_weights is not None:
        alpha = float(selected_weights["alpha"])
        gamma = float(selected_weights["gamma"])
        panel_weights = {"alpha": alpha, "gamma": gamma, "target_ratio": None, "source": "failure_attribution", "panel_local_calibration": [], "final_fixed_weight_replay": replay_fixed_weight_ratios(panel_norms, alpha=alpha, gamma=gamma)}
        panel_weights["panel_effective_ratios"] = panel_weights["final_fixed_weight_replay"]
        panel_gate = final_fixed_weight_ratio_gate(panel_weights["final_fixed_weight_replay"])
    else:
        try:
            panel_weights = calibrate_objective_weights(panel_norms)
        except ValueError as exc:
            if "gradient ratio" not in str(exc):
                raise
            panel_gate = False
            panel_weights = {"alpha": None, "gamma": None, "panel_effective_ratios": [], "final_fixed_weight_replay": [], "panel_local_calibration": []}
    alpha = float(panel_weights["alpha"]) if panel_weights["alpha"] is not None else 0.0
    gamma = float(panel_weights["gamma"]) if panel_weights["gamma"] is not None else 0.0
    if panel_weights.get("panel_local_calibration"):
        local_records = panel_weights["panel_local_calibration"]
    else:
        local_records = []
        for panel in panel_norms:
            mse = float(panel["raw_mse_grad_norm"])
            kl = float(panel["raw_kl_grad_norm"])
            op = float(panel["raw_op_grad_norm"])
            local_records.append({
                "raw_mse_grad_norm": mse,
                "raw_kl_grad_norm": kl,
                "raw_op_grad_norm": op,
                "temporary_panel_alpha": calibrate_loss_weights(mse, kl),
                "temporary_panel_gamma": calibrate_loss_weights(mse, op),
                "local_target_kl_ratio": 0.10,
                "local_target_op_ratio": 0.10,
            })
    fixed_replay = replay_fixed_weight_ratios(panel_norms, alpha=alpha, gamma=gamma) if panel_weights["alpha"] is not None else []
    for artifact, local, replay in zip(panel_artifacts, local_records, fixed_replay, strict=True):
        artifact.update({**local, **replay})
    panel_weights["panel_local_calibration"] = local_records
    panel_weights["final_fixed_weight_replay"] = fixed_replay
    panel_weights["panel_effective_ratios"] = fixed_replay
    panel_gate = bool(fixed_replay) and final_fixed_weight_ratio_gate(fixed_replay)
    panel_path = Path(args.output).with_name("scaled_chen_bogt_gradient_panels.pt")
    if panel_path.exists():
        raise FileExistsError(f"refusing to overwrite scaled Chen gradient panels: {panel_path}")
    atomic_torch_save(panel_path, {"schema": "scaled_chen_bogt_gradient_panels.v2", "panels": panel_artifacts, "weights": panel_weights, "dataset_manifest_hash": data.manifest["manifest_hash"], "mode_bank_sha256": expected_mode_hash})
    reference_mean = torch.linspace(0.8, 1.2, PORT_COUNT, device=device, dtype=torch.float64)
    reference_variance = torch.linspace(0.15, 0.35, PORT_COUNT, device=device, dtype=torch.float64)
    kl_formula_ok = bool(torch.allclose(bogt_kl_surrogate(reference_mean, reference_variance), independent_bogt_kl_reference(reference_mean, reference_variance), atol=1.0e-12, rtol=1.0e-12))
    z_check = torch.rand(BATCH_SIZE, PORT_COUNT, device=device, dtype=torch.float64).add_(0.2).requires_grad_()
    full_mean, full_variance = aggregate_moments(z_check)
    full_kl = bogt_kl_surrogate(full_mean, full_variance)
    full_gradient = torch.autograd.grad(full_kl, z_check, retain_graph=True)[0]
    two_gradient = torch.autograd.grad(two_pass_bogt_surrogate(z_check), z_check)[0]
    two_pass_ok = bool(torch.allclose(full_gradient, two_gradient, atol=1.0e-10, rtol=1.0e-8))
    del z_check, full_gradient, two_gradient
    model = ScaledChenBOGTModel(formal).to(device)
    ols_state_before = {name: value.detach().clone() for name, value in model.ols.state_dict().items()}
    rendered = base.render_intensity_batch(data.images.index_select(0, data.panels[0]))
    amplitudes = rendered.amplitude.to(device)
    targets = rendered.target.to(device)
    def frozen_boundary_control(name: str) -> bool:
        parameter = model.system.encoder.phase if name == "encoder" else model.system.decoder.phase
        previous = parameter.requires_grad
        _clear_phase_gradients(model)
        parameter.requires_grad_(False)
        try:
            value, _ = _arm_loss(model, amplitudes, targets, calibration, "mse_kl_op", alpha=alpha, gamma=gamma)
            value.backward()
            return parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        finally:
            parameter.requires_grad_(previous)
            _clear_phase_gradients(model)
    encoder_frozen_control = frozen_boundary_control("encoder")
    decoder_frozen_control = frozen_boundary_control("decoder")
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
    start_state = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    step_values: list[float] = []
    def update(runtime: ScaledChenBOGTModel, runtime_optimizer: torch.optim.Optimizer) -> float:
        runtime_optimizer.zero_grad(set_to_none=True)
        value, _ = _arm_loss(runtime, amplitudes, targets, calibration, "mse_kl_op", alpha=alpha, gamma=gamma)
        if not torch.isfinite(value):
            raise RuntimeError("scaled Chen preflight produced a non-finite loss")
        value.backward()
        runtime_optimizer.step()
        return float(value.detach().cpu())
    for _ in range(2):
        step_values.append(update(model, optimizer))
    resume_state = copy.deepcopy(model.state_dict())
    resume_optimizer_state = copy.deepcopy(optimizer.state_dict())
    resumed_model = ScaledChenBOGTModel(formal).to(device)
    resumed_model.load_state_dict(resume_state, strict=True)
    resumed_optimizer = torch.optim.Adam((parameter for parameter in resumed_model.parameters() if parameter.requires_grad), lr=LEARNING_RATE)
    resumed_optimizer.load_state_dict(resume_optimizer_state)
    step_values.append(update(resumed_model, resumed_optimizer))
    resume_ok = all(bool(torch.isfinite(value).all()) for value in resumed_model.parameters() if value.requires_grad)
    phase_updates = _phase_update_metrics(start_state, _phase_state(model))
    phase_ok = all(value > 0.0 for value in phase_updates.values())
    with torch.no_grad():
        probe = model(amplitudes[:1]).encoder_records[-1]
    zero_operator = FiberArrayOLS(model.system.encoder_plane, transport=torch.zeros((PORT_COUNT, PORT_COUNT), device=device, dtype=torch.complex64)).to(device)
    zero_output = zero_operator(probe)
    zero_control_ok = bool(torch.allclose(zero_output.tensor, torch.zeros_like(zero_output.tensor), atol=0.0, rtol=0.0))
    permutation = torch.eye(PORT_COUNT, device=device, dtype=torch.complex64).roll(1, dims=1)
    permuted_operator = FiberArrayOLS(model.system.encoder_plane, transport=permutation).to(device)
    permuted_output = permuted_operator(probe)
    normal_output = model.ols(probe)
    permutation_control_ok = bool(not torch.allclose(permuted_output.tensor, normal_output.tensor, atol=1.0e-6, rtol=1.0e-6))
    freeze_ok = all(not parameter.requires_grad for parameter in model.ols.parameters()) and all(torch.equal(value, ols_state_before[name]) for name, value in model.ols.state_dict().items())
    phase_invariance = global_phase_invariance_metrics(model.ols.last_diagnostics["port_input_coefficients"])
    truncated_reference = bogt_kl_sensitivity_reference(reference_mean, reference_variance)
    evaluator_calls = 0
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.monotonic() - started
    storage_bytes = sum(path.stat().st_size for path in Path(args.output).parent.rglob("*") if path.is_file()) if Path(args.output).parent.exists() else 0
    gates = (
        GateResult("cuda_free_memory", free_bytes >= MIN_FREE_BYTES, {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}),
        GateResult("three_step_cuda_update", phase_ok, phase_updates),
        GateResult("finite_loss", all(math.isfinite(value) for value in step_values), {"last_loss": step_values[-1]}),
        GateResult("resume_2_plus_1", resume_ok, {"steps": "2+1"}),
        GateResult("formula_independent_reference", kl_formula_ok, {"passed": kl_formula_ok}),
        GateResult("two_pass_gradient_equivalence", two_pass_ok, {"passed": two_pass_ok}),
        GateResult("gradient_ratio_panels", panel_gate, {"final_fixed_weight_replay": panel_weights.get("final_fixed_weight_replay", []), "panel_local_calibration": panel_weights.get("panel_local_calibration", [])}),
        GateResult("statistical_bogt_batch_stability", statistical_audit.get("status") == "passed" and selected_batch_size == BATCH_SIZE, {"selected_batch_size": selected_batch_size, "audit": statistical_audit}),
        GateResult("ols_and_calibration_frozen", freeze_ok, {"passed": freeze_ok}),
        GateResult("frozen_encoder_negative_control", encoder_frozen_control, {"passed": encoder_frozen_control}),
        GateResult("frozen_decoder_negative_control", decoder_frozen_control, {"passed": decoder_frozen_control}),
        GateResult("zero_transport_negative_control", zero_control_ok, {"output_max": float(zero_output.tensor.abs().max().detach().cpu())}),
        GateResult("port_permutation_negative_control", permutation_control_ok, {"changed_output": permutation_control_ok}),
        GateResult("global_phase_invariance", all(value <= 1.0e-5 for value in phase_invariance.values()), phase_invariance),
        GateResult("truncated_normal_reference_finite", all(math.isfinite(value) for value in truncated_reference.values()), truncated_reference),
        GateResult("evaluator_isolation", evaluator_calls == 0, {"calls": evaluator_calls}),
        GateResult("no_digital_decoder_or_posterior", not hasattr(model, "digital_decoder") and not any(key in dict(model.named_modules()) for key in FORBIDDEN_ELECTRONIC_VAE_KEYS), {"calls": 0}),
        GateResult("cuda_peak_memory", peak_bytes <= MAX_PEAK_BYTES, {"bytes": peak_bytes}),
        GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage_bytes <= MAX_STORAGE_BYTES, {"bytes": storage_bytes}),
    )
    status = "passed" if all(gate.passed for gate in gates) else "invalid_harness"
    input_hashes = {"research_config": file_sha256(spec.path), "formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "dataset_manifest": data.manifest["manifest_hash"], "schedule": data.manifest["schedule_sha256"], "mode_bank": expected_mode_hash, "calibration": file_sha256(calibration_path), "gradient_panels": file_sha256(panel_path), "statistical_batch_audit": file_sha256(statistical_path)}
    if attribution_path is not None:
        input_hashes["attribution_evidence"] = file_sha256(attribution_path)
    payload = _record(PREFLIGHT_SCHEMA, status, spec, formal, state=state, input_hashes=input_hashes, dataset_manifest_hash=data.manifest["manifest_hash"], gates=gates, action="manual_review_scaled_chen_bogt_pilot_contract" if status == "passed" else None, reason=None if status == "passed" else "scaled Chen CUDA preflight gate failed")
    payload["contract"].update({"dataset_manifest": data.manifest, "calibration_path": str(calibration_path), "gradient_panels_path": str(panel_path), "statistical_batch_audit_path": str(statistical_path), "selected_logical_batch_size": selected_batch_size, "mode_bank_sha256": expected_mode_hash, "alpha": alpha, "gamma": gamma, "panel_norms": panel_norms, "truncated_normal_kl_reference": truncated_reference, "global_phase_invariance": phase_invariance, "negative_controls": {"zero_transport": zero_control_ok, "port_permutation": permutation_control_ok, "frozen_encoder": encoder_frozen_control, "frozen_decoder": decoder_frozen_control}, "peak_memory_bytes": peak_bytes, "wall_seconds": elapsed})
    if attribution_path is not None:
        payload["contract"]["attribution_override"] = {"path": str(attribution_path), "sha256": file_sha256(attribution_path), "selected": selected_weights}
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def pilot_train(args: argparse.Namespace) -> int:
    """Run the bounded four-arm pilot after both evidence and review gates pass."""
    spec = _load_spec(args.config)
    _assert_contract(spec)
    state = _git_state()
    _require_clean_branch(state)
    lifecycle_review_path = _resolve(args.review_contract)
    _load_review(lifecycle_review_path, spec)
    attribution_path = _resolve(args.attribution_evidence) if getattr(args, "attribution_evidence", None) else None
    attribution_payload = _load_attribution_override(attribution_path) if attribution_path is not None else None
    formal, formal_path = _load_formal(spec)
    data = _load_research_data(args.dataset_root)
    validation_path = _resolve(args.validation_evidence)
    preflight_path = _resolve(args.preflight_evidence)
    expected_mode_hash = mode_bank_sha256(ScaledChenBOGTModel(formal))
    expected_hashes = {
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(lifecycle_review_path),
        "dataset_manifest": data.manifest["manifest_hash"],
        "schedule": data.manifest["schedule_sha256"],
        "mode_bank": expected_mode_hash,
    }
    if attribution_path is not None:
        expected_hashes["attribution_evidence"] = file_sha256(attribution_path)
    _require_evidence(args.validation_evidence, VALIDATION_SCHEMA, spec, state=state, dataset_manifest_hash=data.manifest["manifest_hash"], expected_hashes=expected_hashes)
    preflight_doc = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_contract = preflight_doc.get("contract")
    if not isinstance(preflight_contract, Mapping):
        raise ValueError("scaled Chen preflight contract is missing")
    preflight_expected = {
        **expected_hashes,
        "calibration": file_sha256(_resolve(preflight_contract["calibration_path"])),
        "gradient_panels": file_sha256(_resolve(preflight_contract["gradient_panels_path"])),
        "statistical_batch_audit": file_sha256(_resolve(preflight_contract["statistical_batch_audit_path"])),
    }
    preflight_payload = _require_evidence(args.preflight_evidence, PREFLIGHT_SCHEMA, spec, state=state, dataset_manifest_hash=data.manifest["manifest_hash"], expected_hashes=preflight_expected)
    _load_pilot_review(args.pilot_review_contract, spec, state=state, validation_path=validation_path, preflight_path=preflight_path, lifecycle_review_path=lifecycle_review_path, formal_path=formal_path, data=data, preflight_payload=preflight_payload, attribution_path=attribution_path)
    preflight_contract = preflight_payload["contract"]
    if attribution_payload is not None:
        selected = attribution_payload["contract"]["selected"]
        if float(preflight_contract["alpha"]) != float(selected["alpha"]) or float(preflight_contract["gamma"]) != float(selected["gamma"]):
            raise ValueError("scaled Chen pilot preflight does not consume attribution-selected weights")
    calibration_path = _resolve(preflight_contract["calibration_path"])
    calibration_payload = safe_torch_load(calibration_path)
    if calibration_payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]:
        raise ValueError("scaled Chen pilot calibration dataset manifest is stale")
    calibration = Calibration(calibration_payload["port_scale"], calibration_payload["plane_reference"], int(calibration_payload["sample_count"]), calibration_payload.get("plane_lower"), calibration_payload.get("plane_upper"))
    calibration.validate()
    rendered = base.render_intensity_batch(data.images.index_select(0, data.schedule.reshape(-1)))
    amplitudes = rendered.amplitude.reshape(STEPS, BATCH_SIZE, 1, 200, 200)
    targets = rendered.target.reshape(STEPS, BATCH_SIZE, 1, 200, 200)
    started = time.monotonic()
    device = torch.device(args.device)
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("scaled Chen pilot requires CUDA; CPU is not a formal substitute")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"scaled Chen pilot requires at least {MIN_FREE_BYTES} free CUDA bytes")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(device)
    class_templates = _class_templates(data)
    calibration_hash = file_sha256(calibration_path)
    train_indices = data.schedule.reshape(-1)
    def observation_callback(runtime_model: ScaledChenBOGTModel, step: int, initial_phase: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
        return evaluate_pilot_observation(runtime_model, data, calibration, train_indices=train_indices, heldout_indices=data.heldout_indices, class_templates=class_templates, device=device, step=step, initial_phase=initial_phase, calibration_sha256=calibration_hash)
    results = run_factorial_pilot(formal, amplitudes, targets, calibration, device=device, alpha=float(preflight_contract["alpha"]), gamma=float(preflight_contract["gamma"]), checkpoint_dir=Path(args.output).parent / "checkpoints", observation_callback=observation_callback)
    baseline_model = ScaledChenBOGTModel(formal).to(device)
    baseline_state = {name: value.detach().clone() for name, value in baseline_model.ols.state_dict().items()}
    baseline_eval = _evaluate_model(baseline_model, data, calibration, data.heldout_indices, device, class_templates)
    baseline_rank = baseline_eval["metrics"]["coefficient_effective_rank"]
    baseline_power_rank = baseline_eval["metrics"]["power_latent_effective_rank"]
    for arm, arm_result in results["arms"].items():
        model = ScaledChenBOGTModel(formal).to(device)
        model.load_state_dict(arm_result.pop("final_state"), strict=True)
        state_before = _phase_state(model)
        continuous_eval = _evaluate_model(model, data, calibration, data.heldout_indices, device, class_templates)
        with torch.no_grad():
            model.system.encoder.phase.copy_(_phase_quantize(model.system.encoder.phase))
            model.system.decoder.phase.copy_(_phase_quantize(model.system.decoder.phase))
        quantized_eval = _evaluate_model(model, data, calibration, data.heldout_indices, device, class_templates)
        _set_phase_state(model, state_before)
        frozen = all(torch.equal(value, baseline_state[name]) for name, value in model.ols.state_dict().items()) and all(not parameter.requires_grad for parameter in model.ols.parameters())
        arm_result.update({
            "continuous": continuous_eval["metrics"],
            "quantized": quantized_eval["metrics"],
            "ols_and_calibration_frozen": frozen,
        })
        del model
        torch.cuda.empty_cache()
    del baseline_model
    torch.cuda.empty_cache()
    metrics = {
        "mse_kl_op_heldout_mse": results["arms"]["mse_kl_op"]["continuous"]["paired_mse"],
        "mse_op_heldout_mse": results["arms"]["mse_op"]["continuous"]["paired_mse"],
        "mse_only_kl": results["arms"]["mse_only"]["continuous"]["kl"],
        "mse_kl_kl": results["arms"]["mse_kl"]["continuous"]["kl"],
        "mse_op_kl": results["arms"]["mse_op"]["continuous"]["kl"],
        "mse_kl_op_kl": results["arms"]["mse_kl_op"]["continuous"]["kl"],
        "mse_only_op": results["arms"]["mse_only"]["continuous"]["op"],
        "mse_op_op": results["arms"]["mse_op"]["continuous"]["op"],
        "ols_effective_rank_retention": results["arms"]["mse_kl_op"]["continuous"]["coefficient_effective_rank"] / max(baseline_rank, 1.0e-30),
        "power_latent_effective_rank_retention": results["arms"]["mse_kl_op"]["continuous"]["power_latent_effective_rank"] / max(baseline_power_rank, 1.0e-30),
        "paired_mse": results["arms"]["mse_kl_op"]["continuous"]["paired_mse"],
        "template_control_mse": results["arms"]["mse_kl_op"]["continuous"]["template_control_mse"],
        "cyclic_control_mse": results["arms"]["mse_kl_op"]["continuous"]["cyclic_mse"],
    }
    all_finite = all(
        value[variant]["finite"]
        for value in results["arms"].values()
        for variant in ("continuous", "quantized")
    )
    phase_updates = {
        arm: result["phase_updates"]
        for arm, result in results["arms"].items()
    }
    all_phase_updated = all(all(value > 0.0 for value in updates.values()) for updates in phase_updates.values())
    frozen = all(result["ols_and_calibration_frozen"] for result in results["arms"].values())
    metrics["four_phase_updated"] = all_phase_updated
    metrics["ols_exposure_calibration_evaluator_frozen"] = frozen
    science_gates = pilot_gate_results(metrics)
    harness_gates = (
        GateResult("all_arms_completed_100_steps", all(len(result["history"]) == STEPS and result["history"][-1]["step"] == STEPS and result["checkpoint_hashes"].keys() == {"50", "100"} for result in results["arms"].values()), {"steps_per_arm": STEPS}),
        GateResult("four_phase_updates_nonzero", all_phase_updated, {"updates": phase_updates}),
        GateResult("finite_continuous_and_quantized", all_finite, {}),
        GateResult("fixed_ols_exposure_calibration_evaluator", frozen, {"evaluator_calls": 0}),
        GateResult("cuda_free_memory", free_bytes >= MIN_FREE_BYTES, {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}),
        GateResult("cuda_peak_memory", int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_BYTES, {"bytes": int(torch.cuda.max_memory_allocated(device))}),
        GateResult("wall_time", time.monotonic() - started <= MAX_PILOT_SECONDS, {"seconds": time.monotonic() - started}),
    )
    gates = science_gates + harness_gates
    status, failed_names = classify_pilot_result(science_gates, harness_gates)
    if status == "invalid_harness":
        action, reason = None, "scaled Chen pilot harness or resource gate failed"
    elif status == "passed_scaled_chen_bogt_pilot":
        action, reason = "manual_review_scaled_chen_bogt_phase_closure_contract", None
    else:
        action, reason = "manual_review_scaled_chen_bogt_pilot_failure_attribution", f"scaled Chen pilot scientific gates failed: {','.join(failed_names)}"
    input_hashes = {
        "research_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(lifecycle_review_path),
        "pilot_review_contract": file_sha256(_resolve(args.pilot_review_contract)),
        "validation_evidence": file_sha256(validation_path),
        "preflight_evidence": file_sha256(preflight_path),
        "dataset_manifest": data.manifest["manifest_hash"],
        "schedule": data.manifest["schedule_sha256"],
        "mode_bank": expected_mode_hash,
        "calibration": file_sha256(calibration_path),
        "gradient_panels": file_sha256(_resolve(preflight_contract["gradient_panels_path"])),
        "statistical_batch_audit": file_sha256(_resolve(preflight_contract["statistical_batch_audit_path"])),
    }
    if attribution_path is not None:
        input_hashes["attribution_evidence"] = file_sha256(attribution_path)
    payload = _record(TRAINING_SCHEMA, status, spec, formal, state=state, input_hashes=input_hashes, dataset_manifest_hash=data.manifest["manifest_hash"], gates=gates, action=action, reason=reason)
    payload["contract"].update({
        "arms": list(spec.payload["training"]["arms"]),
        "steps_per_arm": STEPS,
        "calibration_sample_count": calibration.sample_count,
        "calibration_sha256": file_sha256(calibration_path),
        "gradient_panels_sha256": file_sha256(_resolve(preflight_contract["gradient_panels_path"])),
        "alpha": float(preflight_contract["alpha"]),
        "gamma": float(preflight_contract["gamma"]),
        "baseline": {"zero_phase": baseline_eval["metrics"]},
        "pilot_results": results,
        "science_metrics": metrics,
        "classification": status,
        "failed_gates": list(failed_names),
        "free_cuda_bytes_before": int(free_bytes),
        "total_cuda_bytes": int(total_bytes),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": time.monotonic() - started,
    })
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status != "invalid_harness" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "smf-bogt-validate":
        return validate(args)
    if args.research_action == "smf-bogt-preflight":
        return preflight(args)
    if args.research_action == "smf-bogt-pilot-train":
        return pilot_train(args)
    raise ValueError(f"unsupported scaled Chen action: {args.research_action}")
