"""Zero-column target, bias, and spatial semantics for the frozen RMAL probe.

This is a diagnostic-only, target-rendering consumer of the immutable RMAL
detector readout probe.  It deliberately does not load the optical model,
replay CMOS, regenerate detector features, fit a readout, or create an
optimizer.  The scientific unit is the exact zero dynamic-weight output
column, with ``dynamic_mass_j = sum_i abs(W_ij)`` evaluated on the frozen
float32 probe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml

from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_write_new_json, canonical_sha256, file_sha256, tensor_sha256
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_detector_readout_structure_attribution as rwsa


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit.v1"
COMPACT_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_compact_report.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S5-ZCSA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
BASELINE_HEAD = "0dead1b73655b3556450cedbb5d0a2c579385209"

PRIMARY_CHECKPOINT = "scaled_chen_step_5000"
TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
PROBE_INPUT_DIM = 257
EXPECTED_ZERO_W_COUNT = 6498
EXPECTED_ACTIVE_COUNT = 33502
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
TARGET_SHAPE = (200, 200)
TARGET_MICROBATCH_SIZE = 32
PIXEL_PITCH_M = 2.5e-6
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

CLASSIFICATION = "zero_column_semantics_attribution_completed"
NEXT_ACTION_RUN = "run_scaled_chen_detector_readout_zero_column_semantics_audit"
NEXT_ACTION_RESULT_REVIEW = "manual_review_ZCSA_result"

# These are intentionally copied as immutable bindings.  The ZCSA consumer
# does not resolve a latest artifact and does not alter the prior RWSA result.
RMAL_CONSUMER = dict(rwsa.RMAL_CONSUMER)
RWSA_IDENTITY = {
    "config": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_v1.yaml",
    "config_sha256": "076e447cd2a574cdaf6196bfecfa7071d3fd66957501d61e06bdb0847998171a",
    "review_contract": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_review_v1.json",
    "review_sha256": "b616d431e3727b59c6dbaa146be84513f04c1fd89e1457b523f91f5ddbbb77ae",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_v1_attempt_02/validation.json",
    "validation_sha256": "498903c577b6997a3f57fabe21c27217fcc29abc4eff6e1de029f9b9ba692763",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_detector_readout_structure_attribution_v1_attempt_02/audit.json",
    "audit_sha256": "4d0154eb002cb812056eed40fce11d7f8e2e9a2f6cfc998d4393b1a14f2d654f",
    "implementation_module_sha256": "f9ef7f32d8d02051fcb11286e02c6e06aaed66a8724a79fec5e63ddacf59844c",
    "implementation_commit": "5b274b926474784a1ad49744e75190afff31747c",
    "result_git_commit": BASELINE_HEAD,
    "validation_status": "passed",
    "audit_status": "invalid_harness",
    "audit_classification": "invalid_harness",
}
REFERENCES = {
    "scaled_chen_checkpoint": {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
        "producer_commit": "019b76ef911e0bd16b8d5d4169807323427684af",
        "global_step": 5000,
        "role": "frozen_scaled_chen_reference",
    }
}
FIT_PANEL = dict(rwsa.FIT_PANEL)
EVALUATION_PANELS = json.loads(json.dumps(rwsa.EVALUATION_PANELS))

TARGET = {
    "kind": "camera_intensity_same_fov",
    "shape": [1, 200, 200],
    "preprocessing": "base.render_intensity_batch",
    "normalization": "MNIST_uint8_div_255_to_float32",
    "interpolation": "bilinear",
    "align_corners": False,
    "clamp": [0.0, 1.0],
    "paired_target": True,
    "exact_constant": "all(target[:,j] == target[0,j])",
    "population_variance": "unbiased=false",
    "sample_order": "registered_panel_order",
}
PROBE_CONTRACT = {
    "weights_shape": [257, 40000],
    "dynamic_weights_shape": [256, 40000],
    "bias_shape": [40000],
    "storage_dtype": "torch.float32",
    "dynamic_mass": "sum_i abs(W_ij)",
    "zero_identity": "dynamic_mass_j == 0 exact",
    "expected_zero_w_count": EXPECTED_ZERO_W_COUNT,
    "expected_dynamic_active_count": EXPECTED_ACTIVE_COUNT,
    "active_minimum_mass_telemetry": True,
    "weight_mutation": False,
}
TAXONOMY = {
    "priority": [
        "dynamic_active",
        "target_zero_static",
        "bias_only_constant",
        "bias_only_variable_target",
        "zero_W_constant_target_bias_mismatch",
    ],
    "definitions": {
        "dynamic_active": "dynamic_mass_j > 0",
        "target_zero_static": "dynamic_mass_j == 0 AND b_j == 0 AND fit target column exactly zero",
        "bias_only_constant": "dynamic_mass_j == 0 AND fit target exact constant AND fit constant != 0",
        "bias_only_variable_target": "dynamic_mass_j == 0 AND fit target not constant",
        "zero_W_constant_target_bias_mismatch": "dynamic_mass_j == 0 AND fit target exact zero constant AND b_j != 0",
    },
    "mutually_exclusive": True,
    "collectively_exhaustive": True,
}
SPATIAL = {
    "mask_shape": [200, 200],
    "reshape": "row-major",
    "pixel_pitch_m": [PIXEL_PITCH_M, PIXEL_PITCH_M],
    "origin": "centered",
    "row_axis": "y",
    "column_axis": "x",
    "edge_definition": "outermost one-pixel image boundary",
    "connectivity": "deterministic 4-connected",
    "morphology": None,
    "neighborhood_tuning": None,
}
MEASUREMENT = {
    "objective": "zero_dynamic_weight_column_target_bias_and_spatial_semantics",
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "fit_count": FIT_COUNT,
    "selection_count": SELECTION_COUNT,
    "confirmation_count": CONFIRMATION_COUNT,
    "candidate_prefix_consumed": False,
    "target_renderer": "base.render_intensity_batch",
    "target_only": True,
    "no_optical_forward": True,
    "no_cmos_replay": True,
    "no_feature_regeneration": True,
    "no_raw_cmos_bank": True,
    "no_large_feature_bank_artifact": True,
    "microbatch_size": TARGET_MICROBATCH_SIZE,
    "descriptive_only": True,
}
GATES = {
    "expected_zero_w_count": EXPECTED_ZERO_W_COUNT,
    "expected_dynamic_active_count": EXPECTED_ACTIVE_COUNT,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "backward_calls": 0,
    "evaluator_calls": 0,
    "checkpoint_written": False,
    "prior_sampled": False,
    "generation_started": False,
    "physical_forward_modified": False,
    "formal_promotion": False,
}
AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_zero_column_semantics_audit",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "descriptive_only": True,
    "prohibited": [
        "optimizer",
        "gradient",
        "backward",
        "probe_refit",
        "probe_regularization",
        "weight_mutation",
        "checkpoint_write",
        "prior_sampling",
        "generation",
        "evaluator",
        "CMOS_replay",
        "pooled_feature_regeneration",
        "large_feature_bank",
        "DOAA_reconstruction_candidate",
        "W_locality",
        "local_global_readout_classification",
        "detector_coordinate_mismatch",
        "optical_bottleneck",
        "training",
        "promotion",
        "physical_forward_modification",
        "new_samples",
    ],
}
TARGET_RENDERER_SOURCE_SHA256 = "b183dd4656cea9a84c91320c4369167a87694889806e9ca6b5fec5a6b10f4c4b"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Panel:
    name: str
    source: str
    indices: torch.Tensor
    images: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True)
class PanelStats:
    name: str
    count: int
    minimum: torch.Tensor
    maximum: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    nonzero_count: torch.Tensor
    positive_fraction: torch.Tensor
    exact_constant: torch.Tensor
    first_value: torch.Tensor
    residual: dict[str, Any]

    @property
    def ever_nonzero(self) -> torch.Tensor:
        return self.nonzero_count > 0

    @property
    def variable(self) -> torch.Tensor:
        return ~self.exact_constant

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "column_count": TOTAL_COLUMNS,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "population_variance_unbiased_false": self.variance.tolist(),
            "exact_nonzero_count": self.nonzero_count.tolist(),
            "exact_positive_fraction": self.positive_fraction.tolist(),
            "exact_constant": self.exact_constant.tolist(),
            "residual_target_minus_bias": self.residual,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "column_count": TOTAL_COLUMNS,
            "columns_ever_nonzero": int(self.ever_nonzero.sum()),
            "columns_always_zero": int((~self.ever_nonzero).sum()),
            "columns_variable": int(self.variable.sum()),
            "columns_exact_constant": int(self.exact_constant.sum()),
            "minimum_range": [float(self.minimum.min()), float(self.minimum.max())],
            "maximum_range": [float(self.maximum.min()), float(self.maximum.max())],
            "mean_range": [float(self.mean.min()), float(self.mean.max())],
            "population_variance_range": [float(self.variance.min()), float(self.variance.max())],
            "exact_nonzero_total": int(self.nonzero_count.sum()),
            "exact_positive_fraction_mean": float(self.positive_fraction.mean()),
            "residual_target_minus_bias": self.residual,
        }


class ZCSAHarnessFailure(RuntimeError):
    """The registered ZCSA measurement boundary failed before interpretation."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"ZCSA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "rmal_consumer",
        "rwsa_identity",
        "references",
        "fit_panel",
        "evaluation_panels",
        "target",
        "probe_contract",
        "taxonomy",
        "spatial",
        "measurement",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("ZCSA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("ZCSA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "references": REFERENCES,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "target": TARGET,
        "probe_contract": PROBE_CONTRACT,
        "taxonomy": TAXONOMY,
        "spatial": SPATIAL,
        "measurement": MEASUREMENT,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"ZCSA {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "rmal_consumer": RMAL_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "primary_checkpoint": {
                "name": PRIMARY_CHECKPOINT,
                "sha256": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            },
            "fit_panel": {
                "source": FIT_PANEL["source"],
                "count": FIT_PANEL["count"],
                "indices_sha256": FIT_PANEL["indices_sha256"],
            },
            "evaluation_panels": {
                "selection": EVALUATION_PANELS["selection"],
                "confirmation": EVALUATION_PANELS["confirmation"],
                "candidate_prefix": EVALUATION_PANELS["candidate_prefix"],
                "no_new_samples": EVALUATION_PANELS["no_new_samples"],
            },
            "target": TARGET,
            "probe_contract": PROBE_CONTRACT,
            "taxonomy": TAXONOMY,
            "spatial": SPATIAL,
            "measurement": MEASUREMENT,
            "no_update": True,
            "descriptive_only": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "descriptive_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    if json.loads(target.read_text(encoding="utf-8")) != _review_payload():
        raise ValueError("ZCSA review contract does not match scope")
    return target


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ZCSAHarnessFailure(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ZCSAHarnessFailure(f"{label} is not a mapping: {path}")
    return payload


def _verify_evidence_payload(payload: Mapping[str, Any], *, expected_status: str, expected_classification: str | None, label: str) -> EvidenceRecordV2:
    try:
        record = EvidenceRecordV2.from_dict(payload["evidence"])
        record.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise ZCSAHarnessFailure(f"{label} evidence record is invalid") from exc
    if payload.get("status") != expected_status or payload.get("classification") != expected_classification or record.status != expected_status:
        raise ZCSAHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_rwsa_identity(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(RWSA_IDENTITY["config"], base_path=spec.path.parent)
    rwsa_spec = rwsa._load_spec(config_path)
    rwsa._assert_contract(rwsa_spec)
    review_path = _resolve(RWSA_IDENTITY["review_contract"], base_path=spec.path.parent)
    rwsa._load_review(review_path, rwsa_spec)
    validation_path = _resolve(RWSA_IDENTITY["validation"])
    validation = _read_json_mapping(validation_path, "RWSA attempt-02 validation")
    validation_record = _verify_evidence_payload(validation, expected_status="passed", expected_classification=None, label="RWSA attempt-02 validation")
    audit_path = _resolve(RWSA_IDENTITY["audit"])
    audit = _read_json_mapping(audit_path, "RWSA attempt-02 audit")
    audit_record = _verify_evidence_payload(audit, expected_status="invalid_harness", expected_classification="invalid_harness", label="RWSA attempt-02 audit")
    implementation_module = _resolve("src/ldm_project/v2/application/research/optical_manifold/optical_manifold_smf_bogt_detector_readout_structure_attribution.py")
    return {
        "config_path": config_path,
        "review_path": review_path,
        "validation_path": validation_path,
        "audit_path": audit_path,
        "validation": validation,
        "audit": audit,
        "validation_record": validation_record,
        "audit_record": audit_record,
        "implementation_module": implementation_module,
    }


def _load_rmal_consumer(spec: Spec, dataset_root: str | Path) -> tuple[Any, dict[str, Any], rwsa.ReadoutProbe]:
    consumer, artifacts, probe = rwsa._load_rmal_consumer(spec, dataset_root, load_checkpoints=False)
    if artifacts["probe_path"].as_posix().replace("/", "\\") != _resolve(RMAL_CONSUMER["probe"]).as_posix().replace("/", "\\"):
        raise ZCSAHarnessFailure("RMAL probe path is not the registered probe")
    return consumer, artifacts, probe


def _probe_identity(probe: rwsa.ReadoutProbe) -> dict[str, Any]:
    full = probe.full_weights
    if tuple(full.shape) != (PROBE_INPUT_DIM, TOTAL_COLUMNS) or full.dtype != torch.float32:
        raise ZCSAHarnessFailure("RMAL probe full weight shape or dtype changed")
    W = full[:-1, :]
    b = full[-1, :]
    if tuple(W.shape) != (FEATURE_DIM, TOTAL_COLUMNS) or tuple(b.shape) != (TOTAL_COLUMNS,):
        raise ZCSAHarnessFailure("RMAL probe W or b shape changed")
    if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(b).all()):
        raise ZCSAHarnessFailure("RMAL probe W or b is non-finite")
    dynamic_mass = W.abs().to(torch.float64).sum(dim=0)
    zero = dynamic_mass.eq(0)
    active = dynamic_mass.gt(0)
    if not bool(torch.isfinite(dynamic_mass).all()):
        raise ZCSAHarnessFailure("RMAL dynamic mass is non-finite")
    if not bool(W[:, zero].eq(0).all()):
        raise ZCSAHarnessFailure("zero-W columns are not exactly zero in every dynamic row")
    if not bool(dynamic_mass[active].gt(0).all()):
        raise ZCSAHarnessFailure("active dynamic columns do not have positive mass")
    zero_count = int(zero.sum())
    active_count = int(active.sum())
    if zero_count != EXPECTED_ZERO_W_COUNT or active_count != EXPECTED_ACTIVE_COUNT or zero_count + active_count != TOTAL_COLUMNS:
        raise ZCSAHarnessFailure("RMAL zero-W or active-column count changed")
    return {
        "full_weights": full,
        "W": W,
        "b": b,
        "dynamic_mass": dynamic_mass,
        "zero_mask": zero,
        "active_mask": active,
        "zero_count": zero_count,
        "active_count": active_count,
        "active_minimum_mass": float(dynamic_mass[active].min()),
        "active_maximum_mass": float(dynamic_mass[active].max()),
        "all_finite": True,
        "zero_dynamic_entries_nonzero_count": int(torch.count_nonzero(W[:, zero])),
        "probe_sha256": probe.artifact_sha256,
    }


def _fit_indices(consumer: Any, dataset_root: str | Path) -> torch.Tensor:
    return rwsa._fit_indices(consumer, dataset_root)


def _panel_identity(panel: Panel) -> dict[str, Any]:
    return {
        "name": panel.name,
        "source": panel.source,
        "count": int(panel.indices.numel()),
        "indices_sha256": tensor_sha256(panel.indices),
        "labels_sha256": tensor_sha256(panel.labels),
        "image_sha256": tensor_sha256(panel.images),
        "image_shape": list(panel.images.shape),
        "image_dtype": str(panel.images.dtype),
        "index_min": int(panel.indices.min()),
        "index_max": int(panel.indices.max()),
    }


def _build_panels(consumer: Any, fit_indices: torch.Tensor) -> dict[str, Panel]:
    data = consumer.bundle.parent.parent.data
    selection = data.selection_indices.detach().cpu().to(torch.long)
    confirmation = data.confirmation_indices.detach().cpu().to(torch.long)
    fit_indices = fit_indices.detach().cpu().to(torch.long)
    expected_confirmation = torch.arange(55000, 60000, dtype=torch.long)
    if fit_indices.numel() != FIT_COUNT or tensor_sha256(fit_indices) != FIT_PANEL["indices_sha256"]:
        raise ZCSAHarnessFailure("fit panel identity changed")
    if confirmation.numel() != CONFIRMATION_COUNT or not torch.equal(confirmation, expected_confirmation):
        raise ZCSAHarnessFailure("confirmation panel identity changed")
    if selection.numel() != SELECTION_COUNT or not bool(((selection >= 5000) & (selection < 10000)).all()):
        raise ZCSAHarnessFailure("selection panel range or count changed")
    if torch.unique(selection).numel() != selection.numel() or torch.unique(fit_indices).numel() != fit_indices.numel():
        raise ZCSAHarnessFailure("registered panel contains duplicate indices")
    selection_labels = data.test_labels.index_select(0, selection)
    if not torch.equal(torch.bincount(selection_labels, minlength=10), torch.full((10,), 100, dtype=torch.long)):
        raise ZCSAHarnessFailure("selection panel class balance changed")
    if bool(torch.isin(fit_indices, confirmation).any()) or bool(torch.isin(selection, confirmation).any()):
        raise ZCSAHarnessFailure("registered panels overlap")
    panels = {
        "fit": Panel("fit", FIT_PANEL["source"], fit_indices, data.images.index_select(0, fit_indices), data.labels.index_select(0, fit_indices)),
        "selection": Panel("selection", EVALUATION_PANELS["selection"]["source"], selection, data.test_images.index_select(0, selection), selection_labels),
        "confirmation": Panel("confirmation", EVALUATION_PANELS["confirmation"]["source"], confirmation, data.images.index_select(0, confirmation), data.labels.index_select(0, confirmation)),
    }
    for panel in panels.values():
        if panel.images.ndim != 4 or tuple(panel.images.shape[1:]) != (1, 28, 28) or panel.images.dtype != torch.float32:
            raise ZCSAHarnessFailure(f"{panel.name} image tensor contract changed")
        if not bool(torch.isfinite(panel.images).all()) or bool(panel.images.min() < 0) or bool(panel.images.max() > 1):
            raise ZCSAHarnessFailure(f"{panel.name} image normalization is invalid")
    return panels


def _renderer_identity() -> dict[str, Any]:
    fixture = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    fixture[0, 0] = torch.arange(784, dtype=torch.float32).reshape(28, 28).div(783.0)
    fixture[1, 0, ::2, ::2] = 1.0
    expected = F.interpolate(fixture, size=TARGET_SHAPE, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    rendered_one = base.render_intensity_batch(fixture, output_shape=TARGET_SHAPE).target
    rendered_two = base.render_intensity_batch(fixture, output_shape=TARGET_SHAPE).target
    if rendered_one.dtype != torch.float32 or tuple(rendered_one.shape) != (2, 1, 200, 200):
        raise ZCSAHarnessFailure("target renderer shape or dtype changed")
    if not torch.equal(rendered_one, expected) or not torch.equal(rendered_one, rendered_two):
        raise ZCSAHarnessFailure("target renderer is not the exact registered deterministic renderer")
    renderer_path = _resolve("src/ldm_project/v2/application/research/optical_manifold/optical_manifold.py")
    renderer_hash = file_sha256(renderer_path)
    return {
        "source": "base.render_intensity_batch",
        "source_path": str(renderer_path.resolve()),
        "source_sha256": renderer_hash,
        "normalization": "MNIST_uint8_div_255_to_float32",
        "interpolation": "bilinear",
        "output_shape": list(TARGET_SHAPE),
        "align_corners": False,
        "clamp": [0.0, 1.0],
        "fixture_sha256": tensor_sha256(fixture),
        "fixture_rendered_target_sha256": tensor_sha256(rendered_one),
        "exact_repeat": True,
    }


def _merge_vector_accumulator(
    count: int,
    mean: torch.Tensor,
    m2: torch.Tensor,
    batch: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    batch64 = batch.to(torch.float64)
    batch_count = int(batch64.shape[0])
    batch_mean = batch64.mean(dim=0)
    batch_m2 = (batch64 - batch_mean).square().sum(dim=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
    return total, merged_mean, merged_m2


def _merge_scalar_accumulator(
    count: int,
    mean: float,
    m2: float,
    batch: torch.Tensor,
) -> tuple[int, float, float]:
    values = batch.detach().cpu().to(torch.float64).reshape(-1)
    if values.numel() == 0:
        return count, mean, m2
    batch_count = int(values.numel())
    batch_mean = float(values.mean())
    batch_m2 = float((values - batch_mean).square().sum())
    if count == 0:
        return batch_count, batch_mean, batch_m2
    total = count + batch_count
    delta = batch_mean - mean
    return total, mean + delta * (batch_count / total), m2 + batch_m2 + delta * delta * (count * batch_count / total)


def _residual_accumulator_summary(
    count: int,
    mean: float,
    m2: float,
    minimum: float,
    maximum: float,
    abs_sum: float,
    abs_maximum: float,
) -> dict[str, Any]:
    if count == 0:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "population_variance_unbiased_false": None,
            "mse": None,
            "absolute_mean": None,
            "absolute_maximum": None,
        }
    variance = m2 / count
    if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in (mean, variance, minimum, maximum, abs_sum, abs_maximum)) or variance < 0:
        raise ZCSAHarnessFailure("target-minus-bias residual statistics are non-finite")
    return {
        "count": count,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "population_variance_unbiased_false": variance,
        "mse": variance + mean * mean,
        "absolute_mean": abs_sum / count,
        "absolute_maximum": abs_maximum,
    }


def _panel_stats(panel: Panel, zero_mask: torch.Tensor, bias: torch.Tensor) -> PanelStats:
    count = 0
    mean = torch.zeros(TOTAL_COLUMNS, dtype=torch.float64)
    m2 = torch.zeros(TOTAL_COLUMNS, dtype=torch.float64)
    minimum = torch.full((TOTAL_COLUMNS,), float("inf"), dtype=torch.float64)
    maximum = torch.full((TOTAL_COLUMNS,), float("-inf"), dtype=torch.float64)
    nonzero_count = torch.zeros(TOTAL_COLUMNS, dtype=torch.int64)
    positive_count = torch.zeros(TOTAL_COLUMNS, dtype=torch.int64)
    exact_constant = torch.ones(TOTAL_COLUMNS, dtype=torch.bool)
    first_value: torch.Tensor | None = None
    residual_count = 0
    residual_mean = 0.0
    residual_m2 = 0.0
    residual_min = float("inf")
    residual_max = float("-inf")
    residual_abs_sum = 0.0
    residual_abs_max = 0.0
    zero_bias = bias[zero_mask]
    for batch_indices in torch.arange(panel.indices.numel(), dtype=torch.long).split(TARGET_MICROBATCH_SIZE):
        images = panel.images.index_select(0, batch_indices)
        target = base.render_intensity_batch(images, output_shape=TARGET_SHAPE).target
        if target.dtype != torch.float32 or tuple(target.shape) != (batch_indices.numel(), 1, 200, 200):
            raise ZCSAHarnessFailure(f"{panel.name} target renderer output changed")
        if not bool(torch.isfinite(target).all()) or bool(target.min() < 0) or bool(target.max() > 1):
            raise ZCSAHarnessFailure(f"{panel.name} target is non-finite or outside [0,1]")
        flat = target.reshape(batch_indices.numel(), TOTAL_COLUMNS)
        if first_value is None:
            first_value = flat[0].detach().cpu().clone()
        exact_constant &= flat.eq(first_value).all(dim=0).cpu()
        count, mean, m2 = _merge_vector_accumulator(count, mean, m2, flat)
        flat64 = flat.to(torch.float64)
        minimum = torch.minimum(minimum, flat64.min(dim=0).values)
        maximum = torch.maximum(maximum, flat64.max(dim=0).values)
        nonzero_count += flat.ne(0).sum(dim=0).to(torch.int64)
        positive_count += flat.gt(0).sum(dim=0).to(torch.int64)
        if zero_bias.numel():
            residual = flat[:, zero_mask].to(torch.float64) - zero_bias.to(torch.float64)
            residual_count, residual_mean, residual_m2 = _merge_scalar_accumulator(residual_count, residual_mean, residual_m2, residual)
            residual_min = min(residual_min, float(residual.min()))
            residual_max = max(residual_max, float(residual.max()))
            residual_abs_sum += float(residual.abs().sum())
            residual_abs_max = max(residual_abs_max, float(residual.abs().max()))
    if count != panel.indices.numel() or first_value is None:
        raise ZCSAHarnessFailure(f"{panel.name} target statistics did not consume the registered panel")
    variance = m2 / count
    positive_fraction = positive_count.to(torch.float64).div(count)
    if not bool(torch.isfinite(torch.cat((minimum, maximum, mean, variance, positive_fraction))).all()) or bool(variance.lt(0).any()):
        raise ZCSAHarnessFailure(f"{panel.name} target statistics are non-finite")
    residual_summary = _residual_accumulator_summary(
        residual_count,
        residual_mean,
        residual_m2,
        residual_min,
        residual_max,
        residual_abs_sum,
        residual_abs_max,
    )
    return PanelStats(
        panel.name,
        count,
        minimum,
        maximum,
        mean,
        variance,
        nonzero_count,
        positive_fraction,
        exact_constant,
        first_value,
        residual_summary,
    )


def _vector_summary(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "population_variance_unbiased_false": None}
    variance = flat.var(unbiased=False)
    if not bool(torch.isfinite(torch.cat((flat, variance.reshape(1)))).all()) or bool(variance < 0):
        raise ZCSAHarnessFailure("vector summary is non-finite")
    return {
        "count": int(flat.numel()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "population_variance_unbiased_false": float(variance),
    }


def _taxonomy(probe: Mapping[str, Any], fit: PanelStats) -> dict[str, torch.Tensor]:
    zero = probe["zero_mask"]
    active = probe["active_mask"]
    b = probe["b"]
    fit_zero_constant = fit.exact_constant & fit.first_value.eq(0)
    categories = {
        "dynamic_active": active.clone(),
        "target_zero_static": zero & b.eq(0) & fit_zero_constant,
        "bias_only_constant": zero & fit.exact_constant & fit.first_value.ne(0),
        "bias_only_variable_target": zero & ~fit.exact_constant,
        "zero_W_constant_target_bias_mismatch": zero & fit_zero_constant & b.ne(0),
    }
    masks = list(categories.values())
    if any(bool((left & right).any()) for index, left in enumerate(masks) for right in masks[index + 1 :]):
        raise ZCSAHarnessFailure("ZCSA taxonomy categories overlap")
    union = torch.zeros(TOTAL_COLUMNS, dtype=torch.bool)
    for mask in masks:
        union |= mask
    if not bool(union.all()) or int(union.sum()) != TOTAL_COLUMNS:
        raise ZCSAHarnessFailure("ZCSA taxonomy is not collectively exhaustive")
    return categories


def _cross_panel_semantics(categories: Mapping[str, torch.Tensor], panel_stats: Mapping[str, PanelStats], zero_mask: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    fit = panel_stats["fit"]
    selection = panel_stats["selection"]
    confirmation = panel_stats["confirmation"]
    for name in TAXONOMY["priority"]:
        mask = categories[name]
        result[name] = {
            "zero_w_category_count": int(mask.sum()),
            "fit": {
                "count": int(mask.sum()),
                "target_ever_nonzero_count": int((mask & fit.ever_nonzero).sum()),
                "target_variable_count": int((mask & fit.variable).sum()),
            },
            "selection": {
                "target_ever_nonzero_count": int((mask & selection.ever_nonzero).sum()),
                "target_variable_count": int((mask & selection.variable).sum()),
            },
            "confirmation": {
                "target_ever_nonzero_count": int((mask & confirmation.ever_nonzero).sum()),
                "target_variable_count": int((mask & confirmation.variable).sum()),
            },
        }
    fit_always_zero = zero_mask & ~fit.ever_nonzero
    fit_variable = zero_mask & fit.variable
    selection_active = zero_mask & selection.ever_nonzero
    confirmation_active = zero_mask & confirmation.ever_nonzero
    result["zero_w_intersections"] = {
        "zero_w_fit_always_zero": int(fit_always_zero.sum()),
        "zero_w_fit_variable": int(fit_variable.sum()),
        "zero_w_selection_ever_nonzero": int(selection_active.sum()),
        "zero_w_confirmation_ever_nonzero": int(confirmation_active.sum()),
    }
    result["transitions"] = {
        "fit_zero_to_selection_active": int((fit_always_zero & selection.ever_nonzero).sum()),
        "fit_zero_to_confirmation_active": int((fit_always_zero & confirmation.ever_nonzero).sum()),
    }
    return result


def _bias_semantics(probe: Mapping[str, Any], panel_stats: Mapping[str, PanelStats]) -> dict[str, Any]:
    zero = probe["zero_mask"]
    b = probe["b"]
    zero_bias = b[zero]
    fit_mean = panel_stats["fit"].mean[zero]
    result = {
        "prediction": "bias_only yhat_j = b_j; no refit",
        "residual_definition": "target_minus_bias_prediction",
        "zero_w_column_count": int(zero.sum()),
        "b_exact_zero_count": int(zero_bias.eq(0).sum()),
        "b_nonzero_count": int(zero_bias.ne(0).sum()),
        "b_statistics": _vector_summary(zero_bias),
        "b_minus_fit_target_mean": _vector_summary(zero_bias - fit_mean),
        "panel_bias_only": {},
    }
    for name, stats in panel_stats.items():
        result["panel_bias_only"][name] = {
            "mse": stats.residual["mse"],
            "residual": stats.residual,
        }
    return result


def _component_summary(mask: torch.Tensor) -> dict[str, Any]:
    flat = mask.reshape(-1).tolist()
    visited = [False] * len(flat)
    sizes: list[int] = []
    boxes: list[list[int]] = []
    for start, present in enumerate(flat):
        if not present or visited[start]:
            continue
        visited[start] = True
        stack = [start]
        size = 0
        min_row = min_col = TOTAL_COLUMNS
        max_row = max_col = -1
        while stack:
            index = stack.pop()
            row, column = divmod(index, TARGET_SHAPE[1])
            size += 1
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_col = min(min_col, column)
            max_col = max(max_col, column)
            neighbors: list[int] = []
            if row > 0:
                neighbors.append(index - TARGET_SHAPE[1])
            if row + 1 < TARGET_SHAPE[0]:
                neighbors.append(index + TARGET_SHAPE[1])
            if column > 0:
                neighbors.append(index - 1)
            if column + 1 < TARGET_SHAPE[1]:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        sizes.append(size)
        boxes.append([min_row, max_row, min_col, max_col])
    size_tensor = torch.tensor(sizes, dtype=torch.float64)
    largest_index = max(range(len(sizes)), key=sizes.__getitem__) if sizes else None
    return {
        "connectivity": "4-connected",
        "count": len(sizes),
        "largest_size": max(sizes, default=0),
        "size_summary": _vector_summary(size_tensor),
        "largest_bbox_row_min_max_column_min_max": boxes[largest_index] if largest_index is not None else None,
    }


def _spatial_semantics(zero_mask: torch.Tensor, panel_stats: Mapping[str, PanelStats]) -> dict[str, Any]:
    if zero_mask.numel() != TOTAL_COLUMNS:
        raise ZCSAHarnessFailure("zero-W mask size changed")
    mask = zero_mask.reshape(TARGET_SHAPE)
    rows = torch.arange(TARGET_SHAPE[0], dtype=torch.long).reshape(-1, 1).expand(TARGET_SHAPE)
    columns = torch.arange(TARGET_SHAPE[1], dtype=torch.long).reshape(1, -1).expand(TARGET_SHAPE)
    edge = (rows == 0) | (rows == TARGET_SHAPE[0] - 1) | (columns == 0) | (columns == TARGET_SHAPE[1] - 1)
    coordinates = (torch.arange(200, dtype=torch.float64) - (200 - 1) / 2.0) * PIXEL_PITCH_M
    y = coordinates[rows]
    x = coordinates[columns]
    radius = torch.sqrt(x.square() + y.square())
    selected_radius = radius[mask]
    if selected_radius.numel() == 0:
        raise ZCSAHarnessFailure("zero-W spatial mask is empty")
    quantile_tensor = torch.tensor(QUANTILES, dtype=torch.float64)
    overlap: dict[str, Any] = {}
    for name, stats in panel_stats.items():
        support = stats.ever_nonzero.reshape(TARGET_SHAPE)
        overlap_count = int((mask & support).sum())
        overlap[name] = {
            "zero_w_overlap_count": overlap_count,
            "zero_w_overlap_fraction": overlap_count / int(mask.sum()),
            "panel_support_count": int(support.sum()),
            "overlap_fraction_of_panel_support": overlap_count / int(support.sum()) if int(support.sum()) else None,
        }
    return {
        "mask_shape": list(TARGET_SHAPE),
        "reshape": "row-major",
        "pixel_count": int(mask.sum()),
        "bbox": {
            "row_min": int(rows[mask].min()),
            "row_max": int(rows[mask].max()),
            "column_min": int(columns[mask].min()),
            "column_max": int(columns[mask].max()),
        },
        "bbox_physical_m": {
            "y_min": float(y[mask].min()),
            "y_max": float(y[mask].max()),
            "x_min": float(x[mask].min()),
            "x_max": float(x[mask].max()),
        },
        "pixel_pitch_m": [PIXEL_PITCH_M, PIXEL_PITCH_M],
        "origin": "centered",
        "row_axis": "y",
        "column_axis": "x",
        "edge_definition": "outermost one-pixel image boundary",
        "edge_count": int((mask & edge).sum()),
        "interior_count": int((mask & ~edge).sum()),
        "radial_distance_m": _vector_summary(selected_radius),
        "radial_quantiles_m": {str(q): float(value) for q, value in zip(QUANTILES, torch.quantile(selected_radius, quantile_tensor).tolist())},
        "connected_components": _component_summary(mask),
        "support_overlap": overlap,
        "zero_mask_tensor_sha256": tensor_sha256(zero_mask),
    }


def _input_hashes(
    spec: Spec,
    review_path: Path,
    consumer: Any,
    probe: rwsa.ReadoutProbe,
    fit_indices: torch.Tensor,
    panels: Mapping[str, Panel],
) -> dict[str, str]:
    rwsaconfig = _resolve(RWSA_IDENTITY["config"])
    rwsaspec = rwsa._load_spec(rwsaconfig)
    rwsareview = _resolve(RWSA_IDENTITY["review_contract"])
    hashes = rwsa._input_hashes(rwsaspec, rwsareview, consumer, probe, fit_indices)
    hashes.update(
        {
            "research_config": file_sha256(spec.path),
            "review_contract": file_sha256(review_path),
            "implementation_module": file_sha256(Path(__file__).resolve()),
            "target_renderer_module": TARGET_RENDERER_SOURCE_SHA256,
            "rmal_config": file_sha256(consumer.spec.path),
            "rmal_review_contract": file_sha256(consumer.review_path),
            "rmal_validation": file_sha256(consumer.validation_path),
            "rmal_audit": file_sha256(consumer.audit_path),
            "rmal_compact_report": file_sha256(consumer.compact_path),
            "rmal_probe": probe.artifact_sha256,
            "rwsa_config": file_sha256(rwsaconfig),
            "rwsa_review_contract": file_sha256(rwsareview),
            "rwsa_attempt_02_validation": file_sha256(_resolve(RWSA_IDENTITY["validation"])),
            "rwsa_attempt_02_audit": file_sha256(_resolve(RWSA_IDENTITY["audit"])),
            "rwsa_implementation_module": RWSA_IDENTITY["implementation_module_sha256"],
            "target_renderer_contract": canonical_sha256(TARGET),
            "probe_contract": canonical_sha256(PROBE_CONTRACT),
            "taxonomy_contract": canonical_sha256(TAXONOMY),
            "spatial_contract": canonical_sha256(SPATIAL),
            "measurement_contract": canonical_sha256(MEASUREMENT),
            "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
            "rwsa_identity_contract": canonical_sha256(RWSA_IDENTITY),
            "dataset_manifest": consumer.bundle.parent.parent.data.manifest["manifest_hash"],
            "scaled_chen_checkpoint": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            "fit_indices": tensor_sha256(fit_indices),
            "selection_indices": tensor_sha256(panels["selection"].indices),
            "confirmation_indices": tensor_sha256(panels["confirmation"].indices),
        }
    )
    return hashes


def _context(spec: Spec, consumer: Any) -> EvidenceContextV2:
    signature = canonical_sha256({"hypothesis": HYPOTHESIS, "target": TARGET, "probe": PROBE_CONTRACT, "taxonomy": TAXONOMY, "spatial": SPATIAL, "measurement": MEASUREMENT})
    manifest = consumer.bundle.parent.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(signature, spec.config_hash, dataset_hash=manifest, checkpoint_hash=REFERENCES["scaled_chen_checkpoint"]["sha256"], dataset_split_manifest_hash=manifest)


def _record(
    schema: str,
    status: str,
    classification: str | None,
    spec: Spec,
    consumer: Any,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
    gates: Sequence[GateResult],
    actions: Sequence[str],
    contract: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    record = EvidenceRecordV2(2, schema, "diagnostic", status, str(state["commit"]), None, spec.config_hash, _context(spec, consumer), dict(hashes), tuple(gates), tuple(actions), reason)
    record.validate()
    return {
        "hypothesis_id": HYPOTHESIS,
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "generation_started": False,
        "physical_forward_modified": False,
        "optical_replay_executed": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse ZCSA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("ZCSA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"ZCSA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("ZCSA attempt directory is not a fresh validation-only slot")


def _validation_gates(
    spec: Spec,
    consumer: Any,
    rmal_artifacts: Mapping[str, Any],
    probe: rwsa.ReadoutProbe,
    probe_identity: Mapping[str, Any],
    panels: Mapping[str, Panel],
    renderer: Mapping[str, Any],
    rwsa_identity: Mapping[str, Any],
    fit_indices: torch.Tensor,
    output_path: Path,
) -> tuple[GateResult, ...]:
    data = consumer.bundle.parent.parent.data
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("rmal_consumer_identity", file_sha256(rmal_artifacts["probe_path"]) == RMAL_CONSUMER["probe_sha256"], {}),
        GateResult("rwsa_attempt_02_identity", rwsa_identity["audit"].get("status") == "invalid_harness" and rwsa_identity["audit"].get("classification") == "invalid_harness", {}),
        GateResult("checkpoint_identity", file_sha256(checkpoint_path) == REFERENCES["scaled_chen_checkpoint"]["sha256"], {}),
        GateResult("probe_shape_dtype", tuple(probe.full_weights.shape) == (257, 40000) and probe.full_weights.dtype == torch.float32 and tuple(probe_identity["W"].shape) == (256, 40000) and tuple(probe_identity["b"].shape) == (40000,), {}),
        GateResult("probe_finite", bool(probe_identity["all_finite"]), {}),
        GateResult("exact_zero_dynamic_columns", probe_identity["zero_count"] == EXPECTED_ZERO_W_COUNT and probe_identity["zero_dynamic_entries_nonzero_count"] == 0, {"zero_count": probe_identity["zero_count"]}),
        GateResult("active_dynamic_columns", probe_identity["active_count"] == EXPECTED_ACTIVE_COUNT and probe_identity["active_minimum_mass"] > 0, {"active_count": probe_identity["active_count"], "active_minimum_mass": probe_identity["active_minimum_mass"]}),
        GateResult("fit_panel_identity", panels["fit"].indices.numel() == FIT_COUNT and tensor_sha256(fit_indices) == FIT_PANEL["indices_sha256"], {}),
        GateResult("selection_panel_identity", panels["selection"].indices.numel() == SELECTION_COUNT and bool(((panels["selection"].indices >= 5000) & (panels["selection"].indices < 10000)).all()), {}),
        GateResult("confirmation_panel_identity", panels["confirmation"].indices.numel() == CONFIRMATION_COUNT and torch.equal(panels["confirmation"].indices, torch.arange(55000, 60000, dtype=torch.long)), {}),
        GateResult("candidate_prefix_not_consumed", EVALUATION_PANELS["candidate_prefix"]["consumed"] is False, {}),
        GateResult("target_renderer_behavior", renderer["exact_repeat"] and renderer["output_shape"] == list(TARGET_SHAPE) and renderer["interpolation"] == "bilinear" and renderer["align_corners"] is False and renderer["clamp"] == [0.0, 1.0], {}),
        GateResult("target_only_no_optical_forward", MEASUREMENT["target_only"] and MEASUREMENT["no_optical_forward"] and MEASUREMENT["no_cmos_replay"], {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
        GateResult("windows_native_checkout", os.name == "nt" and Path.cwd().drive.upper() == "D:", {}),
        GateResult("disk_free_precondition", disk.free >= MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES}),
        GateResult("output_slot_absent", not output_path.parent.exists(), {"path": str(output_path.parent)}),
        GateResult("dataset_panel_source", data.images.dtype == torch.float32 and data.test_images.dtype == torch.float32, {}),
    )


def _load_validation(path: str | Path, spec: Spec, review_path: Path, consumer: Any, probe: rwsa.ReadoutProbe, fit_indices: torch.Tensor, panels: Mapping[str, Panel], state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, consumer, probe, fit_indices, panels)
    if payload.get("schema") != VALIDATION_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("status") != "passed" or record.status != "passed" or record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.input_hashes != expected_hashes or record.permitted_next_actions != (NEXT_ACTION_RUN,):
        raise ValueError("ZCSA validation evidence is stale or mismatched")


def _artifact_snapshot(spec: Spec, review_path: Path, consumer: Any, rwsa_identity: Mapping[str, Any]) -> dict[str, str]:
    paths = {
        "research_config": spec.path,
        "review_contract": review_path,
        "implementation_module": Path(__file__).resolve(),
        "target_renderer_module": _resolve("src/ldm_project/v2/application/research/optical_manifold/optical_manifold.py"),
        "rmal_config": consumer.spec.path,
        "rmal_review_contract": consumer.review_path,
        "rmal_validation": consumer.validation_path,
        "rmal_audit": consumer.audit_path,
        "rmal_compact_report": consumer.compact_path,
        "rmal_probe": consumer.probe_path,
        "rwsa_config": rwsa_identity["config_path"],
        "rwsa_review_contract": rwsa_identity["review_path"],
        "rwsa_attempt_02_validation": rwsa_identity["validation_path"],
        "rwsa_attempt_02_audit": rwsa_identity["audit_path"],
        "scaled_chen_checkpoint": _resolve(REFERENCES["scaled_chen_checkpoint"]["path"]),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _invalid_harness_payload(output_path: Path, spec: Spec, review_path: Path, consumer: Any, probe: rwsa.ReadoutProbe, fit_indices: torch.Tensor, panels: Mapping[str, Panel], state: Mapping[str, Any], reason: str) -> None:
    gates = (GateResult("zero_column_semantics_harness", False, {"reason": reason}), GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}))
    payload = _record(AUDIT_SCHEMA, "invalid_harness", "invalid_harness", spec, consumer, state, _input_hashes(spec, review_path, consumer, probe, fit_indices, panels), gates, (), {"reason": reason, "target_only": True}, reason)
    _write_new(output_path, payload, artifact_kind="ZCSA invalid-harness audit evidence")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer, rmal_artifacts, probe = _load_rmal_consumer(spec, args.dataset_root)
    rwsa_identity = _verify_rwsa_identity(spec)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    panels = _build_panels(consumer, fit_indices)
    renderer = _renderer_identity()
    probe_id = _probe_identity(probe)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, consumer, rmal_artifacts, probe, probe_id, panels, renderer, rwsa_identity, fit_indices, output_path)
    valid = all(gate.passed for gate in gates)
    contract = {
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "probe_contract": PROBE_CONTRACT,
        "target": TARGET,
        "taxonomy": TAXONOMY,
        "spatial": SPATIAL,
        "measurement": MEASUREMENT,
        "authorization": AUTHORIZATION,
    }
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None if valid else "invalid_harness", spec, consumer, state, _input_hashes(spec, review_path, consumer, probe, fit_indices, panels), gates, (NEXT_ACTION_RUN,) if valid else (), contract, None if valid else "ZCSA static validation failed")
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="ZCSA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer, rmal_artifacts, probe = _load_rmal_consumer(spec, args.dataset_root)
    rwsa_identity = _verify_rwsa_identity(spec)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    panels = _build_panels(consumer, fit_indices)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    _load_validation(validation_path, spec, review_path, consumer, probe, fit_indices, panels, state)
    renderer = _renderer_identity()
    probe_id = _probe_identity(probe)
    before = _artifact_snapshot(spec, review_path, consumer, rwsa_identity)
    panel_stats = {name: _panel_stats(panel, probe_id["zero_mask"], probe_id["b"]) for name, panel in panels.items()}
    categories = _taxonomy(probe_id, panel_stats["fit"])
    cross_panel = _cross_panel_semantics(categories, panel_stats, probe_id["zero_mask"])
    bias = _bias_semantics(probe_id, panel_stats)
    spatial = _spatial_semantics(probe_id["zero_mask"], panel_stats)
    after = _artifact_snapshot(spec, review_path, consumer, rwsa_identity)
    artifacts_unchanged = before == after
    taxonomy_counts = {name: int(mask.sum()) for name, mask in categories.items()}
    taxonomy_complete = sum(taxonomy_counts.values()) == TOTAL_COLUMNS
    if not taxonomy_complete:
        raise ZCSAHarnessFailure("ZCSA taxonomy count is incomplete")
    compact = {
        "schema": COMPACT_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "status": "passed",
        "classification": CLASSIFICATION,
        "next_action": NEXT_ACTION_RESULT_REVIEW,
        "git": {"branch": state["branch"], "commit": state["commit"], "clean": state["clean"]},
        "probe": {
            "artifact_sha256": probe.artifact_sha256,
            "weight_shape": [257, 40000],
            "dynamic_weight_shape": [256, 40000],
            "bias_shape": [40000],
            "dtype": "torch.float32",
            "dynamic_mass_definition": "sum_i abs(W_ij)",
            "zero_identity": "dynamic_mass_j == 0 exact",
            "zero_w_count": probe_id["zero_count"],
            "dynamic_active_count": probe_id["active_count"],
            "active_minimum_dynamic_mass": probe_id["active_minimum_mass"],
            "active_maximum_dynamic_mass": probe_id["active_maximum_mass"],
            "zero_dynamic_entries_nonzero_count": probe_id["zero_dynamic_entries_nonzero_count"],
            "all_finite": probe_id["all_finite"],
            "zero_w_flat_indices": torch.where(probe_id["zero_mask"])[0].tolist(),
        },
        "panels": {name: {**_panel_identity(panel), "target_statistics_summary": panel_stats[name].summary(), "target_statistics": panel_stats[name].to_dict()} for name, panel in panels.items()},
        "taxonomy": {"counts": taxonomy_counts, "total_columns": TOTAL_COLUMNS, "complete": taxonomy_complete, "priority": TAXONOMY["priority"]},
        "cross_panel_semantics": cross_panel,
        "bias_semantics": bias,
        "spatial_semantics": spatial,
        "renderer": renderer,
        "artifact_integrity": {"before": before, "after": after, "unchanged": artifacts_unchanged},
        "optimizer_created": False,
        "backward_calls": 0,
        "evaluator_calls": 0,
        "training_authorized": False,
        "formal_promotion": False,
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
        "optical_replay_executed": False,
        "raw_cmos_bank_saved": False,
        "pooled_feature_bank_saved": False,
    }
    compact_path = output_path.parent / "compact_report.json"
    _write_new(compact_path, compact, artifact_kind="ZCSA compact report")
    compact_hash = file_sha256(compact_path)
    output_bytes = int(sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()))
    gates = (
        GateResult("consumer_and_lineage_identity", True, {"rmal_probe": probe.artifact_sha256, "rwsa_audit_status": "invalid_harness"}),
        GateResult("probe_zero_column_semantics", probe_id["zero_count"] == EXPECTED_ZERO_W_COUNT and probe_id["active_count"] == EXPECTED_ACTIVE_COUNT and probe_id["zero_dynamic_entries_nonzero_count"] == 0, {"zero_w_count": probe_id["zero_count"], "active_count": probe_id["active_count"]}),
        GateResult("target_renderer_behavior", renderer["exact_repeat"] and renderer["output_shape"] == list(TARGET_SHAPE) and renderer["interpolation"] == "bilinear" and renderer["align_corners"] is False and renderer["clamp"] == [0.0, 1.0], {}),
        GateResult("panel_identity", all(_panel_identity(p)["count"] == expected for p, expected in ((panels["fit"], FIT_COUNT), (panels["selection"], SELECTION_COUNT), (panels["confirmation"], CONFIRMATION_COUNT))), {}),
        GateResult("finite_target_measurements", all(bool(torch.isfinite(stats.mean).all() and torch.isfinite(stats.variance).all()) for stats in panel_stats.values()), {}),
        GateResult("taxonomy_complete", taxonomy_complete, {"counts": taxonomy_counts, "total_columns": TOTAL_COLUMNS}),
        GateResult("bias_statistics_finite", all(value["mse"] is not None for value in bias["panel_bias_only"].values()), {}),
        GateResult("spatial_statistics_complete", spatial["pixel_count"] == EXPECTED_ZERO_W_COUNT and spatial["connected_components"]["count"] > 0, {}),
        GateResult("artifacts_unchanged", artifacts_unchanged, {}),
        GateResult("target_only_no_optical_replay", True, {"optical_replay_executed": False, "optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
        GateResult("storage_limit", output_bytes <= MAX_STORAGE_BYTES, {"output_bytes": output_bytes, "maximum": MAX_STORAGE_BYTES}),
    )
    if not all(gate.passed for gate in gates) or not artifacts_unchanged:
        raise ZCSAHarnessFailure("ZCSA result gate failed")
    audit_contract = {
        "probe": compact["probe"],
        "panels": {name: {**_panel_identity(panel), "target_statistics_summary": panel_stats[name].summary()} for name, panel in panels.items()},
        "taxonomy": compact["taxonomy"],
        "cross_panel_semantics": cross_panel,
        "bias_semantics": bias,
        "spatial_semantics": spatial,
        "renderer": renderer,
        "artifact_integrity": compact["artifact_integrity"],
        "compact_report_path": str(compact_path.resolve()),
        "compact_report_sha256": compact_hash,
        "target_only": True,
        "optical_replay_executed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
    }
    payload = _record(AUDIT_SCHEMA, "passed", CLASSIFICATION, spec, consumer, state, _input_hashes(spec, review_path, consumer, probe, fit_indices, panels), gates, (NEXT_ACTION_RESULT_REVIEW,), audit_contract, "ZCSA is descriptive-only; it does not classify locality, bottlenecks, or readout validity")
    _write_new(output_path, payload, artifact_kind="ZCSA audit evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-detector-readout-zero-column-semantics-audit-validate":
        return validate(args)
    if args.research_action == "scaled-chen-detector-readout-zero-column-semantics-audit-run":
        return run(args)
    raise ValueError(f"unsupported ZCSA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "CLASSIFICATION",
    "CONFIG_SCHEMA",
    "EXPECTED_ACTIVE_COUNT",
    "EXPECTED_ZERO_W_COUNT",
    "FIT_PANEL",
    "HYPOTHESIS",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "PROBE_CONTRACT",
    "REFERENCES",
    "RMAL_CONSUMER",
    "REVIEW_SCHEMA",
    "RWSA_IDENTITY",
    "SPATIAL",
    "TARGET",
    "TARGET_RENDERER_SOURCE_SHA256",
    "TAXONOMY",
    "VALIDATION_SCHEMA",
    "ZCSAHarnessFailure",
    "Panel",
    "PanelStats",
    "Spec",
    "_assert_contract",
    "_bias_semantics",
    "_build_panels",
    "_component_summary",
    "_cross_panel_semantics",
    "_load_review",
    "_load_spec",
    "_probe_identity",
    "_renderer_identity",
    "_spatial_semantics",
    "_taxonomy",
    "dispatch",
    "run",
    "validate",
]
