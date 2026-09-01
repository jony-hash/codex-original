"""Descriptive structure and bias attribution for the frozen RMAL detector readout.

This module is a diagnostic-only consumer of the immutable RMAL CMOS probe.  It
does not fit, refit, or insert a readout, and it does not alter the optical
model, checkpoint, evaluator, or formal runtime.  The only model execution is
an exact-panel, CMOS-only replay used to recover the features that the RMAL
probe artifact intentionally did not store.
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

from ...config import FormalSystemConfig
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_horizon as horizon
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution.v1"
COMPACT_SCHEMA = "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_compact_report.v1"
RMAL_PROBE_SCHEMA = "scaled_chen_reconstruction_mapping_alignment_localization_probes.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S5-RWSA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
BASELINE_HEAD = "3b4133ad8c6e00941d64f7de9f0570c33086ada9"

PRIMARY_CHECKPOINT = "scaled_chen_step_5000"
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
FIT_COUNT = 1000
FIT_TRAIN_LIMIT = 55000
MICROBATCH_SIZE = 1
PERMUTATION_NULL_COUNT = 256
PERMUTATION_NULL_SEED = 0
TOP_K = 4
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
PHYSICAL_BASELINE_TOLERANCE = 1.0e-6
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2

CLASSIFICATION = "descriptive_structural_attribution_completed"
NEXT_ACTION_RUN = "run_scaled_chen_detector_readout_structure_attribution"
NEXT_ACTION_RESULT_REVIEW = "manual_review_RWSA_structural_result"

RMAL_CONSUMER = {
    "config": "configs/research/lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_v1.yaml",
    "config_sha256": "c50b5b2d12830c40fbe619f137aca40f36f799f277e637d1ec20ac7e3403859f",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_review_v1.json",
    "review_sha256": "73ba944ac5766c88591e0f5125cc031a1c9c55bd84aa2cbc30ea457bd4fb30e8",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/validation.json",
    "validation_sha256": "affb4961626a2caaf4dc2c8bb04c06ac4f8de51d1d4a1ab418858eaa11e536ec",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/audit.json",
    "audit_sha256": "5ba51e041a266865fc0443b729d31a0c97c35f2b86d6ef45f11dacf6617a15fa",
    "compact_report": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/compact_report.json",
    "compact_report_sha256": "c335f75124666e9423d66b7a4537c74ff642ef2d71044bda86be16f9333a2137",
    "probe": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_mapping_alignment_localization_v1_attempt_01/probes.pt",
    "probe_sha256": "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed",
    "implementation_commit": "5b274b926474784a1ad49744e75190afff31747c",
    "audit_status": "passed",
    "audit_classification": "bounded_recoverability_established_at_detector",
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

FIT_PANEL = dict(rmal.FIT_PANEL)
EVALUATION_PANELS = json.loads(json.dumps(rmal.EVALUATION_PANELS))
TARGET = dict(rmal.TARGET)

POOLING = {
    "input_shape": [200, 200],
    "output_shape": [16, 16],
    "bin_formula": "B_i=[floor(i*200/16),ceil((i+1)*200/16))",
    "source_order": "row-major i_then_j",
    "target_order": "row-major r_then_c",
    "overlap_semantics": "inclusive_membership_in_each_adaptive_pool_bin",
    "overlap_is_contractual": True,
}

COORDINATE_CONTRACT = {
    "detector_shape": [200, 200],
    "target_shape": [200, 200],
    "pixel_pitch_m": [2.5e-6, 2.5e-6],
    "physical_fov_m": [500.0e-6, 500.0e-6],
    "origin": "centered",
    "row_axis": "y",
    "column_axis": "x",
    "overall_orientation": "identity",
    "front_magnification": -0.4,
    "back_magnification": -2.5,
    "net_magnification": 1.0,
    "expected_support": "E(r,c)={(i,j):r in B_i and c in B_j}",
    "expected_distribution": "q(i,j|r,c) proportional to 1/(|B_i||B_j|) over E only",
    "cell_center_definition": "mean of actual covered raw detector pixel-center coordinates",
}

D4_CONTROLS = (
    "identity",
    "x_flip",
    "y_flip",
    "xy_flip",
    "transpose",
    "transpose_x_flip",
    "transpose_y_flip",
    "transpose_xy_flip",
)

PERMUTATION_NULL = {
    "count": PERMUTATION_NULL_COUNT,
    "seed": PERMUTATION_NULL_SEED,
    "rng": "torch.Generator(device='cpu').manual_seed(seed)",
    "operation": "source_feature_rows_only",
    "marginal_preservation": "each_target_column_abs_weight_values_unchanged",
    "statistic": "identity_expected_support_locality_mass",
}

MEASUREMENT = {
    "objective": "descriptive_16x16_pooled_detector_to_200x200_target_structure_and_bias_attribution",
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "selection_count": SELECTION_COUNT,
    "confirmation_count": CONFIRMATION_COUNT,
    "fit_count": FIT_COUNT,
    "candidate_prefix_consumed": False,
    "physical_forward_unchanged": True,
    "cmos_source": "output.camera",
    "no_six_stage_capture": True,
    "no_stage_search": True,
    "no_raw_cmos_bank": True,
    "no_large_feature_bank_artifact": True,
    "microbatch_size": MICROBATCH_SIZE,
    "descriptive_only": True,
}

BIAS_ATTRIBUTION = {
    "full": "zW_plus_b",
    "dynamic": "zW",
    "bias_only": "b_broadcast_over_samples",
    "feature_source": "exact_RMAL_cmos_intensity_pooled_features",
    "postprocessing": "none",
    "refit": False,
    "instance_constant_label": "not_applicable_degenerate_constant_output",
}

GATES = {
    "physical_baseline_tolerance": PHYSICAL_BASELINE_TOLERANCE,
    "permutation_null_count": PERMUTATION_NULL_COUNT,
    "permutation_null_seed": PERMUTATION_NULL_SEED,
    "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
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
    "decision": "approve_no_update_scaled_chen_detector_readout_structure_attribution",
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
        "checkpoint_write",
        "prior_sampling",
        "generation",
        "evaluator",
        "DOAA_reconstruction_candidate",
        "FMF",
        "modal_sweep",
        "OLS_redesign",
        "decoder_redesign",
        "relay_tuning",
        "geometry_sweep",
        "training",
        "promotion",
        "physical_forward_modification",
        "new_samples",
    ],
}


PHYSICAL_ANCHORS = rmfa.PHYSICAL_ANCHORS


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ReadoutProbe:
    full_weights: torch.Tensor
    weights: torch.Tensor
    bias: torch.Tensor
    fit_indices: torch.Tensor
    metadata: Mapping[str, Any]
    artifact_sha256: str


class RWSAHarnessFailure(RuntimeError):
    """A registered RWSA measurement boundary failed before interpretation."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"RWSA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "rmal_consumer",
        "references",
        "fit_panel",
        "evaluation_panels",
        "pooling",
        "coordinate_contract",
        "d4_controls",
        "permutation_null",
        "measurement",
        "bias_attribution",
        "target",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("RWSA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RWSA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "rmal_consumer": RMAL_CONSUMER,
        "references": REFERENCES,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "pooling": POOLING,
        "coordinate_contract": COORDINATE_CONTRACT,
        "d4_controls": list(D4_CONTROLS),
        "permutation_null": PERMUTATION_NULL,
        "measurement": MEASUREMENT,
        "bias_attribution": BIAS_ATTRIBUTION,
        "target": TARGET,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"RWSA {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "rmal_consumer": RMAL_CONSUMER,
            "references": {
                "primary_checkpoint": PRIMARY_CHECKPOINT,
                "checkpoint_sha256": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            },
            "fit_panel": {
                key: FIT_PANEL[key]
                for key in ("source", "count", "per_label", "construction", "indices_sha256")
            },
            "evaluation_panels": {
                "selection": EVALUATION_PANELS["selection"]["source"],
                "confirmation": EVALUATION_PANELS["confirmation"]["source"],
                "candidate_prefix_consumed": EVALUATION_PANELS["candidate_prefix"]["consumed"],
            },
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": list(D4_CONTROLS),
            "permutation_null": PERMUTATION_NULL,
            "measurement": MEASUREMENT,
            "bias_attribution": BIAS_ATTRIBUTION,
            "target": TARGET,
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
        raise ValueError("RWSA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return payload


def _load_rmal_consumer(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> tuple[rmal.RMFAConsumer, dict[str, Any], ReadoutProbe]:
    rmal_config = _resolve(RMAL_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(rmal_config) != RMAL_CONSUMER["config_sha256"]:
        raise ValueError("RMAL consumer config is missing, stale, or tampered")
    rmal_spec = rmal._load_spec(rmal_config)
    rmal._assert_contract(rmal_spec)
    rmal_review = _resolve(RMAL_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(rmal_review) != RMAL_CONSUMER["review_sha256"]:
        raise ValueError("RMAL consumer review is missing, stale, or tampered")
    rmal._load_review(rmal_review, rmal_spec)
    consumer = rmal._load_rmfa_consumer(rmal_spec, dataset_root, load_checkpoints=load_checkpoints)

    validation_path = _resolve(RMAL_CONSUMER["validation"])
    validation = _read_json_hash(validation_path, RMAL_CONSUMER["validation_sha256"], "RMAL validation")
    validation_record = EvidenceRecordV2.from_dict(validation["evidence"])
    validation_record.validate()
    if (
        validation.get("status") != "passed"
        or validation_record.status != "passed"
        or validation_record.git_commit != RMAL_CONSUMER["implementation_commit"]
        or validation_record.config_hash != rmal_spec.config_hash
    ):
        raise ValueError("RMAL validation terminal state is stale")

    audit_path = _resolve(RMAL_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, RMAL_CONSUMER["audit_sha256"], "RMAL audit")
    audit_record = EvidenceRecordV2.from_dict(audit["evidence"])
    audit_record.validate()
    if (
        audit.get("status") != RMAL_CONSUMER["audit_status"]
        or audit.get("classification") != RMAL_CONSUMER["audit_classification"]
        or audit_record.git_commit != RMAL_CONSUMER["implementation_commit"]
    ):
        raise ValueError("RMAL audit terminal state is stale or mismatched")

    compact_path = _resolve(RMAL_CONSUMER["compact_report"])
    compact = _read_json_hash(compact_path, RMAL_CONSUMER["compact_report_sha256"], "RMAL compact report")
    if compact.get("classification") != RMAL_CONSUMER["audit_classification"] or compact.get("generation_readiness") != "not_ready":
        raise ValueError("RMAL compact terminal interpretation changed")

    probe_path = _resolve(RMAL_CONSUMER["probe"])
    probe_payload = safe_torch_load(probe_path, expected_sha256=RMAL_CONSUMER["probe_sha256"], map_location="cpu")
    probe = _load_rmal_probe(probe_payload, rmal._fit_indices(consumer, dataset_root), RMAL_CONSUMER["probe_sha256"])
    return consumer, {"validation": validation, "audit": audit, "compact": compact}, probe


def _load_rmal_probe(payload: Mapping[str, Any], fit_indices: torch.Tensor, artifact_sha256: str = "") -> ReadoutProbe:
    if payload.get("schema") != RMAL_PROBE_SCHEMA or set(payload) != {"schema", "fit_indices", "probes", "fit_count", "mechanism_evidence_only", "physical_forward_modified"}:
        raise ValueError("RMAL probe artifact schema or fields changed")
    stored_indices = payload.get("fit_indices")
    probes = payload.get("probes")
    if not torch.is_tensor(stored_indices) or not isinstance(probes, Mapping) or set(probes) != {"cmos_intensity"}:
        raise ValueError("RMAL probe artifact fields are malformed")
    stored_indices = stored_indices.detach().cpu().to(torch.long)
    if not torch.equal(stored_indices, fit_indices) or int(payload.get("fit_count", -1)) != FIT_COUNT:
        raise ValueError("RMAL probe fit panel is not the registered fit panel")
    item = probes["cmos_intensity"]
    if not isinstance(item, Mapping) or not torch.is_tensor(item.get("weights")) or not isinstance(item.get("metadata"), Mapping):
        raise ValueError("RMAL CMOS probe entry is malformed")
    full = item["weights"].detach().cpu().contiguous()
    metadata = item["metadata"]
    if tuple(full.shape) != (257, 40000) or full.dtype != torch.float32:
        raise ValueError("RMAL CMOS probe tensor shape or dtype changed")
    expected_metadata = {
        "stage": "cmos_intensity",
        "feature_dim": 256,
        "input_dim": 257,
        "output_dim": 40000,
        "fit_panel_hash": tensor_sha256(fit_indices),
        "probe_hash": tensor_sha256(full),
        "frozen": True,
    }
    for name, value in expected_metadata.items():
        if metadata.get(name) != value:
            raise ValueError(f"RMAL probe metadata changed for {name}")
    if not torch.isfinite(full).all():
        raise RWSAHarnessFailure("RMAL probe weights are non-finite")
    return ReadoutProbe(full, full[:-1, :].clone(), full[-1, :].clone(), stored_indices.clone(), dict(metadata), artifact_sha256)


def _fit_indices(consumer: rmal.RMFAConsumer, dataset_root: str | Path) -> torch.Tensor:
    horizon_data = horizon._load_data(dataset_root)
    return rmfa._fit_indices(
        consumer.bundle.parent.parent.data,
        horizon_data,
        rmfa._load_plan_indices(rmfa.PLAN3),
        rmfa._load_plan_indices(rmfa.PLAN4),
    )


def _input_hashes(
    spec: Spec,
    review_path: Path,
    consumer: rmal.RMFAConsumer,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
) -> dict[str, str]:
    hashes = rmal._input_hashes(rmal._load_spec(consumer.spec.path), consumer.review_path, consumer, fit_indices)
    hashes.update(
        {
            "research_config": file_sha256(spec.path),
            "review_contract": file_sha256(review_path),
            "implementation_module": file_sha256(Path(__file__).resolve()),
            "rmal_config": file_sha256(consumer.spec.path),
            "rmal_review_contract": file_sha256(consumer.review_path),
            "rmal_validation": file_sha256(consumer.validation_path),
            "rmal_audit": file_sha256(consumer.audit_path),
            "rmal_compact_report": file_sha256(consumer.compact_path),
            "rmal_probe": probe.artifact_sha256,
            "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
            "pooling_contract": canonical_sha256(POOLING),
            "coordinate_contract": canonical_sha256(COORDINATE_CONTRACT),
            "d4_control_contract": canonical_sha256(D4_CONTROLS),
            "permutation_null_contract": canonical_sha256(PERMUTATION_NULL),
            "bias_attribution_contract": canonical_sha256(BIAS_ATTRIBUTION),
            "target_contract": canonical_sha256(TARGET),
        }
    )
    hashes["fit_indices"] = tensor_sha256(fit_indices)
    hashes["scaled_chen_checkpoint"] = REFERENCES["scaled_chen_checkpoint"]["sha256"]
    return hashes


def _context(spec: Spec, consumer: rmal.RMFAConsumer) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": D4_CONTROLS,
            "permutation_null": PERMUTATION_NULL,
            "bias_attribution": BIAS_ATTRIBUTION,
        }
    )
    manifest = consumer.bundle.parent.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=manifest,
        checkpoint_hash=REFERENCES["scaled_chen_checkpoint"]["sha256"],
        dataset_split_manifest_hash=manifest,
    )


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _state_equal(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> bool:
    return set(before) == set(after) and all(torch.equal(before[name], after[name]) for name in before)


def _pool_bins(input_size: int = 200, output_size: int = 16) -> tuple[tuple[int, int], ...]:
    if input_size <= 0 or output_size <= 0:
        raise ValueError("pool dimensions must be positive")
    return tuple(
        (
            (index * input_size) // output_size,
            ((index + 1) * input_size + output_size - 1) // output_size,
        )
        for index in range(output_size)
    )


def _bin_membership(size: int, bins: Sequence[tuple[int, int]]) -> torch.Tensor:
    coordinates = torch.arange(size, dtype=torch.long)
    return torch.stack([(coordinates >= start) & (coordinates < end) for start, end in bins])


def _transform_coordinates(rows: torch.Tensor, columns: torch.Tensor, control: str, size: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    if control not in D4_CONTROLS:
        raise ValueError(f"unknown D4 control {control}")
    result_rows, result_columns = rows.clone(), columns.clone()
    if control.startswith("transpose"):
        result_rows, result_columns = result_columns, result_rows
    suffix = control.removeprefix("transpose")
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if suffix in {"x_flip", "xy_flip"}:
        result_columns = size - 1 - result_columns
    if suffix in {"y_flip", "xy_flip"}:
        result_rows = size - 1 - result_rows
    return result_rows, result_columns


def _expected_support_indices(
    rows: torch.Tensor,
    columns: torch.Tensor,
    bins: Sequence[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rows.ndim != 1 or columns.ndim != 1 or rows.numel() != columns.numel():
        raise ValueError("target coordinates must be equal-length vectors")
    bins = tuple(bins or _pool_bins())
    bin_count = len(bins)
    row_membership = _bin_membership(200, bins).index_select(1, rows).T
    column_membership = _bin_membership(200, bins).index_select(1, columns).T
    row_width = int(row_membership.sum(dim=1).max()) if rows.numel() else 0
    column_width = int(column_membership.sum(dim=1).max()) if columns.numel() else 0
    row_ids = torch.arange(bin_count, dtype=torch.long).expand(rows.numel(), -1).masked_fill(~row_membership, bin_count)
    column_ids = torch.arange(bin_count, dtype=torch.long).expand(columns.numel(), -1).masked_fill(~column_membership, bin_count)
    row_ids = row_ids.sort(dim=1).values[:, :row_width]
    column_ids = column_ids.sort(dim=1).values[:, :column_width]
    indices = (row_ids.unsqueeze(2) * bin_count + column_ids.unsqueeze(1)).reshape(rows.numel(), row_width * column_width)
    mask = ((row_ids.unsqueeze(2) < bin_count) & (column_ids.unsqueeze(1) < bin_count)).reshape(rows.numel(), row_width * column_width)
    return indices.masked_fill(~mask, -1), mask


def _expected_support(rows: torch.Tensor, columns: torch.Tensor, bins: Sequence[tuple[int, int]] | None = None) -> torch.Tensor:
    indices, mask = _expected_support_indices(rows, columns, bins)
    support = torch.zeros((rows.numel(), 16 * 16), dtype=torch.bool)
    if indices.numel():
        support.scatter_(1, indices.clamp_min(0), mask)
    return support


def _physical_cell_centers(
    bins: Sequence[tuple[int, int]] | None = None,
    *,
    size: int = 200,
    pixel_pitch_m: float = 2.5e-6,
) -> torch.Tensor:
    bins = tuple(bins or _pool_bins(size, 16))
    raw_centers = (torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0) * pixel_pitch_m
    cell_centers = torch.stack([raw_centers[start:end].mean() for start, end in bins])
    return torch.stack(
        (
            cell_centers.repeat_interleave(len(bins)),
            cell_centers.repeat(len(bins)),
        ),
        dim=1,
    )


def _split_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(weights) or tuple(weights.shape) != (257, 40000) or weights.dtype != torch.float32:
        raise ValueError("RWSA readout must have shape [257,40000] and float32 storage")
    return weights[:-1, :], weights[-1, :]


def _normalize_abs_columns(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.isfinite(weights).all():
        raise RWSAHarnessFailure("RWSA readout contains non-finite weights")
    mass = weights.abs().to(torch.float64).T.contiguous()
    denominator = mass.sum(dim=1)
    if not torch.isfinite(denominator).all() or bool((denominator <= 0.0).any()):
        raise RWSAHarnessFailure("RWSA absolute-weight denominator is non-finite or non-positive")
    return mass / denominator.unsqueeze(1), denominator


def _scalar_summary(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().to(torch.float64).flatten()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "quantiles": {f"q{int(q * 100):02d}": float(flat.quantile(q)) for q in QUANTILES},
    }


def _vector_summary(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().to(torch.float64)
    return {
        "mean": flat.mean(dim=0).tolist(),
        "std": flat.std(dim=0, unbiased=False).tolist(),
        "quantiles": {f"q{int(q * 100):02d}": flat.quantile(q, dim=0).tolist() for q in QUANTILES},
    }


def _mean_distribution_summary(values: torch.Tensor) -> dict[str, Any]:
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)), "quantiles": _scalar_summary(values)["quantiles"]}


def _locality_mass(probabilities: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if indices.shape != mask.shape:
        raise ValueError("support index and mask shapes differ")
    if indices.shape[1] == 0:
        return torch.zeros(probabilities.shape[0], dtype=probabilities.dtype)
    selected = probabilities.gather(1, indices.clamp_min(0))
    return (selected * mask.to(probabilities.dtype)).sum(dim=1)


def _permutation_bank(
    *,
    count: int = PERMUTATION_NULL_COUNT,
    seed: int = PERMUTATION_NULL_SEED,
    source_dim: int = 256,
) -> torch.Tensor:
    if count != PERMUTATION_NULL_COUNT or seed != PERMUTATION_NULL_SEED or source_dim != 256:
        raise ValueError("permutation bank parameters are not contract values")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.stack([torch.randperm(source_dim, generator=generator) for _ in range(count)])


def _apply_source_row_permutation(weights: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2 or tuple(weights.shape) != (256, 40000) or permutation.shape != (256,) or permutation.dtype != torch.long:
        raise ValueError("source-row permutation shape is invalid")
    if torch.unique(permutation).numel() != 256 or permutation.min() != 0 or permutation.max() != 255:
        raise ValueError("source-row permutation is not a complete permutation")
    return weights.index_select(0, permutation)


def _permutation_null_summary(
    abs_mass: torch.Tensor,
    denominator: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    observed: float,
) -> dict[str, Any]:
    bank = _permutation_bank()
    null_values: list[torch.Tensor] = []
    for permutation in bank:
        permuted_support = permutation.index_select(0, support_indices.clamp_min(0).reshape(-1)).reshape_as(support_indices)
        selected = abs_mass.gather(1, permuted_support.clamp_min(0))
        value = ((selected * support_mask.to(abs_mass.dtype)).sum(dim=1) / denominator).mean()
        null_values.append(value)
    null = torch.stack(null_values)
    mean = float(null.mean())
    std = float(null.std(unbiased=False))
    effect = 0.0 if std == 0.0 else (observed - mean) / std
    return {
        "statistic": PERMUTATION_NULL["statistic"],
        "observed": observed,
        "null_mean": mean,
        "null_std": std,
        "null_quantiles": _scalar_summary(null)["quantiles"],
        "empirical_rank": 1 + int((null <= observed).sum()),
        "empirical_rank_definition": "1_plus_count(null_statistic<=observed), higher_is_more_local",
        "empirical_p_value": float((1 + (null >= observed).sum()) / (PERMUTATION_NULL_COUNT + 1)),
        "standardized_effect_size": float(effect),
        "count": PERMUTATION_NULL_COUNT,
        "seed": PERMUTATION_NULL_SEED,
        "bank_sha256": tensor_sha256(bank),
        "interpretation": "descriptive_only_no_significance_threshold",
    }


def _bias_spatial_summary(bias: torch.Tensor) -> dict[str, Any]:
    if bias.ndim != 1 or bias.numel() != 40000 or not torch.isfinite(bias).all():
        raise RWSAHarnessFailure("bias vector is malformed or non-finite")
    signed = bias.to(torch.float64).reshape(200, 200)
    absolute = signed.abs()
    coordinates = (torch.arange(200, dtype=torch.float64) - 99.5) * 2.5e-6
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    denominator = absolute.sum()
    centroid = torch.stack(((absolute * yy).sum() / denominator.clamp_min(1.0e-30), (absolute * xx).sum() / denominator.clamp_min(1.0e-30)))
    row_profile = absolute.mean(dim=1)
    column_profile = absolute.mean(dim=0)
    return {
        "mean": float(signed.mean()),
        "variance": float(signed.var(unbiased=False)),
        "rms": float(signed.square().mean().sqrt()),
        "min": float(signed.min()),
        "max": float(signed.max()),
        "positive_abs_mass_fraction": float(signed.clamp_min(0.0).sum() / denominator.clamp_min(1.0e-30)),
        "negative_abs_mass_fraction": float((-signed).clamp_min(0.0).sum() / denominator.clamp_min(1.0e-30)),
        "absolute_weight_centroid_um": (centroid * 1.0e6).tolist(),
        "absolute_spatial_rms_radius_um": float((((yy - centroid[0]).square() + (xx - centroid[1]).square()) * absolute).sum().div(denominator.clamp_min(1.0e-30)).sqrt() * 1.0e6),
        "row_profile_summary": _scalar_summary(row_profile),
        "column_profile_summary": _scalar_summary(column_profile),
        "signed_cancellation": {
            "signed_to_absolute_ratio": float(signed.sum().abs() / denominator.clamp_min(1.0e-30)),
            "cancellation_fraction": float(1.0 - signed.sum().abs() / denominator.clamp_min(1.0e-30)),
        },
    }


def _structural_metrics(weights: torch.Tensor, bias: torch.Tensor | None = None) -> dict[str, Any]:
    source_weights, bias_from_full = _split_weights(weights)
    if bias is not None and not torch.equal(bias.detach().cpu(), bias_from_full.detach().cpu()):
        raise ValueError("supplied bias does not match the frozen full readout")
    bias = bias_from_full
    probabilities, denominator = _normalize_abs_columns(source_weights)
    bins = _pool_bins()
    rows = torch.arange(200, dtype=torch.long).repeat_interleave(200)
    columns = torch.arange(200, dtype=torch.long).repeat(200)
    support_indices, support_mask = _expected_support_indices(rows, columns, bins)
    source_indices = torch.stack(
        (
            torch.arange(16, dtype=torch.float64).repeat_interleave(16),
            torch.arange(16, dtype=torch.float64).repeat(16),
        ),
        dim=1,
    )
    cell_centers = _physical_cell_centers(bins)
    source_row = source_indices[:, 0].long()
    source_column = source_indices[:, 1].long()
    cell_lengths = torch.tensor([end - start for start, end in bins], dtype=torch.float64)
    support_source = support_indices.clamp_min(0)
    support_row = source_row.index_select(0, support_source.reshape(-1)).reshape_as(support_source)
    support_column = source_column.index_select(0, support_source.reshape(-1)).reshape_as(support_source)
    q_unnormalized = 1.0 / (cell_lengths.index_select(0, support_row.reshape(-1)).reshape_as(support_row) * cell_lengths.index_select(0, support_column.reshape(-1)).reshape_as(support_column))
    q_unnormalized = q_unnormalized * support_mask
    q = q_unnormalized / q_unnormalized.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    expected_coarse = (q.unsqueeze(-1) * source_indices.index_select(0, support_source.reshape(-1)).reshape(support_source.shape[0], support_source.shape[1], 2)).sum(dim=1)
    expected_physical = (q.unsqueeze(-1) * cell_centers.index_select(0, support_source.reshape(-1)).reshape(support_source.shape[0], support_source.shape[1], 2)).sum(dim=1)
    absolute_centroid = probabilities.mm(cell_centers)
    coarse_centroid = probabilities.mm(source_indices)
    displacement_physical = absolute_centroid - expected_physical
    displacement_coarse = coarse_centroid - expected_coarse
    argmax = probabilities.argmax(dim=1)
    expected_locality = _locality_mass(probabilities, support_indices, support_mask)
    radius_distance = (source_indices.unsqueeze(0) - expected_coarse.unsqueeze(1)).abs().amax(dim=2)
    radius_mass = {
        str(radius): (probabilities * (radius_distance <= radius + 0.5)).sum(dim=1)
        for radius in (0, 1, 2)
    }
    weighted_spatial_variance = (probabilities * (cell_centers.unsqueeze(0) - absolute_centroid.unsqueeze(1)).square().sum(dim=2)).sum(dim=1)
    row_mass = torch.zeros((probabilities.shape[0], 16), dtype=probabilities.dtype).scatter_add_(1, source_row.unsqueeze(0).expand(probabilities.shape[0], -1), probabilities)
    column_mass = torch.zeros((probabilities.shape[0], 16), dtype=probabilities.dtype).scatter_add_(1, source_column.unsqueeze(0).expand(probabilities.shape[0], -1), probabilities)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-30).log()).sum(dim=1) / math.log(256.0)
    effective_support = 1.0 / probabilities.square().sum(dim=1).clamp_min(1.0e-30)
    positive_mass = source_weights.to(torch.float64).clamp_min(0.0).T.sum(dim=1) / denominator
    negative_mass = (-source_weights.to(torch.float64)).clamp_min(0.0).T.sum(dim=1) / denominator
    signed_ratio = source_weights.to(torch.float64).T.sum(dim=1).abs() / denominator
    d4_summary: dict[str, Any] = {}
    for control in D4_CONTROLS:
        control_rows, control_columns = _transform_coordinates(rows, columns, control)
        control_indices, control_mask = _expected_support_indices(control_rows, control_columns, bins)
        control_locality = _locality_mass(probabilities, control_indices, control_mask)
        d4_summary[control] = {
            "locality_mass": float(control_locality.mean()),
            "locality_mass_summary": _scalar_summary(control_locality),
            "difference_from_identity": float(control_locality.mean() - expected_locality.mean()),
        }
    position_rows = rows * 4 // 200
    position_columns = columns * 4 // 200
    locality_position = torch.zeros((4, 4), dtype=torch.float64)
    for row in range(4):
        for column in range(4):
            selection = (position_rows == row) & (position_columns == column)
            locality_position[row, column] = expected_locality[selection].mean()
    translation_mean = displacement_coarse.mean(dim=0)
    translation_residual = displacement_coarse - translation_mean
    argmax_histogram = torch.bincount(argmax, minlength=256).reshape(16, 16)
    target_blocks = torch.zeros((16, 16), dtype=torch.long)
    for row in range(16):
        for column in range(16):
            selection = (rows * 16 // 200 == row) & (columns * 16 // 200 == column)
            target_blocks[row, column] = torch.bincount(argmax[selection], minlength=256).argmax()
    observed = float(expected_locality.mean())
    argmax_inside_expected_support = (support_indices == argmax.unsqueeze(1)).any(dim=1)
    return {
        "mapping_description": "16x16 pooled detector to 200x200 target; no raw-pixel registration claim",
        "pool_bins": [list(item) for item in bins],
        "argmax_source_coarse_cell": {
            "mode_cell": argmax_histogram.flatten().argmax().item() // 16,
            "mode_cell_pair": source_indices[argmax_histogram.flatten().argmax()].tolist(),
            "occupancy_fraction": float(argmax_histogram.max() / argmax.numel()),
            "occupancy_entropy": float(-(argmax_histogram.flatten().to(torch.float64) / argmax.numel()).clamp_min(1.0e-30).mul(((argmax_histogram.flatten().to(torch.float64) / argmax.numel()).clamp_min(1.0e-30)).log()).sum() / math.log(256.0)),
            "target_block_mode_map": target_blocks.tolist(),
        },
        "absolute_weight_centroid": {
            "physical_um": _vector_summary(absolute_centroid * 1.0e6),
            "coarse_cell": _vector_summary(coarse_centroid),
        },
        "centroid_displacement": {
            "physical_um": _vector_summary(displacement_physical * 1.0e6),
            "coarse_cell": _vector_summary(displacement_coarse),
        },
        "top_1_mass": _scalar_summary(probabilities.max(dim=1).values),
        "top_k_mass": {"k": TOP_K, **_scalar_summary(probabilities.topk(TOP_K, dim=1).values.sum(dim=1))},
        "expected_support_locality_mass": _scalar_summary(expected_locality),
        "coarse_radius_locality_mass": {f"radius_{radius}": _scalar_summary(values) for radius, values in radius_mass.items()},
        "spatial_variance_m2": _scalar_summary(weighted_spatial_variance),
        "rms_support_radius_um": _scalar_summary(weighted_spatial_variance.sqrt() * 1.0e6),
        "effective_support_size": _scalar_summary(effective_support),
        "normalized_weight_entropy": _scalar_summary(entropy),
        "source_row_concentration": {
            "max_mass": _scalar_summary(row_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(-(row_mass * row_mass.clamp_min(1.0e-30).log()).sum(dim=1) / math.log(16.0)),
        },
        "source_column_concentration": {
            "max_mass": _scalar_summary(column_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(-(column_mass * column_mass.clamp_min(1.0e-30).log()).sum(dim=1) / math.log(16.0)),
        },
        "identity_coordinate_consistency": {
            "locality_mass": observed,
            "argmax_inside_expected_support_fraction": float(argmax_inside_expected_support.float().mean()),
            "expected_centroid_physical_um": _vector_summary(expected_physical * 1.0e6),
            "observed_centroid_physical_um": _vector_summary(absolute_centroid * 1.0e6),
        },
        "d4_control_consistency": d4_summary,
        "translation_field": {
            "mean_coarse_cell": translation_mean.tolist(),
            "residual_variance_coarse_cell": translation_residual.var(dim=0, unbiased=False).tolist(),
            "residual_rms_coarse_cell": translation_residual.square().mean(dim=0).sqrt().tolist(),
        },
        "locality_vs_target_spatial_position": {
            "grid_shape": [4, 4],
            "aggregation": "mean_expected_support_locality_mass",
            "grid": locality_position.tolist(),
        },
        "positive_weight_mass": _scalar_summary(positive_mass),
        "negative_weight_mass": _scalar_summary(negative_mass),
        "signed_cancellation": {
            "signed_to_absolute_ratio": _scalar_summary(signed_ratio),
            "cancellation_fraction": _scalar_summary(1.0 - signed_ratio),
        },
        "bias_spatial_summary": _bias_spatial_summary(bias),
        "permutation_null": _permutation_null_summary(abs_mass=probabilities * denominator.unsqueeze(1), denominator=denominator, support_indices=support_indices, support_mask=support_mask, observed=observed),
    }


def _decompose_readout(features: torch.Tensor, probe: ReadoutProbe | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(probe, ReadoutProbe):
        weights, bias = probe.weights, probe.bias
    else:
        weights, bias = _split_weights(probe)
    if features.ndim != 2 or features.shape[1] != 256 or not torch.is_floating_point(features):
        raise ValueError("RWSA features must be [N,256] real values")
    dynamic = features.float().mm(weights)
    static = bias.unsqueeze(0).expand(features.shape[0], -1)
    full = dynamic + static
    if not torch.isfinite(full).all():
        raise RWSAHarnessFailure("RWSA readout decomposition is non-finite")
    return full, dynamic, static


def _panel_templates(images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or labels.ndim != 1 or images.shape[0] != labels.numel() or images.shape[1:] != (1, 28, 28):
        raise ValueError("RWSA panel has invalid image or label shapes")
    if not torch.equal(torch.unique(labels).to(torch.long), torch.arange(10)):
        raise ValueError("RWSA panel must contain all ten labels")
    sums = torch.zeros((10, 1, 200, 200), dtype=torch.float32)
    global_sum = torch.zeros((1, 200, 200), dtype=torch.float32)
    counts = torch.bincount(labels, minlength=10)
    for index in range(images.shape[0]):
        target = base.render_intensity_batch(images[index : index + 1]).target.float()[0]
        sums[labels[index]] += target
        global_sum += target
    if bool((counts == 0).any()):
        raise ValueError("RWSA panel has an empty class")
    return sums / counts.reshape(10, 1, 1, 1), global_sum / images.shape[0]


def _variant_metrics(
    errors: torch.Tensor,
    template_errors: torch.Tensor,
    cyclic_errors: torch.Tensor,
    labels: torch.Tensor,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "paired_mse": float(errors.mean()),
        "class_template_mse": float(template_errors.mean()),
        "cyclic_mse": float(cyclic_errors.mean()),
        "mse_by_label": [float(errors[labels == label].mean()) for label in range(10)],
        "class_template_mse_by_label": [float(template_errors[labels == label].mean()) for label in range(10)],
        "cyclic_mse_by_label": [float(cyclic_errors[labels == label].mean()) for label in range(10)],
        "instance_sensitive_metrics": dict(context),
        "finite": bool(torch.isfinite(errors).all() and torch.isfinite(template_errors).all() and torch.isfinite(cyclic_errors).all()),
    }


def _bias_metrics(
    errors: torch.Tensor,
    class_errors: torch.Tensor,
    global_errors: torch.Tensor,
    labels: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, Any]:
    return {
        "paired_mse": float(errors.mean()),
        "class_template_mse": float(class_errors.mean()),
        "global_template_mse": float(global_errors.mean()),
        "class_global_template_relation": {
            "class_template_minus_global_template_mse": float(class_errors.mean() - global_errors.mean()),
            "class_to_global_template_mse_ratio": float(class_errors.mean() / global_errors.mean().clamp_min(1.0e-30)),
        },
        "mse_by_label": [float(errors[labels == label].mean()) for label in range(10)],
        "mean": float(bias.mean()),
        "variance": float(bias.var(unbiased=False)),
        "spatial_structure": _bias_spatial_summary(bias),
        "instance_sensitive_metrics": "not_applicable_degenerate_constant_output",
        "finite": bool(torch.isfinite(errors).all() and torch.isfinite(class_errors).all() and torch.isfinite(global_errors).all()),
    }


def _panel_replay(
    model: bogt.ScaledChenBOGTModel,
    probe: ReadoutProbe,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    panel_name: str,
) -> dict[str, Any]:
    templates, global_template = _panel_templates(images, labels)
    first_target = base.render_intensity_batch(images[:1]).target.float()[0]
    last_target = base.render_intensity_batch(images[-1:]).target.float()[0]
    previous_target: torch.Tensor | None = None
    physical_errors: list[torch.Tensor] = []
    full_errors: list[torch.Tensor] = []
    dynamic_errors: list[torch.Tensor] = []
    bias_errors: list[torch.Tensor] = []
    physical_template_errors: list[torch.Tensor] = []
    full_template_errors: list[torch.Tensor] = []
    dynamic_template_errors: list[torch.Tensor] = []
    bias_template_errors: list[torch.Tensor] = []
    bias_global_errors: list[torch.Tensor] = []
    physical_cyclic_errors: list[torch.Tensor] = []
    full_cyclic_errors: list[torch.Tensor] = []
    dynamic_cyclic_errors: list[torch.Tensor] = []
    physical_features: list[torch.Tensor] = []
    full_features: list[torch.Tensor] = []
    dynamic_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    dynamic_sum = torch.zeros(40000, dtype=torch.float64)
    dynamic_square_sum = torch.zeros(40000, dtype=torch.float64)
    dynamic_bias_inner_sum = 0.0
    dynamic_square_total = 0.0
    bias_square_total = 0.0
    full_square_total = 0.0
    count = 0
    for index in range(images.shape[0]):
        rendered = base.render_intensity_batch(images[index : index + 1])
        target = rendered.target.float()
        cyclic = last_target.unsqueeze(0) if index == 0 else previous_target.unsqueeze(0)  # type: ignore[union-attr]
        previous_target = target[0]
        with torch.no_grad():
            output = model(rendered.amplitude.to(device), return_trace=False)
        camera = output.camera.detach().float().cpu()
        if tuple(camera.shape) != (1, 1, 200, 200) or not torch.isfinite(camera).all():
            raise RWSAHarnessFailure("CMOS-only replay produced an invalid output.camera")
        features = mra._feature_projection(camera, complex_value=False).float()
        full, dynamic, static = _decompose_readout(features, probe)
        target_flat = target.flatten(1)
        cyclic_flat = cyclic.flatten(1)
        template_flat = templates.index_select(0, labels[index : index + 1]).flatten(1)
        global_flat = global_template.unsqueeze(0).flatten(1)
        physical_flat = camera.flatten(1)
        physical_errors.append((physical_flat - target_flat).square().mean(dim=1))
        full_errors.append((full - target_flat).square().mean(dim=1))
        dynamic_errors.append((dynamic - target_flat).square().mean(dim=1))
        bias_errors.append((static - target_flat).square().mean(dim=1))
        physical_template_errors.append((physical_flat - template_flat).square().mean(dim=1))
        full_template_errors.append((full - template_flat).square().mean(dim=1))
        dynamic_template_errors.append((dynamic - template_flat).square().mean(dim=1))
        bias_template_errors.append((static - template_flat).square().mean(dim=1))
        bias_global_errors.append((static - global_flat).square().mean(dim=1))
        physical_cyclic_errors.append((physical_flat - cyclic_flat).square().mean(dim=1))
        full_cyclic_errors.append((full - cyclic_flat).square().mean(dim=1))
        dynamic_cyclic_errors.append((dynamic - cyclic_flat).square().mean(dim=1))
        physical_features.append(mra._feature_projection(camera, complex_value=False))
        full_features.append(mra._feature_projection(full.reshape(1, 1, 200, 200), complex_value=False))
        dynamic_features.append(mra._feature_projection(dynamic.reshape(1, 1, 200, 200), complex_value=False))
        target_features.append(mra._feature_projection(target, complex_value=False))
        dynamic_sum += dynamic[0].to(torch.float64)
        dynamic_square_sum += dynamic[0].to(torch.float64).square()
        dynamic_square_total += float(dynamic.square().sum())
        bias_square_total += float(static.square().sum())
        full_square_total += float(full.square().sum())
        dynamic_bias_inner_sum += float((dynamic * static).sum())
        count += 1
    physical = torch.cat(physical_errors)
    full = torch.cat(full_errors)
    dynamic = torch.cat(dynamic_errors)
    bias = torch.cat(bias_errors)
    physical_template = torch.cat(physical_template_errors)
    full_template = torch.cat(full_template_errors)
    dynamic_template = torch.cat(dynamic_template_errors)
    bias_template = torch.cat(bias_template_errors)
    bias_global = torch.cat(bias_global_errors)
    physical_cyclic = torch.cat(physical_cyclic_errors)
    full_cyclic = torch.cat(full_cyclic_errors)
    dynamic_cyclic = torch.cat(dynamic_cyclic_errors)
    target_compact = torch.cat(target_features)
    full_context = rmfa._context_metrics_from_compact(torch.cat(full_features), target_compact, labels)
    dynamic_context = rmfa._context_metrics_from_compact(torch.cat(dynamic_features), target_compact, labels)
    physical_context = rmfa._context_metrics_from_compact(torch.cat(physical_features), target_compact, labels)
    anchor = PHYSICAL_ANCHORS[panel_name]
    physical_values = {
        "paired_mse": float(physical.mean()),
        "class_template_mse": float(physical_template.mean()),
        "cyclic_mse": float(physical_cyclic.mean()),
    }
    replay = {key: abs(physical_values[key] - value) <= PHYSICAL_BASELINE_TOLERANCE for key, value in anchor.items()}
    dynamic_mean = dynamic_sum / count
    dynamic_variance = (dynamic_square_sum / count - dynamic_mean.square()).clamp_min(0.0)
    bias_rms = math.sqrt(bias_square_total / (count * 40000))
    dynamic_rms = math.sqrt(dynamic_square_total / (count * 40000))
    full_rms = math.sqrt(full_square_total / (count * 40000))
    return {
        "panel": panel_name,
        "evaluation_count": count,
        "physical": {**physical_values, "mse_by_label": [float(physical[labels == label].mean()) for label in range(10)], "instance_sensitive_metrics": physical_context},
        "full": _variant_metrics(full, full_template, full_cyclic, labels, full_context),
        "dynamic": _variant_metrics(dynamic, dynamic_template, dynamic_cyclic, labels, dynamic_context),
        "bias_only": _bias_metrics(bias, bias_template, bias_global, labels, probe.bias),
        "native_physical_anchor": anchor,
        "native_replay": replay,
        "finite": bool(all(replay.values()) and torch.isfinite(physical).all() and torch.isfinite(full).all() and torch.isfinite(dynamic).all() and torch.isfinite(bias).all()),
        "decomposition": {
            "dynamic_Wz_rms": dynamic_rms,
            "bias_b_rms": bias_rms,
            "full_rms": full_rms,
            "dynamic_Wz_norm_mean": float((dynamic.square().sum(dim=1).sqrt()).mean()),
            "bias_b_norm": float(probe.bias.square().sum().sqrt()),
            "dynamic_prediction_variance_mean": float(dynamic_variance.mean()),
            "dynamic_to_bias_rms_ratio": float(dynamic_rms / max(bias_rms, 1.0e-30)),
            "bias_to_dynamic_rms_ratio": float(bias_rms / max(dynamic_rms, 1.0e-30)),
            "signed_interaction_mean": dynamic_bias_inner_sum / (count * 40000),
            "signed_interaction_twice_mean": 2.0 * dynamic_bias_inner_sum / (count * 40000),
            "energy_identity_residual": full_rms * full_rms - dynamic_rms * dynamic_rms - bias_rms * bias_rms - 2.0 * dynamic_bias_inner_sum / (count * 40000),
        },
        "replay_contract": {
            "source": "output.camera",
            "feature_projection": "mra._feature_projection(real, complex_value=False)",
            "postprocessing": "none",
            "raw_cmos_saved": False,
            "pooled_feature_bank_saved": False,
            "candidate_prefix_consumed": False,
            "microbatch_size": MICROBATCH_SIZE,
            "first_target_sha256": tensor_sha256(first_target),
        },
    }


def _record(
    schema: str,
    status: str,
    classification: str | None,
    spec: Spec,
    consumer: rmal.RMFAConsumer,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
    gates: Sequence[GateResult],
    actions: Sequence[str],
    contract: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    record = EvidenceRecordV2(
        2,
        schema,
        "diagnostic",
        status,
        str(state["commit"]),
        None,
        spec.config_hash,
        _context(spec, consumer),
        dict(hashes),
        tuple(gates),
        tuple(actions),
        reason,
    )
    record.validate()
    return {
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
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse RWSA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("RWSA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"RWSA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("RWSA attempt directory is not a fresh validation-only slot")


def _resource_values(device: torch.device) -> tuple[bool, int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, 0, 0
    free, total = torch.cuda.mem_get_info(device)
    return True, int(free), int(total)


def _validation_gates(
    spec: Spec,
    consumer: rmal.RMFAConsumer,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
    formal: FormalSystemConfig,
    output_path: Path,
) -> tuple[GateResult, ...]:
    data = consumer.bundle.parent.parent.data
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    counts = torch.bincount(selection_labels, minlength=10)
    operator = mra._build_research_ols(formal)
    cuda_available, free_cuda, _ = _resource_values(torch.device("cuda"))
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    bins = _pool_bins()
    rows = torch.arange(200, dtype=torch.long).repeat_interleave(200)
    columns = torch.arange(200, dtype=torch.long).repeat(200)
    support_indices, support_mask = _expected_support_indices(rows, columns, bins)
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("rmal_consumer_identity", file_sha256(consumer.spec.path) == RMAL_CONSUMER["config_sha256"] and file_sha256(consumer.probe_path) == RMAL_CONSUMER["probe_sha256"], {}),
        GateResult("rmal_terminal_classification", consumer.audit.get("classification") == RMAL_CONSUMER["audit_classification"], {}),
        GateResult("checkpoint_identity", file_sha256(checkpoint_path) == REFERENCES["scaled_chen_checkpoint"]["sha256"], {}),
        GateResult("selection_balanced", bool(torch.equal(counts, torch.full((10,), 100, dtype=torch.long))), {"counts": counts.tolist()}),
        GateResult("confirmation_count_locked", data.confirmation_indices.numel() == CONFIRMATION_COUNT and torch.equal(data.confirmation_indices, torch.arange(55000, 60000, dtype=torch.long)), {}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_confirmation_disjoint", not bool(torch.isin(fit_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_panel_locked", fit_indices.numel() == FIT_COUNT and tensor_sha256(fit_indices) == FIT_PANEL["indices_sha256"], {}),
        GateResult("probe_shape_locked", tuple(probe.full_weights.shape) == (257, 40000) and tuple(probe.weights.shape) == (256, 40000) and tuple(probe.bias.shape) == (40000,), {}),
        GateResult("probe_feature_order_locked", probe.metadata.get("stage") == "cmos_intensity" and probe.metadata.get("feature_dim") == 256, {}),
        GateResult("mode_bank_rank_100", mra._mode_rank(operator) == 100, {"rank": mra._mode_rank(operator)}),
        GateResult("finite_mode_bank", bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all()), {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("pooling_contract_locked", spec.payload["pooling"] == POOLING and _pool_bins(POOLING["input_shape"][0], POOLING["output_shape"][0]) == bins, {"bins": [list(item) for item in bins]}),
        GateResult("overlapping_bins_present", any(end - start > 0 for start, end in bins) and bool(_bin_membership(200, bins).sum(dim=0).max() > 1), {}),
        GateResult("expected_support_locked", support_indices.shape == (40000, 4) and support_mask.shape == (40000, 4), {}),
        GateResult("d4_controls_locked", tuple(spec.payload["d4_controls"]) == D4_CONTROLS, {}),
        GateResult("permutation_null_locked", spec.payload["permutation_null"] == PERMUTATION_NULL and _permutation_bank().shape == (PERMUTATION_NULL_COUNT, 256), {}),
        GateResult("candidate_prefix_not_consumed", EVALUATION_PANELS["candidate_prefix"]["consumed"] is False, {}),
        GateResult("no_update_scope", os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
        GateResult("windows_native_checkout", os.name == "nt" and Path.cwd().drive.upper() == "D:", {}),
        GateResult("cuda_available", cuda_available, {"free_cuda_bytes": free_cuda}),
        GateResult("free_cuda_precondition", free_cuda >= MIN_FREE_CUDA_BYTES, {"free_cuda_bytes": free_cuda, "minimum": MIN_FREE_CUDA_BYTES}),
        GateResult("disk_free_precondition", disk.free >= MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES}),
        GateResult("output_slot_absent", not output_path.parent.exists(), {"path": str(output_path.parent)}),
    )


def _load_validation(path: str | Path, spec: Spec, review_path: Path, consumer: rmal.RMFAConsumer, probe: ReadoutProbe, fit_indices: torch.Tensor, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, consumer, probe, fit_indices)
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
        raise ValueError("RWSA validation evidence is stale or mismatched")


def _invalid_harness_payload(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    consumer: rmal.RMFAConsumer,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
    state: Mapping[str, Any],
    reason: str,
) -> None:
    gates = (
        GateResult("rmal_physical_anchor_replay", False, {"reason": reason}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
    )
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        consumer,
        state,
        _input_hashes(spec, review_path, consumer, probe, fit_indices),
        gates,
        (),
        {"reason": reason, "structural_result_available": False},
        reason,
    )
    _write_new(output_path, payload, artifact_kind="RWSA invalid-harness audit evidence")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer, _, probe = _load_rmal_consumer(spec, args.dataset_root, load_checkpoints=False)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    formal, _ = mra._load_formal(spec)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, consumer, probe, fit_indices, formal, output_path)
    valid = all(gate.passed for gate in gates)
    contract = {
        "rmal_consumer": RMAL_CONSUMER,
        "pooling": POOLING,
        "coordinate_contract": COORDINATE_CONTRACT,
        "d4_controls": list(D4_CONTROLS),
        "permutation_null": PERMUTATION_NULL,
        "measurement": MEASUREMENT,
        "bias_attribution": BIAS_ATTRIBUTION,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        consumer,
        state,
        _input_hashes(spec, review_path, consumer, probe, fit_indices),
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "RWSA static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="RWSA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    available, free, total = _resource_values(device)
    if not available:
        raise RWSAHarnessFailure("RWSA CMOS replay requires CUDA; CPU is not a formal fallback")
    if free < MIN_FREE_CUDA_BYTES:
        raise RWSAHarnessFailure("RWSA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RWSAHarnessFailure("RWSA requires at least 10 GiB free storage")
    return free, total


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("RWSA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer, _, probe = _load_rmal_consumer(spec, args.dataset_root, load_checkpoints=True)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    formal, _ = mra._load_formal(spec)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    _load_validation(validation_path, spec, review_path, consumer, probe, fit_indices, state)
    free_cuda, total_cuda = _resource_preflight(output_path, torch.device(args.device))
    data = consumer.bundle.parent.parent.data
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    checkpoint_hash_before = file_sha256(checkpoint_path)
    model = mra._load_scaled_model(formal, consumer.source.checkpoints[PRIMARY_CHECKPOINT], torch.device(args.device))
    model_state_before = _state_cpu(model)
    ols_state_before = _state_cpu(model.ols)
    try:
        structural = _structural_metrics(probe.full_weights, probe.bias)
        selection_images = data.test_images.index_select(0, data.selection_indices)
        selection_labels = data.test_labels.index_select(0, data.selection_indices)
        confirmation_images = data.images.index_select(0, data.confirmation_indices)
        confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        started = time.perf_counter()
        selection = _panel_replay(model, probe, selection_images, selection_labels, torch.device(args.device), "selection")
        confirmation = _panel_replay(model, probe, confirmation_images, confirmation_labels, torch.device(args.device), "confirmation")
        elapsed = time.perf_counter() - started
        model_state_after = _state_cpu(model)
        ols_state_after = _state_cpu(model.ols)
        checkpoint_hash_after = file_sha256(checkpoint_path)
        peak = int(torch.cuda.max_memory_allocated(torch.device(args.device))) if torch.cuda.is_available() else 0
        state_unchanged = _state_equal(model_state_before, model_state_after)
        ols_unchanged = _state_equal(ols_state_before, ols_state_after)
        physical_anchor_replay = bool(all(selection["native_replay"].values()) and all(confirmation["native_replay"].values()))
        resource_ok = peak <= MAX_PEAK_CUDA_BYTES
        storage_estimate = len(json.dumps(structural)) + len(json.dumps(selection)) + len(json.dumps(confirmation)) + 256 * 1024
        if not physical_anchor_replay:
            raise RWSAHarnessFailure("RWSA physical anchor replay exceeded the registered RMAL tolerance")
        if not state_unchanged or not ols_unchanged:
            raise RWSAHarnessFailure("RWSA changed the frozen model or OLS state")
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RWSAHarnessFailure("RWSA changed the frozen checkpoint")
        if not resource_ok or storage_estimate > MAX_STORAGE_BYTES:
            raise RWSAHarnessFailure("RWSA resource gate failed")
        common_contract = {
            "rmal_consumer": RMAL_CONSUMER,
            "probe": {
                "artifact_sha256": probe.artifact_sha256,
                "weight_shape": list(probe.full_weights.shape),
                "feature_shape": list(probe.weights.shape),
                "bias_shape": list(probe.bias.shape),
                "fit_panel_hash": tensor_sha256(fit_indices),
                "metadata": dict(probe.metadata),
            },
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": list(D4_CONTROLS),
            "permutation_null": PERMUTATION_NULL,
            "bias_attribution": BIAS_ATTRIBUTION,
            "candidate_prefix_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "mechanism_evidence_only": True,
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
        }
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": "passed",
            "classification": CLASSIFICATION,
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "structural": structural,
            "panels": {"selection": selection, "confirmation": confirmation},
            "contract": common_contract,
            "optimizer_created": False,
            "backward_calls": 0,
            "evaluator_calls": 0,
            "checkpoint_written": False,
            "prior_sampled": False,
            "generation_started": False,
            "physical_forward_modified": False,
            "raw_cmos_bank_saved": False,
            "pooled_feature_bank_saved": False,
        }
        compact_path = output_path.parent / "compact_report.json"
        _write_new(compact_path, compact, artifact_kind="RWSA compact report")
        compact_hash = file_sha256(compact_path)
        output_bytes_before_audit = int(sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()))
        gates = (
            GateResult("rmal_physical_anchor_replay", physical_anchor_replay, {"selection": selection["native_replay"], "confirmation": confirmation["native_replay"]}),
            GateResult("finite_measurements", bool(selection["finite"] and confirmation["finite"]), {}),
            GateResult("frozen_model_and_OLS", state_unchanged and ols_unchanged, {}),
            GateResult("checkpoint_unchanged", checkpoint_hash_before == checkpoint_hash_after, {"before": checkpoint_hash_before, "after": checkpoint_hash_after}),
            GateResult("readout_frozen", probe.metadata.get("frozen") is True and probe.metadata.get("probe_hash") == tensor_sha256(probe.full_weights), {}),
            GateResult("permutation_contract", PERMUTATION_NULL_COUNT == 256 and PERMUTATION_NULL_SEED == 0, {}),
            GateResult("no_update_isolation", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
            GateResult("resource_limits", resource_ok and output_bytes_before_audit <= MAX_STORAGE_BYTES, {"peak_bytes": peak, "output_bytes_before_audit": output_bytes_before_audit}),
        )
        audit_contract = {
            **common_contract,
            "structural": structural,
            "panels": {"selection": selection, "confirmation": confirmation},
            "model_state_before": _state_digest(model_state_before),
            "model_state_after": _state_digest(model_state_after),
            "ols_state_before": _state_digest(ols_state_before),
            "ols_state_after": _state_digest(ols_state_after),
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "resource": {
                "free_cuda_bytes_before": free_cuda,
                "total_cuda_bytes": total_cuda,
                "peak_memory_bytes": peak,
                "wall_clock_seconds": elapsed,
                "wall_clock": "observational_only",
                "output_bytes_before_audit": output_bytes_before_audit,
                "storage_estimate_bytes": storage_estimate,
            },
            "compact_report_path": str(compact_path.resolve()),
            "compact_report_sha256": compact_hash,
        }
        payload = _record(
            AUDIT_SCHEMA,
            "passed",
            CLASSIFICATION,
            spec,
            consumer,
            state,
            _input_hashes(spec, review_path, consumer, probe, fit_indices),
            gates,
            (NEXT_ACTION_RESULT_REVIEW,),
            audit_contract,
            "RWSA is descriptive-only; no structural metric authorizes a mechanism classification",
        )
        _write_new(output_path, payload, artifact_kind="RWSA audit evidence")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except RWSAHarnessFailure as exc:
        if not output_path.exists():
            _invalid_harness_payload(output_path, spec, review_path, consumer, probe, fit_indices, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-detector-readout-structure-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-detector-readout-structure-attribution-run":
        return run(args)
    raise ValueError(f"unsupported RWSA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "BIAS_ATTRIBUTION",
    "CLASSIFICATION",
    "COORDINATE_CONTRACT",
    "D4_CONTROLS",
    "GATES",
    "HYPOTHESIS",
    "MEASUREMENT",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "PERMUTATION_NULL",
    "PERMUTATION_NULL_COUNT",
    "PERMUTATION_NULL_SEED",
    "POOLING",
    "PRIMARY_CHECKPOINT",
    "RMAL_PROBE_SCHEMA",
    "ReadoutProbe",
    "REFERENCES",
    "RMAL_CONSUMER",
    "REVIEW_SCHEMA",
    "Spec",
    "VALIDATION_SCHEMA",
    "_apply_source_row_permutation",
    "_assert_contract",
    "_decompose_readout",
    "_expected_support",
    "_expected_support_indices",
    "_load_review",
    "_load_rmal_probe",
    "_load_spec",
    "_panel_replay",
    "_permutation_bank",
    "_physical_cell_centers",
    "_pool_bins",
    "_structural_metrics",
    "dispatch",
    "run",
    "validate",
]
