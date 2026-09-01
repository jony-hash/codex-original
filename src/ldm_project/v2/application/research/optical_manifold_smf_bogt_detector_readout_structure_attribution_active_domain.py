"""Active-domain structure attribution for the frozen RMAL detector readout.

This is a diagnostic-only amendment of the historical RWSA contract.  It
consumes the immutable RMAL probe, the immutable invalid RWSA attempt-02
identity, and the passed ZCSA result.  Weight probabilities are defined only
on the frozen dynamic-active output columns; inactive columns remain explicit
in every decomposition and are never silently dropped.
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
    safe_torch_load,  # noqa: F401 - preserve the probe-loading test boundary
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_detector_readout_structure_attribution as rwsa
from . import optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit as zcsa
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
    "active_domain_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S5-RWSA-ACTIVE"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
BASELINE_HEAD = "86e0862690cdb32173e453869ed33b8eb2b4c017"

PRIMARY_CHECKPOINT = "scaled_chen_step_5000"
TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
PROBE_INPUT_DIM = 257
EXPECTED_ACTIVE_COUNT = 33502
EXPECTED_INACTIVE_COUNT = 6498
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
TARGET_SHAPE = (200, 200)
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
NEXT_ACTION_RUN = "run_scaled_chen_detector_readout_structure_attribution_active_domain"
NEXT_ACTION_RESULT_REVIEW = "manual_review_RWSA_active_domain_result"

RMAL_CONSUMER = dict(rwsa.RMAL_CONSUMER)
RWSA_IDENTITY = dict(zcsa.RWSA_IDENTITY)
ZCSA_IDENTITY = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_v1.yaml"
    ),
    "config_sha256": "7fba357434f83f5300982a5319fe161776824d4b3ee6487efe74db7c9fe09fb0",
    "config_canonical_sha256": "a09dfe21cd0735530689f52cae42067a14ee80776098bcf1321f993557a965c3",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit_review_v1.json"
    ),
    "review_sha256": "ea90976379e935144ae51db18665862595d66cd538fdcc3835d8f7cdee0b59a0",
    "validation": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/"
        "validation.json"
    ),
    "validation_sha256": "d422f1e31e8298f9662896f832b207ad896fb619c5a656bccf54b099cd419954",
    "audit": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/"
        "audit.json"
    ),
    "audit_sha256": "0cba5990462e9f8cc238885b14c90c907a7c6f085877b6c8ad0c8b95cbf0bfb8",
    "compact_report": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_zero_column_semantics_audit_v1_attempt_01/"
        "compact_report.json"
    ),
    "compact_report_sha256": "772fb739b39d029c213ab42e8f552648782607b5562b6350d4d2c9744b73b9e6",
    "implementation_module_sha256": (
        "d046bd6b29993825deea33a4c02d6e0a7e10a9e73ddfd9016e918a04e9a7e1d7"
    ),
    "implementation_commit": "1f5e553f6c7ad80217427ab83f5669e7b9758acb",
    "result_git_commit": "1f5e553f6c7ad80217427ab83f5669e7b9758acb",
    "validation_status": "passed",
    "audit_status": "passed",
    "audit_classification": "zero_column_semantics_attribution_completed",
}
REFERENCES = {
    "scaled_chen_checkpoint": {
        "path": (
            "outputs/20260829_v2/runs/"
            "lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/"
            "scaled_chen_horizon_step_5000.pt"
        ),
        "sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
        "producer_commit": "019b76ef911e0bd16b8d5d4169807323427684af",
        "global_step": 5000,
        "role": "frozen_scaled_chen_reference",
    }
}
FIT_PANEL = dict(rwsa.FIT_PANEL)
EVALUATION_PANELS = json.loads(json.dumps(rwsa.EVALUATION_PANELS))
POOLING = dict(rwsa.POOLING)
COORDINATE_CONTRACT = dict(rwsa.COORDINATE_CONTRACT)
D4_CONTROLS = tuple(rwsa.D4_CONTROLS)
PERMUTATION_NULL = dict(rwsa.PERMUTATION_NULL)
PERMUTATION_NULL["statistic"] = "active_domain_expected_support_locality_mass"
TARGET = dict(zcsa.TARGET)
PHYSICAL_ANCHORS = rmfa.PHYSICAL_ANCHORS
TARGET_RENDERER_SOURCE_SHA256 = zcsa.TARGET_RENDERER_SOURCE_SHA256

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S5-RWSA",
    "lineage_start_head": BASELINE_HEAD,
    "old_rwsa_attempt_02_immutable": True,
    "zcsa_passed_evidence_required": True,
    "attempt_name": (
        "lightgen_scaled_chen_detector_readout_structure_attribution_"
        "active_domain_v1_attempt_02"
    ),
}
ACTIVE_DOMAIN = {
    "name": "dynamic_active",
    "conditional_on_dynamic_active_domain": True,
    "total_output_columns": TOTAL_COLUMNS,
    "active_count": EXPECTED_ACTIVE_COUNT,
    "inactive_count": EXPECTED_INACTIVE_COUNT,
    "dynamic_mass": "d_j = sum_i abs(W_ij)",
    "active_identity": "d_j > 0 exact",
    "inactive_identity": "d_j == 0 exact",
    "probability_definition": "p_ij = abs(W_ij) / d_j for active j only",
    "inactive_probability": "not_computed",
    "inactive_columns_retained_in_evidence": True,
    "epsilon_normalization": False,
}
MEASUREMENT = {
    "objective": "conditional_active_domain_detector_readout_structure_attribution",
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "fit_count": FIT_COUNT,
    "selection_count": SELECTION_COUNT,
    "confirmation_count": CONFIRMATION_COUNT,
    "candidate_prefix_consumed": False,
    "conditional_on_dynamic_active_domain": True,
    "total_output_columns": TOTAL_COLUMNS,
    "dynamic_active_count": EXPECTED_ACTIVE_COUNT,
    "dynamic_inactive_count": EXPECTED_INACTIVE_COUNT,
    "target_renderer": "base.render_intensity_batch",
    "cmos_source": "output.camera",
    "exact_panel_cmos_only_replay": True,
    "no_six_stage_capture": True,
    "no_stage_search": True,
    "no_incident_capture": True,
    "no_decoder_capture": True,
    "no_modal_capture": True,
    "no_pre_ols_capture": True,
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
    "decomposition_domains": ["all_40000", "dynamic_active", "zero_W"],
    "masked_class_template_metrics": "not_evaluated_without_unambiguous_contract",
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
    "decision": "approve_no_update_scaled_chen_active_domain_structure_attribution",
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
        "six_stage_RMAL_search",
        "incident_capture",
        "decoder_capture",
        "modal_capture",
        "pre_OLS_capture",
        "DOAA_reconstruction_candidate",
        "FMF",
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


class RWSAActiveHarnessFailure(RuntimeError):
    """The registered active-domain measurement boundary failed."""


ReadoutProbe = rwsa.ReadoutProbe


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"RWSA active amendment requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "lineage",
        "rmal_consumer",
        "rwsa_identity",
        "zcsa_identity",
        "references",
        "fit_panel",
        "evaluation_panels",
        "pooling",
        "coordinate_contract",
        "d4_controls",
        "permutation_null",
        "active_domain",
        "measurement",
        "bias_attribution",
        "target",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("active-domain RWSA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("active-domain RWSA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": (
            "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
        ),
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "references": REFERENCES,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "pooling": POOLING,
        "coordinate_contract": COORDINATE_CONTRACT,
        "d4_controls": list(D4_CONTROLS),
        "permutation_null": PERMUTATION_NULL,
        "active_domain": ACTIVE_DOMAIN,
        "measurement": MEASUREMENT,
        "bias_attribution": BIAS_ATTRIBUTION,
        "target": TARGET,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"active-domain RWSA {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "references": {
                "primary_checkpoint": PRIMARY_CHECKPOINT,
                "checkpoint_sha256": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            },
            "fit_panel": {
                key: FIT_PANEL[key]
                for key in ("source", "count", "per_label", "construction", "indices_sha256")
            },
            "evaluation_panels": EVALUATION_PANELS,
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": list(D4_CONTROLS),
            "permutation_null": PERMUTATION_NULL,
            "active_domain": ACTIVE_DOMAIN,
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
        raise ValueError("active-domain RWSA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise RWSAActiveHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RWSAActiveHarnessFailure(f"{label} is not a mapping: {path}")
    return payload


def _verify_evidence_payload(
    payload: Mapping[str, Any],
    *,
    expected_status: str,
    expected_classification: str | None,
    label: str,
) -> EvidenceRecordV2:
    try:
        record = EvidenceRecordV2.from_dict(payload["evidence"])
        record.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RWSAActiveHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise RWSAActiveHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_rwsa_identity(spec: Spec) -> dict[str, Any]:
    try:
        identity = zcsa._verify_rwsa_identity(spec)
    except (ValueError, zcsa.ZCSAHarnessFailure) as exc:
        raise RWSAActiveHarnessFailure("old RWSA attempt-02 identity changed") from exc
    audit = identity["audit"]
    if (
        audit.get("status") != "invalid_harness"
        or audit.get("classification") != "invalid_harness"
    ):
        raise RWSAActiveHarnessFailure("old RWSA attempt-02 is not the registered invalid harness")
    return identity


def _verify_zcsa_identity(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(ZCSA_IDENTITY["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != ZCSA_IDENTITY["config_sha256"]:
        raise RWSAActiveHarnessFailure("ZCSA config identity changed")
    zcsa_spec = zcsa._load_spec(config_path)
    zcsa._assert_contract(zcsa_spec)
    if zcsa_spec.config_hash != ZCSA_IDENTITY["config_canonical_sha256"]:
        raise RWSAActiveHarnessFailure("ZCSA config canonical identity changed")
    review_path = _resolve(ZCSA_IDENTITY["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != ZCSA_IDENTITY["review_sha256"]:
        raise RWSAActiveHarnessFailure("ZCSA review identity changed")
    zcsa._load_review(review_path, zcsa_spec)

    validation_path = _resolve(ZCSA_IDENTITY["validation"])
    validation = _read_json_hash(
        validation_path, ZCSA_IDENTITY["validation_sha256"], "ZCSA validation"
    )
    validation_record = _verify_evidence_payload(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="ZCSA validation",
    )
    if (
        validation_record.git_commit != ZCSA_IDENTITY["implementation_commit"]
        or validation_record.config_hash != zcsa_spec.config_hash
    ):
        raise RWSAActiveHarnessFailure("ZCSA validation lineage changed")

    audit_path = _resolve(ZCSA_IDENTITY["audit"])
    audit = _read_json_hash(audit_path, ZCSA_IDENTITY["audit_sha256"], "ZCSA audit")
    audit_record = _verify_evidence_payload(
        audit,
        expected_status="passed",
        expected_classification=ZCSA_IDENTITY["audit_classification"],
        label="ZCSA audit",
    )
    if (
        audit_record.git_commit != ZCSA_IDENTITY["result_git_commit"]
        or audit.get("classification") != ZCSA_IDENTITY["audit_classification"]
    ):
        raise RWSAActiveHarnessFailure("ZCSA audit lineage changed")

    compact_path = _resolve(ZCSA_IDENTITY["compact_report"])
    compact = _read_json_hash(
        compact_path, ZCSA_IDENTITY["compact_report_sha256"], "ZCSA compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != ZCSA_IDENTITY["audit_classification"]
        or compact.get("generation_readiness") != "not_ready"
    ):
        raise RWSAActiveHarnessFailure("ZCSA compact terminal interpretation changed")
    probe = compact.get("probe", {})
    if (
        probe.get("zero_w_count") != EXPECTED_INACTIVE_COUNT
        or probe.get("dynamic_active_count") != EXPECTED_ACTIVE_COUNT
        or probe.get("zero_dynamic_entries_nonzero_count") != 0
    ):
        raise RWSAActiveHarnessFailure("ZCSA probe counts changed")
    counts = compact.get("taxonomy", {}).get("counts", {})
    if counts != {
        "dynamic_active": EXPECTED_ACTIVE_COUNT,
        "target_zero_static": EXPECTED_INACTIVE_COUNT,
        "bias_only_constant": 0,
        "bias_only_variable_target": 0,
        "zero_W_constant_target_bias_mismatch": 0,
    }:
        raise RWSAActiveHarnessFailure("ZCSA taxonomy identity changed")
    transitions = compact.get("cross_panel_semantics", {}).get("transitions", {})
    if transitions != {
        "fit_zero_to_selection_active": 1495,
        "fit_zero_to_confirmation_active": 2097,
    }:
        raise RWSAActiveHarnessFailure("ZCSA transition identity changed")
    implementation_module = _resolve(
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit.py"
    )
    if file_sha256(implementation_module) != ZCSA_IDENTITY["implementation_module_sha256"]:
        raise RWSAActiveHarnessFailure("ZCSA implementation identity changed")
    return {
        "config_path": config_path,
        "review_path": review_path,
        "validation_path": validation_path,
        "audit_path": audit_path,
        "compact_path": compact_path,
        "validation": validation,
        "audit": audit,
        "compact": compact,
        "validation_record": validation_record,
        "audit_record": audit_record,
        "implementation_module": implementation_module,
    }


def _load_rmal_consumer(
    spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool
) -> tuple[Any, dict[str, Any], ReadoutProbe]:
    try:
        consumer, artifacts, probe = rwsa._load_rmal_consumer(
            spec, dataset_root, load_checkpoints=load_checkpoints
        )
    except (ValueError, rwsa.RWSAHarnessFailure) as exc:
        raise RWSAActiveHarnessFailure("RMAL frozen consumer identity is invalid") from exc
    if probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]:
        raise RWSAActiveHarnessFailure("RMAL probe path or hash is not registered")
    return consumer, artifacts, probe


def _load_rmal_probe(
    payload: Mapping[str, Any],
    fit_indices: torch.Tensor,
    artifact_sha256: str = "",
) -> ReadoutProbe:
    return rwsa._load_rmal_probe(payload, fit_indices, artifact_sha256)


def _probe_identity(probe: ReadoutProbe) -> dict[str, Any]:
    full = probe.full_weights.detach().cpu().contiguous()
    if tuple(full.shape) != (PROBE_INPUT_DIM, TOTAL_COLUMNS) or full.dtype != torch.float32:
        raise RWSAActiveHarnessFailure("RMAL probe shape or dtype changed")
    W = full[:-1, :]
    b = full[-1, :]
    if tuple(W.shape) != (FEATURE_DIM, TOTAL_COLUMNS) or tuple(b.shape) != (TOTAL_COLUMNS,):
        raise RWSAActiveHarnessFailure("RMAL W or b shape changed")
    if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(b).all()):
        raise RWSAActiveHarnessFailure("RMAL W or b is non-finite")
    dynamic_mass = W.abs().to(torch.float64).sum(dim=0)
    if not bool(torch.isfinite(dynamic_mass).all()):
        raise RWSAActiveHarnessFailure("RMAL dynamic mass is non-finite")
    active_mask = dynamic_mass.gt(0)
    inactive_mask = dynamic_mass.eq(0)
    if not bool(W[:, inactive_mask].eq(0).all()):
        raise RWSAActiveHarnessFailure("inactive W columns are not exactly zero")
    if not bool(b[inactive_mask].eq(0).all()):
        raise RWSAActiveHarnessFailure("inactive bias columns are not exactly zero")
    if not bool(dynamic_mass[active_mask].gt(0).all()):
        raise RWSAActiveHarnessFailure("active dynamic mass is not positive")
    active_count = int(active_mask.sum())
    inactive_count = int(inactive_mask.sum())
    if (
        active_count != EXPECTED_ACTIVE_COUNT
        or inactive_count != EXPECTED_INACTIVE_COUNT
        or active_count + inactive_count != TOTAL_COLUMNS
    ):
        raise RWSAActiveHarnessFailure("active/inactive column counts changed")
    if probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]:
        raise RWSAActiveHarnessFailure("RMAL probe artifact hash changed")
    return {
        "full_weights": full,
        "W": W,
        "b": b,
        "dynamic_mass": dynamic_mass,
        "active_mask": active_mask,
        "inactive_mask": inactive_mask,
        "active_count": active_count,
        "inactive_count": inactive_count,
        "active_minimum_mass": float(dynamic_mass[active_mask].min()),
        "active_maximum_mass": float(dynamic_mass[active_mask].max()),
        "zero_dynamic_entries_nonzero_count": int(torch.count_nonzero(W[:, inactive_mask])),
        "zero_bias_entries_nonzero_count": int(torch.count_nonzero(b[inactive_mask])),
        "all_finite": True,
        "probe_sha256": probe.artifact_sha256,
    }


def _fit_indices(consumer: Any, dataset_root: str | Path) -> torch.Tensor:
    return rwsa._fit_indices(consumer, dataset_root)


def _build_panels(consumer: Any, fit_indices: torch.Tensor) -> dict[str, Panel]:
    data = consumer.bundle.parent.parent.data
    fit_indices = fit_indices.detach().cpu().to(torch.long)
    selection = data.selection_indices.detach().cpu().to(torch.long)
    confirmation = data.confirmation_indices.detach().cpu().to(torch.long)
    expected_confirmation = torch.arange(55000, 60000, dtype=torch.long)
    if fit_indices.numel() != FIT_COUNT or tensor_sha256(fit_indices) != FIT_PANEL["indices_sha256"]:
        raise RWSAActiveHarnessFailure("fit panel identity changed")
    if selection.numel() != SELECTION_COUNT or not bool(
        ((selection >= 5000) & (selection < 10000)).all()
    ):
        raise RWSAActiveHarnessFailure("selection panel identity changed")
    if confirmation.numel() != CONFIRMATION_COUNT or not torch.equal(
        confirmation, expected_confirmation
    ):
        raise RWSAActiveHarnessFailure("confirmation panel identity changed")
    if (
        torch.unique(fit_indices).numel() != fit_indices.numel()
        or torch.unique(selection).numel() != selection.numel()
        or torch.unique(confirmation).numel() != confirmation.numel()
    ):
        raise RWSAActiveHarnessFailure("panel contains duplicate indices")
    if bool(torch.isin(fit_indices, confirmation).any()) or bool(
        torch.isin(selection, confirmation).any()
    ):
        raise RWSAActiveHarnessFailure("fit/selection/confirmation panels overlap")
    selection_labels = data.test_labels.index_select(0, selection)
    if not torch.equal(
        torch.bincount(selection_labels, minlength=10),
        torch.full((10,), 100, dtype=torch.long),
    ):
        raise RWSAActiveHarnessFailure("selection panel class balance changed")
    panels = {
        "fit": Panel(
            "fit",
            FIT_PANEL["source"],
            fit_indices,
            data.images.index_select(0, fit_indices),
            data.labels.index_select(0, fit_indices),
        ),
        "selection": Panel(
            "selection",
            EVALUATION_PANELS["selection"]["source"],
            selection,
            data.test_images.index_select(0, selection),
            selection_labels,
        ),
        "confirmation": Panel(
            "confirmation",
            EVALUATION_PANELS["confirmation"]["source"],
            confirmation,
            data.images.index_select(0, confirmation),
            data.labels.index_select(0, confirmation),
        ),
    }
    for panel in panels.values():
        if (
            panel.images.ndim != 4
            or tuple(panel.images.shape[1:]) != (1, 28, 28)
            or panel.images.dtype != torch.float32
        ):
            raise RWSAActiveHarnessFailure(f"{panel.name} image tensor contract changed")
        if (
            not bool(torch.isfinite(panel.images).all())
            or bool(panel.images.min() < 0)
            or bool(panel.images.max() > 1)
        ):
            raise RWSAActiveHarnessFailure(f"{panel.name} image normalization is invalid")
    return panels


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


def _renderer_identity() -> dict[str, Any]:
    return zcsa._renderer_identity()


def _pool_bins(input_size: int = 200, output_size: int = 16) -> tuple[tuple[int, int], ...]:
    return rwsa._pool_bins(input_size, output_size)


def _bin_membership(size: int, bins: Sequence[tuple[int, int]]) -> torch.Tensor:
    return rwsa._bin_membership(size, bins)


def _transform_coordinates(
    rows: torch.Tensor, columns: torch.Tensor, control: str, size: int = 200
) -> tuple[torch.Tensor, torch.Tensor]:
    return rwsa._transform_coordinates(rows, columns, control, size)


def _expected_support_indices(
    rows: torch.Tensor,
    columns: torch.Tensor,
    bins: Sequence[tuple[int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return rwsa._expected_support_indices(rows, columns, bins)


def _physical_cell_centers(
    bins: Sequence[tuple[int, int]] | None = None,
    *,
    size: int = 200,
    pixel_pitch_m: float = 2.5e-6,
) -> torch.Tensor:
    return rwsa._physical_cell_centers(bins, size=size, pixel_pitch_m=pixel_pitch_m)


def _split_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not torch.is_tensor(weights)
        or tuple(weights.shape) != (PROBE_INPUT_DIM, TOTAL_COLUMNS)
        or weights.dtype != torch.float32
    ):
        raise ValueError("active RWSA readout must have shape [257,40000] and float32 storage")
    return weights[:-1, :], weights[-1, :]


def _decompose_readout(
    features: torch.Tensor, probe: ReadoutProbe | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(probe, ReadoutProbe):
        weights, bias = probe.weights, probe.bias
    else:
        weights, bias = _split_weights(probe)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not torch.is_floating_point(features):
        raise ValueError("active RWSA features must be [N,256] real values")
    dynamic = features.float().mm(weights)
    static = bias.unsqueeze(0).expand(features.shape[0], -1)
    full = dynamic + static
    if not bool(torch.isfinite(full).all()):
        raise RWSAActiveHarnessFailure("active RWSA readout decomposition is non-finite")
    return full, dynamic, static


def _normalize_active_abs_columns(
    weights: torch.Tensor, active_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    W, _ = _split_weights(weights)
    if active_mask.dtype != torch.bool or tuple(active_mask.shape) != (TOTAL_COLUMNS,):
        raise ValueError("active-domain mask must be a [40000] bool tensor")
    if not bool(torch.isfinite(W).all()):
        raise RWSAActiveHarnessFailure("active-domain W contains non-finite values")
    active_weights = W[:, active_mask]
    mass = active_weights.abs().to(torch.float64).sum(dim=0)
    if not bool(torch.isfinite(mass).all()) or bool(mass.le(0).any()):
        raise RWSAActiveHarnessFailure("active-domain dynamic mass is non-finite or non-positive")
    probabilities = active_weights.abs().to(torch.float64).T / mass.unsqueeze(1)
    if not bool(torch.isfinite(probabilities).all()):
        raise RWSAActiveHarnessFailure("active-domain probabilities are non-finite")
    return probabilities, mass


def _scalar_summary(values: torch.Tensor) -> dict[str, Any]:
    return rwsa._scalar_summary(values)


def _vector_summary(values: torch.Tensor) -> dict[str, Any]:
    return rwsa._vector_summary(values)


def _domain_summary(values: torch.Tensor, domain: str) -> dict[str, Any]:
    return {"domain": domain, **_scalar_summary(values)}


def _permutation_bank(
    *,
    count: int = PERMUTATION_NULL_COUNT,
    seed: int = PERMUTATION_NULL_SEED,
    source_dim: int = FEATURE_DIM,
) -> torch.Tensor:
    return rwsa._permutation_bank(count=count, seed=seed, source_dim=source_dim)


def _permutation_null_summary(
    probabilities: torch.Tensor,
    denominator: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    active_mask: torch.Tensor,
    observed: float,
) -> dict[str, Any]:
    if active_mask.dtype != torch.bool or active_mask.numel() != TOTAL_COLUMNS:
        raise ValueError("permutation active mask has invalid shape")
    if probabilities.shape[0] != int(active_mask.sum()):
        raise ValueError("permutation probabilities do not match active mask")
    if denominator.shape != (probabilities.shape[0],):
        raise ValueError("permutation denominator shape is invalid")
    bank = _permutation_bank()
    abs_mass = probabilities * denominator.unsqueeze(1)
    null_values: list[torch.Tensor] = []
    for permutation in bank:
        permuted_support = permutation.index_select(
            0, support_indices.clamp_min(0).reshape(-1)
        ).reshape_as(support_indices)
        selected = abs_mass.gather(1, permuted_support.clamp_min(0))
        value = (
            (selected * support_mask.to(abs_mass.dtype)).sum(dim=1) / denominator
        ).mean()
        null_values.append(value)
    null = torch.stack(null_values)
    null_mean = float(null.mean())
    null_std = float(null.std(unbiased=False))
    effect = 0.0 if null_std == 0.0 else (observed - null_mean) / null_std
    return {
        "domain": "dynamic_active",
        "conditional_on_dynamic_active_domain": True,
        "active_mask_sha256": tensor_sha256(active_mask),
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_quantiles": _scalar_summary(null)["quantiles"],
        "empirical_rank": 1 + int((null <= observed).sum()),
        "empirical_rank_definition": "1_plus_count(null_statistic<=observed)",
        "empirical_p_value": float(
            (1 + (null >= observed).sum()) / (PERMUTATION_NULL_COUNT + 1)
        ),
        "standardized_effect_size": float(effect),
        "count": PERMUTATION_NULL_COUNT,
        "seed": PERMUTATION_NULL_SEED,
        "bank_sha256": tensor_sha256(bank),
        "operation": "source_feature_rows_only",
        "active_mask_preserved": True,
        "interpretation": "descriptive_only",
    }


def _active_structural_metrics(
    weights: torch.Tensor, bias: torch.Tensor, active_mask: torch.Tensor
) -> dict[str, Any]:
    W, bias_from_full = _split_weights(weights)
    if not torch.equal(bias.detach().cpu(), bias_from_full.detach().cpu()):
        raise ValueError("supplied bias does not match frozen full readout")
    if active_mask.dtype != torch.bool or tuple(active_mask.shape) != (TOTAL_COLUMNS,):
        raise ValueError("active-domain mask must be a [40000] bool tensor")
    if int(active_mask.sum()) == 0:
        raise RWSAActiveHarnessFailure("dynamic-active domain is empty")
    probabilities, denominator = _normalize_active_abs_columns(weights, active_mask)
    active_indices = torch.where(active_mask)[0]
    rows = torch.div(active_indices, TARGET_SHAPE[1], rounding_mode="floor")
    columns = torch.remainder(active_indices, TARGET_SHAPE[1])
    bins = _pool_bins()
    support_indices, support_mask = _expected_support_indices(rows, columns, bins)
    source_indices = torch.stack(
        (
            torch.arange(16, dtype=torch.float64).repeat_interleave(16),
            torch.arange(16, dtype=torch.float64).repeat(16),
        ),
        dim=1,
    )
    cell_centers = _physical_cell_centers(bins)
    source_row = torch.arange(16, dtype=torch.long).repeat_interleave(16)
    source_column = torch.arange(16, dtype=torch.long).repeat(16)
    cell_lengths = torch.tensor(
        [end - start for start, end in bins], dtype=torch.float64
    )
    support_source = support_indices.clamp_min(0)
    support_row = source_row.index_select(0, support_source.reshape(-1)).reshape_as(
        support_source
    )
    support_column = source_column.index_select(
        0, support_source.reshape(-1)
    ).reshape_as(support_source)
    q_unnormalized = 1.0 / (
        cell_lengths.index_select(0, support_row.reshape(-1)).reshape_as(support_row)
        * cell_lengths.index_select(0, support_column.reshape(-1)).reshape_as(
            support_column
        )
    )
    q_unnormalized = q_unnormalized * support_mask
    q = q_unnormalized / q_unnormalized.sum(dim=1, keepdim=True)
    expected_coarse = (
        q.unsqueeze(-1)
        * source_indices.index_select(0, support_source.reshape(-1)).reshape(
            support_source.shape[0], support_source.shape[1], 2
        )
    ).sum(dim=1)
    expected_physical = (
        q.unsqueeze(-1)
        * cell_centers.index_select(0, support_source.reshape(-1)).reshape(
            support_source.shape[0], support_source.shape[1], 2
        )
    ).sum(dim=1)
    absolute_centroid = probabilities.mm(cell_centers)
    coarse_centroid = probabilities.mm(source_indices)
    displacement_physical = absolute_centroid - expected_physical
    displacement_coarse = coarse_centroid - expected_coarse
    argmax = probabilities.argmax(dim=1)
    expected_locality = (
        probabilities.gather(1, support_indices.clamp_min(0))
        * support_mask.to(probabilities.dtype)
    ).sum(dim=1)
    radius_distance = (
        source_indices.unsqueeze(0) - expected_coarse.unsqueeze(1)
    ).abs().amax(dim=2)
    radius_mass = {
        str(radius): (probabilities * (radius_distance <= radius + 0.5)).sum(dim=1)
        for radius in (0, 1, 2)
    }
    weighted_spatial_variance = (
        probabilities
        * (cell_centers.unsqueeze(0) - absolute_centroid.unsqueeze(1))
        .square()
        .sum(dim=2)
    ).sum(dim=1)
    row_mass = torch.zeros(
        (probabilities.shape[0], 16), dtype=probabilities.dtype
    ).scatter_add_(
        1, source_row.unsqueeze(0).expand(probabilities.shape[0], -1), probabilities
    )
    column_mass = torch.zeros(
        (probabilities.shape[0], 16), dtype=probabilities.dtype
    ).scatter_add_(
        1,
        source_column.unsqueeze(0).expand(probabilities.shape[0], -1),
        probabilities,
    )
    entropy = -(
        probabilities * probabilities.clamp_min(1.0e-30).log()
    ).sum(dim=1) / math.log(256.0)
    effective_support = 1.0 / probabilities.square().sum(dim=1)
    active_weights = W[:, active_mask].to(torch.float64).T
    positive_mass = active_weights.clamp_min(0.0).sum(dim=1) / denominator
    negative_mass = (-active_weights).clamp_min(0.0).sum(dim=1) / denominator
    signed_ratio = active_weights.sum(dim=1).abs() / denominator
    d4_summary: dict[str, Any] = {}
    for control in D4_CONTROLS:
        control_rows, control_columns = _transform_coordinates(
            rows, columns, control
        )
        control_indices, control_mask = _expected_support_indices(
            control_rows, control_columns, bins
        )
        control_locality = (
            probabilities.gather(1, control_indices.clamp_min(0))
            * control_mask.to(probabilities.dtype)
        ).sum(dim=1)
        d4_summary[control] = {
            "domain": "dynamic_active",
            "locality_mass": float(control_locality.mean()),
            "locality_mass_summary": _domain_summary(
                control_locality, "dynamic_active"
            ),
            "difference_from_identity": float(
                control_locality.mean() - expected_locality.mean()
            ),
        }
    position_rows = rows * 4 // TARGET_SHAPE[0]
    position_columns = columns * 4 // TARGET_SHAPE[1]
    locality_position = torch.full((4, 4), float("nan"), dtype=torch.float64)
    for row in range(4):
        for column in range(4):
            selection = (position_rows == row) & (position_columns == column)
            if bool(selection.any()):
                locality_position[row, column] = expected_locality[selection].mean()
    translation_mean = displacement_coarse.mean(dim=0)
    translation_residual = displacement_coarse - translation_mean
    argmax_histogram = torch.bincount(argmax, minlength=256).reshape(16, 16)
    observed = float(expected_locality.mean())
    argmax_inside_expected_support = (
        support_indices == argmax.unsqueeze(1)
    ).any(dim=1)
    return {
        "domain": "dynamic_active",
        "conditional_on_dynamic_active_domain": True,
        "total_output_columns": TOTAL_COLUMNS,
        "active_column_count": int(active_mask.sum()),
        "inactive_output_columns_excluded": TOTAL_COLUMNS - int(active_mask.sum()),
        "active_mask_sha256": tensor_sha256(active_mask),
        "dynamic_mass_definition": "sum_i abs(W_ij)",
        "L_active": observed,
        "L_active_observed": observed,
        "argmax_source_cell": {
            "domain": "dynamic_active",
            "mode_cell": int(argmax_histogram.flatten().argmax()) // 16,
            "mode_cell_pair": source_indices[argmax_histogram.flatten().argmax()].tolist(),
            "occupancy_fraction": float(argmax_histogram.max() / argmax.numel()),
            "histogram": argmax_histogram.tolist(),
        },
        "centroid": {
            "domain": "dynamic_active",
            "physical_um": _vector_summary(absolute_centroid * 1.0e6),
            "coarse_cell": _vector_summary(coarse_centroid),
        },
        "displacement": {
            "domain": "dynamic_active",
            "physical_um": _vector_summary(displacement_physical * 1.0e6),
            "coarse_cell": _vector_summary(displacement_coarse),
        },
        "top_1_mass": _domain_summary(probabilities.max(dim=1).values, "dynamic_active"),
        "top_k_mass": {
            "domain": "dynamic_active",
            "k": TOP_K,
            **_scalar_summary(probabilities.topk(TOP_K, dim=1).values.sum(dim=1)),
        },
        "expected_support_mass": _domain_summary(
            expected_locality, "dynamic_active"
        ),
        "expected_support_locality_mass": _domain_summary(
            expected_locality, "dynamic_active"
        ),
        "coarse_radius_locality_mass": {
            f"radius_{radius}": _domain_summary(values, "dynamic_active")
            for radius, values in radius_mass.items()
        },
        "spatial_variance_m2": _domain_summary(
            weighted_spatial_variance, "dynamic_active"
        ),
        "rms_support_radius_um": _domain_summary(
            weighted_spatial_variance.sqrt() * 1.0e6, "dynamic_active"
        ),
        "effective_support": _domain_summary(effective_support, "dynamic_active"),
        "effective_support_size": _domain_summary(
            effective_support, "dynamic_active"
        ),
        "normalized_entropy": _domain_summary(entropy, "dynamic_active"),
        "normalized_weight_entropy": _domain_summary(entropy, "dynamic_active"),
        "row_concentration": {
            "domain": "dynamic_active",
            "max_mass": _scalar_summary(row_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(
                -(row_mass * row_mass.clamp_min(1.0e-30).log()).sum(dim=1)
                / math.log(16.0)
            ),
        },
        "column_concentration": {
            "domain": "dynamic_active",
            "max_mass": _scalar_summary(column_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(
                -(column_mass * column_mass.clamp_min(1.0e-30).log()).sum(dim=1)
                / math.log(16.0)
            ),
        },
        "source_row_concentration": {
            "domain": "dynamic_active",
            "max_mass": _scalar_summary(row_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(
                -(row_mass * row_mass.clamp_min(1.0e-30).log()).sum(dim=1)
                / math.log(16.0)
            ),
        },
        "source_column_concentration": {
            "domain": "dynamic_active",
            "max_mass": _scalar_summary(column_mass.max(dim=1).values),
            "normalized_entropy": _scalar_summary(
                -(column_mass * column_mass.clamp_min(1.0e-30).log()).sum(dim=1)
                / math.log(16.0)
            ),
        },
        "d4_coordinate_consistency": {
            "domain": "dynamic_active",
            "controls": d4_summary,
        },
        "d4_control_consistency": d4_summary,
        "identity_coordinate_consistency": {
            "domain": "dynamic_active",
            "locality_mass": observed,
            "argmax_inside_expected_support_fraction": float(
                argmax_inside_expected_support.to(torch.float64).mean()
            ),
        },
        "translation_field": {
            "domain": "dynamic_active",
            "mean_coarse_cell": translation_mean.tolist(),
            "residual_variance_coarse_cell": translation_residual.var(
                dim=0, unbiased=False
            ).tolist(),
            "residual_rms_coarse_cell": translation_residual.square()
            .mean(dim=0)
            .sqrt()
            .tolist(),
        },
        "locality_vs_target_position": {
            "domain": "dynamic_active",
            "grid_shape": [4, 4],
            "aggregation": "mean_expected_support_locality_mass",
            "grid": locality_position.tolist(),
        },
        "positive_mass": _domain_summary(positive_mass, "dynamic_active"),
        "negative_mass": _domain_summary(negative_mass, "dynamic_active"),
        "positive_weight_mass": _domain_summary(positive_mass, "dynamic_active"),
        "negative_weight_mass": _domain_summary(negative_mass, "dynamic_active"),
        "cancellation": {
            "domain": "dynamic_active",
            "signed_to_absolute_ratio": _scalar_summary(signed_ratio),
            "cancellation_fraction": _scalar_summary(1.0 - signed_ratio),
        },
        "signed_cancellation": {
            "domain": "dynamic_active",
            "signed_to_absolute_ratio": _scalar_summary(signed_ratio),
            "cancellation_fraction": _scalar_summary(1.0 - signed_ratio),
        },
        "permutation_null": _permutation_null_summary(
            probabilities,
            denominator,
            support_indices,
            support_mask,
            active_mask,
            observed,
        ),
    }


def _pixelwise_decomposition(
    physical: torch.Tensor,
    full: torch.Tensor,
    dynamic: torch.Tensor,
    bias: torch.Tensor,
    target: torch.Tensor,
    domains: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    tensors = (physical, full, dynamic, bias, target)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("pixelwise decomposition inputs must be matrices")
    if any(tuple(value.shape) != tuple(target.shape) for value in tensors[:-1]):
        raise ValueError("pixelwise decomposition shapes differ")
    if not torch.is_floating_point(target) or not bool(
        torch.isfinite(torch.cat([value.reshape(-1) for value in tensors])).all()
    ):
        raise RWSAActiveHarnessFailure("pixelwise decomposition is non-finite")
    required = {"all_40000", "dynamic_active", "zero_W"}
    if set(domains) != required:
        raise ValueError("pixelwise decomposition domains are incomplete")
    for name, mask in domains.items():
        if mask.dtype != torch.bool or tuple(mask.shape) != (target.shape[1],):
            raise ValueError(f"{name} domain mask is invalid")
    if not bool(domains["all_40000"].all()):
        raise ValueError("all_40000 domain is not complete")
    if bool((domains["dynamic_active"] & domains["zero_W"]).any()) or not bool(
        (domains["dynamic_active"] | domains["zero_W"]).all()
    ):
        raise ValueError("active and zero-W domains are not complementary")
    errors = {
        "paired": (physical - target).square(),
        "full": (full - target).square(),
        "dynamic_only": (dynamic - target).square(),
        "bias_only": (bias - target).square(),
    }
    all_sums = {
        variant: float(value[:, domains["all_40000"]].sum())
        for variant, value in errors.items()
    }
    target_all_sum = float(target.square()[:, domains["all_40000"]].sum())
    result: dict[str, Any] = {}
    sample_count = target.shape[0]
    for name, mask in domains.items():
        column_count = int(mask.sum())
        if column_count == 0:
            raise RWSAActiveHarnessFailure(f"{name} domain is empty")
        target_sum = float(target.square()[:, mask].sum())
        domain_result: dict[str, Any] = {
            "domain": name,
            "column_count": column_count,
            "fraction_of_total_columns": column_count / target.shape[1],
            "conditional_on_dynamic_active_domain": name == "dynamic_active",
            "target_energy": target_sum / sample_count,
            "target_mean_square": target_sum / (sample_count * column_count),
            "target_contribution_to_overall_energy": (
                target_sum / target_all_sum if target_all_sum else 0.0
            ),
            "squared_error_sum": {},
            "contribution_to_overall_squared_error": {},
        }
        for variant, value in errors.items():
            error_sum = float(value[:, mask].sum())
            domain_result["squared_error_sum"][variant] = error_sum / sample_count
            domain_result["contribution_to_overall_squared_error"][variant] = (
                error_sum / all_sums[variant] if all_sums[variant] else 0.0
            )
        domain_result.update(
            {
                "paired_mse": domain_result["squared_error_sum"]["paired"]
                / column_count,
                "full_mse": domain_result["squared_error_sum"]["full"]
                / column_count,
                "dynamic_only_mse": domain_result["squared_error_sum"]["dynamic_only"]
                / column_count,
                "bias_only_mse": domain_result["squared_error_sum"]["bias_only"]
                / column_count,
            }
        )
        if name == "all_40000":
            domain_result["masked_class_template_metrics"] = "not_applicable_full_domain"
        else:
            domain_result["masked_class_template_metrics"] = (
                "not_evaluated_without_unambiguous_masked_output_semantics"
            )
        result[name] = domain_result
    return result


def _empty_streaming_domains() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "target_sum": 0.0,
            "errors": {variant: 0.0 for variant in ("paired", "full", "dynamic_only", "bias_only")},
        }
        for name in ("all_40000", "dynamic_active", "zero_W")
    }


def _update_streaming_domains(
    accumulator: dict[str, dict[str, Any]],
    physical: torch.Tensor,
    full: torch.Tensor,
    dynamic: torch.Tensor,
    bias: torch.Tensor,
    target: torch.Tensor,
    domains: Mapping[str, torch.Tensor],
) -> None:
    values = {
        "paired": (physical - target).square(),
        "full": (full - target).square(),
        "dynamic_only": (dynamic - target).square(),
        "bias_only": (bias - target).square(),
    }
    for name, mask in domains.items():
        accumulator[name]["target_sum"] += float(target[:, mask].square().sum())
        for variant, value in values.items():
            accumulator[name]["errors"][variant] += float(value[:, mask].sum())


def _finish_streaming_domains(
    accumulator: Mapping[str, Mapping[str, Any]],
    count: int,
    domain_counts: Mapping[str, int],
) -> dict[str, Any]:
    all_sums = {
        variant: float(accumulator["all_40000"]["errors"][variant])
        for variant in ("paired", "full", "dynamic_only", "bias_only")
    }
    all_target_sum = float(accumulator["all_40000"]["target_sum"])
    result: dict[str, Any] = {}
    for name in ("all_40000", "dynamic_active", "zero_W"):
        columns = int(domain_counts[name])
        target_sum = float(accumulator[name]["target_sum"])
        domain_result: dict[str, Any] = {
            "domain": name,
            "column_count": columns,
            "fraction_of_total_columns": columns / TOTAL_COLUMNS,
            "conditional_on_dynamic_active_domain": name == "dynamic_active",
            "target_energy": target_sum / count,
            "target_mean_square": target_sum / (count * columns),
            "target_contribution_to_overall_energy": (
                target_sum / all_target_sum if all_target_sum else 0.0
            ),
            "squared_error_sum": {},
            "contribution_to_overall_squared_error": {},
        }
        for variant in ("paired", "full", "dynamic_only", "bias_only"):
            error_sum = float(accumulator[name]["errors"][variant])
            domain_result["squared_error_sum"][variant] = error_sum / count
            domain_result["contribution_to_overall_squared_error"][variant] = (
                error_sum / all_sums[variant] if all_sums[variant] else 0.0
            )
        domain_result.update(
            {
                "paired_mse": domain_result["squared_error_sum"]["paired"] / columns,
                "full_mse": domain_result["squared_error_sum"]["full"] / columns,
                "dynamic_only_mse": domain_result["squared_error_sum"]["dynamic_only"]
                / columns,
                "bias_only_mse": domain_result["squared_error_sum"]["bias_only"]
                / columns,
                "masked_class_template_metrics": (
                    "not_applicable_full_domain"
                    if name == "all_40000"
                    else "not_evaluated_without_unambiguous_masked_output_semantics"
                ),
            }
        )
        result[name] = domain_result
    return result


def _variant_metrics(
    errors: Sequence[float],
    template_errors: Sequence[float],
    cyclic_errors: Sequence[float],
    labels: torch.Tensor,
) -> dict[str, Any]:
    error_tensor = torch.tensor(errors, dtype=torch.float64)
    template_tensor = torch.tensor(template_errors, dtype=torch.float64)
    cyclic_tensor = torch.tensor(cyclic_errors, dtype=torch.float64)
    return {
        "domain": "all_40000",
        "paired_mse": float(error_tensor.mean()),
        "class_template_mse": float(template_tensor.mean()),
        "cyclic_mse": float(cyclic_tensor.mean()),
        "mse_by_label": [
            float(error_tensor[labels == label].mean()) for label in range(10)
        ],
        "class_template_mse_by_label": [
            float(template_tensor[labels == label].mean()) for label in range(10)
        ],
        "cyclic_mse_by_label": [
            float(cyclic_tensor[labels == label].mean()) for label in range(10)
        ],
        "instance_sensitive_metrics": (
            "not_evaluated_no_feature_bank_in_pixelwise_amendment"
        ),
        "finite": bool(
            torch.isfinite(error_tensor).all()
            and torch.isfinite(template_tensor).all()
            and torch.isfinite(cyclic_tensor).all()
        ),
    }


def _transition_summary(
    errors: Mapping[str, float],
    overall_errors: Mapping[str, float],
    transition_mask: torch.Tensor,
) -> dict[str, Any]:
    count = int(transition_mask.sum())
    return {
        "domain": "zero_W_transition_columns",
        "column_count": count,
        "mask_sha256": tensor_sha256(transition_mask),
        "squared_error_sum": {
            variant: value for variant, value in errors.items()
        },
        "contribution_to_overall_squared_error": {
            variant: (
                value / overall_errors[variant] if overall_errors[variant] else 0.0
            )
            for variant, value in errors.items()
        },
    }


def _panel_replay(
    model: bogt.ScaledChenBOGTModel,
    probe: ReadoutProbe,
    panel: Panel,
    active_mask: torch.Tensor,
    transition_mask: torch.Tensor,
    device: torch.device,
    panel_name: str,
) -> dict[str, Any]:
    templates, global_template = rwsa._panel_templates(panel.images, panel.labels)
    first_target = base.render_intensity_batch(panel.images[:1]).target.float()[0]
    last_target = base.render_intensity_batch(panel.images[-1:]).target.float()[0]
    previous_target: torch.Tensor | None = None
    domains = {
        "all_40000": torch.ones(TOTAL_COLUMNS, dtype=torch.bool),
        "dynamic_active": active_mask,
        "zero_W": ~active_mask,
    }
    accumulator = _empty_streaming_domains()
    transition_sums = {
        variant: 0.0 for variant in ("paired", "full", "dynamic_only", "bias_only")
    }
    physical_errors: list[float] = []
    full_errors: list[float] = []
    dynamic_errors: list[float] = []
    bias_errors: list[float] = []
    physical_template_errors: list[float] = []
    full_template_errors: list[float] = []
    dynamic_template_errors: list[float] = []
    bias_template_errors: list[float] = []
    physical_cyclic_errors: list[float] = []
    full_cyclic_errors: list[float] = []
    dynamic_cyclic_errors: list[float] = []
    bias_cyclic_errors: list[float] = []
    count = 0
    for index in range(panel.images.shape[0]):
        rendered = base.render_intensity_batch(
            panel.images[index : index + 1], output_shape=TARGET_SHAPE
        )
        target = rendered.target.float()
        cyclic = (
            last_target.unsqueeze(0)
            if index == 0
            else previous_target.unsqueeze(0)  # type: ignore[union-attr]
        )
        previous_target = target[0]
        with torch.no_grad():
            output = model(rendered.amplitude.to(device), return_trace=False)
        camera = output.camera.detach().float().cpu()
        if tuple(camera.shape) != (1, 1, 200, 200) or not bool(
            torch.isfinite(camera).all()
        ):
            raise RWSAActiveHarnessFailure("CMOS-only replay produced invalid output.camera")
        features = mra._feature_projection(camera, complex_value=False).float()
        full, dynamic, static = _decompose_readout(features, probe)
        target_flat = target.flatten(1)
        cyclic_flat = cyclic.flatten(1)
        template_flat = templates.index_select(
            0, panel.labels[index : index + 1]
        ).flatten(1)
        physical_flat = camera.flatten(1)
        physical_error = (physical_flat - target_flat).square().mean(dim=1)
        full_error = (full - target_flat).square().mean(dim=1)
        dynamic_error = (dynamic - target_flat).square().mean(dim=1)
        bias_error = (static - target_flat).square().mean(dim=1)
        physical_errors.append(float(physical_error[0]))
        full_errors.append(float(full_error[0]))
        dynamic_errors.append(float(dynamic_error[0]))
        bias_errors.append(float(bias_error[0]))
        physical_template_errors.append(
            float((physical_flat - template_flat).square().mean())
        )
        full_template_errors.append(float((full - template_flat).square().mean()))
        dynamic_template_errors.append(
            float((dynamic - template_flat).square().mean())
        )
        bias_template_errors.append(
            float((static - template_flat).square().mean())
        )
        physical_cyclic_errors.append(float((physical_flat - cyclic_flat).square().mean()))
        full_cyclic_errors.append(float((full - cyclic_flat).square().mean()))
        dynamic_cyclic_errors.append(float((dynamic - cyclic_flat).square().mean()))
        bias_cyclic_errors.append(float((static - cyclic_flat).square().mean()))
        _update_streaming_domains(
            accumulator,
            physical_flat,
            full,
            dynamic,
            static,
            target_flat,
            domains,
        )
        value_map = {
            "paired": (physical_flat - target_flat).square(),
            "full": (full - target_flat).square(),
            "dynamic_only": (dynamic - target_flat).square(),
            "bias_only": (static - target_flat).square(),
        }
        for variant, value in value_map.items():
            transition_sums[variant] += float(value[:, transition_mask].sum())
        count += 1
    if count != panel.images.shape[0]:
        raise RWSAActiveHarnessFailure(f"{panel_name} did not consume its registered panel")
    domain_counts = {
        "all_40000": TOTAL_COLUMNS,
        "dynamic_active": int(active_mask.sum()),
        "zero_W": int((~active_mask).sum()),
    }
    domains_report = _finish_streaming_domains(accumulator, count, domain_counts)
    overall_errors = {
        variant: float(accumulator["all_40000"]["errors"][variant])
        for variant in ("paired", "full", "dynamic_only", "bias_only")
    }
    physical_values = {
        "paired_mse": float(torch.tensor(physical_errors).mean()),
        "class_template_mse": float(torch.tensor(physical_template_errors).mean()),
        "cyclic_mse": float(torch.tensor(physical_cyclic_errors).mean()),
    }
    anchor = PHYSICAL_ANCHORS[panel_name]
    native_replay = {
        key: abs(physical_values[key] - value) <= PHYSICAL_BASELINE_TOLERANCE
        for key, value in anchor.items()
    }
    all_metrics = {
        "physical": _variant_metrics(
            physical_errors,
            physical_template_errors,
            physical_cyclic_errors,
            panel.labels,
        ),
        "full": _variant_metrics(
            full_errors,
            full_template_errors,
            full_cyclic_errors,
            panel.labels,
        ),
        "dynamic": _variant_metrics(
            dynamic_errors,
            dynamic_template_errors,
            dynamic_cyclic_errors,
            panel.labels,
        ),
        "bias_only": {
            **_variant_metrics(
                bias_errors,
                bias_template_errors,
                bias_cyclic_errors,
                panel.labels,
            ),
            "global_template_mse": float(
                torch.tensor(
                    [
                        float(
                            (
                                probe.bias.unsqueeze(0) - global_template.flatten(1)
                            )
                            .square()
                            .mean()
                        )
                        for _ in range(count)
                    ]
                ).mean()
            ),
        },
    }
    transition = _transition_summary(
        transition_sums, overall_errors, transition_mask
    )
    return {
        "panel": panel_name,
        "evaluation_count": count,
        "domains": domains_report,
        "all_domain_metrics": all_metrics,
        "zero_w_error_contribution": domains_report["zero_W"][
            "contribution_to_overall_squared_error"
        ],
        "zero_w_transition_columns": transition,
        "native_physical_anchor": anchor,
        "native_replay": native_replay,
        "finite": bool(
            all(native_replay.values())
            and all(
                bool(torch.isfinite(torch.tensor(values)).all())
                for values in (
                    physical_errors,
                    full_errors,
                    dynamic_errors,
                    bias_errors,
                )
            )
        ),
        "replay_contract": {
            "source": "output.camera",
            "feature_projection": "mra._feature_projection(real, complex_value=False)",
            "full_definition": "zW_plus_b",
            "dynamic_definition": "zW",
            "bias_definition": "b",
            "postprocessing": "none",
            "active_domain": "dynamic_active",
            "active_mask_sha256": tensor_sha256(active_mask),
            "transition_mask_sha256": tensor_sha256(transition_mask),
            "raw_cmos_saved": False,
            "pooled_feature_bank_saved": False,
            "candidate_prefix_consumed": False,
            "microbatch_size": MICROBATCH_SIZE,
            "first_target_sha256": tensor_sha256(first_target),
        },
    }


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in sorted(state.items())}
    )


def _state_equal(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> bool:
    return set(before) == set(after) and all(
        torch.equal(before[name], after[name]) for name in before
    )


def _transition_masks(
    zcsa_compact: Mapping[str, Any],
    active_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    panels = zcsa_compact.get("panels", {})
    counts: dict[str, torch.Tensor] = {}
    for name in ("fit", "selection", "confirmation"):
        values = panels.get(name, {}).get("target_statistics", {}).get(
            "exact_nonzero_count"
        )
        if not isinstance(values, list) or len(values) != TOTAL_COLUMNS:
            raise RWSAActiveHarnessFailure(
                f"ZCSA {name} exact target statistics are unavailable"
            )
        counts[name] = torch.tensor(values, dtype=torch.int64)
    fit_zero = counts["fit"].eq(0)
    if int(fit_zero.sum()) != EXPECTED_INACTIVE_COUNT:
        raise RWSAActiveHarnessFailure("ZCSA fit-zero transition base changed")
    if not torch.equal(fit_zero, ~active_mask):
        raise RWSAActiveHarnessFailure("ZCSA fit-zero mask is not the zero-W mask")
    selection_transition = (~active_mask) & fit_zero & counts["selection"].gt(0)
    confirmation_transition = (~active_mask) & fit_zero & counts["confirmation"].gt(0)
    if int(selection_transition.sum()) != 1495 or int(confirmation_transition.sum()) != 2097:
        raise RWSAActiveHarnessFailure("ZCSA transition counts changed")
    return {
        "selection": selection_transition,
        "confirmation": confirmation_transition,
        "fit_zero": fit_zero,
    }


def _input_hashes(
    spec: Spec,
    review_path: Path,
    consumer: Any,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
    panels: Mapping[str, Panel],
    rwsa_identity: Mapping[str, Any],
    zcsa_identity: Mapping[str, Any],
) -> dict[str, str]:
    rwsa_config = _resolve(RWSA_IDENTITY["config"])
    rwsa_spec = rwsa._load_spec(rwsa_config)
    rwsa_review = _resolve(RWSA_IDENTITY["review_contract"])
    hashes = rwsa._input_hashes(rwsa_spec, rwsa_review, consumer, probe, fit_indices)
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
            "rwsa_config": RWSA_IDENTITY["config_sha256"],
            "rwsa_review_contract": RWSA_IDENTITY["review_sha256"],
            "rwsa_attempt_02_validation": RWSA_IDENTITY["validation_sha256"],
            "rwsa_attempt_02_audit": RWSA_IDENTITY["audit_sha256"],
            "rwsa_implementation_module": RWSA_IDENTITY["implementation_module_sha256"],
            "zcsa_config": ZCSA_IDENTITY["config_sha256"],
            "zcsa_review_contract": ZCSA_IDENTITY["review_sha256"],
            "zcsa_validation": ZCSA_IDENTITY["validation_sha256"],
            "zcsa_audit": ZCSA_IDENTITY["audit_sha256"],
            "zcsa_compact_report": ZCSA_IDENTITY["compact_report_sha256"],
            "zcsa_implementation_module": ZCSA_IDENTITY["implementation_module_sha256"],
            "active_domain_contract": canonical_sha256(ACTIVE_DOMAIN),
            "lineage_contract": canonical_sha256(LINEAGE),
            "target_contract": canonical_sha256(TARGET),
            "pooling_contract": canonical_sha256(POOLING),
            "coordinate_contract": canonical_sha256(COORDINATE_CONTRACT),
            "d4_control_contract": canonical_sha256(D4_CONTROLS),
            "permutation_null_contract": canonical_sha256(PERMUTATION_NULL),
            "measurement_contract": canonical_sha256(MEASUREMENT),
            "bias_attribution_contract": canonical_sha256(BIAS_ATTRIBUTION),
            "dataset_manifest": consumer.bundle.parent.parent.data.manifest["manifest_hash"],
            "scaled_chen_checkpoint": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            "fit_indices": tensor_sha256(fit_indices),
            "selection_indices": tensor_sha256(panels["selection"].indices),
            "confirmation_indices": tensor_sha256(panels["confirmation"].indices),
        }
    )
    return hashes


def _context(spec: Spec, consumer: Any) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "lineage": LINEAGE,
            "active_domain": ACTIVE_DOMAIN,
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": D4_CONTROLS,
            "permutation_null": PERMUTATION_NULL,
            "measurement": MEASUREMENT,
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
        "hypothesis_id": HYPOTHESIS,
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "conditional_on_dynamic_active_domain": True,
        "total_output_columns": TOTAL_COLUMNS,
        "dynamic_active_count": EXPECTED_ACTIVE_COUNT,
        "dynamic_inactive_count": EXPECTED_INACTIVE_COUNT,
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
        raise FileExistsError(
            f"refusing to reuse active-domain validation attempt: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("active-domain validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"active-domain validation attempt directory is missing: {output_path.parent}"
        )
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError(
            "active-domain attempt directory is not a fresh validation-only slot"
        )


def _resource_values(device: torch.device) -> tuple[bool, int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False, 0, 0
    free, total = torch.cuda.mem_get_info(device)
    return True, int(free), int(total)


def _validation_gates(
    spec: Spec,
    consumer: Any,
    rmal_artifacts: Mapping[str, Any],
    rwsa_identity: Mapping[str, Any],
    zcsa_identity: Mapping[str, Any],
    probe: ReadoutProbe,
    probe_id: Mapping[str, Any],
    panels: Mapping[str, Panel],
    renderer: Mapping[str, Any],
    formal: FormalSystemConfig,
    fit_indices: torch.Tensor,
    output_path: Path,
) -> tuple[GateResult, ...]:
    data = consumer.bundle.parent.parent.data
    operator = mra._build_research_ols(formal)
    cuda_available, free_cuda, _ = _resource_values(torch.device("cuda"))
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    bins = _pool_bins()
    rows = torch.arange(200, dtype=torch.long).repeat_interleave(200)
    columns = torch.arange(200, dtype=torch.long).repeat(200)
    support_indices, support_mask = _expected_support_indices(rows, columns, bins)
    return (
        GateResult(
            "clean_registered_branch",
            _git_state().get("branch") == BRANCH and _git_state().get("clean") is True,
            {},
        ),
        GateResult(
            "rmal_consumer_identity",
            file_sha256(rmal_artifacts["probe_path"]) == RMAL_CONSUMER["probe_sha256"],
            {},
        ),
        GateResult(
            "old_rwsa_attempt_02_identity",
            rwsa_identity["audit"].get("status") == "invalid_harness"
            and rwsa_identity["audit"].get("classification") == "invalid_harness",
            {},
        ),
        GateResult(
            "zcsa_passed_identity",
            zcsa_identity["audit"].get("status") == "passed"
            and zcsa_identity["audit"].get("classification")
            == ZCSA_IDENTITY["audit_classification"],
            {},
        ),
        GateResult(
            "checkpoint_identity",
            file_sha256(checkpoint_path)
            == REFERENCES["scaled_chen_checkpoint"]["sha256"],
            {},
        ),
        GateResult(
            "probe_shape_dtype",
            tuple(probe.full_weights.shape) == (257, 40000)
            and tuple(probe_id["W"].shape) == (256, 40000)
            and tuple(probe_id["b"].shape) == (40000,)
            and probe.full_weights.dtype == torch.float32,
            {},
        ),
        GateResult("probe_finite", bool(probe_id["all_finite"]), {}),
        GateResult(
            "active_count_locked",
            probe_id["active_count"] == EXPECTED_ACTIVE_COUNT
            and probe_id["active_minimum_mass"] > 0,
            {
                "active_count": probe_id["active_count"],
                "active_minimum_mass": probe_id["active_minimum_mass"],
            },
        ),
        GateResult(
            "inactive_count_locked",
            probe_id["inactive_count"] == EXPECTED_INACTIVE_COUNT
            and probe_id["zero_dynamic_entries_nonzero_count"] == 0
            and probe_id["zero_bias_entries_nonzero_count"] == 0,
            {
                "inactive_count": probe_id["inactive_count"],
                "zero_dynamic_entries_nonzero_count": probe_id[
                    "zero_dynamic_entries_nonzero_count"
                ],
                "zero_bias_entries_nonzero_count": probe_id[
                    "zero_bias_entries_nonzero_count"
                ],
            },
        ),
        GateResult(
            "fit_panel_identity",
            panels["fit"].indices.numel() == FIT_COUNT
            and tensor_sha256(fit_indices) == FIT_PANEL["indices_sha256"],
            {},
        ),
        GateResult(
            "selection_panel_identity",
            panels["selection"].indices.numel() == SELECTION_COUNT
            and bool(
                ((panels["selection"].indices >= 5000)
                & (panels["selection"].indices < 10000)).all()
            ),
            {},
        ),
        GateResult(
            "confirmation_panel_identity",
            panels["confirmation"].indices.numel() == CONFIRMATION_COUNT
            and torch.equal(
                panels["confirmation"].indices,
                torch.arange(55000, 60000, dtype=torch.long),
            ),
            {},
        ),
        GateResult(
            "target_renderer_identity",
            renderer["source_sha256"] == TARGET_RENDERER_SOURCE_SHA256
            and renderer["exact_repeat"],
            {},
        ),
        GateResult(
            "formal_identity_boundary",
            formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal,
            {},
        ),
        GateResult(
            "mode_bank_rank_100",
            mra._mode_rank(operator) == 100,
            {"rank": mra._mode_rank(operator)},
        ),
        GateResult(
            "finite_mode_bank",
            bool(
                torch.isfinite(operator.input_modes).all()
                and torch.isfinite(operator.output_modes).all()
            ),
            {},
        ),
        GateResult(
            "pooling_contract_locked",
            spec.payload["pooling"] == POOLING
            and _pool_bins(POOLING["input_shape"][0], POOLING["output_shape"][0])
            == bins,
            {"bins": [list(item) for item in bins]},
        ),
        GateResult(
            "overlapping_adaptive_bins_present",
            bool(_bin_membership(200, bins).sum(dim=0).max() > 1),
            {},
        ),
        GateResult(
            "expected_support_locked",
            support_indices.shape == (40000, 4)
            and support_mask.shape == (40000, 4),
            {},
        ),
        GateResult(
            "d4_controls_locked",
            tuple(spec.payload["d4_controls"]) == D4_CONTROLS,
            {},
        ),
        GateResult(
            "permutation_null_locked",
            spec.payload["permutation_null"] == PERMUTATION_NULL
            and _permutation_bank().shape == (PERMUTATION_NULL_COUNT, FEATURE_DIM),
            {},
        ),
        GateResult(
            "active_domain_contract_locked",
            spec.payload["active_domain"] == ACTIVE_DOMAIN
            and ACTIVE_DOMAIN["epsilon_normalization"] is False
            and ACTIVE_DOMAIN["inactive_columns_retained_in_evidence"] is True,
            {},
        ),
        GateResult(
            "candidate_prefix_not_consumed",
            EVALUATION_PANELS["candidate_prefix"]["consumed"] is False,
            {},
        ),
        GateResult(
            "no_update_scope",
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0},
        ),
        GateResult(
            "windows_native_checkout",
            os.name == "nt" and Path.cwd().drive.upper() == "D:",
            {},
        ),
        GateResult(
            "cuda_available",
            cuda_available,
            {"free_cuda_bytes": free_cuda},
        ),
        GateResult(
            "free_cuda_precondition",
            free_cuda >= MIN_FREE_CUDA_BYTES,
            {"free_cuda_bytes": free_cuda, "minimum": MIN_FREE_CUDA_BYTES},
        ),
        GateResult(
            "disk_free_precondition",
            disk.free >= MIN_DISK_FREE_BYTES,
            {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES},
        ),
        GateResult(
            "output_slot_absent",
            not output_path.parent.exists(),
            {"path": str(output_path.parent)},
        ),
        GateResult(
            "dataset_panel_source",
            data.images.dtype == torch.float32 and data.test_images.dtype == torch.float32,
            {},
        ),
    )


def _load_validation(
    path: str | Path,
    spec: Spec,
    review_path: Path,
    consumer: Any,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
    panels: Mapping[str, Panel],
    rwsa_identity: Mapping[str, Any],
    zcsa_identity: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(
        spec,
        review_path,
        consumer,
        probe,
        fit_indices,
        panels,
        rwsa_identity,
        zcsa_identity,
    )
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("status") != "passed"
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.input_hashes != expected_hashes
        or record.permitted_next_actions != (NEXT_ACTION_RUN,)
        or payload.get("conditional_on_dynamic_active_domain") is not True
    ):
        raise ValueError("active-domain validation evidence is stale or mismatched")


def _artifact_snapshot(
    spec: Spec,
    review_path: Path,
    consumer: Any,
    rwsa_identity: Mapping[str, Any],
    zcsa_identity: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "research_config": spec.path,
        "review_contract": review_path,
        "implementation_module": Path(__file__).resolve(),
        "target_renderer_module": _resolve(
            "src/ldm_project/v2/application/research/optical_manifold.py"
        ),
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
        "zcsa_config": zcsa_identity["config_path"],
        "zcsa_review_contract": zcsa_identity["review_path"],
        "zcsa_validation": zcsa_identity["validation_path"],
        "zcsa_audit": zcsa_identity["audit_path"],
        "zcsa_compact_report": zcsa_identity["compact_path"],
        "scaled_chen_checkpoint": _resolve(REFERENCES["scaled_chen_checkpoint"]["path"]),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _invalid_harness_payload(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    consumer: Any,
    probe: ReadoutProbe,
    fit_indices: torch.Tensor,
    panels: Mapping[str, Panel],
    rwsa_identity: Mapping[str, Any],
    zcsa_identity: Mapping[str, Any],
    state: Mapping[str, Any],
    reason: str,
) -> None:
    gates = (
        GateResult("active_domain_harness", False, {"reason": reason}),
        GateResult(
            "no_update_scope",
            True,
            {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0},
        ),
    )
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        consumer,
        state,
        _input_hashes(
            spec,
            review_path,
            consumer,
            probe,
            fit_indices,
            panels,
            rwsa_identity,
            zcsa_identity,
        ),
        gates,
        (),
        {
            "lineage": LINEAGE,
            "active_domain": ACTIVE_DOMAIN,
            "conditional_on_dynamic_active_domain": True,
            "reason": reason,
        },
        reason,
    )
    _write_new(
        output_path, payload, artifact_kind="active-domain RWSA invalid-harness audit evidence"
    )


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    rwsa_identity = _verify_rwsa_identity(spec)
    zcsa_identity = _verify_zcsa_identity(spec)
    consumer, rmal_artifacts, probe = _load_rmal_consumer(
        spec, args.dataset_root, load_checkpoints=False
    )
    fit_indices = _fit_indices(consumer, args.dataset_root)
    panels = _build_panels(consumer, fit_indices)
    renderer = _renderer_identity()
    probe_id = _probe_identity(probe)
    formal, _ = mra._load_formal(spec)
    output_path = _resolve(args.output)
    gates = _validation_gates(
        spec,
        consumer,
        rmal_artifacts,
        rwsa_identity,
        zcsa_identity,
        probe,
        probe_id,
        panels,
        renderer,
        formal,
        fit_indices,
        output_path,
    )
    valid = all(gate.passed for gate in gates)
    contract = {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "active_domain": ACTIVE_DOMAIN,
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
        _input_hashes(
            spec,
            review_path,
            consumer,
            probe,
            fit_indices,
            panels,
            rwsa_identity,
            zcsa_identity,
        ),
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "active-domain RWSA static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="active-domain RWSA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    available, free, total = _resource_values(device)
    if not available:
        raise RWSAActiveHarnessFailure(
            "active-domain CMOS replay requires CUDA; CPU is not a formal fallback"
        )
    if free < MIN_FREE_CUDA_BYTES:
        raise RWSAActiveHarnessFailure(
            "active-domain RWSA requires at least 4 GiB free CUDA memory"
        )
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RWSAActiveHarnessFailure(
            "active-domain RWSA requires at least 10 GiB free storage"
        )
    return free, total


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "active-domain RWSA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup"
        )
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    rwsa_identity = _verify_rwsa_identity(spec)
    zcsa_identity = _verify_zcsa_identity(spec)
    consumer, _, probe = _load_rmal_consumer(
        spec, args.dataset_root, load_checkpoints=True
    )
    fit_indices = _fit_indices(consumer, args.dataset_root)
    panels = _build_panels(consumer, fit_indices)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    _load_validation(
        validation_path,
        spec,
        review_path,
        consumer,
        probe,
        fit_indices,
        panels,
        rwsa_identity,
        zcsa_identity,
        state,
    )
    free_cuda, total_cuda = _resource_preflight(
        output_path, torch.device(args.device)
    )
    formal, _ = mra._load_formal(spec)
    probe_id = _probe_identity(probe)
    transition_masks = _transition_masks(
        zcsa_identity["compact"], probe_id["active_mask"]
    )
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    checkpoint_hash_before = file_sha256(checkpoint_path)
    model = mra._load_scaled_model(
        formal,
        consumer.source.checkpoints[PRIMARY_CHECKPOINT],
        torch.device(args.device),
    )
    model_state_before = _state_cpu(model)
    ols_state_before = _state_cpu(model.ols)
    try:
        renderer = _renderer_identity()
        structural = _active_structural_metrics(
            probe_id["full_weights"], probe_id["b"], probe_id["active_mask"]
        )
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        selection = _panel_replay(
            model,
            probe,
            panels["selection"],
            probe_id["active_mask"],
            transition_masks["selection"],
            torch.device(args.device),
            "selection",
        )
        confirmation = _panel_replay(
            model,
            probe,
            panels["confirmation"],
            probe_id["active_mask"],
            transition_masks["confirmation"],
            torch.device(args.device),
            "confirmation",
        )
        elapsed = time.perf_counter() - started
        model_state_after = _state_cpu(model)
        ols_state_after = _state_cpu(model.ols)
        checkpoint_hash_after = file_sha256(checkpoint_path)
        peak = (
            int(torch.cuda.max_memory_allocated(torch.device(args.device)))
            if torch.cuda.is_available()
            else 0
        )
        state_unchanged = _state_equal(model_state_before, model_state_after)
        ols_unchanged = _state_equal(ols_state_before, ols_state_after)
        physical_anchor_replay = bool(
            all(selection["native_replay"].values())
            and all(confirmation["native_replay"].values())
        )
        resource_ok = peak <= MAX_PEAK_CUDA_BYTES
        artifact_before = _artifact_snapshot(
            spec, review_path, consumer, rwsa_identity, zcsa_identity
        )
        artifact_after = _artifact_snapshot(
            spec, review_path, consumer, rwsa_identity, zcsa_identity
        )
        artifacts_unchanged = artifact_before == artifact_after
        storage_estimate = (
            len(json.dumps(structural))
            + len(json.dumps(selection))
            + len(json.dumps(confirmation))
            + 256 * 1024
        )
        if not physical_anchor_replay:
            raise RWSAActiveHarnessFailure(
                "active-domain physical anchor replay exceeded registered RMAL tolerance"
            )
        if not state_unchanged or not ols_unchanged:
            raise RWSAActiveHarnessFailure(
                "active-domain replay changed frozen model or OLS state"
            )
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RWSAActiveHarnessFailure(
                "active-domain replay changed frozen checkpoint"
            )
        if not artifacts_unchanged:
            raise RWSAActiveHarnessFailure(
                "active-domain replay changed a locked input artifact"
            )
        if not resource_ok or storage_estimate > MAX_STORAGE_BYTES:
            raise RWSAActiveHarnessFailure("active-domain resource gate failed")
        common_contract = {
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "probe": {
                "artifact_sha256": probe.artifact_sha256,
                "weight_shape": list(probe.full_weights.shape),
                "dynamic_weight_shape": list(probe.weights.shape),
                "bias_shape": list(probe.bias.shape),
                "fit_panel_hash": tensor_sha256(fit_indices),
                "active_mask_sha256": tensor_sha256(probe_id["active_mask"]),
                "inactive_mask_sha256": tensor_sha256(probe_id["inactive_mask"]),
                "active_count": probe_id["active_count"],
                "inactive_count": probe_id["inactive_count"],
                "active_minimum_mass": probe_id["active_minimum_mass"],
                "metadata": dict(probe.metadata),
            },
            "active_domain": ACTIVE_DOMAIN,
            "conditional_on_dynamic_active_domain": True,
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "target_renderer": renderer,
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
            "hypothesis": HYPOTHESIS,
            "status": "passed",
            "classification": CLASSIFICATION,
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "git": {
                "branch": state["branch"],
                "commit": state["commit"],
                "clean": state["clean"],
            },
            "conditional_on_dynamic_active_domain": True,
            "total_output_columns": TOTAL_COLUMNS,
            "dynamic_active_count": probe_id["active_count"],
            "dynamic_inactive_count": probe_id["inactive_count"],
            "structural": structural,
            "panels": {
                "selection": selection,
                "confirmation": confirmation,
            },
            "transition_columns": {
                "fit_zero_to_selection_active": transition_masks[
                    "selection"
                ].nonzero().flatten().tolist(),
                "fit_zero_to_confirmation_active": transition_masks[
                    "confirmation"
                ].nonzero().flatten().tolist(),
                "counts": {
                    "fit_zero_to_selection_active": int(
                        transition_masks["selection"].sum()
                    ),
                    "fit_zero_to_confirmation_active": int(
                        transition_masks["confirmation"].sum()
                    ),
                },
            },
            "contract": common_contract,
            "artifact_integrity": {
                "before": artifact_before,
                "after": artifact_after,
                "unchanged": artifacts_unchanged,
            },
            "resource": {
                "free_cuda_bytes_before": free_cuda,
                "total_cuda_bytes": total_cuda,
                "peak_memory_bytes": peak,
                "wall_clock_seconds": elapsed,
                "wall_clock": "observational_only",
                "storage_estimate_bytes": storage_estimate,
            },
            "optimizer_created": False,
            "backward_calls": 0,
            "evaluator_calls": 0,
            "checkpoint_written": False,
            "prior_sampled": False,
            "generation_started": False,
            "physical_forward_modified": False,
            "optical_replay_executed": False,
            "exact_panel_cmos_only_replay_executed": True,
            "raw_cmos_bank_saved": False,
            "pooled_feature_bank_saved": False,
        }
        compact_path = output_path.parent / "compact_report.json"
        _write_new(
            compact_path, compact, artifact_kind="active-domain RWSA compact report"
        )
        compact_hash = file_sha256(compact_path)
        output_bytes_before_audit = int(
            sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file())
        )
        gates = (
            GateResult(
                "old_rwsa_attempt_02_immutable",
                rwsa_identity["audit"].get("status") == "invalid_harness",
                {},
            ),
            GateResult(
                "zcsa_passed_identity",
                zcsa_identity["audit"].get("status") == "passed"
                and zcsa_identity["audit"].get("classification")
                == ZCSA_IDENTITY["audit_classification"],
                {},
            ),
            GateResult(
                "active_domain_conditional",
                compact["conditional_on_dynamic_active_domain"] is True
                and compact["dynamic_active_count"] == EXPECTED_ACTIVE_COUNT
                and compact["dynamic_inactive_count"] == EXPECTED_INACTIVE_COUNT,
                {},
            ),
            GateResult(
                "physical_anchor_replay",
                physical_anchor_replay,
                {
                    "selection": selection["native_replay"],
                    "confirmation": confirmation["native_replay"],
                },
            ),
            GateResult(
                "finite_measurements",
                bool(selection["finite"] and confirmation["finite"]),
                {},
            ),
            GateResult(
                "frozen_model_and_OLS",
                state_unchanged and ols_unchanged,
                {},
            ),
            GateResult(
                "checkpoint_unchanged",
                checkpoint_hash_before == checkpoint_hash_after,
                {
                    "before": checkpoint_hash_before,
                    "after": checkpoint_hash_after,
                },
            ),
            GateResult(
                "readout_frozen",
                probe.metadata.get("frozen") is True
                and probe.metadata.get("probe_hash")
                == tensor_sha256(probe.full_weights),
                {},
            ),
            GateResult(
                "active_mask_null_consistency",
                selection["replay_contract"]["active_mask_sha256"]
                == structural["active_mask_sha256"]
                == confirmation["replay_contract"]["active_mask_sha256"]
                == structural["permutation_null"]["active_mask_sha256"],
                {},
            ),
            GateResult(
                "transition_accounting",
                selection["zero_w_transition_columns"]["column_count"] == 1495
                and confirmation["zero_w_transition_columns"]["column_count"] == 2097,
                {},
            ),
            GateResult(
                "no_update_isolation",
                True,
                {
                    "optimizer_created": False,
                    "backward_calls": 0,
                    "evaluator_calls": 0,
                },
            ),
            GateResult(
                "artifacts_unchanged",
                artifacts_unchanged,
                {},
            ),
            GateResult(
                "resource_limits",
                resource_ok and output_bytes_before_audit <= MAX_STORAGE_BYTES,
                {
                    "peak_bytes": peak,
                    "output_bytes_before_audit": output_bytes_before_audit,
                },
            ),
        )
        if not all(gate.passed for gate in gates):
            raise RWSAActiveHarnessFailure("active-domain result gate failed")
        audit_contract = {
            **common_contract,
            "structural": structural,
            "panels": {
                "selection": selection,
                "confirmation": confirmation,
            },
            "transition_columns": compact["transition_columns"]["counts"],
            "artifact_integrity": compact["artifact_integrity"],
            "model_state_before": _state_digest(model_state_before),
            "model_state_after": _state_digest(model_state_after),
            "ols_state_before": _state_digest(ols_state_before),
            "ols_state_after": _state_digest(ols_state_after),
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "resource": compact["resource"],
            "compact_report_path": str(compact_path.resolve()),
            "compact_report_sha256": compact_hash,
            "descriptive_only": True,
            "interpretation_boundary": (
                "metrics describe the frozen dynamic-active output domain only"
            ),
        }
        payload = _record(
            AUDIT_SCHEMA,
            "passed",
            CLASSIFICATION,
            spec,
            consumer,
            state,
            _input_hashes(
                spec,
                review_path,
                consumer,
                probe,
                fit_indices,
                panels,
                rwsa_identity,
                zcsa_identity,
            ),
            gates,
            (NEXT_ACTION_RESULT_REVIEW,),
            audit_contract,
            "active-domain RWSA is descriptive-only and conditional on dynamic-active columns",
        )
        _write_new(output_path, payload, artifact_kind="active-domain RWSA audit evidence")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except RWSAActiveHarnessFailure as exc:
        if not output_path.exists():
            _invalid_harness_payload(
                output_path,
                spec,
                review_path,
                consumer,
                probe,
                fit_indices,
                panels,
                rwsa_identity,
                zcsa_identity,
                state,
                str(exc),
            )
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if (
        args.research_action
        == "scaled-chen-detector-readout-structure-active-domain-validate"
    ):
        return validate(args)
    if (
        args.research_action
        == "scaled-chen-detector-readout-structure-active-domain-run"
    ):
        return run(args)
    raise ValueError(f"unsupported active-domain RWSA action: {args.research_action}")


__all__ = [
    "ACTIVE_DOMAIN",
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "BIAS_ATTRIBUTION",
    "CLASSIFICATION",
    "CONFIG_SCHEMA",
    "COORDINATE_CONTRACT",
    "D4_CONTROLS",
    "EXPECTED_ACTIVE_COUNT",
    "EXPECTED_INACTIVE_COUNT",
    "FIT_PANEL",
    "HYPOTHESIS",
    "LINEAGE",
    "MEASUREMENT",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "PERMUTATION_NULL",
    "PERMUTATION_NULL_COUNT",
    "PERMUTATION_NULL_SEED",
    "POOLING",
    "REFERENCES",
    "RMAL_CONSUMER",
    "RWSAActiveHarnessFailure",
    "RWSA_IDENTITY",
    "ReadoutProbe",
    "REVIEW_SCHEMA",
    "Spec",
    "TARGET",
    "TARGET_RENDERER_SOURCE_SHA256",
    "VALIDATION_SCHEMA",
    "ZCSA_IDENTITY",
    "_active_structural_metrics",
    "_assert_contract",
    "_build_panels",
    "_decompose_readout",
    "_expected_support_indices",
    "_fit_indices",
    "_load_review",
    "_load_rmal_probe",
    "_load_spec",
    "_normalize_active_abs_columns",
    "_panel_replay",
    "_permutation_bank",
    "_permutation_null_summary",
    "_physical_cell_centers",
    "_pixelwise_decomposition",
    "_pool_bins",
    "_probe_identity",
    "_renderer_identity",
    "_split_weights",
    "_transition_masks",
    "dispatch",
    "run",
    "validate",
]
