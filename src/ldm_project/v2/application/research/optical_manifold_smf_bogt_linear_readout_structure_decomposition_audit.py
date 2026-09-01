"""Frozen-W linear readout spectral and operator attribution.

This module is a diagnostic-only consumer of the registered RMAL detector
probe.  It performs a CPU float64 eigendecomposition of the two 256 by 256
Gram matrices induced by the immutable float32 dynamic readout.  It does not
construct an optical model, fit or refit a probe, run a decoder, or evaluate a
selection/confirmation panel.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

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
from . import optical_manifold_smf_bogt_detector_readout_structure_attribution as rwsa
from . import optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain as rwsa_active
from . import optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit as zcsa
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S6-LSDA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
BASELINE_HEAD = "bd967f64ffb046677fa4204e81e911b226047e6e"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
PROBE_INPUT_DIM = 257
SOURCE_SHAPE = (16, 16)
TARGET_SHAPE = (200, 200)
MODE_PREVIEW_K = 16
FIT_COUNT = 1000
EXPECTED_ACTIVE_COUNT = 33502
EXPECTED_INACTIVE_COUNT = 6498
TARGET_MICROBATCH_SIZE = 32
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_WORKING_SET_BYTES = 512 * 1024**2

CLASSIFICATION = "linear_readout_spectral_attribution_completed"
NEXT_ACTION_RUN = "run_scaled_chen_linear_readout_structure_decomposition"
NEXT_ACTION_RESULT_REVIEW = "manual_review_LSDA_spectral_result"

DATASET_MANIFEST_SHA256 = "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
ACTIVE_MASK_SHA256 = "afbda35eca136c1275c5771bc9540f7985dd66f77a4c6ea596940649848352ea"
INACTIVE_MASK_SHA256 = "98a3637529cbe9ca0c7f90311608e9ab7be3d3b060d46a419079093633412175"
TARGET_RENDERER_SOURCE_SHA256 = zcsa.TARGET_RENDERER_SOURCE_SHA256

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
RWSA_ACTIVE_IDENTITY = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
        "active_domain_v1.yaml"
    ),
    "config_sha256": "f5ae22615259888c1cb64b89dc11005d99c5412285461889118fed041cf2d18f",
    "config_canonical_sha256": "2115380c676fc710470429abed9438d154a1758db99e5635d6295eeaae03a25d",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_readout_structure_attribution_"
        "active_domain_review_v1.json"
    ),
    "review_sha256": "77ceffe03f135cbf20c81a164b353242d5a755bbd104a7dd22becbd2d4cb4b9f",
    "validation": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_02/"
        "validation.json"
    ),
    "validation_sha256": "686d99f331c99b30a73cb3a70cc1002aa0a21a33b014e2a59a5e6c33f8a5e15d",
    "audit": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_02/"
        "audit.json"
    ),
    "audit_sha256": "cb3feabbfc0b70e23d43ae18a354dabc112319eb11d2c8cea2c13f6bd5c0337b",
    "compact_report": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_detector_readout_structure_attribution_active_domain_v1_attempt_02/"
        "compact_report.json"
    ),
    "compact_report_sha256": "23b38d3231594cb0eb3ebab21972b8979ffaf825e17c901f4b86d46eeddc1c11",
    "implementation_module_sha256": (
        "a357d89c0d06d98f48e06c76e6b1e58c660ba4c1cd0f5f8e8d9e2e7843388572"
    ),
    "implementation_commit": "e0c442f92dd23b71d048ceb647570b76406c0435",
    "result_git_commit": "e0c442f92dd23b71d048ceb647570b76406c0435",
    "validation_status": "passed",
    "audit_status": "passed",
    "audit_classification": "descriptive_structural_attribution_completed",
}

FIT_PANEL = dict(rwsa.FIT_PANEL)
POOLING = dict(rwsa.POOLING)
COORDINATE_CONTRACT = dict(rwsa.COORDINATE_CONTRACT)
D4_CONTROLS = tuple(rwsa.D4_CONTROLS)
TARGET = dict(zcsa.TARGET)

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S5-RWSA-ACTIVE",
    "lineage_start_head": BASELINE_HEAD,
    "rmal_probe_frozen": True,
    "rws_active_passed_required": True,
    "zcsa_passed_required": True,
    "selection_confirmation_not_consumed": True,
    "attempt_name": (
        "lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01"
    ),
}
MODE_PREVIEW = {
    "k": MODE_PREVIEW_K,
    "kind": "descriptive_preview_only",
    "basis_dependent_mode_descriptor": True,
    "near_degenerate_individual_vectors_may_rotate": True,
    "no_degeneracy_threshold_gate": True,
    "no_numerical_rank_cutoff": True,
}
SIGN_CANONICALIZATION = {
    "name": "largest_abs_source_entry_positive",
    "tie_break": "lowest_row_major_source_feature_index",
    "paired_target_sign_flip": True,
    "target_based_sign_selection": False,
}
PSD_SANITY = {
    "dtype": "float64",
    "machine_epsilon": "torch.finfo(torch.float64).eps",
    "tau_formula": "100 * eps64 * 256 * max(lambda_max, 1)",
    "material_negative_rule": "lambda < -tau_psd -> invalid_harness",
    "tiny_negative_rule": "-tau_psd <= lambda < 0 -> record_and_clamp_for_sqrt_only",
    "rank_cutoff": None,
}
SPECTRAL_CONTRACT = {
    "matrix": "W64 @ W64.T",
    "full_shape": [FEATURE_DIM, FEATURE_DIM],
    "active_shape": [FEATURE_DIM, FEATURE_DIM],
    "eigendecomposition": "torch.linalg.eigh_cpu_float64_symmetric",
    "ordering": "eigenvalues_descending",
    "singular_value_formula": "sqrt(max(lambda, 0))",
    "probability_formula": "sigma_squared / sum_sigma_squared",
    "effective_rank_formula": "exp(-sum(p_k * log(p_k))) for p_k > 0",
    "stable_rank_formula": "sum_sigma_squared / sigma_1_squared",
    "adjacent_gap_formula": "(sigma_k - sigma_k_plus_1) / sigma_k",
    "randomized_svd": False,
    "large_target_gram": False,
    "numerical_rank_classification": False,
}
GLOBAL_TARGET_REFERENCE = {
    "kind": "exact_RMAL_fit_panel_unlabeled_target_mean",
    "count": FIT_COUNT,
    "renderer": "base.render_intensity_batch",
    "streaming": True,
    "target_bank_saved": False,
    "labels_used": False,
    "selection_used": False,
    "confirmation_used": False,
    "class_template_reference": "not_evaluated",
}
BIAS_ATTRIBUTION = {
    "source": "frozen_RMAL_bias_b",
    "included_in_gram": False,
    "projection": "dot(v_k, b)",
    "normalized_projection": "c_k_squared / norm_b_squared",
    "preview_k": MODE_PREVIEW_K,
    "refit": False,
    "dominant_subspace_gate": False,
}
MEASUREMENT = {
    "objective": "frozen_W_spectral_operator_descriptive_attribution",
    "probe_source": "exact_RMAL_detector_cmos_intensity_probe",
    "active_domain_retained": True,
    "full_output_columns": TOTAL_COLUMNS,
    "active_output_columns": EXPECTED_ACTIVE_COUNT,
    "inactive_output_columns": EXPECTED_INACTIVE_COUNT,
    "source_order": "row-major i_then_j",
    "target_order": "row-major r_then_c",
    "source_reshape": list(SOURCE_SHAPE),
    "target_reshape": list(TARGET_SHAPE),
    "no_training": True,
    "no_optimizer": True,
    "no_backward": True,
    "no_probe_refit": True,
    "no_rank_truncated_reconstruction": True,
    "no_decoder": True,
    "no_physical_forward": True,
    "no_rmal_replay": True,
    "descriptive_only": True,
}
GATES = {
    "mode_preview_k": MODE_PREVIEW_K,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_working_set_bytes": MAX_WORKING_SET_BYTES,
    "psd_material_negative_is_invalid_harness": True,
    "optimizer_created": False,
    "backward_calls": 0,
    "probe_refit": False,
    "decoder_called": False,
    "physical_forward_called": False,
    "formal_promotion": False,
}
AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_frozen_W_linear_readout_spectral_attribution",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "descriptive_only": True,
    "prohibited": [
        "training",
        "optimizer",
        "backward",
        "probe_refit",
        "probe_regularization",
        "weight_mutation",
        "rank_truncated_reconstruction",
        "decoder",
        "physical_forward",
        "RMAL_replay",
        "architecture_modification",
        "selection_panel_consumption",
        "confirmation_panel_consumption",
        "scientific_numerical_rank_classification",
        "checkpoint_write",
        "generation",
        "promotion",
    ],
}


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class SpectralSummary:
    label: str
    dimension: int
    raw_eigenvalues: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    singular_values: torch.Tensor
    squared_singular_values: torch.Tensor
    normalized_energy: torch.Tensor
    cumulative_energy: torch.Tensor
    sigma_ratios: torch.Tensor
    log10_sigma_ratios: tuple[float | None, ...]
    adjacent_relative_gaps: tuple[float | None, ...]
    effective_rank: float
    stable_rank: float
    condition_number: float
    psd: Mapping[str, Any]

    def compact(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "dimension": self.dimension,
            "raw_eigenvalues": self.raw_eigenvalues.tolist(),
            "squared_singular_values": self.squared_singular_values.tolist(),
            "singular_values": self.singular_values.tolist(),
            "normalized_energy": self.normalized_energy.tolist(),
            "cumulative_energy": self.cumulative_energy.tolist(),
            "sigma_over_sigma1": self.sigma_ratios.tolist(),
            "log10_sigma_over_sigma1": list(self.log10_sigma_ratios),
            "adjacent_relative_gaps": list(self.adjacent_relative_gaps),
            "effective_rank_entropy": self.effective_rank,
            "stable_rank": self.stable_rank,
            "condition_number": (
                self.condition_number if math.isfinite(self.condition_number) else None
            ),
            "condition_number_is_infinite": not math.isfinite(self.condition_number),
            "condition_profile_complete": True,
            "psd": dict(self.psd),
        }


class LSDAHarnessFailure(RuntimeError):
    """The frozen LSDA evidence boundary failed closed."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"LSDA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "lineage",
        "rmal_consumer",
        "rwsa_identity",
        "rwsa_active_identity",
        "zcsa_identity",
        "references",
        "fit_panel",
        "pooling",
        "coordinate_contract",
        "d4_controls",
        "mode_preview",
        "sign_canonicalization",
        "psd_sanity",
        "spectral_contract",
        "global_target_reference",
        "bias_attribution",
        "measurement",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("LSDA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("LSDA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "references": {},
        "fit_panel": FIT_PANEL,
        "pooling": POOLING,
        "coordinate_contract": COORDINATE_CONTRACT,
        "d4_controls": list(D4_CONTROLS),
        "mode_preview": MODE_PREVIEW,
        "sign_canonicalization": SIGN_CANONICALIZATION,
        "psd_sanity": PSD_SANITY,
        "spectral_contract": SPECTRAL_CONTRACT,
        "global_target_reference": GLOBAL_TARGET_REFERENCE,
        "bias_attribution": BIAS_ATTRIBUTION,
        "measurement": MEASUREMENT,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"LSDA {name} contract is not locked")


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
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "fit_panel": {
                key: FIT_PANEL[key]
                for key in ("source", "count", "per_label", "construction", "indices_sha256")
            },
            "pooling": POOLING,
            "coordinate_contract": COORDINATE_CONTRACT,
            "d4_controls": list(D4_CONTROLS),
            "mode_preview": MODE_PREVIEW,
            "sign_canonicalization": SIGN_CANONICALIZATION,
            "psd_sanity": PSD_SANITY,
            "spectral_contract": SPECTRAL_CONTRACT,
            "global_target_reference": GLOBAL_TARGET_REFERENCE,
            "bias_attribution": BIAS_ATTRIBUTION,
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
        raise ValueError("LSDA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise LSDAHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LSDAHarnessFailure(f"{label} is not a mapping: {path}")
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
        raise LSDAHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise LSDAHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_rmal_identity(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(RMAL_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != RMAL_CONSUMER["config_sha256"]:
        raise LSDAHarnessFailure("RMAL config identity changed")
    rmal_spec = rmal._load_spec(config_path)
    rmal._assert_contract(rmal_spec)
    review_path = _resolve(RMAL_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != RMAL_CONSUMER["review_sha256"]:
        raise LSDAHarnessFailure("RMAL review identity changed")
    rmal._load_review(review_path, rmal_spec)
    validation_path = _resolve(RMAL_CONSUMER["validation"])
    validation = _read_json_hash(
        validation_path, RMAL_CONSUMER["validation_sha256"], "RMAL validation"
    )
    validation_record = _verify_evidence_payload(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="RMAL validation",
    )
    if (
        validation_record.git_commit != RMAL_CONSUMER["implementation_commit"]
        or validation_record.config_hash != rmal_spec.config_hash
    ):
        raise LSDAHarnessFailure("RMAL validation lineage changed")
    audit_path = _resolve(RMAL_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, RMAL_CONSUMER["audit_sha256"], "RMAL audit")
    audit_record = _verify_evidence_payload(
        audit,
        expected_status="passed",
        expected_classification=RMAL_CONSUMER["audit_classification"],
        label="RMAL audit",
    )
    if audit_record.git_commit != RMAL_CONSUMER["implementation_commit"]:
        raise LSDAHarnessFailure("RMAL audit lineage changed")
    compact_path = _resolve(RMAL_CONSUMER["compact_report"])
    compact = _read_json_hash(
        compact_path, RMAL_CONSUMER["compact_report_sha256"], "RMAL compact report"
    )
    if (
        compact.get("classification") != RMAL_CONSUMER["audit_classification"]
        or compact.get("generation_readiness") != "not_ready"
    ):
        raise LSDAHarnessFailure("RMAL compact terminal interpretation changed")
    probe_path = _resolve(RMAL_CONSUMER["probe"])
    return {
        "config_path": config_path,
        "review_path": review_path,
        "validation_path": validation_path,
        "audit_path": audit_path,
        "compact_path": compact_path,
        "probe_path": probe_path,
        "validation": validation,
        "audit": audit,
        "compact": compact,
        "validation_record": validation_record,
        "audit_record": audit_record,
    }


def _verify_rwsa_identity(spec: Spec) -> dict[str, Any]:
    try:
        identity = zcsa._verify_rwsa_identity(spec)
    except (ValueError, zcsa.ZCSAHarnessFailure) as exc:
        raise LSDAHarnessFailure("historical RWSA attempt-02 identity changed") from exc
    if identity["audit"].get("status") != "invalid_harness":
        raise LSDAHarnessFailure("historical RWSA attempt-02 terminal state changed")
    return identity


def _verify_zcsa_identity(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(ZCSA_IDENTITY["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != ZCSA_IDENTITY["config_sha256"]:
        raise LSDAHarnessFailure("ZCSA config identity changed")
    zcsa_spec = zcsa._load_spec(config_path)
    zcsa._assert_contract(zcsa_spec)
    if zcsa_spec.config_hash != ZCSA_IDENTITY["config_canonical_sha256"]:
        raise LSDAHarnessFailure("ZCSA config canonical identity changed")
    review_path = _resolve(ZCSA_IDENTITY["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != ZCSA_IDENTITY["review_sha256"]:
        raise LSDAHarnessFailure("ZCSA review identity changed")
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
        raise LSDAHarnessFailure("ZCSA validation lineage changed")
    audit_path = _resolve(ZCSA_IDENTITY["audit"])
    audit = _read_json_hash(audit_path, ZCSA_IDENTITY["audit_sha256"], "ZCSA audit")
    audit_record = _verify_evidence_payload(
        audit,
        expected_status="passed",
        expected_classification=ZCSA_IDENTITY["audit_classification"],
        label="ZCSA audit",
    )
    if audit_record.git_commit != ZCSA_IDENTITY["result_git_commit"]:
        raise LSDAHarnessFailure("ZCSA audit lineage changed")
    compact_path = _resolve(ZCSA_IDENTITY["compact_report"])
    compact = _read_json_hash(
        compact_path, ZCSA_IDENTITY["compact_report_sha256"], "ZCSA compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != ZCSA_IDENTITY["audit_classification"]
        or compact.get("generation_readiness") != "not_ready"
    ):
        raise LSDAHarnessFailure("ZCSA compact terminal interpretation changed")
    if compact.get("probe", {}).get("zero_w_count") != EXPECTED_INACTIVE_COUNT:
        raise LSDAHarnessFailure("ZCSA zero-W count changed")
    if compact.get("probe", {}).get("dynamic_active_count") != EXPECTED_ACTIVE_COUNT:
        raise LSDAHarnessFailure("ZCSA active count changed")
    if compact.get("cross_panel_semantics", {}).get("transitions") != {
        "fit_zero_to_selection_active": 1495,
        "fit_zero_to_confirmation_active": 2097,
    }:
        raise LSDAHarnessFailure("ZCSA transition identity changed")
    implementation_module = _resolve(
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_detector_readout_zero_column_semantics_audit.py"
    )
    if file_sha256(implementation_module) != ZCSA_IDENTITY["implementation_module_sha256"]:
        raise LSDAHarnessFailure("ZCSA implementation identity changed")
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


def _verify_rwsa_active_identity(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(RWSA_ACTIVE_IDENTITY["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != RWSA_ACTIVE_IDENTITY["config_sha256"]:
        raise LSDAHarnessFailure("RWSA-ACTIVE config identity changed")
    active_spec = rwsa_active._load_spec(config_path)
    rwsa_active._assert_contract(active_spec)
    if active_spec.config_hash != RWSA_ACTIVE_IDENTITY["config_canonical_sha256"]:
        raise LSDAHarnessFailure("RWSA-ACTIVE config canonical identity changed")
    review_path = _resolve(
        RWSA_ACTIVE_IDENTITY["review_contract"], base_path=spec.path.parent
    )
    if file_sha256(review_path) != RWSA_ACTIVE_IDENTITY["review_sha256"]:
        raise LSDAHarnessFailure("RWSA-ACTIVE review identity changed")
    rwsa_active._load_review(review_path, active_spec)
    validation_path = _resolve(RWSA_ACTIVE_IDENTITY["validation"])
    validation = _read_json_hash(
        validation_path,
        RWSA_ACTIVE_IDENTITY["validation_sha256"],
        "RWSA-ACTIVE validation",
    )
    validation_record = _verify_evidence_payload(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="RWSA-ACTIVE validation",
    )
    if (
        validation_record.git_commit != RWSA_ACTIVE_IDENTITY["implementation_commit"]
        or validation_record.config_hash != active_spec.config_hash
    ):
        raise LSDAHarnessFailure("RWSA-ACTIVE validation lineage changed")
    audit_path = _resolve(RWSA_ACTIVE_IDENTITY["audit"])
    audit = _read_json_hash(
        audit_path, RWSA_ACTIVE_IDENTITY["audit_sha256"], "RWSA-ACTIVE audit"
    )
    audit_record = _verify_evidence_payload(
        audit,
        expected_status="passed",
        expected_classification=RWSA_ACTIVE_IDENTITY["audit_classification"],
        label="RWSA-ACTIVE audit",
    )
    if audit_record.git_commit != RWSA_ACTIVE_IDENTITY["result_git_commit"]:
        raise LSDAHarnessFailure("RWSA-ACTIVE audit lineage changed")
    compact_path = _resolve(RWSA_ACTIVE_IDENTITY["compact_report"])
    compact = _read_json_hash(
        compact_path,
        RWSA_ACTIVE_IDENTITY["compact_report_sha256"],
        "RWSA-ACTIVE compact report",
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != RWSA_ACTIVE_IDENTITY["audit_classification"]
        or compact.get("conditional_on_dynamic_active_domain") is not True
        or compact.get("dynamic_active_count") != EXPECTED_ACTIVE_COUNT
        or compact.get("dynamic_inactive_count") != EXPECTED_INACTIVE_COUNT
    ):
        raise LSDAHarnessFailure("RWSA-ACTIVE compact terminal interpretation changed")
    implementation_module = _resolve(
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_detector_readout_structure_attribution_active_domain.py"
    )
    if (
        file_sha256(implementation_module)
        != RWSA_ACTIVE_IDENTITY["implementation_module_sha256"]
    ):
        raise LSDAHarnessFailure("RWSA-ACTIVE implementation identity changed")
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


def _load_probe() -> tuple[rwsa.ReadoutProbe, torch.Tensor, Path]:
    probe_path = _resolve(RMAL_CONSUMER["probe"])
    payload = safe_torch_load(
        probe_path,
        expected_sha256=RMAL_CONSUMER["probe_sha256"],
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise LSDAHarnessFailure("RMAL probe artifact is not a mapping")
    stored_indices = payload.get("fit_indices")
    if not torch.is_tensor(stored_indices):
        raise LSDAHarnessFailure("RMAL probe fit indices are unavailable")
    fit_indices = stored_indices.detach().cpu().to(torch.long).contiguous()
    if (
        fit_indices.ndim != 1
        or fit_indices.numel() != FIT_COUNT
        or tensor_sha256(fit_indices) != FIT_PANEL["indices_sha256"]
    ):
        raise LSDAHarnessFailure("RMAL probe fit-panel identity changed")
    try:
        probe = rwsa._load_rmal_probe(
            payload, fit_indices, RMAL_CONSUMER["probe_sha256"]
        )
    except (ValueError, rwsa.RWSAHarnessFailure) as exc:
        raise LSDAHarnessFailure("RMAL probe contract is invalid") from exc
    return probe, fit_indices, probe_path


def _probe_identity(probe: rwsa.ReadoutProbe) -> dict[str, Any]:
    full = probe.full_weights.detach().cpu().contiguous()
    if tuple(full.shape) != (PROBE_INPUT_DIM, TOTAL_COLUMNS):
        raise LSDAHarnessFailure("RMAL probe full shape changed")
    if full.dtype != torch.float32:
        raise LSDAHarnessFailure("RMAL probe storage dtype changed")
    W = full[:-1, :]
    bias = full[-1, :]
    if tuple(W.shape) != (FEATURE_DIM, TOTAL_COLUMNS) or bias.shape != (TOTAL_COLUMNS,):
        raise LSDAHarnessFailure("RMAL W or b shape changed")
    if not bool(torch.isfinite(full).all()):
        raise LSDAHarnessFailure("RMAL probe contains non-finite values")
    dynamic_mass = W.abs().to(torch.float64).sum(dim=0)
    active_mask = dynamic_mass.gt(0)
    inactive_mask = dynamic_mass.eq(0)
    if not bool(torch.isfinite(dynamic_mass).all()):
        raise LSDAHarnessFailure("RMAL dynamic mass is non-finite")
    if not bool(W[:, inactive_mask].eq(0).all()):
        raise LSDAHarnessFailure("inactive dynamic columns are not exact zero")
    if not bool(bias[inactive_mask].eq(0).all()):
        raise LSDAHarnessFailure("inactive bias columns are not exact zero")
    if int(active_mask.sum()) != EXPECTED_ACTIVE_COUNT:
        raise LSDAHarnessFailure("active column count changed")
    if int(inactive_mask.sum()) != EXPECTED_INACTIVE_COUNT:
        raise LSDAHarnessFailure("inactive column count changed")
    if tensor_sha256(active_mask) != ACTIVE_MASK_SHA256:
        raise LSDAHarnessFailure("active mask identity changed")
    if tensor_sha256(inactive_mask) != INACTIVE_MASK_SHA256:
        raise LSDAHarnessFailure("inactive mask identity changed")
    return {
        "full": full,
        "W": W,
        "bias": bias,
        "dynamic_mass": dynamic_mass,
        "active_mask": active_mask,
        "inactive_mask": inactive_mask,
        "full_shape": list(full.shape),
        "W_shape": list(W.shape),
        "bias_shape": list(bias.shape),
        "storage_dtype": str(full.dtype),
        "source_order": POOLING["source_order"],
        "target_order": POOLING["target_order"],
        "active_count": int(active_mask.sum()),
        "inactive_count": int(inactive_mask.sum()),
        "active_mask_sha256": tensor_sha256(active_mask),
        "inactive_mask_sha256": tensor_sha256(inactive_mask),
        "active_minimum_mass": float(dynamic_mass[active_mask].min()),
        "active_maximum_mass": float(dynamic_mass[active_mask].max()),
        "zero_dynamic_entries_nonzero_count": int(torch.count_nonzero(W[:, inactive_mask])),
        "zero_bias_entries_nonzero_count": int(torch.count_nonzero(bias[inactive_mask])),
        "probe_sha256": probe.artifact_sha256,
        "source_float32_immutable": True,
    }


def _load_fit_images(
    dataset_root: str | Path, fit_indices: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        data = mra._load_data(dataset_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise LSDAHarnessFailure("registered MNIST fit-panel data is unavailable") from exc
    if data.manifest.get("manifest_hash") != DATASET_MANIFEST_SHA256:
        raise LSDAHarnessFailure("dataset manifest identity changed")
    if fit_indices.numel() != FIT_COUNT or tensor_sha256(fit_indices) != FIT_PANEL["indices_sha256"]:
        raise LSDAHarnessFailure("fit-panel index identity changed")
    if data.images.ndim != 4 or tuple(data.images.shape[1:]) != (1, 28, 28):
        raise LSDAHarnessFailure("MNIST image shape changed")
    if data.images.dtype != torch.float32 or not bool(torch.isfinite(data.images).all()):
        raise LSDAHarnessFailure("MNIST image dtype or finiteness changed")
    fit_images = data.images.index_select(0, fit_indices).contiguous()
    if fit_images.shape[0] != FIT_COUNT:
        raise LSDAHarnessFailure("fit-panel image count changed")
    return fit_images, {
        "manifest_hash": data.manifest["manifest_hash"],
        "image_shape": list(fit_images.shape),
        "image_dtype": str(fit_images.dtype),
        "fit_count": int(fit_images.shape[0]),
        "labels_used_for_target_mean": False,
        "selection_targets_consumed": False,
        "confirmation_targets_consumed": False,
    }


def _renderer_identity() -> dict[str, Any]:
    try:
        identity = zcsa._renderer_identity()
    except (ValueError, zcsa.ZCSAHarnessFailure) as exc:
        raise LSDAHarnessFailure("target renderer identity changed") from exc
    if identity.get("source_sha256") != TARGET_RENDERER_SOURCE_SHA256:
        raise LSDAHarnessFailure("target renderer source hash changed")
    return identity


def _scalar_summary(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    quantile_values = torch.quantile(flat, torch.tensor(QUANTILES, dtype=torch.float64))
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "quantiles": {
            str(quantile): float(value)
            for quantile, value in zip(QUANTILES, quantile_values, strict=True)
        },
    }


def _finite_or_none(value: torch.Tensor | float) -> float | None:
    scalar = float(value) if not torch.is_tensor(value) else float(value)
    return scalar if math.isfinite(scalar) else None


def _gram_matrix(W64: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    if W64.dtype != torch.float64 or W64.device.type != "cpu" or W64.ndim != 2:
        raise ValueError("analysis W must be a CPU float64 matrix")
    columns = W64 if active_mask is None else W64[:, active_mask]
    if columns.ndim != 2 or columns.shape[0] == 0:
        raise ValueError("analysis W column selection is invalid")
    return columns @ columns.T


def _spectrum_from_gram(gram: torch.Tensor, label: str = "") -> SpectralSummary:
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square")
    if gram.dtype != torch.float64 or gram.device.type != "cpu":
        raise ValueError("Gram matrix must be CPU float64")
    if not bool(torch.isfinite(gram).all()):
        raise LSDAHarnessFailure("Gram matrix is non-finite")
    raw_ascending, vectors_ascending = torch.linalg.eigh(gram)
    raw = raw_ascending.flip(0).contiguous()
    vectors = vectors_ascending.flip(1).contiguous()
    lambda_max = float(raw[0]) if raw.numel() else 0.0
    eps64 = torch.finfo(torch.float64).eps
    dimension = int(gram.shape[0])
    tau_psd = 100.0 * eps64 * dimension * max(lambda_max, 1.0)
    negative = raw.lt(0)
    material = raw.lt(-tau_psd)
    if bool(material.any()):
        raise LSDAHarnessFailure("Gram eigendecomposition has a material negative eigenvalue")
    tiny_negative = negative & ~material
    eigenvalues = torch.where(negative, torch.zeros_like(raw), raw)
    singular = eigenvalues.clamp_min(0).sqrt()
    squared = eigenvalues.clone()
    total_energy = float(squared.sum())
    if total_energy > 0.0:
        probability = squared / total_energy
        cumulative = torch.cumsum(probability, dim=0)
        positive = probability > 0
        entropy = float(-(probability[positive] * probability[positive].log()).sum())
        effective = math.exp(entropy)
    else:
        probability = torch.zeros_like(squared)
        cumulative = torch.zeros_like(squared)
        effective = 0.0
    sigma1 = float(singular[0]) if singular.numel() else 0.0
    if sigma1 > 0.0:
        ratios = singular / sigma1
        stable = float(squared.sum() / (sigma1 * sigma1))
        condition = float(sigma1 / singular[-1]) if float(singular[-1]) > 0 else math.inf
    else:
        ratios = torch.zeros_like(singular)
        stable = 0.0
        condition = math.inf
    log_ratios = tuple(
        math.log10(float(value)) if float(value) > 0 else None for value in ratios
    )
    gaps = tuple(
        _finite_or_none((singular[index] - singular[index + 1]) / singular[index])
        if float(singular[index]) > 0
        else None
        for index in range(dimension - 1)
    )
    negative_values = raw[negative]
    psd = {
        "tau_psd": tau_psd,
        "minimum_raw_eigenvalue": float(raw.min()),
        "negative_eigenvalue_count": int(negative.sum()),
        "clamped_tiny_negative_count": int(tiny_negative.sum()),
        "maximum_negative_magnitude": (
            float((-negative_values).max()) if negative_values.numel() else 0.0
        ),
        "material_negative_count": int(material.sum()),
        "numerical_rank_cutoff_used": False,
    }
    return SpectralSummary(
        label,
        dimension,
        raw,
        eigenvalues,
        vectors,
        singular,
        squared,
        probability,
        cumulative,
        ratios,
        log_ratios,
        gaps,
        effective,
        stable,
        condition,
        psd,
    )


def _spectral_analysis(
    W: torch.Tensor, active_mask: torch.Tensor
) -> tuple[torch.Tensor, SpectralSummary, SpectralSummary, dict[str, Any]]:
    if W.dtype != torch.float32 or W.device.type != "cpu":
        raise ValueError("frozen source W must remain CPU float32")
    W64 = W.to(torch.float64).contiguous()
    full_gram = _gram_matrix(W64)
    active_gram = _gram_matrix(W64, active_mask)
    full = _spectrum_from_gram(full_gram, "full")
    active = _spectrum_from_gram(active_gram, "active")
    eig_diff = (full.raw_eigenvalues - active.raw_eigenvalues).abs()
    eig_denominator = full.raw_eigenvalues.abs().clamp_min(torch.finfo(torch.float64).eps)
    sigma_diff = (full.singular_values - active.singular_values).abs()
    differences = {
        "max_absolute_eigenvalue_difference": float(eig_diff.max()),
        "max_relative_eigenvalue_difference": float((eig_diff / eig_denominator).max()),
        "max_singular_value_difference": float(sigma_diff.max()),
        "effective_rank_difference": abs(full.effective_rank - active.effective_rank),
        "stable_rank_difference": abs(full.stable_rank - active.stable_rank),
        "full_output_columns": TOTAL_COLUMNS,
        "active_output_columns": EXPECTED_ACTIVE_COUNT,
        "inactive_output_columns": EXPECTED_INACTIVE_COUNT,
        "inactive_exact_zero_identity_retained": True,
    }
    return W64, full, active, differences


def _frequency_coordinates(shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("frequency shape must be positive")
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=1.0, device="cpu"))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=1.0, device="cpu"))
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(grid_x.square() + grid_y.square()) / math.sqrt(0.5**2 + 0.5**2)
    return fx, fy, rho


def _radial_fft_energy(mode: torch.Tensor) -> dict[str, Any]:
    if mode.ndim != 2:
        raise ValueError("mode must be a 2D array")
    spectrum = torch.fft.fftshift(torch.fft.fft2(mode.to(torch.float64)))
    power = spectrum.abs().square()
    _, _, rho = _frequency_coordinates((int(mode.shape[0]), int(mode.shape[1])))
    total = power.sum()
    bands = {
        "low": (rho >= 0.0) & (rho < 0.25),
        "mid": (rho >= 0.25) & (rho < 0.50),
        "high": (rho >= 0.50) & (rho <= 1.00),
    }
    if float(total) == 0.0:
        fractions = {name: None for name in bands}
    else:
        fractions = {name: float(power[mask].sum() / total) for name, mask in bands.items()}
    return {
        "frequency_shape": [int(mode.shape[0]), int(mode.shape[1])],
        "frequency_range": "[-0.5,0.5)",
        "rho_definition": "sqrt(fx^2+fy^2)/sqrt(0.5^2+0.5^2)",
        "fixed_bands": {"low": [0.0, 0.25], "mid": [0.25, 0.50], "high": [0.50, 1.00]},
        "band_boundary_semantics": "low_mid_high = [0,.25),[.25,.50),[.50,1.00]",
        "fractions": fractions,
        "total_power": float(total),
    }


def _source_coordinate_grid() -> torch.Tensor:
    centers = rwsa._physical_cell_centers(
        rwsa._pool_bins(POOLING["input_shape"][0], POOLING["output_shape"][0]),
        size=200,
    )
    return centers.reshape(*SOURCE_SHAPE, 2).to(torch.float64)


def _pixel_coordinate_grid(shape: tuple[int, int], pixel_pitch_m: float) -> torch.Tensor:
    height, width = shape
    y = (torch.arange(height, dtype=torch.float64) - (height - 1) / 2.0) * pixel_pitch_m
    x = (torch.arange(width, dtype=torch.float64) - (width - 1) / 2.0) * pixel_pitch_m
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1)


def _spatial_masks(coordinates: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [H,W,2]")
    y = coordinates[..., 0]
    x = coordinates[..., 1]
    x_max = float(x.abs().max()) or 1.0
    y_max = float(y.abs().max()) or 1.0
    rho = torch.maximum(x.abs() / x_max, y.abs() / y_max)
    masks = {
        "center": rho <= 0.50,
        "middle": (rho > 0.50) & (rho < 0.75),
        "edge": rho >= 0.75,
    }
    if not bool(torch.stack(tuple(masks.values())).sum(dim=0).eq(1).all()):
        raise ValueError("spatial center/middle/edge masks are not exhaustive")
    return rho, masks


def _spatial_energy(mode: torch.Tensor, coordinates: torch.Tensor) -> dict[str, Any]:
    if tuple(mode.shape) != tuple(coordinates.shape[:2]):
        raise ValueError("mode and coordinate grid shapes differ")
    energy = mode.abs().to(torch.float64).square()
    total = energy.sum()
    _, masks = _spatial_masks(coordinates)
    if float(total) == 0.0:
        fractions = {name: None for name in masks}
        centroid = torch.zeros(2, dtype=torch.float64)
        rms_radius = None
        participation = None
    else:
        fractions = {name: float(energy[mask].sum() / total) for name, mask in masks.items()}
        centroid = (energy.unsqueeze(-1) * coordinates).sum(dim=(0, 1)) / total
        radius_squared = (coordinates - centroid).square().sum(dim=-1)
        rms_radius = float((energy * radius_squared).sum().div(total).sqrt())
        participation = float(total.square() / energy.square().sum())
    pixels = int(energy.numel())
    normalized_participation = None if participation is None else participation / pixels
    return {
        "rho_definition": "max(abs(x)/x_max,abs(y)/y_max)",
        "fixed_regions": {
            "center": "rho <= 0.50",
            "middle": "0.50 < rho < 0.75",
            "edge": "rho >= 0.75",
        },
        "squared_mode_energy_fractions": fractions,
        "centroid_m": centroid.tolist(),
        "centroid_um": (centroid * 1.0e6).tolist(),
        "rms_radius_m": rms_radius,
        "rms_radius_um": None if rms_radius is None else rms_radius * 1.0e6,
        "participation_size": participation,
        "normalized_participation": normalized_participation,
        "pixel_count": pixels,
    }


def _finite_difference_energy(mode: torch.Tensor) -> dict[str, float]:
    value = mode.to(torch.float64)
    if value.shape[0] > 1:
        y_energy = value[1:, :] - value[:-1, :]
        y_mean = float(y_energy.abs().square().mean())
    else:
        y_mean = 0.0
    if value.shape[1] > 1:
        x_energy = value[:, 1:] - value[:, :-1]
        x_mean = float(x_energy.abs().square().mean())
    else:
        x_mean = 0.0
    total = float(value.abs().square().mean())
    finite_difference = x_mean + y_mean
    normalized = finite_difference / total if total > 0 else 0.0
    return {
        "finite_difference_energy": finite_difference,
        "normalized_finite_difference_energy": normalized,
        "spatial_smoothness_score": 1.0 / (1.0 + normalized),
    }


def _d4_transform(mode: torch.Tensor, control: str) -> torch.Tensor:
    if control not in D4_CONTROLS:
        raise ValueError(f"unknown D4 control {control}")
    transformed = mode
    if control.startswith("transpose"):
        transformed = transformed.transpose(-1, -2)
    suffix = control.removeprefix("transpose").removeprefix("_")
    if suffix in {"x_flip", "xy_flip"}:
        transformed = transformed.flip(-1)
    if suffix in {"y_flip", "xy_flip"}:
        transformed = transformed.flip(-2)
    return transformed


def _d4_sign_invariant_residual(mode: torch.Tensor) -> dict[str, float]:
    denominator = float(torch.linalg.vector_norm(mode.to(torch.float64)))
    if denominator == 0.0:
        return {control: 0.0 for control in D4_CONTROLS}
    values: dict[str, float] = {}
    for control in D4_CONTROLS:
        transformed = _d4_transform(mode, control)
        direct = torch.linalg.vector_norm(mode - transformed)
        signed = torch.linalg.vector_norm(mode + transformed)
        values[control] = float(torch.minimum(direct, signed) / denominator)
    return values


def _signed_structure(mode: torch.Tensor) -> dict[str, Any]:
    real = mode.real.to(torch.float64)
    absolute = real.abs().sum()
    if float(absolute) == 0.0:
        return {
            "positive_mass": None,
            "negative_mass": None,
            "cancellation": "not_applicable_zero_mode",
        }
    positive = real.clamp_min(0).sum()
    negative = (-real).clamp_min(0).sum()
    signed_to_absolute = real.sum().abs() / absolute
    return {
        "positive_mass": float(positive / absolute),
        "negative_mass": float(negative / absolute),
        "signed_to_absolute_ratio": float(signed_to_absolute),
        "cancellation_fraction": float(1.0 - signed_to_absolute),
        "cancellation": "absolute_signed_cancellation_descriptor",
    }


def _mode_metrics(
    mode: torch.Tensor,
    *,
    mode_index: int,
    domain: str,
    coordinates: torch.Tensor,
) -> dict[str, Any]:
    if mode.ndim != 2:
        raise ValueError("mode descriptor requires a 2D mode")
    result = {
        "mode_index": mode_index,
        "domain": domain,
        "basis_dependent_mode_descriptor": True,
        "reshape": list(mode.shape),
        "finite_difference": _finite_difference_energy(mode),
        "radial_fft": _radial_fft_energy(mode),
        "spatial": _spatial_energy(mode, coordinates),
        "d4_sign_invariant_residual": _d4_sign_invariant_residual(mode),
        "signed_structure": _signed_structure(mode),
    }
    return result


def _orthogonality_residual(modes: torch.Tensor) -> float:
    if modes.ndim != 2:
        raise ValueError("mode bank must be a matrix")
    if modes.shape[1] == 0:
        return 0.0
    gram = modes.T @ modes
    identity = torch.eye(modes.shape[1], dtype=modes.dtype)
    return float(torch.linalg.matrix_norm(gram - identity) / torch.linalg.matrix_norm(identity))


def _canonicalize_source_modes(
    source_modes: torch.Tensor, target_modes: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None, list[int], list[int]]:
    if source_modes.ndim != 2:
        raise ValueError("source mode bank must be [feature,k]")
    source = source_modes.clone()
    target = None if target_modes is None else target_modes.clone()
    if target is not None and target.ndim != 2:
        raise ValueError("target mode bank must be [output,k]")
    indices: list[int] = []
    signs: list[int] = []
    for column in range(source.shape[1]):
        absolute = source[:, column].abs()
        maximum = absolute.max()
        candidates = torch.where(absolute == maximum)[0]
        if candidates.numel() == 0:
            raise ValueError("source mode has no canonical entry")
        selected = int(candidates[0])
        sign = -1 if float(source[selected, column]) < 0 else 1
        if sign < 0:
            source[:, column].mul_(-1)
            if target is not None:
                target[:, column].mul_(-1)
        indices.append(selected)
        signs.append(sign)
    return source, target, indices, signs


def _target_modes(
    W64: torch.Tensor,
    spectral: SpectralSummary,
    source_modes: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    if W64.dtype != torch.float64 or W64.device.type != "cpu" or W64.ndim != 2:
        raise ValueError("target-mode W must be a CPU float64 matrix")
    if source_modes.shape[0] != W64.shape[0]:
        raise ValueError("source modes do not match W feature rows")
    numerators = W64.T @ source_modes
    target = torch.zeros((W64.shape[1], source_modes.shape[1]), dtype=torch.float64)
    available: list[int] = []
    for column in range(source_modes.shape[1]):
        sigma = float(spectral.singular_values[column])
        if sigma > 0.0:
            target[:, column] = numerators[:, column] / sigma
            available.append(column)
    if available and not bool(torch.isfinite(target[:, available]).all()):
        raise LSDAHarnessFailure("target singular modes are non-finite")
    return target, available


def _global_target_mean(
    fit_images: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if fit_images.ndim != 4 or tuple(fit_images.shape[1:]) != (1, 28, 28):
        raise ValueError("fit images do not match the registered MNIST shape")
    total = torch.zeros(TOTAL_COLUMNS, dtype=torch.float64)
    first_target: torch.Tensor | None = None
    count = 0
    for indices in torch.arange(fit_images.shape[0], dtype=torch.long).split(
        TARGET_MICROBATCH_SIZE
    ):
        batch = fit_images.index_select(0, indices)
        rendered = base.render_intensity_batch(batch, output_shape=TARGET_SHAPE).target
        if rendered.dtype != torch.float32 or tuple(rendered.shape[1:]) != (1, *TARGET_SHAPE):
            raise LSDAHarnessFailure("target renderer output shape or dtype changed")
        if not bool(torch.isfinite(rendered).all()):
            raise LSDAHarnessFailure("target renderer produced non-finite values")
        flattened = rendered.reshape(rendered.shape[0], TOTAL_COLUMNS)
        if first_target is None:
            first_target = flattened[0].detach().clone()
        total.add_(flattened.to(torch.float64).sum(dim=0))
        count += int(flattened.shape[0])
    if count != FIT_COUNT:
        raise LSDAHarnessFailure("global target reference count changed")
    mean_target = total / count
    if not bool(torch.isfinite(mean_target).all()):
        raise LSDAHarnessFailure("global target mean is non-finite")
    return mean_target, {
        "kind": GLOBAL_TARGET_REFERENCE["kind"],
        "count": count,
        "shape": list(mean_target.shape),
        "dtype": str(mean_target.dtype),
        "target_mean_sha256": tensor_sha256(mean_target),
        "mean_squared_norm": float(mean_target.square().sum()),
        "mean_min": float(mean_target.min()),
        "mean_max": float(mean_target.max()),
        "first_target_sha256": tensor_sha256(first_target) if first_target is not None else None,
        "target_bank_saved": False,
        "labels_used": False,
        "selection_used": False,
        "confirmation_used": False,
        "class_template_reference": "not_evaluated",
    }


def _global_mean_projections(
    target_modes: torch.Tensor,
    available: Sequence[int],
    mean_target: torch.Tensor,
) -> dict[str, Any]:
    denominator = float(mean_target.square().sum())
    reports: list[dict[str, Any]] = []
    available_set = set(available)
    for index in range(MODE_PREVIEW_K):
        if index not in available_set:
            reports.append(
                {
                    "mode_index": index,
                    "descriptor": "not_evaluated_zero_singular_value",
                }
            )
            continue
        vector = target_modes[:, index]
        dot = float(torch.dot(vector, mean_target))
        squared = dot * dot
        reports.append(
            {
                "mode_index": index,
                "dot": dot,
                "normalized_correlation": (
                    dot / math.sqrt(denominator) if denominator > 0 else None
                ),
                "squared_projection": squared,
                "squared_projection_contribution": squared / denominator if denominator > 0 else None,
            }
        )
    return {
        "reference": GLOBAL_TARGET_REFERENCE,
        "mean_target_norm_squared": denominator,
        "class_template_reference": "not_evaluated",
        "modes": reports,
    }


def _bias_projection(
    target_modes: torch.Tensor,
    available: Sequence[int],
    bias: torch.Tensor,
) -> dict[str, Any]:
    bias64 = bias.to(torch.float64)
    norm_squared = float(bias64.square().sum())
    available_set = set(available)
    coefficients: list[dict[str, Any]] = []
    squared_values: list[float] = []
    for index in range(MODE_PREVIEW_K):
        if index not in available_set:
            coefficients.append(
                {
                    "mode_index": index,
                    "descriptor": "not_evaluated_zero_singular_value",
                }
            )
            continue
        coefficient = float(torch.dot(target_modes[:, index], bias64))
        squared = coefficient * coefficient
        squared_values.append(squared)
        coefficients.append(
            {
                "mode_index": index,
                "coefficient": coefficient,
                "coefficient_squared": squared,
                "normalized_projection": squared / norm_squared if norm_squared > 0 else None,
            }
        )
    cumulative = []
    running = 0.0
    for item in coefficients:
        if "coefficient_squared" in item:
            running += float(item["coefficient_squared"])
        cumulative.append(running / norm_squared if norm_squared > 0 else None)
    return {
        "bias_norm_squared": norm_squared,
        "bias_norm": math.sqrt(norm_squared),
        "coefficients": coefficients,
        "cumulative_top_k_projection_energy": cumulative,
        "preview_projection_energy": sum(squared_values),
        "residual_bias_energy_outside_preview": max(norm_squared - sum(squared_values), 0.0),
        "zero_bias_descriptor": "not_applicable_zero_bias" if norm_squared == 0 else None,
        "refit": False,
    }


def _preview_report(
    W64: torch.Tensor,
    bias: torch.Tensor,
    spectral: SpectralSummary,
    mean_target: torch.Tensor,
) -> dict[str, Any]:
    source_raw = spectral.eigenvectors[:, :MODE_PREVIEW_K].contiguous()
    target_raw, available = _target_modes(W64, spectral, source_raw)
    source_modes, target_modes, sign_indices, signs = _canonicalize_source_modes(
        source_raw, target_raw
    )
    source_coordinates = _source_coordinate_grid()
    target_coordinates = _pixel_coordinate_grid(TARGET_SHAPE, 2.5e-6)
    source_reports = [
        _mode_metrics(
            source_modes[:, index].reshape(*SOURCE_SHAPE),
            mode_index=index,
            domain="source",
            coordinates=source_coordinates,
        )
        for index in range(MODE_PREVIEW_K)
    ]
    available_reports: list[dict[str, Any]] = []
    available_set = set(available)
    for index in range(MODE_PREVIEW_K):
        if index not in available_set:
            available_reports.append(
                {
                    "mode_index": index,
                    "descriptor": "not_evaluated_zero_singular_value",
                    "basis_dependent_mode_descriptor": True,
                }
            )
            continue
        target_mode = target_modes[:, index].reshape(*TARGET_SHAPE)
        available_reports.append(
            _mode_metrics(
                target_mode,
                mode_index=index,
                domain="target",
                coordinates=target_coordinates,
            )
        )
    available_target_modes = target_modes[:, available] if available else target_modes[:, :0]
    return {
        "mode_preview_k": MODE_PREVIEW_K,
        "basis_dependent_mode_descriptor": True,
        "sign_canonicalization": SIGN_CANONICALIZATION,
        "sign_canonicalization_entry_indices": sign_indices,
        "signs_applied_to_source_and_target": signs,
        "adjacent_relative_gaps": list(spectral.adjacent_relative_gaps[: MODE_PREVIEW_K - 1]),
        "source_modes": source_reports,
        "target_modes": available_reports,
        "source_orthogonality_relative_residual": _orthogonality_residual(source_modes),
        "target_orthogonality_relative_residual": (
            _orthogonality_residual(available_target_modes)
            if available_target_modes.shape[1]
            else 0.0
        ),
        "available_target_mode_indices": available,
        "global_mean_projection": _global_mean_projections(
            target_modes, available, mean_target
        ),
        "bias_projection": _bias_projection(target_modes, available, bias),
        "raw_source_mode_bank_saved": False,
        "raw_target_mode_bank_saved": False,
    }


def _runtime_metadata() -> dict[str, Any]:
    torch_config = torch.__config__.show()
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "analysis_dtype": "torch.float64",
        "source_dtype": "torch.float32",
        "backend_identity": {
            "torch_config_sha256": canonical_sha256(torch_config),
            "mkl_available": bool(torch.backends.mkl.is_available()),
            "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
            "openmp_available": bool(torch.backends.openmp.is_available()),
        },
        "thread_configuration": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "randomized_svd_used": False,
        "cuda_used": False,
    }


def _estimated_working_set_bytes() -> int:
    w64 = FEATURE_DIM * TOTAL_COLUMNS * 8
    grams = 2 * FEATURE_DIM * FEATURE_DIM * 8
    target_modes = TOTAL_COLUMNS * MODE_PREVIEW_K * 8
    target_mean = TOTAL_COLUMNS * 8
    fit_images = FIT_COUNT * 28 * 28 * 4
    target_batch = TARGET_MICROBATCH_SIZE * TOTAL_COLUMNS * 4
    return w64 + grams + target_modes + target_mean + fit_images + target_batch


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Mapping[str, Any]],
    probe_path: Path,
    fit_indices: torch.Tensor,
    dataset_manifest: str,
) -> dict[str, str]:
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "implementation_module": file_sha256(Path(__file__).resolve()),
        "target_renderer_module": file_sha256(
            _resolve("src/ldm_project/v2/application/research/optical_manifold.py")
        ),
        "rmal_config": file_sha256(upstream["rmal"]["config_path"]),
        "rmal_review_contract": file_sha256(upstream["rmal"]["review_path"]),
        "rmal_validation": file_sha256(upstream["rmal"]["validation_path"]),
        "rmal_audit": file_sha256(upstream["rmal"]["audit_path"]),
        "rmal_compact_report": file_sha256(upstream["rmal"]["compact_path"]),
        "rmal_probe": file_sha256(probe_path),
        "rwsa_config": file_sha256(upstream["rwsa"]["config_path"]),
        "rwsa_review_contract": file_sha256(upstream["rwsa"]["review_path"]),
        "rwsa_validation": file_sha256(upstream["rwsa"]["validation_path"]),
        "rwsa_audit": file_sha256(upstream["rwsa"]["audit_path"]),
        "rwsa_implementation_module": RWSA_IDENTITY["implementation_module_sha256"],
        "rwsa_active_config": file_sha256(upstream["rwsa_active"]["config_path"]),
        "rwsa_active_review_contract": file_sha256(upstream["rwsa_active"]["review_path"]),
        "rwsa_active_validation": file_sha256(upstream["rwsa_active"]["validation_path"]),
        "rwsa_active_audit": file_sha256(upstream["rwsa_active"]["audit_path"]),
        "rwsa_active_compact_report": file_sha256(upstream["rwsa_active"]["compact_path"]),
        "rwsa_active_implementation_module": RWSA_ACTIVE_IDENTITY[
            "implementation_module_sha256"
        ],
        "zcsa_config": file_sha256(upstream["zcsa"]["config_path"]),
        "zcsa_review_contract": file_sha256(upstream["zcsa"]["review_path"]),
        "zcsa_validation": file_sha256(upstream["zcsa"]["validation_path"]),
        "zcsa_audit": file_sha256(upstream["zcsa"]["audit_path"]),
        "zcsa_compact_report": file_sha256(upstream["zcsa"]["compact_path"]),
        "zcsa_implementation_module": ZCSA_IDENTITY["implementation_module_sha256"],
        "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
        "rwsa_identity_contract": canonical_sha256(RWSA_IDENTITY),
        "rwsa_active_identity_contract": canonical_sha256(RWSA_ACTIVE_IDENTITY),
        "zcsa_identity_contract": canonical_sha256(ZCSA_IDENTITY),
        "fit_indices": tensor_sha256(fit_indices),
        "dataset_manifest": dataset_manifest,
        "pooling_contract": canonical_sha256(POOLING),
        "coordinate_contract": canonical_sha256(COORDINATE_CONTRACT),
        "d4_control_contract": canonical_sha256(D4_CONTROLS),
        "mode_preview_contract": canonical_sha256(MODE_PREVIEW),
        "sign_canonicalization_contract": canonical_sha256(SIGN_CANONICALIZATION),
        "psd_sanity_contract": canonical_sha256(PSD_SANITY),
        "spectral_contract": canonical_sha256(SPECTRAL_CONTRACT),
        "target_contract": canonical_sha256(TARGET),
        "target_renderer_contract": TARGET_RENDERER_SOURCE_SHA256,
    }
    return hashes


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "spectral_contract": SPECTRAL_CONTRACT,
            "mode_preview": MODE_PREVIEW,
            "sign_canonicalization": SIGN_CANONICALIZATION,
            "global_target_reference": GLOBAL_TARGET_REFERENCE,
            "bias_attribution": BIAS_ATTRIBUTION,
        }
    )
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=dataset_manifest,
        checkpoint_hash=RMAL_CONSUMER.get("checkpoint_sha256"),
        dataset_split_manifest_hash=dataset_manifest,
    )


def _record(
    schema: str,
    status: str,
    classification: str | None,
    spec: Spec,
    state: Mapping[str, Any],
    dataset_manifest: str,
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
        _context(spec, dataset_manifest),
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
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "probe_refit": False,
        "decoder_called": False,
        "physical_forward_called": False,
        "cuda_used": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse LSDA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("LSDA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"LSDA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("LSDA attempt directory is not a fresh validation-only slot")


def _validation_gates(
    state: Mapping[str, Any],
    probe_id: Mapping[str, Any],
    fit_images: torch.Tensor,
    dataset_info: Mapping[str, Any],
    renderer: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    estimated = _estimated_working_set_bytes()
    return (
        GateResult(
            "clean_registered_branch",
            state.get("branch") == BRANCH and state.get("clean") is True,
            {},
        ),
        GateResult(
            "probe_shape_dtype_and_order",
            probe_id["full_shape"] == [PROBE_INPUT_DIM, TOTAL_COLUMNS]
            and probe_id["W_shape"] == [FEATURE_DIM, TOTAL_COLUMNS]
            and probe_id["bias_shape"] == [TOTAL_COLUMNS]
            and probe_id["storage_dtype"] == "torch.float32"
            and probe_id["source_order"] == "row-major i_then_j"
            and probe_id["target_order"] == "row-major r_then_c",
            {},
        ),
        GateResult(
            "active_inactive_identity",
            probe_id["active_count"] == EXPECTED_ACTIVE_COUNT
            and probe_id["inactive_count"] == EXPECTED_INACTIVE_COUNT
            and probe_id["zero_dynamic_entries_nonzero_count"] == 0
            and probe_id["zero_bias_entries_nonzero_count"] == 0
            and probe_id["active_mask_sha256"] == ACTIVE_MASK_SHA256
            and probe_id["inactive_mask_sha256"] == INACTIVE_MASK_SHA256,
            {},
        ),
        GateResult(
            "fit_panel_identity",
            fit_images.shape[0] == FIT_COUNT
            and dataset_info["manifest_hash"] == DATASET_MANIFEST_SHA256,
            {},
        ),
        GateResult(
            "target_renderer_identity",
            renderer.get("source_sha256") == TARGET_RENDERER_SOURCE_SHA256
            and renderer.get("exact_repeat") is True,
            {},
        ),
        GateResult(
            "mode_preview_and_spectral_contract",
            MODE_PREVIEW["k"] == MODE_PREVIEW_K
            and SPECTRAL_CONTRACT["randomized_svd"] is False
            and SPECTRAL_CONTRACT["large_target_gram"] is False
            and PSD_SANITY["rank_cutoff"] is None,
            {},
        ),
        GateResult(
            "cpu_only_no_update_scope",
            MEASUREMENT["no_training"]
            and MEASUREMENT["no_optimizer"]
            and MEASUREMENT["no_backward"]
            and MEASUREMENT["no_probe_refit"]
            and MEASUREMENT["no_decoder"]
            and MEASUREMENT["no_physical_forward"],
            {"cuda_used": False, "optimizer_created": False, "backward_calls": 0},
        ),
        GateResult(
            "working_set_bound",
            estimated < MAX_WORKING_SET_BYTES,
            {"estimated_bytes": estimated, "maximum_bytes": MAX_WORKING_SET_BYTES},
        ),
        GateResult(
            "disk_free_precondition",
            disk.free >= MIN_DISK_FREE_BYTES,
            {"free_bytes": int(disk.free), "minimum_bytes": MIN_DISK_FREE_BYTES},
        ),
        GateResult(
            "output_slot_absent",
            not output_path.parent.exists(),
            {"path": str(output_path.parent)},
        ),
    )


def _upstream_bundle(
    spec: Spec,
) -> dict[str, dict[str, Any]]:
    return {
        "rmal": _verify_rmal_identity(spec),
        "rwsa": _verify_rwsa_identity(spec),
        "zcsa": _verify_zcsa_identity(spec),
        "rwsa_active": _verify_rwsa_active_identity(spec),
    }


def _load_validation(
    path: str | Path,
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Mapping[str, Any]],
    probe_path: Path,
    fit_indices: torch.Tensor,
    dataset_manifest: str,
    state: Mapping[str, Any],
) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(
        spec,
        review_path,
        upstream,
        probe_path,
        fit_indices,
        dataset_manifest,
    )
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("status") != "passed"
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or dict(record.input_hashes) != expected_hashes
        or record.permitted_next_actions != (NEXT_ACTION_RUN,)
    ):
        raise ValueError("LSDA validation evidence is stale or mismatched")


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _upstream_bundle(spec)
    probe, fit_indices, probe_path = _load_probe()
    probe_id = _probe_identity(probe)
    fit_images, dataset_info = _load_fit_images(args.dataset_root, fit_indices)
    renderer = _renderer_identity()
    dataset_manifest = dataset_info["manifest_hash"]
    output_path = _resolve(args.output)
    gates = _validation_gates(
        state,
        probe_id,
        fit_images,
        dataset_info,
        renderer,
        output_path,
    )
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(
        spec,
        review_path,
        upstream,
        probe_path,
        fit_indices,
        dataset_manifest,
    )
    contract = {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "probe": {
            "artifact_sha256": RMAL_CONSUMER["probe_sha256"],
            "full_shape": probe_id["full_shape"],
            "W_shape": probe_id["W_shape"],
            "bias_shape": probe_id["bias_shape"],
            "storage_dtype": probe_id["storage_dtype"],
            "active_count": probe_id["active_count"],
            "inactive_count": probe_id["inactive_count"],
            "active_mask_sha256": probe_id["active_mask_sha256"],
            "inactive_mask_sha256": probe_id["inactive_mask_sha256"],
        },
        "dataset": dataset_info,
        "renderer": renderer,
        "runtime": _runtime_metadata(),
        "spectral_contract": SPECTRAL_CONTRACT,
        "mode_preview": MODE_PREVIEW,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        state,
        dataset_manifest,
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "LSDA static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="LSDA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _check_completeness(
    full: SpectralSummary,
    active: SpectralSummary,
    preview: Mapping[str, Any],
    global_mean: Mapping[str, Any],
    bias: Mapping[str, Any],
) -> tuple[GateResult, ...]:
    source_count = len(preview["source_modes"])
    target_count = len(preview["target_modes"])
    return (
        GateResult(
            "full_spectrum_complete",
            len(full.singular_values) == FEATURE_DIM
            and len(full.normalized_energy) == FEATURE_DIM
            and len(full.cumulative_energy) == FEATURE_DIM
            and len(full.sigma_ratios) == FEATURE_DIM
            and len(full.adjacent_relative_gaps) == FEATURE_DIM - 1,
            {},
        ),
        GateResult(
            "active_spectrum_complete",
            len(active.singular_values) == FEATURE_DIM
            and len(active.normalized_energy) == FEATURE_DIM
            and len(active.cumulative_energy) == FEATURE_DIM
            and len(active.sigma_ratios) == FEATURE_DIM
            and len(active.adjacent_relative_gaps) == FEATURE_DIM - 1,
            {},
        ),
        GateResult("mode_preview_complete", source_count == MODE_PREVIEW_K and target_count == MODE_PREVIEW_K, {}),
        GateResult(
            "global_mean_projection_complete",
            len(global_mean["modes"]) == MODE_PREVIEW_K
            and global_mean["reference"]["class_template_reference"] == "not_evaluated",
            {},
        ),
        GateResult(
            "bias_projection_complete",
            len(bias["coefficients"]) == MODE_PREVIEW_K
            and bias["refit"] is False,
            {},
        ),
    )


def _compact_report(
    state: Mapping[str, Any],
    spec: Spec,
    probe_id: Mapping[str, Any],
    runtime: Mapping[str, Any],
    full: SpectralSummary,
    active: SpectralSummary,
    differences: Mapping[str, Any],
    preview: Mapping[str, Any],
    global_mean: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": COMPACT_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "status": "pending_audit_gate",
        "classification": None,
        "producer": {"git_commit": state["commit"], "config_hash": spec.config_hash},
        "consumer": {
            "branch": BRANCH,
            "rmal_consumer": RMAL_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "probe_sha256": probe_id["probe_sha256"],
            "full_shape": probe_id["full_shape"],
            "dynamic_weight_shape": probe_id["W_shape"],
            "bias_shape": probe_id["bias_shape"],
            "storage_dtype": probe_id["storage_dtype"],
            "source_order": probe_id["source_order"],
            "target_order": probe_id["target_order"],
            "active_count": probe_id["active_count"],
            "inactive_count": probe_id["inactive_count"],
            "active_mask_sha256": probe_id["active_mask_sha256"],
            "inactive_mask_sha256": probe_id["inactive_mask_sha256"],
            "dataset_manifest": dataset_info["manifest_hash"],
        },
        "runtime": runtime,
        "numerical": {
            "analysis_dtype": "torch.float64",
            "source_dtype": "torch.float32",
            "psd_sanity": PSD_SANITY,
            "full_psd": dict(full.psd),
            "active_psd": dict(active.psd),
            "large_target_gram_constructed": False,
            "modified_probe_saved": False,
        },
        "spectrum_full": full.compact(),
        "spectrum_active": active.compact(),
        "full_vs_active": dict(differences),
        "preview": preview,
        "global_mean": global_mean,
        "bias_projection": preview["bias_projection"],
        "classification_boundary": {
            "descriptive_only": True,
            "numerical_rank_classification": False,
            "rank_truncated_reconstruction": False,
            "training_authorized": False,
            "formal_promotion": False,
        },
        "selection_confirmation_reference": {
            "selection": "not_consumed",
            "confirmation": "not_consumed",
            "class_template": "not_evaluated",
        },
        "raw_source_mode_bank_saved": False,
        "raw_target_mode_bank_saved": False,
        "target_bank_saved": False,
    }


def run(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _upstream_bundle(spec)
    probe, fit_indices, probe_path = _load_probe()
    probe_id = _probe_identity(probe)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    fit_images, dataset_info = _load_fit_images(args.dataset_root, fit_indices)
    dataset_manifest = dataset_info["manifest_hash"]
    _load_validation(
        validation_path,
        spec,
        review_path,
        upstream,
        probe_path,
        fit_indices,
        dataset_manifest,
        state,
    )
    runtime = _runtime_metadata()
    try:
        W64, full, active, differences = _spectral_analysis(
            probe_id["W"], probe_id["active_mask"]
        )
        mean_target, global_mean = _global_target_mean(fit_images)
        preview = _preview_report(W64, probe_id["bias"], active, mean_target)
    except LSDAHarnessFailure as exc:
        hashes = _input_hashes(
            spec,
            review_path,
            upstream,
            probe_path,
            fit_indices,
            dataset_manifest,
        )
        payload = _record(
            AUDIT_SCHEMA,
            "invalid_harness",
            "invalid_harness",
            spec,
            state,
            dataset_manifest,
            hashes,
            (
                GateResult("lsda_analysis", False, {"reason": str(exc)}),
                GateResult("no_update_scope", True, {"cuda_used": False}),
            ),
            (),
            {"lineage": LINEAGE, "reason": str(exc)},
            str(exc),
        )
        _write_new(output_path, payload, artifact_kind="LSDA invalid-harness audit evidence")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    completeness = _check_completeness(
        full,
        active,
        preview,
        preview["global_mean_projection"],
        preview["bias_projection"],
    )
    compact = _compact_report(
        state,
        spec,
        probe_id,
        runtime,
        full,
        active,
        differences,
        preview,
        global_mean,
        dataset_info,
    )
    finite = _all_finite(compact)
    gates = completeness + (
        GateResult("finite_measurements", finite, {}),
        GateResult(
            "no_update_scope",
            True,
            {
                "cuda_used": False,
                "optimizer_created": False,
                "backward_calls": 0,
                "probe_refit": False,
                "decoder_called": False,
                "physical_forward_called": False,
            },
        ),
        GateResult(
            "storage_limit",
            len(json.dumps(compact, sort_keys=True).encode("utf-8"))
            <= MAX_WORKING_SET_BYTES,
            {
                "estimated_compact_bytes": len(
                    json.dumps(compact, sort_keys=True).encode("utf-8")
                ),
                "maximum_bytes": MAX_WORKING_SET_BYTES,
            },
        ),
    )
    valid = all(gate.passed for gate in gates)
    compact["status"] = "passed" if valid else "invalid_harness"
    compact["classification"] = CLASSIFICATION if valid else "invalid_harness"
    compact_path = output_path.parent / "compact_report.json"
    _write_new(compact_path, compact, artifact_kind="LSDA compact report")
    compact_hash = file_sha256(compact_path)
    output_bytes = int(
        sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file())
    )
    hashes = _input_hashes(
        spec,
        review_path,
        upstream,
        probe_path,
        fit_indices,
        dataset_manifest,
    )
    hashes["lsda_compact_report"] = compact_hash
    audit_contract = {
        "lineage": LINEAGE,
        "classification": CLASSIFICATION if valid else "invalid_harness",
        "consumer": compact["consumer"],
        "runtime": runtime,
        "numerical": compact["numerical"],
        "full_vs_active": differences,
        "mode_preview": MODE_PREVIEW,
        "global_target_reference": global_mean,
        "bias_attribution": preview["bias_projection"],
        "completeness": [gate.name for gate in completeness],
        "compact_report_sha256": compact_hash,
        "compact_report_size_bytes": output_bytes,
        "selection_confirmation_not_consumed": True,
    }
    payload = _record(
        AUDIT_SCHEMA,
        "passed" if valid else "invalid_harness",
        CLASSIFICATION if valid else "invalid_harness",
        spec,
        state,
        dataset_manifest,
        hashes,
        gates,
        (NEXT_ACTION_RESULT_REVIEW,) if valid else (),
        audit_contract,
        None if valid else "LSDA completeness or resource gate failed",
    )
    _write_new(output_path, payload, artifact_kind="LSDA audit evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-linear-readout-structure-decomposition-validate":
        return validate(args)
    if args.research_action == "scaled-chen-linear-readout-structure-decomposition-run":
        return run(args)
    raise ValueError(f"unsupported LSDA action: {args.research_action}")


__all__ = [
    "ACTIVE_MASK_SHA256",
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
    "GLOBAL_TARGET_REFERENCE",
    "HYPOTHESIS",
    "INACTIVE_MASK_SHA256",
    "LINEAGE",
    "LSDAHarnessFailure",
    "MEASUREMENT",
    "MODE_PREVIEW",
    "MODE_PREVIEW_K",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "POOLING",
    "PSD_SANITY",
    "RMAL_CONSUMER",
    "RWSA_ACTIVE_IDENTITY",
    "RWSA_IDENTITY",
    "SIGN_CANONICALIZATION",
    "SPECTRAL_CONTRACT",
    "SpectralSummary",
    "TARGET",
    "VALIDATION_SCHEMA",
    "ZCSA_IDENTITY",
    "_canonicalize_source_modes",
    "_d4_sign_invariant_residual",
    "_frequency_coordinates",
    "_global_target_mean",
    "_gram_matrix",
    "_mode_metrics",
    "_orthogonality_residual",
    "_pixel_coordinate_grid",
    "_radial_fft_energy",
    "_scalar_summary",
    "_spatial_masks",
    "_spectrum_from_gram",
    "_spectral_analysis",
    "_target_modes",
    "dispatch",
    "run",
    "validate",
]
