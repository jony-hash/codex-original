"""Attribute the frozen RMAL readout under detector data second moments.

This is a descriptive, no-update consumer of the registered RMAL detector
readout.  It reuses the exact DSFA CMOS-only replay path and the LSDA ordered
CPU float64 eigensystem, while keeping fit, selection, and confirmation panels
separate.  No reconstruction truncation, rank search, refit, or feature bank
is part of this stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
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
    tensor_sha256,
)
from . import optical_manifold as base
from . import (
    optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution as dsfa,
)
from . import (
    optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit as lsda,
)
from . import (
    optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc,
)


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S6-DMSA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "39b0ac449986df31b9102c0b1296dda912f7b39c"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10
EXPECTED_ACTIVE_COUNT = 33502
EXPECTED_INACTIVE_COUNT = 6498
LSDA_SIGMA_RTOL = 1.0e-11
LSDA_SIGMA_ATOL = 1.0e-8
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
MOMENT_IDENTITY_RTOL = LSDA_SIGMA_RTOL
MOMENT_IDENTITY_ATOL = LSDA_SIGMA_ATOL
SQUARE_ROOT_RTOL = LSDA_SIGMA_RTOL
SQUARE_ROOT_ATOL = LSDA_SIGMA_ATOL
LANDMARKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

RMAL_CONSUMER = dict(dsfa.RMAL_CONSUMER)
LSDA_CONSUMER = dict(dsfa.LSDA_CONSUMER)
RWSA_IDENTITY = dict(dsfa.RWSA_IDENTITY)
RWSA_ACTIVE_IDENTITY = dict(dsfa.RWSA_ACTIVE_IDENTITY)
ZCSA_IDENTITY = dict(dsfa.ZCSA_IDENTITY)
LRRC_CONSUMER = dict(dsfa.LRRC_CONSUMER)
DSFA_CONSUMER = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_v1.yaml"
    ),
    "config_sha256": "bdaa2a8a5f3f67e57df16383cfbf1a8ebe164c12fba4db2e8be4b557fcc67ded",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_review_v1.json"
    ),
    "review_sha256": "ec1aad5424ef75381d6906df809242c8e0dfb568dc9b26edc13e7f2b6293fffd",
    "validation": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/validation.json"
    ),
    "validation_sha256": "ff15d6da857f1db12b67b2abb0a182aff0a622b15718c44113ad01389fe7efbe",
    "audit": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/audit.json"
    ),
    "audit_sha256": "2521e0ec4138d55cfaf17f153a493c40aef5151bf5628a4daa9852dc198a265e",
    "compact_report": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_data_weighted_spectral_functional_attribution_v1_attempt_01/compact_report.json"
    ),
    "compact_report_sha256": "8aec1e2b718eb6de499f87d81b68203ebbd7d8483bff0c36066ed9e376dea3c3",
    "implementation_module": (
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution.py"
    ),
    "implementation_module_sha256": "ba05ed7b2ba06f4943d8e0188613f2d609eb5c128effb301c1ffe83dbb044237",
    "implementation_commit": "d032738c0f02af5ada85554224e045d238416e0a",
    "status": "passed",
    "classification": "data_weighted_spectral_functional_attribution_completed",
}

FIT_PANEL = {
    "role": "probe_fit_origin_reference",
    "source": "mnist_train_indices_0_54999",
    "count": FIT_COUNT,
    "per_label": 100,
    "indices_sha256": "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d",
}

PANEL_CONTRACT = {
    "fit": FIT_PANEL,
    "selection": {
        "role": "independent_registered_selection",
        "count": SELECTION_COUNT,
        "source": "mnist_test_indices_5000_9999",
        "parent": dsfa.PANEL_CONTRACT["selection"]["parent"],
    },
    "confirmation": {
        "role": "independent_registered_confirmation",
        "count": CONFIRMATION_COUNT,
        "source": "horizon_heldout_train_indices_55000_59999",
        "parent": dsfa.PANEL_CONTRACT["confirmation"]["parent"],
    },
    "feature_path": "output.camera -> mra._feature_projection(real, complex_value=False)",
    "streaming": True,
    "population_denominator": "N",
    "fit_is_reference_only": True,
    "selection_confirmation_independent": True,
    "pooled_feature_bank_saved": False,
    "target_bank_saved": False,
    "raw_cmos_bank_saved": False,
    "internal_stage_capture": False,
}

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S6-DSFA",
    "lineage_start_head": LINEAGE_START_HEAD,
    "rmal_probe_frozen": True,
    "lsda_passed_required": True,
    "lrrc_passed_required": True,
    "dsfa_passed_required": True,
    "zcsa_passed_required": True,
    "rwsa_active_passed_required": True,
    "fit_selection_confirmation_separate": True,
}

SPECTRAL_REPRESENTATION = {
    "source_W_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "eigensystem": "LSDA exact ordered CPU float64 symmetric eigensolver",
    "raw_operator": "G_raw=W64 @ W64.T",
    "mode_order": "LSDA sigma descending, k=1..256",
    "raw_basis": "U=LSDA ordered left singular vectors",
    "weighted_basis_semantics": "raw, second_moment, centered_variance are distinct bases",
    "weighted_eigenvalue_interpretation": "squared singular values of S_half W or C_half W",
    "randomized_svd": False,
    "basis_refit": False,
    "rank_truncation": False,
}

MOMENT_CONTRACT = {
    "feature_dim": FEATURE_DIM,
    "population_denominator": "N",
    "mu": "sum_z/N",
    "second_moment": "sum_zzT/N",
    "centered_second_moment": "M2/N",
    "centered_accumulator": "deterministic_float64_Welford_Chan_merge",
    "identity": "S=C+mu muT",
    "unbiased_N_minus_1": False,
    "streaming": True,
    "feature_bank_saved": False,
}

DECOMPOSITION_CONTRACT = {
    "G_raw": "W64 @ W64.T",
    "E_total": "tr(S G_raw)",
    "E_mean": "mu.T @ G_raw @ mu",
    "E_var": "tr(C G_raw)",
    "identity": "E_total=E_mean+E_var",
    "mean_fraction": "E_mean/E_total when finite nonzero",
    "variance_fraction": "E_var/E_total when finite nonzero",
    "no_epsilon": True,
    "no_scientific_dominance_gate": True,
}

PSD_SANITY = {
    "dtype": "float64",
    "machine_epsilon": "torch.finfo(torch.float64).eps",
    "tau_formula": "100 * eps64 * 256 * max(lambda_max, 1)",
    "material_negative_rule": "lambda < -tau_psd -> invalid_harness",
    "tiny_negative_rule": "-tau_psd <= lambda < 0 -> record_and_clamp_for_sqrt_only",
    "rank_cutoff": None,
}

WEIGHTED_SPECTRUM_CONTRACT = {
    "second_moment_operator": "S_half @ G_raw @ S_half",
    "centered_variance_operator": "C_half @ G_raw @ C_half",
    "square_root": "Q diag(sqrt(clamped_lambda)) Q.T",
    "squared_singular_values": True,
    "normalized_energy": "lambda/sum(lambda)",
    "cumulative_landmarks": list(LANDMARKS),
    "effective_rank": "exp(-sum(p*log(p))) for p>0",
    "stable_rank": "sum(lambda)/lambda_1",
    "normalized_condition_profile": "sqrt(lambda_k)/sqrt(lambda_1)",
    "no_reconstruction_truncation": True,
    "no_rank_classification": True,
}

OFF_DIAGONAL_CONTRACT = {
    "raw_basis_second_moment": "B=U.T @ S @ U",
    "raw_basis_centered_variance": "B_C=U.T @ C @ U",
    "diagonal_frobenius_squared": "sum(diag(B)^2)",
    "total_frobenius_squared": "||B||_F^2",
    "offdiagonal_frobenius_squared": "||B-diag(diag(B))||_F^2",
    "diagonal_fraction": "diagonal_frobenius_squared/total_frobenius_squared",
    "offdiagonal_fraction": "offdiagonal_frobenius_squared/total_frobenius_squared",
    "normalized_offdiagonal_norm": "||offdiag(B)||_F/||B||_F",
    "zero_denominator": "undefined",
    "scientific_threshold": False,
}

NUMERICAL_CONTRACT = {
    "source_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "moment_identity_atol": MOMENT_IDENTITY_ATOL,
    "moment_identity_rtol": MOMENT_IDENTITY_RTOL,
    "square_root_atol": SQUARE_ROOT_ATOL,
    "square_root_rtol": SQUARE_ROOT_RTOL,
    "lsda_sigma_atol": LSDA_SIGMA_ATOL,
    "lsda_sigma_rtol": LSDA_SIGMA_RTOL,
    "psd_sanity": PSD_SANITY,
    "no_epsilon": True,
    "no_renormalization": True,
    "no_N_minus_1": True,
}

MEASUREMENT = {
    "objective": "detector_data_metric_spectral_attribution",
    "primary_question": "DSFA amplification mean versus centered variance",
    "panels": ["fit", "selection", "confirmation"],
    "mode_count": FEATURE_DIM,
    "full_output_columns": TOTAL_COLUMNS,
    "active_output_columns": EXPECTED_ACTIVE_COUNT,
    "inactive_output_columns": EXPECTED_INACTIVE_COUNT,
    "selection_confirmation_independent": True,
    "fit_reference_only": True,
    "no_training": True,
    "no_optimizer": True,
    "no_backward": True,
    "no_refit": True,
    "no_rank_search": True,
    "no_rank_truncated_reconstruction": True,
    "no_decoder": True,
    "no_physical_forward_modification": True,
    "descriptive_only": True,
}

RESOURCE_CONTRACT = {
    "microbatch_size": MICROBATCH_SIZE,
    "moment_state_complexity": "O(256^2)",
    "w64_bytes_approx": FEATURE_DIM * TOTAL_COLUMNS * 8,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "min_free_cuda_bytes_if_replay": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes_if_replay": MAX_PEAK_CUDA_BYTES,
    "feature_bank_saved": False,
    "target_bank_saved": False,
    "mode_output_bank_saved": False,
    "large_target_gram": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_detector_data_metric_spectral_attribution",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "descriptive_only": True,
    "prohibited": [
        "training",
        "refit",
        "optimizer",
        "backward",
        "new_decoder",
        "rank_truncated_reconstruction",
        "rank_search",
        "covariance_aware_LRRC",
        "whitening_reconstruction",
        "optical_system_modification",
        "generation",
        "promotion",
        "feature_bank_persistence",
        "target_bank_persistence",
        "mode_output_bank_persistence",
        "scientific_rank_classification",
        "covariance_mismatch_supported",
        "mean_dominated_supported",
        "variance_dominated_supported",
        "data_aware_low_rank_supported",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_detector_data_metric_spectral_attribution"
NEXT_ACTION_RESULT_REVIEW = "manual_review_DMSA_result"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Upstream:
    dsfa: dict[str, Any]
    runtime: dsfa.Upstream


@dataclass(frozen=True)
class MomentResult:
    panel: str
    count: int
    mu: torch.Tensor
    second: torch.Tensor
    centered: torch.Tensor
    identity: dict[str, Any]
    symmetry: dict[str, Any]


class DMSAHarnessFailure(RuntimeError):
    """A DMSA lineage, measurement, or numerical boundary failed closed."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"DMSA requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "lineage",
        "rmal_consumer",
        "lsda_consumer",
        "rwsa_identity",
        "rwsa_active_identity",
        "zcsa_identity",
        "lrrc_consumer",
        "dsfa_consumer",
        "fit_panel",
        "panel_contract",
        "spectral_representation",
        "moment_contract",
        "decomposition_contract",
        "psd_sanity",
        "weighted_spectrum_contract",
        "off_diagonal_contract",
        "numerical",
        "measurement",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("DMSA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("DMSA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _contract_payload() -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "lsda_consumer": LSDA_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "lrrc_consumer": LRRC_CONSUMER,
        "dsfa_consumer": DSFA_CONSUMER,
        "fit_panel": FIT_PANEL,
        "panel_contract": PANEL_CONTRACT,
        "spectral_representation": SPECTRAL_REPRESENTATION,
        "moment_contract": MOMENT_CONTRACT,
        "decomposition_contract": DECOMPOSITION_CONTRACT,
        "psd_sanity": PSD_SANITY,
        "weighted_spectrum_contract": WEIGHTED_SPECTRUM_CONTRACT,
        "off_diagonal_contract": OFF_DIAGONAL_CONTRACT,
        "numerical": NUMERICAL_CONTRACT,
        "measurement": MEASUREMENT,
        "resource": RESOURCE_CONTRACT,
        "authorization": AUTHORIZATION,
    }


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": (
            "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
        ),
        **_contract_payload(),
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"DMSA {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": _contract_payload(),
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "descriptive_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    del spec
    target = _resolve(path)
    if json.loads(target.read_text(encoding="utf-8")) != _review_payload():
        raise ValueError("DMSA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise DMSAHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DMSAHarnessFailure(f"{label} is not a mapping: {path}")
    return payload


def _verify_evidence(
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
        raise DMSAHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise DMSAHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_dsfa(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(DSFA_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != DSFA_CONSUMER["config_sha256"]:
        raise DMSAHarnessFailure("DSFA config identity changed")
    dsfa_spec = dsfa._load_spec(config_path)
    dsfa._assert_contract(dsfa_spec)
    review_path = _resolve(DSFA_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != DSFA_CONSUMER["review_sha256"]:
        raise DMSAHarnessFailure("DSFA review identity changed")
    dsfa._load_review(review_path, dsfa_spec)
    implementation_path = _resolve(DSFA_CONSUMER["implementation_module"])
    if file_sha256(implementation_path) != DSFA_CONSUMER["implementation_module_sha256"]:
        raise DMSAHarnessFailure("DSFA implementation identity changed")
    validation_path = _resolve(DSFA_CONSUMER["validation"])
    validation = _read_json_hash(
        validation_path, DSFA_CONSUMER["validation_sha256"], "DSFA validation"
    )
    validation_record = _verify_evidence(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="DSFA validation",
    )
    if (
        validation_record.git_commit != DSFA_CONSUMER["implementation_commit"]
        or validation_record.config_hash != dsfa_spec.config_hash
    ):
        raise DMSAHarnessFailure("DSFA validation lineage changed")
    audit_path = _resolve(DSFA_CONSUMER["audit"])
    audit = _read_json_hash(
        audit_path, DSFA_CONSUMER["audit_sha256"], "DSFA audit"
    )
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=DSFA_CONSUMER["classification"],
        label="DSFA audit",
    )
    if audit_record.git_commit != DSFA_CONSUMER["implementation_commit"]:
        raise DMSAHarnessFailure("DSFA audit lineage changed")
    compact_path = _resolve(DSFA_CONSUMER["compact_report"])
    compact = _read_json_hash(
        compact_path, DSFA_CONSUMER["compact_report_sha256"], "DSFA compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != DSFA_CONSUMER["classification"]
        or compact.get("probe", {}).get("sha256") != RMAL_CONSUMER["probe_sha256"]
        or compact.get("probe", {}).get("W_shape") != [FEATURE_DIM, TOTAL_COLUMNS]
        or compact.get("probe", {}).get("bias_shape") != [TOTAL_COLUMNS]
        or compact.get("selection", {}).get("identity", {}).get("passed") is not True
        or compact.get("confirmation", {}).get("identity", {}).get("passed") is not True
    ):
        raise DMSAHarnessFailure("DSFA compact terminal identity changed")
    return {
        "config_path": config_path,
        "review_path": review_path,
        "implementation_path": implementation_path,
        "validation_path": validation_path,
        "audit_path": audit_path,
        "compact_path": compact_path,
        "validation": validation,
        "audit": audit,
        "compact": compact,
    }


def _load_upstream(spec: Spec, dataset_root: str | Path) -> Upstream:
    dsfa_identity = _verify_dsfa(spec)
    try:
        runtime = dsfa._load_upstream(spec, dataset_root)
    except (ValueError, dsfa.DSFAHarnessFailure, lrrc.LRRCHarnessFailure) as exc:
        raise DMSAHarnessFailure("RMAL/LSDA/LRRC upstream identity is invalid") from exc
    probe = runtime.runtime.rmal_probe
    if (
        probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]
        or tuple(probe.weights.shape) != (FEATURE_DIM, TOTAL_COLUMNS)
        or tuple(probe.bias.shape) != (TOTAL_COLUMNS,)
        or probe.weights.dtype != torch.float32
        or probe.bias.dtype != torch.float32
    ):
        raise DMSAHarnessFailure("frozen RMAL probe identity changed")
    return Upstream(dsfa_identity, runtime)


def _verify_panel_identity(upstream: Upstream) -> dict[str, Any]:
    runtime = upstream.runtime.runtime
    data = runtime.data
    fit_indices = runtime.fit_indices
    selection = data.selection_indices
    confirmation = data.confirmation_indices
    fit_labels = data.labels.index_select(0, fit_indices)
    label_counts = torch.bincount(fit_labels, minlength=10)
    if (
        fit_indices.numel() != FIT_COUNT
        or tensor_sha256(fit_indices) != FIT_PANEL["indices_sha256"]
        or not bool(torch.equal(runtime.rmal_probe.fit_indices, fit_indices))
        or not bool(torch.equal(label_counts, torch.full((10,), 100, dtype=torch.long)))
        or selection.numel() != SELECTION_COUNT
        or confirmation.numel() != CONFIRMATION_COUNT
        or bool(torch.isin(fit_indices, confirmation).any())
        or bool(torch.isin(selection, confirmation).any())
        or data.manifest.get("manifest_hash") != "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
    ):
        raise DMSAHarnessFailure("fit/selection/confirmation panel identity changed")
    return {
        "fit": {
            **FIT_PANEL,
            "label_counts": label_counts.tolist(),
            "indices_sha256": tensor_sha256(fit_indices),
        },
        "selection": {"count": int(selection.numel()), "source": PANEL_CONTRACT["selection"]["source"]},
        "confirmation": {"count": int(confirmation.numel()), "source": PANEL_CONTRACT["confirmation"]["source"]},
        "selection_confirmation_disjoint": True,
        "fit_confirmation_disjoint": True,
        "dataset_manifest": data.manifest.get("manifest_hash"),
    }


def _canonical_spectral(probe: Any, lsda_compact: Mapping[str, Any]) -> dict[str, Any]:
    try:
        spectral = dsfa._canonical_spectral(probe, lsda_compact)
    except (ValueError, lsda.LSDAHarnessFailure, lrrc.LRRCHarnessFailure) as exc:
        raise DMSAHarnessFailure("LSDA spectral lock failed") from exc
    if (
        tuple(spectral["W64"].shape) != (FEATURE_DIM, TOTAL_COLUMNS)
        or tuple(spectral["U"].shape) != (FEATURE_DIM, FEATURE_DIM)
        or spectral["q_identity"]["passed"] is not True
    ):
        raise DMSAHarnessFailure("LSDA spectral representation is invalid")
    return spectral


def _raw_spectrum_identity(spectral: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    W64 = spectral["W64"]
    raw_gram = W64.mm(W64.T)
    try:
        raw = lsda._spectrum_from_gram(raw_gram, "dmsa_raw")
    except lsda.LSDAHarnessFailure as exc:
        raise DMSAHarnessFailure("raw DMSA spectrum failed PSD validation") from exc
    expected = spectral["full"]
    sigma_residual = (raw.singular_values - expected.singular_values).abs()
    energy_residual = (raw.squared_singular_values - expected.squared_singular_values).abs()
    cumulative_residual = (raw.cumulative_energy - expected.cumulative_energy).abs()
    result = {
        "passed": bool(
            torch.allclose(
                raw.singular_values,
                expected.singular_values,
                rtol=LSDA_SIGMA_RTOL,
                atol=LSDA_SIGMA_ATOL,
            )
            and torch.allclose(
                raw.cumulative_energy,
                expected.cumulative_energy,
                rtol=LSDA_SIGMA_RTOL,
                atol=LSDA_SIGMA_ATOL,
            )
        ),
        "max_abs_sigma_residual": float(sigma_residual.max()),
        "max_abs_squared_singular_value_residual": float(energy_residual.max()),
        "max_abs_cumulative_residual": float(cumulative_residual.max()),
        "raw_spectrum_source": "LSDA exact ordered CPU float64 spectrum",
    }
    if not result["passed"]:
        raise DMSAHarnessFailure("raw spectrum differs from LSDA evidence")
    return raw, result


def _finite_matrix(matrix: torch.Tensor, label: str) -> None:
    if matrix.dtype != torch.float64 or matrix.device.type != "cpu":
        raise DMSAHarnessFailure(f"{label} is not CPU float64")
    if not bool(torch.isfinite(matrix).all()):
        raise DMSAHarnessFailure(f"{label} is non-finite")


def _matrix_symmetry(matrix: torch.Tensor) -> dict[str, Any]:
    anti = matrix - matrix.T
    norm = float(torch.linalg.matrix_norm(matrix))
    residual = float(torch.linalg.matrix_norm(anti))
    return {
        "max_abs_antisymmetry": float(anti.abs().max()),
        "frobenius_antisymmetry_residual": residual,
        "relative_frobenius_antisymmetry": residual / max(norm, 1.0e-30),
    }


def _moment_identity(second: torch.Tensor, centered: torch.Tensor, mu: torch.Tensor) -> dict[str, Any]:
    rhs = centered + torch.outer(mu, mu)
    residual = second - rhs
    lhs_norm = float(torch.linalg.matrix_norm(second))
    rhs_norm = float(torch.linalg.matrix_norm(rhs))
    frob = float(torch.linalg.matrix_norm(residual))
    return {
        "max_absolute_residual": float(residual.abs().max()),
        "frobenius_residual": frob,
        "relative_frobenius_residual": frob / max(lhs_norm, rhs_norm, 1.0e-30),
        "passed": bool(
            torch.allclose(
                second,
                rhs,
                rtol=MOMENT_IDENTITY_RTOL,
                atol=MOMENT_IDENTITY_ATOL,
            )
        ),
    }


@dataclass
class MomentAccumulator:
    feature_dim: int = FEATURE_DIM
    count: int = 0
    sum_z: torch.Tensor = field(init=False)
    sum_zzT: torch.Tensor = field(init=False)
    mean: torch.Tensor = field(init=False)
    M2: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.sum_z = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.sum_zzT = torch.zeros((self.feature_dim, self.feature_dim), dtype=torch.float64)
        self.mean = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.M2 = torch.zeros((self.feature_dim, self.feature_dim), dtype=torch.float64)

    def update(self, batch: torch.Tensor) -> None:
        if batch.ndim != 2 or tuple(batch.shape[1:]) != (self.feature_dim,):
            raise ValueError("moment batch has the wrong shape")
        x = batch.detach().cpu().to(torch.float64).contiguous()
        _finite_matrix(x, "moment feature batch")
        batch_count = int(x.shape[0])
        if batch_count <= 0:
            raise ValueError("moment batch is empty")
        batch_sum = x.sum(dim=0)
        batch_mean = batch_sum / batch_count
        centered = x - batch_mean
        batch_M2 = centered.T.mm(centered)
        self.sum_z += batch_sum
        self.sum_zzT += x.T.mm(x)
        if self.count == 0:
            self.count = batch_count
            self.mean.copy_(batch_mean)
            self.M2.copy_(batch_M2)
            return
        old_count = self.count
        total = old_count + batch_count
        delta = batch_mean - self.mean
        self.M2 += batch_M2 + torch.outer(delta, delta) * (old_count * batch_count / total)
        self.mean += delta * (batch_count / total)
        self.count = total

    def finalize(self, panel: str) -> MomentResult:
        if self.count <= 0:
            raise DMSAHarnessFailure(f"{panel} panel has no features")
        mu = self.sum_z / self.count
        second = self.sum_zzT / self.count
        centered = self.M2 / self.count
        _finite_matrix(second, f"{panel} second moment")
        _finite_matrix(centered, f"{panel} centered moment")
        _finite_matrix(mu, f"{panel} mean")
        symmetry = {
            "second": _matrix_symmetry(second),
            "centered": _matrix_symmetry(centered),
            "pre_symmetrization": True,
        }
        identity = _moment_identity(second, centered, mu)
        if not identity["passed"]:
            raise DMSAHarnessFailure(f"{panel} S=C+mu muT identity failed")
        second = 0.5 * (second + second.T)
        centered = 0.5 * (centered + centered.T)
        return MomentResult(panel, self.count, mu, second, centered, identity, symmetry)


def _panel_images(runtime: dsfa.Upstream, panel: str) -> torch.Tensor:
    data = runtime.runtime.data
    if panel == "fit":
        return data.images.index_select(0, runtime.runtime.fit_indices)
    if panel == "selection":
        return data.test_images.index_select(0, data.selection_indices)
    if panel == "confirmation":
        return data.images.index_select(0, data.confirmation_indices)
    raise ValueError(f"unknown DMSA panel: {panel}")


def _replay_panel(
    model: torch.nn.Module,
    runtime: dsfa.Upstream,
    images: torch.Tensor,
    device: torch.device,
    panel: str,
) -> MomentResult:
    del runtime
    accumulator = MomentAccumulator()
    expected_count = {"fit": FIT_COUNT, "selection": SELECTION_COUNT, "confirmation": CONFIRMATION_COUNT}[panel]
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        camera = lrrc._capture_cmos(model, batch_images, device)
        features = lrrc.mra._feature_projection(camera, complex_value=False).float()
        if tuple(features.shape) != (indices.numel(), FEATURE_DIM):
            raise DMSAHarnessFailure(f"{panel} feature projection shape changed")
        accumulator.update(features)
    result = accumulator.finalize(panel)
    if result.count != expected_count:
        raise DMSAHarnessFailure(f"{panel} streaming count changed")
    return result


def _psd_square_root(matrix: torch.Tensor, label: str) -> tuple[torch.Tensor, dict[str, Any]]:
    _finite_matrix(matrix, label)
    raw, vectors = torch.linalg.eigh(matrix)
    lambda_max = float(raw.max()) if raw.numel() else 0.0
    tau = 100.0 * torch.finfo(torch.float64).eps * FEATURE_DIM * max(lambda_max, 1.0)
    negative = raw.lt(0)
    material = raw.lt(-tau)
    tiny = negative & ~material
    if bool(material.any()):
        raise DMSAHarnessFailure(f"{label} has a material negative eigenvalue")
    clamped = torch.where(negative, torch.zeros_like(raw), raw)
    root = (vectors * clamped.clamp_min(0).sqrt().unsqueeze(0)).mm(vectors.T)
    reconstruction = root.mm(root)
    residual = reconstruction - matrix
    result = {
        "lambda_min_raw": float(raw.min()),
        "lambda_max_raw": lambda_max,
        "negative_eigenvalue_count": int(negative.sum()),
        "tiny_negative_count": int(tiny.sum()),
        "maximum_negative_magnitude": float((-raw[negative]).max()) if bool(negative.any()) else 0.0,
        "tau_psd": tau,
        "square_root_max_abs_residual": float(residual.abs().max()),
        "square_root_frobenius_residual": float(torch.linalg.matrix_norm(residual)),
        "square_root_passed": bool(
            torch.allclose(
                reconstruction,
                matrix,
                rtol=SQUARE_ROOT_RTOL,
                atol=SQUARE_ROOT_ATOL,
            )
        ),
        "rank_cutoff_used": False,
    }
    if not result["square_root_passed"]:
        raise DMSAHarnessFailure(f"{label} square-root reconstruction failed")
    return root, result


def _landmarks(summary: Any) -> dict[str, float]:
    return {str(rank): float(summary.cumulative_energy[rank - 1]) for rank in LANDMARKS}


def _spectrum_report(summary: Any) -> dict[str, Any]:
    payload = summary.compact()
    payload["cumulative_energy_landmarks"] = _landmarks(summary)
    return payload


def _spectrum_comparison(
    selection: Mapping[str, Any], confirmation: Mapping[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name in ("raw_spectrum", "second_moment_spectrum", "centered_variance_spectrum"):
        selection_spectrum = selection[name]
        confirmation_spectrum = confirmation[name]
        selection_landmarks = selection_spectrum["cumulative_energy_landmarks"]
        confirmation_landmarks = confirmation_spectrum["cumulative_energy_landmarks"]
        selection_profile = selection_spectrum["sigma_over_sigma1"]
        confirmation_profile = confirmation_spectrum["sigma_over_sigma1"]
        comparisons[name] = {
            "selection_effective_rank_entropy": selection_spectrum[
                "effective_rank_entropy"
            ],
            "confirmation_effective_rank_entropy": confirmation_spectrum[
                "effective_rank_entropy"
            ],
            "absolute_effective_rank_difference": abs(
                float(selection_spectrum["effective_rank_entropy"])
                - float(confirmation_spectrum["effective_rank_entropy"])
            ),
            "selection_stable_rank": selection_spectrum["stable_rank"],
            "confirmation_stable_rank": confirmation_spectrum["stable_rank"],
            "absolute_stable_rank_difference": abs(
                float(selection_spectrum["stable_rank"])
                - float(confirmation_spectrum["stable_rank"])
            ),
            "absolute_cumulative_landmark_differences": {
                rank: abs(
                    float(selection_landmarks[rank])
                    - float(confirmation_landmarks[rank])
                )
                for rank in selection_landmarks
            },
            "maximum_absolute_condition_profile_difference": max(
                abs(float(left) - float(right))
                for left, right in zip(selection_profile, confirmation_profile)
            ),
        }
    comparisons["raw_panel_identity"] = {
        "singular_values_equal": selection["raw_spectrum"]["singular_values"]
        == confirmation["raw_spectrum"]["singular_values"],
        "cumulative_energy_equal": selection["raw_spectrum"]["cumulative_energy"]
        == confirmation["raw_spectrum"]["cumulative_energy"],
        "descriptive_only": True,
    }
    comparisons["descriptive_only"] = True
    comparisons["panel_agreement_gate"] = False
    return comparisons


def _fraction(value: float, denominator: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    return value / denominator


def _energy_decomposition(moment: MomentResult, G_raw: torch.Tensor) -> dict[str, Any]:
    total = float((moment.second * G_raw.T).sum())
    mean_energy = float(moment.mu.dot(G_raw.mv(moment.mu)))
    variance_energy = float((moment.centered * G_raw.T).sum())
    residual = total - mean_energy - variance_energy
    passed = bool(
        math.isfinite(residual)
        and abs(residual)
        <= MOMENT_IDENTITY_ATOL
        + MOMENT_IDENTITY_RTOL
        * max(abs(total), abs(mean_energy), abs(variance_energy), 1.0)
    )
    if not passed:
        raise DMSAHarnessFailure(f"{moment.panel} output-energy decomposition failed")
    return {
        "E_total": total,
        "E_mean": mean_energy,
        "E_var": variance_energy,
        "identity_residual": residual,
        "identity_passed": passed,
        "mean_fraction": _fraction(mean_energy, total),
        "variance_fraction": _fraction(variance_energy, total),
    }


def _off_diagonal(matrix: torch.Tensor) -> dict[str, Any]:
    diagonal = torch.diag(torch.diag(matrix))
    off = matrix - diagonal
    diagonal_squared = float(torch.diag(matrix).square().sum())
    total_squared = float(matrix.square().sum())
    off_squared = float(off.square().sum())
    return {
        "diagonal_frobenius_squared": diagonal_squared,
        "total_frobenius_squared": total_squared,
        "offdiagonal_frobenius_squared": off_squared,
        "diagonal_frobenius_fraction": _fraction(diagonal_squared, total_squared),
        "offdiagonal_frobenius_fraction": _fraction(off_squared, total_squared),
        "normalized_offdiagonal_norm": (
            math.sqrt(off_squared / total_squared) if total_squared > 0.0 else None
        ),
        "maximum_absolute_offdiagonal": float(off.abs().max()),
    }


def _dsfa_cross_identity(
    panel: str,
    U: torch.Tensor,
    second: torch.Tensor,
    centered: torch.Tensor,
    mu: torch.Tensor,
    dsfa_panel: Mapping[str, Any] | None,
) -> dict[str, Any]:
    B = U.T.mm(second).mm(U)
    B_C = U.T.mm(centered).mm(U)
    diagonal = torch.diag(B)
    centered_component = torch.diag(B_C)
    mean_component = U.T.mv(mu).square()
    mode_residual = diagonal - centered_component - mean_component
    result: dict[str, Any] = {
        "diag_Ut_S_U": diagonal.tolist(),
        "diag_Ut_C_U": centered_component.tolist(),
        "mean_component": mean_component.tolist(),
        "modewise_decomposition_max_abs_residual": float(mode_residual.abs().max()),
        "modewise_decomposition_frobenius_residual": float(torch.linalg.vector_norm(mode_residual)),
        "modewise_decomposition_passed": bool(
            torch.allclose(
                diagonal,
                centered_component + mean_component,
                rtol=MOMENT_IDENTITY_RTOL,
                atol=MOMENT_IDENTITY_ATOL,
            )
        ),
        "B": B,
        "B_C": B_C,
    }
    if dsfa_panel is None:
        result["dsfa_reference"] = "not_evaluated_fit_panel_no_DSFA_reference"
        result["dsfa_identity_passed"] = True
        return result
    rows = dsfa_panel.get("per_mode")
    if not isinstance(rows, list) or len(rows) != FEATURE_DIM:
        raise DMSAHarnessFailure(f"DSFA {panel} per-mode reference is unavailable")
    expected = torch.tensor(
        [float(row["activation_second_moment"]) for row in rows], dtype=torch.float64
    )
    residual = diagonal - expected
    result.update(
        {
            "dsfa_reference": "DSFA attempt-01 per_mode activation_second_moment",
            "dsfa_max_abs_residual": float(residual.abs().max()),
            "dsfa_frobenius_residual": float(torch.linalg.vector_norm(residual)),
            "dsfa_identity_passed": bool(
                torch.allclose(
                    diagonal,
                    expected,
                    rtol=MOMENT_IDENTITY_RTOL,
                    atol=MOMENT_IDENTITY_ATOL,
                )
            ),
        }
    )
    if not result["dsfa_identity_passed"] or not result["modewise_decomposition_passed"]:
        raise DMSAHarnessFailure(f"DMSA {panel} DSFA cross-evidence identity failed")
    return result


def _panel_report(
    moment: MomentResult,
    spectral: Mapping[str, Any],
    raw_spectrum: Any,
    raw_identity: Mapping[str, Any],
    dsfa_panel: Mapping[str, Any] | None,
) -> dict[str, Any]:
    W64 = spectral["W64"]
    U = spectral["U"]
    G_raw = W64.mm(W64.T)
    S_half, S_psd = _psd_square_root(moment.second, f"{moment.panel}.S")
    C_half, C_psd = _psd_square_root(moment.centered, f"{moment.panel}.C")
    raw_second = S_half.mm(G_raw).mm(S_half)
    raw_centered = C_half.mm(G_raw).mm(C_half)
    G_second = 0.5 * (raw_second + raw_second.T)
    G_centered = 0.5 * (raw_centered + raw_centered.T)
    try:
        second_spectrum = lsda._spectrum_from_gram(G_second, f"{moment.panel}.second_moment")
        centered_spectrum = lsda._spectrum_from_gram(G_centered, f"{moment.panel}.centered_variance")
    except lsda.LSDAHarnessFailure as exc:
        raise DMSAHarnessFailure(f"{moment.panel} weighted spectrum failed PSD validation") from exc
    energy = _energy_decomposition(moment, G_raw)
    cross = _dsfa_cross_identity(
        moment.panel,
        U,
        moment.second,
        moment.centered,
        moment.mu,
        dsfa_panel,
    )
    B = cross.pop("B")
    B_C = cross.pop("B_C")
    off_diagonal = {"second_moment": _off_diagonal(B), "centered_variance": _off_diagonal(B_C)}
    matrix_hashes = {
        "mu": tensor_sha256(moment.mu),
        "S": tensor_sha256(moment.second),
        "C": tensor_sha256(moment.centered),
        "B": tensor_sha256(B),
        "B_C": tensor_sha256(B_C),
    }
    panel_report = {
        "panel": moment.panel,
        "role": PANEL_CONTRACT[moment.panel]["role"],
        "evaluation_count": moment.count,
        "feature_dim": FEATURE_DIM,
        "moment_semantics": MOMENT_CONTRACT,
        "mu": moment.mu.tolist(),
        "matrix_hashes": matrix_hashes,
        "moment_identity": moment.identity,
        "symmetry": moment.symmetry,
        "energy_decomposition": energy,
        "psd": {"S": S_psd, "C": C_psd},
        "dsfa_cross_evidence_identity": cross,
        "off_diagonal": off_diagonal,
        "raw_spectrum": _spectrum_report(raw_spectrum),
        "raw_spectrum_identity": dict(raw_identity),
        "second_moment_spectrum": _spectrum_report(second_spectrum),
        "centered_variance_spectrum": _spectrum_report(centered_spectrum),
        "streaming": True,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
        "raw_cmos_bank_saved": False,
        "mode_output_bank_saved": False,
        "internal_stage_capture": False,
        "cmos_only_replay_executed": True,
        "optical_replay_executed": False,
        "finite": True,
    }
    panel_report["harness_passed"] = bool(
        panel_report["moment_identity"]["passed"]
        and panel_report["energy_decomposition"]["identity_passed"]
        and panel_report["psd"]["S"]["square_root_passed"]
        and panel_report["psd"]["C"]["square_root_passed"]
        and panel_report["dsfa_cross_evidence_identity"]["modewise_decomposition_passed"]
        and panel_report["dsfa_cross_evidence_identity"]["dsfa_identity_passed"]
        and panel_report["raw_spectrum_identity"]["passed"]
    )
    return panel_report


def _runtime_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "analysis_dtype": "torch.float64",
        "source_dtype": "torch.float32",
        "microbatch_size": MICROBATCH_SIZE,
        "cuda_used_for_spectral_arithmetic": False,
        "randomized_svd_used": False,
        "large_target_gram_constructed": False,
        "feature_bank_saved": False,
    }


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
) -> dict[str, str]:
    runtime = upstream.runtime.runtime
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "implementation_module": file_sha256(Path(__file__).resolve()),
        "rmal_config": file_sha256(runtime.rmal_consumer.spec.path),
        "rmal_review_contract": file_sha256(runtime.rmal_consumer.review_path),
        "rmal_validation": file_sha256(runtime.rmal_consumer.validation_path),
        "rmal_audit": file_sha256(runtime.rmal_consumer.audit_path),
        "rmal_compact_report": file_sha256(runtime.rmal_consumer.compact_path),
        "rmal_probe": runtime.rmal_probe.artifact_sha256,
        "lsda_config": file_sha256(runtime.lsda_path),
        "lsda_review_contract": file_sha256(runtime.lsda_review_path),
        "lsda_validation": file_sha256(runtime.lsda_validation_path),
        "lsda_audit": file_sha256(runtime.lsda_audit_path),
        "lsda_compact_report": file_sha256(runtime.lsda_compact_path),
        "lrrc_config": file_sha256(upstream.runtime.lrrc["config_path"]),
        "lrrc_review_contract": file_sha256(upstream.runtime.lrrc["review_path"]),
        "lrrc_validation": file_sha256(upstream.runtime.lrrc["validation_path"]),
        "lrrc_audit": file_sha256(upstream.runtime.lrrc["audit_path"]),
        "lrrc_compact_report": file_sha256(upstream.runtime.lrrc["compact_path"]),
        "lrrc_implementation_module": file_sha256(upstream.runtime.lrrc["implementation_path"]),
        "dsfa_config": file_sha256(upstream.dsfa["config_path"]),
        "dsfa_review_contract": file_sha256(upstream.dsfa["review_path"]),
        "dsfa_validation": file_sha256(upstream.dsfa["validation_path"]),
        "dsfa_audit": file_sha256(upstream.dsfa["audit_path"]),
        "dsfa_compact_report": file_sha256(upstream.dsfa["compact_path"]),
        "dataset_manifest": runtime.data.manifest["manifest_hash"],
        "fit_indices": panel_identity["fit"]["indices_sha256"],
        "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
        "lsda_consumer_contract": canonical_sha256(LSDA_CONSUMER),
        "lrrc_consumer_contract": canonical_sha256(LRRC_CONSUMER),
        "dsfa_consumer_contract": canonical_sha256(DSFA_CONSUMER),
        "moment_contract": canonical_sha256(MOMENT_CONTRACT),
        "weighted_spectrum_contract": canonical_sha256(WEIGHTED_SPECTRUM_CONTRACT),
        "off_diagonal_contract": canonical_sha256(OFF_DIAGONAL_CONTRACT),
    }
    return hashes


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "lineage": LINEAGE,
            "moment_contract": MOMENT_CONTRACT,
            "decomposition_contract": DECOMPOSITION_CONTRACT,
            "weighted_spectrum_contract": WEIGHTED_SPECTRUM_CONTRACT,
            "off_diagonal_contract": OFF_DIAGONAL_CONTRACT,
        }
    )
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=dataset_manifest,
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
    *,
    cuda_used: bool = False,
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
        "cuda_used": cuda_used,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse DMSA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("DMSA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"DMSA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("DMSA attempt directory is not a fresh validation-only slot")


def _validation_gates(
    state: Mapping[str, Any],
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
    spectral: Mapping[str, Any],
    raw_identity: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    probe = upstream.runtime.runtime.rmal_probe
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    return (
        GateResult("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True, {}),
        GateResult(
            "frozen_probe_shape_dtype_hash",
            probe.artifact_sha256 == RMAL_CONSUMER["probe_sha256"]
            and tuple(probe.weights.shape) == (FEATURE_DIM, TOTAL_COLUMNS)
            and tuple(probe.bias.shape) == (TOTAL_COLUMNS,)
            and probe.weights.dtype == torch.float32
            and probe.bias.dtype == torch.float32,
            {},
        ),
        GateResult("fit_selection_confirmation_identity", panel_identity["fit"]["indices_sha256"] == FIT_PANEL["indices_sha256"] and panel_identity["selection_confirmation_disjoint"] and panel_identity["fit_confirmation_disjoint"], {}),
        GateResult("lsda_raw_spectrum_inheritance", spectral["q_identity"]["passed"] is True and raw_identity["passed"] is True, raw_identity),
        GateResult("dsfa_terminal_identity", upstream.dsfa["compact"].get("classification") == DSFA_CONSUMER["classification"], {}),
        GateResult("population_moment_contract", MOMENT_CONTRACT["population_denominator"] == "N" and MOMENT_CONTRACT["unbiased_N_minus_1"] is False, {}),
        GateResult(
            "three_panel_isolation",
            tuple(list(PANEL_CONTRACT)[:3]) == ("fit", "selection", "confirmation")
            and PANEL_CONTRACT["fit_is_reference_only"],
            {},
        ),
        GateResult("no_downstream_path", MEASUREMENT["no_rank_search"] and MEASUREMENT["no_rank_truncated_reconstruction"] and MEASUREMENT["descriptive_only"], {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "probe_refit": False, "decoder_called": False, "physical_forward_called": False}),
        GateResult("disk_free_precondition", disk.free >= MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum_bytes": MIN_DISK_FREE_BYTES}),
        GateResult("output_slot_absent", not output_path.parent.exists(), {"path": str(output_path.parent)}),
    )


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    panel_identity = _verify_panel_identity(upstream)
    spectral = _canonical_spectral(upstream.runtime.runtime.rmal_probe, upstream.runtime.runtime.lsda_compact)
    _, raw_identity = _raw_spectrum_identity(spectral)
    output_path = _resolve(args.output)
    gates = _validation_gates(state, upstream, panel_identity, spectral, raw_identity, output_path)
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(spec, review_path, upstream, panel_identity)
    contract = _contract_payload()
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        state,
        upstream.runtime.runtime.data.manifest["manifest_hash"],
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "DMSA static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="DMSA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DMSAHarnessFailure("DMSA CMOS replay requires CUDA")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise DMSAHarnessFailure("DMSA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise DMSAHarnessFailure("DMSA requires at least 10 GiB free storage")
    return int(free), int(total)


def _load_validation(
    path: str | Path,
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, upstream, panel_identity)
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
        raise ValueError("DMSA validation evidence is stale or mismatched")


def _run_invalid_harness(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
    state: Mapping[str, Any],
    reason: str,
) -> None:
    hashes = _input_hashes(spec, review_path, upstream, panel_identity)
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        state,
        upstream.runtime.runtime.data.manifest["manifest_hash"],
        hashes,
        (GateResult("dmsa_harness", False, {"reason": reason}), GateResult("no_update_scope", True, {})),
        (),
        {"lineage": LINEAGE, "reason": reason},
        reason,
    )
    _write_new(output_path, payload, artifact_kind="DMSA invalid-harness audit evidence")


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("DMSA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    panel_identity = _verify_panel_identity(upstream)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    _load_validation(validation_path, spec, review_path, upstream, panel_identity, state)
    device = torch.device(args.device)
    try:
        free_cuda, total_cuda = _resource_preflight(output_path, device)
        source = lrrc.rmfa._load_source(upstream.runtime.runtime.rmal_consumer.bundle, load_checkpoints=True)
        formal = upstream.runtime.runtime.rmal_consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[lrrc.rmal.PRIMARY_CHECKPOINT]
        model = lrrc.mra._load_scaled_model(formal, checkpoint, device)
        model.eval()
        spectral = _canonical_spectral(upstream.runtime.runtime.rmal_probe, upstream.runtime.runtime.lsda_compact)
        raw_spectrum, raw_identity = _raw_spectrum_identity(spectral)
        state_before = lrrc.rmal._state_cpu(model)
        ols_before = lrrc.rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        reports: dict[str, dict[str, Any]] = {}
        for panel in ("fit", "selection", "confirmation"):
            images = _panel_images(upstream.runtime, panel)
            moment = _replay_panel(model, upstream.runtime, images, device, panel)
            dsfa_panel = upstream.dsfa["compact"].get(panel) if panel in {"selection", "confirmation"} else None
            reports[panel] = _panel_report(moment, spectral, raw_spectrum, raw_identity, dsfa_panel)
        state_unchanged = lrrc.rmal._state_equal(state_before, lrrc.rmal._state_cpu(model))
        ols_unchanged = lrrc.rmal._state_equal(ols_before, lrrc.rmal._state_cpu(model.ols))
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        panel_passed = all(report["harness_passed"] for report in reports.values())
        spectrum_comparison = _spectrum_comparison(
            reports["selection"], reports["confirmation"]
        )
        compact = {
            "schema": COMPACT_SCHEMA,
            "hypothesis": HYPOTHESIS,
            "status": "passed",
            "classification": "detector_data_metric_spectral_attribution_completed",
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "lsda_consumer": LSDA_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "lrrc_consumer": LRRC_CONSUMER,
            "dsfa_consumer": DSFA_CONSUMER,
            "fit_panel": panel_identity["fit"],
            "panel_contract": PANEL_CONTRACT,
            "panel_identity": panel_identity,
            "probe": {"sha256": upstream.runtime.runtime.rmal_probe.artifact_sha256, "W_shape": [FEATURE_DIM, TOTAL_COLUMNS], "bias_shape": [TOTAL_COLUMNS], "active_count": EXPECTED_ACTIVE_COUNT, "inactive_count": EXPECTED_INACTIVE_COUNT},
            "spectral_representation": SPECTRAL_REPRESENTATION,
            "raw_spectrum": _spectrum_report(raw_spectrum),
            "raw_spectrum_identity": raw_identity,
            "panels": reports,
            "spectrum_comparison": spectrum_comparison,
            "moment_contract": MOMENT_CONTRACT,
            "decomposition_contract": DECOMPOSITION_CONTRACT,
            "psd_sanity": PSD_SANITY,
            "weighted_spectrum_contract": WEIGHTED_SPECTRUM_CONTRACT,
            "off_diagonal_contract": OFF_DIAGONAL_CONTRACT,
            "runtime": _runtime_metadata(),
            "resource": {"free_cuda_bytes_before": free_cuda, "total_cuda_bytes": total_cuda, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, **RESOURCE_CONTRACT},
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "closed_task_relevant_bottleneck_not_supported": True,
            "training_authorized": False,
            "formal_promotion": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "probe_refit": False,
            "decoder_created": False,
            "checkpoint_mutation": False,
            "generation": False,
            "cuda_used": True,
            "cmos_only_replay_executed": True,
            "optical_replay_executed": False,
            "internal_stage_capture": False,
            "feature_bank_saved": False,
            "target_bank_saved": False,
            "mode_output_bank_saved": False,
            "mechanism_evidence_only": True,
            "descriptive_only": True,
        }
        audit_gates_without_resource = (
            GateResult("panel_replays", panel_passed, {}),
            GateResult("fit_moment_identity", reports["fit"]["moment_identity"]["passed"] is True, reports["fit"]["moment_identity"]),
            GateResult("selection_moment_identity", reports["selection"]["moment_identity"]["passed"] is True, reports["selection"]["moment_identity"]),
            GateResult("confirmation_moment_identity", reports["confirmation"]["moment_identity"]["passed"] is True, reports["confirmation"]["moment_identity"]),
            GateResult("mean_variance_energy_identities", all(report["energy_decomposition"]["identity_passed"] for report in reports.values()), {}),
            GateResult("psd_and_square_root_identities", all(report["psd"]["S"]["square_root_passed"] and report["psd"]["C"]["square_root_passed"] for report in reports.values()), {}),
            GateResult("raw_spectrum_inheritance", raw_identity["passed"] is True, raw_identity),
            GateResult("dsfa_selection_cross_identity", reports["selection"]["dsfa_cross_evidence_identity"]["dsfa_identity_passed"] is True, {}),
            GateResult("dsfa_confirmation_cross_identity", reports["confirmation"]["dsfa_cross_evidence_identity"]["dsfa_identity_passed"] is True, {}),
            GateResult("fit_reference_only", reports["fit"]["dsfa_cross_evidence_identity"]["dsfa_reference"].startswith("not_evaluated"), {}),
            GateResult("selection_confirmation_isolated", panel_identity["selection_confirmation_disjoint"] and panel_identity["fit_confirmation_disjoint"], {}),
            GateResult("state_unchanged", state_unchanged and ols_unchanged, {}),
            GateResult("streaming_no_banks", all(not report["pooled_feature_bank_saved"] and not report["target_bank_saved"] and not report["mode_output_bank_saved"] for report in reports.values()), {}),
            GateResult("finite_measurements", all(report["finite"] and report["harness_passed"] for report in reports.values()), {}),
        )
        compact_size = len(json.dumps(compact, sort_keys=True).encode("utf-8"))
        audit_gates = audit_gates_without_resource + (
            GateResult(
                "resource_limits",
                peak <= MAX_PEAK_CUDA_BYTES and compact_size <= MAX_STORAGE_BYTES,
                {"peak_bytes": peak, "compact_bytes": compact_size},
            ),
        )
        harness_passed = all(gate.passed for gate in audit_gates)
        status = "passed" if harness_passed else "invalid_harness"
        classification = "detector_data_metric_spectral_attribution_completed" if harness_passed else "invalid_harness"
        compact["status"] = status
        compact["classification"] = classification
        compact["next_action"] = NEXT_ACTION_RESULT_REVIEW if harness_passed else None
        compact_path = output_path.with_name("compact_report.json")
        _write_new(compact_path, compact, artifact_kind="DMSA compact report")
        compact_hash = file_sha256(compact_path)
        hashes = _input_hashes(spec, review_path, upstream, panel_identity)
        hashes["dmsa_compact_report"] = compact_hash
        audit_contract = {
            "lineage": LINEAGE,
            "classification": classification,
            "consumer": "project_owner",
            "fit": reports["fit"],
            "selection": reports["selection"],
            "confirmation": reports["confirmation"],
            "raw_spectrum_identity": raw_identity,
            "spectrum_comparison": spectrum_comparison,
            "compact_report_sha256": compact_hash,
            "selection_confirmation_isolated": True,
            "no_update": True,
            "descriptive_only": True,
        }
        audit_payload = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            state,
            upstream.runtime.runtime.data.manifest["manifest_hash"],
            hashes,
            audit_gates,
            (NEXT_ACTION_RESULT_REVIEW,) if harness_passed else (),
            audit_contract,
            None if harness_passed else "DMSA harness gate failed",
            cuda_used=True,
        )
        _write_new(output_path, audit_payload, artifact_kind="DMSA audit evidence")
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return 0 if harness_passed else 2
    except DMSAHarnessFailure as exc:
        if not output_path.exists():
            _run_invalid_harness(output_path, spec, review_path, upstream, panel_identity, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-detector-data-metric-spectral-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-detector-data-metric-spectral-attribution-run":
        return run(args)
    raise ValueError(f"unsupported DMSA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "DECOMPOSITION_CONTRACT",
    "DMSAHarnessFailure",
    "DSFA_CONSUMER",
    "FIT_PANEL",
    "HYPOTHESIS",
    "LINEAGE",
    "MEASUREMENT",
    "MOMENT_CONTRACT",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "NUMERICAL_CONTRACT",
    "OFF_DIAGONAL_CONTRACT",
    "PANEL_CONTRACT",
    "PSD_SANITY",
    "REVIEW_SCHEMA",
    "RMAL_CONSUMER",
    "SPECTRAL_REPRESENTATION",
    "VALIDATION_SCHEMA",
    "WEIGHTED_SPECTRUM_CONTRACT",
    "MomentAccumulator",
    "_assert_contract",
    "_load_review",
    "_load_spec",
    "_moment_identity",
    "_off_diagonal",
    "_spectrum_comparison",
    "_psd_square_root",
    "_review_payload",
    "dispatch",
    "run",
    "validate",
]
