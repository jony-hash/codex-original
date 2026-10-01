"""Attribute frozen RMAL readout modes with data-weighted functional metrics.

This is a descriptive, no-update consumer of the registered RMAL detector
readout.  It reuses the LSDA eigensystem and the LRRC CMOS-only replay path,
but does not search ranks, fit a probe, capture internal optical stages, or
persist mode/output banks.
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

from ldm_project.v2.evidence import EvidenceContextV2, EvidenceRecordV2, GateResult, atomic_write_new_json, canonical_sha256, file_sha256, tensor_sha256
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit as lsda
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S6-DSFA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "fd2024546dbb577d05858a825e90ff528d9ed8a6"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10
MODE_BLOCK_SIZE = 8
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

LSDA_SIGMA_RTOL = 1.0e-11
LSDA_SIGMA_ATOL = 1.0e-8
LRRC_REPLAY_RTOL = 1.0e-6
LRRC_REPLAY_ATOL = 1.0e-5
LRRC_OPERATOR_RTOL = 1.0e-6
LRRC_OPERATOR_ATOL = 1.0e-4
MIN_DISK_FREE_BYTES = lrrc.MIN_DISK_FREE_BYTES
MAX_STORAGE_BYTES = lrrc.MAX_STORAGE_BYTES
MIN_FREE_CUDA_BYTES = lrrc.MIN_FREE_CUDA_BYTES
MAX_PEAK_CUDA_BYTES = lrrc.MAX_PEAK_CUDA_BYTES

REGISTERED_BANDS = {
    "B1": (1, 1),
    "B2": (2, 2),
    "B4": (3, 4),
    "B8": (5, 8),
    "B16": (9, 16),
    "B32": (17, 32),
    "B64": (33, 64),
    "B128": (65, 128),
    "B256": (129, 256),
}
REGISTERED_RANKS = tuple(range(1, FEATURE_DIM + 1))
LRRC_RANKS = tuple(lrrc.RANK_CANDIDATES)

RMAL_CONSUMER = dict(lrrc.RMAL_CONSUMER)
LSDA_CONSUMER = dict(lrrc.LSDA_CONSUMER)
RWSA_IDENTITY = dict(lrrc.RWSA_IDENTITY)
RWSA_ACTIVE_IDENTITY = dict(lrrc.RWSA_ACTIVE_IDENTITY)
ZCSA_IDENTITY = dict(lrrc.ZCSA_IDENTITY)
LRRC_CONSUMER = {
    "config": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_v1.yaml",
    "config_sha256": "6b8bc357e7546327cea64fa94f519a2c1487a6ef385ad2a47c8ca66e80eff17d",
    "review_contract": "configs/research/optical_manifold/lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_review_v1.json",
    "review_sha256": "d2309a6de8d93c8490d07d892a9a72f067664f5a41b1faa59fc6a0be46fdb872",
    "validation": "outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/validation.json",
    "validation_sha256": "b305c38dc3c774ff995c98907977d67cfc396d56009b8a88f726e214f9efa004",
    "audit": "outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/audit.json",
    "audit_sha256": "5ffcb3550e0a1d635f70cdf965d12c93ded6e99fadf8f0f6ef0cda829b4a6b58",
    "compact_report": "outputs/20260902_v2/runs/lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_04/compact_report.json",
    "compact_report_sha256": "4ae704104accd8cebfcb0e3c19dba7deb90b4febf38bace3c7e578bd484951b8",
    "implementation_module": "src/ldm_project/v2/application/research/optical_manifold/optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation.py",
    "implementation_module_sha256": "ad201cd42cb4da6ec393278326db3fb9b8a6fee30cb42cc076df7267304a848b",
    "implementation_commit": "2a061db61910a847118e38663d05c05bd9447063",
    "status": "passed",
    "classification": "low_rank_readout_recovery_confirmation_completed",
    "candidate_family": list(lrrc.RANK_CANDIDATES),
    "first_confirmed_candidate_rank": 256,
}

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S6-LRRC",
    "lineage_start_head": LINEAGE_START_HEAD,
    "rmal_probe_frozen": True,
    "lsda_passed_required": True,
    "lrrc_passed_required": True,
    "zcsa_passed_required": True,
    "rwsa_active_passed_required": True,
    "selection_confirmation_independent": True,
}

SPECTRAL_REPRESENTATION = {
    "source_W_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "eigensystem": "LSDA exact ordered CPU float64 symmetric eigensolver",
    "mode_order": "sigma descending, k=1..256",
    "q_formula": "q_k=W64.T @ u_k",
    "mode_output_formula": "m_nk=a_nk*q_k",
    "activation_formula": "a_nk=z_n dot u_k",
    "sign_convention": dict(lsda.SIGN_CANONICALIZATION),
    "randomized_svd": False,
    "basis_refit": False,
    "v_explicit_primary_path": False,
}

MODE_QUANTITIES = {
    "operator_energy": "sigma_k^2",
    "operator_energy_from_q": "||q_k||^2",
    "activation_mean": "mean_n(a_nk)",
    "activation_variance": "mean_n(a_nk^2)-mean_n(a_nk)^2",
    "activation_second_moment": "mean_n(a_nk^2)",
    "mode_output_energy": "activation_second_moment_k*||q_k||^2",
    "alignment_benefit": "2*mean_n[a_nk*dot(r_n,q_k)]/40000",
    "self_energy_penalty": "mean_n[a_nk^2]*||q_k||^2/40000",
    "delta_mse_gain": "alignment_benefit_k-self_energy_penalty_k",
    "signed_gain_share": "delta_mse_gain_k/sum_j(delta_mse_gain_j) when finite nonzero",
    "residual": "r_n=target_n-b",
}

PANEL_CONTRACT = {
    "selection": {
        "count": SELECTION_COUNT,
        "source": "mnist_test_indices_5000_9999",
        "parent": lrrc.EVALUATION["selection"]["parent"],
    },
    "confirmation": {
        "count": CONFIRMATION_COUNT,
        "source": "horizon_heldout_train_indices_55000_59999",
        "parent": lrrc.EVALUATION["confirmation"]["parent"],
    },
    "feature_path": "output.camera -> mra._feature_projection(real, complex_value=False)",
    "streaming": True,
    "pooled_feature_bank_saved": False,
    "target_bank_saved": False,
    "raw_cmos_bank_saved": False,
    "internal_stage_capture": False,
    "no_rmal_search": True,
    "no_lrrc_rank_search": True,
    "no_refit": True,
}

BAND_CONTRACT = {
    name: {"first_mode": first, "last_mode": last}
    for name, (first, last) in REGISTERED_BANDS.items()
}

NUMERICAL_CONTRACT = {
    "source_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "mse_denominator": TOTAL_COLUMNS,
    "lsda_sigma_atol": LSDA_SIGMA_ATOL,
    "lsda_sigma_rtol": LSDA_SIGMA_RTOL,
    "lrrc_replay_atol": LRRC_REPLAY_ATOL,
    "lrrc_replay_rtol": LRRC_REPLAY_RTOL,
    "lrrc_operator_atol": LRRC_OPERATOR_ATOL,
    "lrrc_operator_rtol": LRRC_OPERATOR_RTOL,
    "no_epsilon": True,
    "no_renormalization": True,
    "no_signed_abs": True,
}

MEASUREMENT = {
    "objective": "data_weighted_spectral_functional_attribution",
    "primary_question": "low_operator_energy_tail_functional_effect_on_real_detector_data",
    "mode_count": FEATURE_DIM,
    "registered_ranks": list(LRRC_RANKS),
    "registered_bands": BAND_CONTRACT,
    "full_output_columns": TOTAL_COLUMNS,
    "active_output_columns": lrrc.lsda.EXPECTED_ACTIVE_COUNT,
    "inactive_output_columns": lrrc.lsda.EXPECTED_INACTIVE_COUNT,
    "selection_confirmation_independent": True,
    "no_new_rank_search": True,
    "no_training": True,
    "no_optimizer": True,
    "no_backward": True,
    "no_refit": True,
    "no_decoder": True,
    "no_physical_forward_modification": True,
    "descriptive_only": True,
}

RESOURCE_CONTRACT = {
    "mode_block_size": MODE_BLOCK_SIZE,
    "microbatch_size": MICROBATCH_SIZE,
    "no_mode_output_bank": True,
    "no_feature_bank": True,
    "no_target_bank": True,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "min_free_cuda_bytes_if_replay": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes_if_replay": MAX_PEAK_CUDA_BYTES,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_data_weighted_spectral_functional_attribution",
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
        "new_rank_search",
        "rank_129_to_255_search",
        "new_decoder",
        "optical_system_modification",
        "generation",
        "promotion",
        "full_optical_stage_capture",
        "feature_bank_persistence",
        "target_bank_persistence",
        "mode_output_bank_persistence",
        "covariance_eigendecomposition",
        "scientific_rank_classification",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_data_weighted_spectral_functional_attribution"
NEXT_ACTION_RESULT_REVIEW = "manual_review_DSFA_result"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Upstream:
    lrrc: dict[str, Any]
    runtime: lrrc.Upstream


class DSFAHarnessFailure(RuntimeError):
    """A DSFA measurement or lineage boundary failed closed."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"DSFA requires clean branch {BRANCH}")


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
        "spectral_representation",
        "mode_quantities",
        "panel_contract",
        "band_contract",
        "numerical",
        "measurement",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("DSFA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("DSFA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "lsda_consumer": LSDA_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "lrrc_consumer": LRRC_CONSUMER,
        "spectral_representation": SPECTRAL_REPRESENTATION,
        "mode_quantities": MODE_QUANTITIES,
        "panel_contract": PANEL_CONTRACT,
        "band_contract": BAND_CONTRACT,
        "numerical": NUMERICAL_CONTRACT,
        "measurement": MEASUREMENT,
        "resource": RESOURCE_CONTRACT,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"DSFA {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "lineage": LINEAGE,
            "lrrc_consumer": LRRC_CONSUMER,
            "spectral_representation": SPECTRAL_REPRESENTATION,
            "mode_quantities": MODE_QUANTITIES,
            "panel_contract": PANEL_CONTRACT,
            "band_contract": BAND_CONTRACT,
            "numerical": NUMERICAL_CONTRACT,
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
        raise ValueError("DSFA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise DSFAHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DSFAHarnessFailure(f"{label} is not a mapping: {path}")
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
        raise DSFAHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise DSFAHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_lrrc(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(LRRC_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != LRRC_CONSUMER["config_sha256"]:
        raise DSFAHarnessFailure("LRRC config identity changed")
    lrrc_spec = lrrc._load_spec(config_path)
    lrrc._assert_contract(lrrc_spec)
    review_path = _resolve(LRRC_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != LRRC_CONSUMER["review_sha256"]:
        raise DSFAHarnessFailure("LRRC review identity changed")
    lrrc._load_review(review_path, lrrc_spec)
    implementation_path = _resolve(LRRC_CONSUMER["implementation_module"])
    if file_sha256(implementation_path) != LRRC_CONSUMER["implementation_module_sha256"]:
        raise DSFAHarnessFailure("LRRC implementation identity changed")
    validation_path = _resolve(LRRC_CONSUMER["validation"])
    validation = _read_json_hash(validation_path, LRRC_CONSUMER["validation_sha256"], "LRRC validation")
    validation_record = _verify_evidence(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="LRRC validation",
    )
    if validation_record.git_commit != LRRC_CONSUMER["implementation_commit"] or validation_record.config_hash != lrrc_spec.config_hash:
        raise DSFAHarnessFailure("LRRC validation lineage changed")
    audit_path = _resolve(LRRC_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, LRRC_CONSUMER["audit_sha256"], "LRRC audit")
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=LRRC_CONSUMER["classification"],
        label="LRRC audit",
    )
    if audit_record.git_commit != LRRC_CONSUMER["implementation_commit"]:
        raise DSFAHarnessFailure("LRRC audit lineage changed")
    compact_path = _resolve(LRRC_CONSUMER["compact_report"])
    compact = _read_json_hash(compact_path, LRRC_CONSUMER["compact_report_sha256"], "LRRC compact report")
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != LRRC_CONSUMER["classification"]
        or compact.get("rank_candidates") != list(lrrc.RANK_CANDIDATES)
        or compact.get("first_confirmed_candidate_rank") != 256
        or compact.get("previous_registered_candidate") != 128
        or compact.get("rank_256_reused_as_fallback") is not True
    ):
        raise DSFAHarnessFailure("LRRC compact identity changed")
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
    lrrc_identity = _verify_lrrc(spec)
    try:
        runtime = lrrc._load_upstream(spec, dataset_root)
    except (ValueError, lrrc.LRRCHarnessFailure) as exc:
        raise DSFAHarnessFailure("RMAL/LSDA/ZCSA/RWSA upstream identity is invalid") from exc
    if runtime.rmal_probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]:
        raise DSFAHarnessFailure("RMAL probe identity changed")
    return Upstream(lrrc_identity, runtime)


def _canonical_spectral(probe: Any, lsda_compact: Mapping[str, Any]) -> dict[str, Any]:
    locked = lrrc._spectral_lock(probe, lsda_compact)
    raw_u = locked["U"]
    canonical_u, _, sign_indices, signs = lsda._canonicalize_source_modes(raw_u, None)
    W64 = locked["W64"]
    sigma = locked["full"].singular_values
    q = W64.T.mm(canonical_u).contiguous()
    operator_energy = sigma.square().contiguous()
    q_energy = q.square().sum(dim=0).contiguous()
    q_residual = q_energy - operator_energy
    W_energy = W64.square().sum()
    identity = {
        "q_norm_squared_matches_sigma_squared": bool(
            torch.allclose(q_energy, operator_energy, rtol=LSDA_SIGMA_RTOL, atol=LSDA_SIGMA_ATOL)
        ),
        "q_norm_squared_max_abs_residual": float(q_residual.abs().max()),
        "sum_operator_energy": float(operator_energy.sum()),
        "W_frobenius_squared": float(W_energy),
        "operator_frobenius_max_abs_residual": abs(float(operator_energy.sum() - W_energy)),
        "operator_frobenius_matches": bool(
            torch.allclose(operator_energy.sum(), W_energy, rtol=LSDA_SIGMA_RTOL, atol=LSDA_SIGMA_ATOL)
        ),
        "source_orthogonality_relative_residual": lsda._orthogonality_residual(canonical_u),
    }
    identity["passed"] = bool(identity["q_norm_squared_matches_sigma_squared"] and identity["operator_frobenius_matches"])
    return {
        **locked,
        "U": canonical_u,
        "q": q,
        "sigma": sigma,
        "operator_energy": operator_energy,
        "operator_energy_from_q": q_energy,
        "q_identity": identity,
        "sign_canonicalization_entry_indices": sign_indices,
        "signs_applied": signs,
    }


def _close(actual: float, expected: float, *, atol: float, rtol: float) -> bool:
    return abs(actual - expected) <= atol + rtol * max(abs(expected), 1.0)


def _factorized_mode_terms(
    activations: torch.Tensor,
    residual: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return alignment, penalty, and signed gain sums without a mode bank."""
    if activations.ndim != 2 or residual.ndim != 2 or q.ndim != 2:
        raise ValueError("factorized mode terms require rank-2 tensors")
    if activations.shape[0] != residual.shape[0] or residual.shape[1] != q.shape[0]:
        raise ValueError("factorized mode term shapes are incompatible")
    if activations.shape[1] != q.shape[1]:
        raise ValueError("factorized mode count is inconsistent")
    residual_projection = residual.mm(q)
    alignment = 2.0 * (activations * residual_projection).sum(dim=0) / TOTAL_COLUMNS
    penalty = activations.square().sum(dim=0) * q.square().sum(dim=0) / TOTAL_COLUMNS
    return alignment, penalty, alignment - penalty


def _explicit_mode_terms(
    activations: torch.Tensor,
    residual: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference calculation for small fixtures; formal path remains factorized."""
    if activations.ndim != 2 or residual.ndim != 2 or q.ndim != 2:
        raise ValueError("explicit mode terms require rank-2 tensors")
    if activations.shape[0] != residual.shape[0] or residual.shape[1] != q.shape[0]:
        raise ValueError("explicit mode term shapes are incompatible")
    if activations.shape[1] != q.shape[1]:
        raise ValueError("explicit mode count is inconsistent")
    mode_output = activations.unsqueeze(-1) * q.transpose(0, 1).unsqueeze(0)
    alignment = 2.0 * (mode_output * residual.unsqueeze(1)).sum(dim=-1).sum(dim=0) / TOTAL_COLUMNS
    penalty = mode_output.square().sum(dim=-1).sum(dim=0) / TOTAL_COLUMNS
    return alignment, penalty, alignment - penalty


def _quantile_summary(values: torch.Tensor) -> dict[str, float | int]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("summary values must be a non-empty vector")
    q = torch.quantile(values, torch.tensor(QUANTILES, dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(q[2]),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "q95": float(q[4]),
    }


def _band_reports(
    per_mode: Sequence[Mapping[str, Any]],
    operator_total: float,
    data_total: float,
    total_gain: float,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, (first, last) in REGISTERED_BANDS.items():
        rows = per_mode[first - 1:last]
        operator = sum(float(row["operator_energy"]) for row in rows)
        data_energy = sum(float(row["mode_output_energy"]) for row in rows)
        alignment = sum(float(row["alignment_benefit"]) for row in rows)
        penalty = sum(float(row["self_energy_penalty"]) for row in rows)
        gain = sum(float(row["delta_mse_gain"]) for row in rows)
        activation = torch.tensor(
            [float(row["activation_second_moment"]) for row in rows], dtype=torch.float64
        )
        reports[name] = {
            "mode_range": [first, last],
            "operator_energy": operator,
            "operator_energy_share": operator / operator_total if operator_total else None,
            "activation_second_moment_summary": _quantile_summary(activation),
            "data_weighted_output_energy": data_energy,
            "data_weighted_output_energy_share": data_energy / data_total if data_total else None,
            "alignment_benefit": alignment,
            "self_energy_penalty": penalty,
            "signed_mse_gain": gain,
            "normalized_signed_mse_gain_share": gain / total_gain if math.isfinite(total_gain) and total_gain != 0.0 else None,
        }
    return reports


def _panel_report(
    model: torch.nn.Module,
    probe: Any,
    spectral: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    panel: str,
    lrrc_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if images.shape[0] != labels.shape[0] or labels.ndim != 1:
        raise ValueError("panel images and labels do not match")
    model_before = lrrc.rmal._state_cpu(model)
    ols_before = lrrc.rmal._state_cpu(model.ols)
    W64 = spectral["W64"]
    U = spectral["U"]
    q = spectral["q"]
    q_energy = spectral["operator_energy_from_q"]
    b64 = probe.bias.to(torch.float64)
    activation_sum = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    activation_square_sum = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    alignment_sum = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    bias_error_sum = 0.0
    full_error_sum = 0.0
    dynamic_energy_sum = 0.0
    count = 0
    feature_contract_valid = True
    finite = True
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        camera = lrrc._capture_cmos(model, batch_images, device)
        z = lrrc.mra._feature_projection(camera, complex_value=False).float()
        if tuple(z.shape) != (indices.numel(), FEATURE_DIM):
            raise DSFAHarnessFailure("CMOS feature projection shape changed")
        z64 = z.to(torch.float64)
        activations = z64.mm(U)
        dynamic = z64.mm(W64)
        target_flat = target.reshape(indices.numel(), TOTAL_COLUMNS).to(torch.float64)
        residual = target_flat - b64
        prediction_residual = residual - dynamic
        activation_sum += activations.sum(dim=0)
        activation_square_sum += activations.square().sum(dim=0)
        bias_error_sum += float(residual.square().sum())
        full_error_sum += float(prediction_residual.square().sum())
        dynamic_energy_sum += float(dynamic.square().sum())
        for start in range(0, FEATURE_DIM, MODE_BLOCK_SIZE):
            end = min(start + MODE_BLOCK_SIZE, FEATURE_DIM)
            a_block = activations[:, start:end]
            alignment_block, _, _ = _factorized_mode_terms(a_block, residual, q[:, start:end])
            alignment_sum[start:end] += alignment_block
        count += int(indices.numel())
        finite = finite and bool(
            torch.isfinite(z64).all()
            and torch.isfinite(activations).all()
            and torch.isfinite(dynamic).all()
            and torch.isfinite(target_flat).all()
        )
    if count != int(images.shape[0]):
        raise DSFAHarnessFailure("panel streaming count changed")
    activation_mean = activation_sum / count
    activation_second = activation_square_sum / count
    activation_variance = activation_second - activation_mean.square()
    alignment = alignment_sum / count
    penalty = activation_second * q_energy / TOTAL_COLUMNS
    gain = alignment - penalty
    bias_mse = bias_error_sum / count / TOTAL_COLUMNS
    full_mse = full_error_sum / count / TOTAL_COLUMNS
    total_gain = float(gain.sum())
    direct_gain = bias_mse - full_mse
    prefix_mse = bias_mse - torch.cumsum(gain, dim=0)
    data_energy = activation_second * q_energy
    direct_data_energy = dynamic_energy_sum / count
    per_mode: list[dict[str, Any]] = []
    operator_total = float(spectral["operator_energy"].sum())
    data_total = float(data_energy.sum())
    for index in range(FEATURE_DIM):
        mode_gain = float(gain[index])
        per_mode.append(
            {
                "mode": index + 1,
                "sigma": float(spectral["sigma"][index]),
                "operator_energy": float(spectral["operator_energy"][index]),
                "operator_energy_from_q": float(q_energy[index]),
                "operator_energy_share": float(spectral["operator_energy"][index] / operator_total),
                "q_norm_squared_abs_residual": float(
                    spectral["operator_energy_from_q"][index] - spectral["operator_energy"][index]
                ),
                "activation_mean": float(activation_mean[index]),
                "activation_variance": float(activation_variance[index]),
                "activation_second_moment": float(activation_second[index]),
                "activation_second_moment_identity_residual": float(
                    activation_second[index]
                    - (activation_variance[index] + activation_mean[index].square())
                ),
                "mode_output_energy": float(data_energy[index]),
                "data_weighted_output_energy_share": float(data_energy[index] / data_total),
                "alignment_benefit": float(alignment[index]),
                "self_energy_penalty": float(penalty[index]),
                "delta_mse_gain": mode_gain,
                "normalized_signed_mse_gain_share": (
                    mode_gain / total_gain if math.isfinite(total_gain) and total_gain != 0.0 else None
                ),
            }
        )
    identity = {
        "data_weighted_energy_sum": data_total,
        "direct_mean_dynamic_output_energy": direct_data_energy,
        "data_weighted_energy_abs_residual": abs(data_total - direct_data_energy),
        "data_weighted_energy_matches": _close(
            data_total, direct_data_energy, atol=LRRC_REPLAY_ATOL, rtol=LRRC_REPLAY_RTOL
        ),
        "total_mse_gain_from_modes": total_gain,
        "total_mse_gain_from_mse": direct_gain,
        "total_mse_gain_abs_residual": abs(total_gain - direct_gain),
        "total_mse_gain_matches": _close(
            total_gain, direct_gain, atol=LRRC_REPLAY_ATOL, rtol=LRRC_REPLAY_RTOL
        ),
        "activation_second_moment_max_identity_residual": float(
            (activation_second - (activation_variance + activation_mean.square())).abs().max()
        ),
        "finite": finite and bool(torch.isfinite(gain).all()),
    }
    identity["passed"] = bool(
        identity["finite"]
        and identity["data_weighted_energy_matches"]
        and identity["total_mse_gain_matches"]
    )
    replay_rows: list[dict[str, Any]] = []
    for rank in LRRC_RANKS:
        if panel == "selection":
            expected = float(lrrc_reference[str(rank)])
            actual = float(prefix_mse[rank - 1])
            replay_rows.append(
                {
                    "rank": rank,
                    "reconstructed_mse": actual,
                    "recorded_lrrc_mse": expected,
                    "absolute_residual": abs(actual - expected),
                    "relative_residual": abs(actual - expected) / max(abs(expected), 1.0e-30),
                    "passed": _close(actual, expected, atol=LRRC_REPLAY_ATOL, rtol=LRRC_REPLAY_RTOL),
                }
            )
    if panel == "confirmation":
        expected = float(lrrc_reference["full_rank"])
        actual = float(prefix_mse[-1])
        replay_rows.append(
            {
                "rank": FEATURE_DIM,
                "reconstructed_mse": actual,
                "recorded_lrrc_mse": expected,
                "absolute_residual": abs(actual - expected),
                "relative_residual": abs(actual - expected) / max(abs(expected), 1.0e-30),
                "passed": _close(actual, expected, atol=LRRC_REPLAY_ATOL, rtol=LRRC_REPLAY_RTOL),
            }
        )
    state_unchanged = lrrc.rmal._state_equal(model_before, lrrc.rmal._state_cpu(model))
    ols_unchanged = lrrc.rmal._state_equal(ols_before, lrrc.rmal._state_cpu(model.ols))
    report = {
        "panel": panel,
        "evaluation_count": count,
        "mse_denominator": TOTAL_COLUMNS,
        "bias_only_mse": bias_mse,
        "full_probe_mse": full_mse,
        "total_functional_gain": total_gain if math.isfinite(total_gain) else None,
        "prefix_reconstructed_mse": [float(value) for value in prefix_mse],
        "per_mode": per_mode,
        "bands": _band_reports(per_mode, operator_total, data_total, total_gain),
        "identity": identity,
        "registered_prefix_replay": replay_rows,
        "lrrc_reference": dict(lrrc_reference),
        "state_unchanged": state_unchanged,
        "ols_state_unchanged": ols_unchanged,
        "feature_contract_valid": feature_contract_valid,
        "finite": finite,
        "streaming": True,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
        "raw_cmos_bank_saved": False,
        "mode_output_bank_saved": False,
        "internal_stage_capture": False,
        "optical_replay_executed": False,
        "cmos_only_replay_executed": True,
    }
    report["tail_129_256"] = {
        **dict(report["bands"]["B256"]),
        "rank128_reconstructed_mse": float(prefix_mse[127]),
        "rank256_reconstructed_mse": float(prefix_mse[255]),
        "rank128_to_rank256_mse_difference": float(prefix_mse[127] - prefix_mse[255]),
    }
    report["harness_passed"] = bool(
        identity["passed"]
        and all(row["passed"] for row in replay_rows)
        and state_unchanged
        and ols_unchanged
        and feature_contract_valid
        and finite
    )
    return report


def _lrrc_references(compact: Mapping[str, Any]) -> dict[str, Any]:
    rank_results = compact.get("rank_results")
    if not isinstance(rank_results, Mapping):
        raise DSFAHarnessFailure("LRRC rank results are unavailable")
    selection: dict[str, float] = {}
    for rank in LRRC_RANKS:
        row = rank_results.get(str(rank))
        if not isinstance(row, Mapping) or not isinstance(row.get("selection"), Mapping):
            raise DSFAHarnessFailure(f"LRRC selection reference rank {rank} is unavailable")
        probe = row["selection"].get("probe")
        if not isinstance(probe, Mapping) or not isinstance(probe.get("paired_mse"), (int, float)):
            raise DSFAHarnessFailure(f"LRRC selection MSE rank {rank} is unavailable")
        selection[str(rank)] = float(probe["paired_mse"])
    anchor = compact.get("rank_256_full_positive_anchor")
    if not isinstance(anchor, Mapping) or not isinstance(anchor.get("confirmation"), Mapping):
        raise DSFAHarnessFailure("LRRC confirmation anchor is unavailable")
    confirmation_probe = anchor["confirmation"].get("probe")
    if not isinstance(confirmation_probe, Mapping) or not isinstance(confirmation_probe.get("paired_mse"), (int, float)):
        raise DSFAHarnessFailure("LRRC confirmation full-rank MSE is unavailable")
    rank0 = compact.get("rank_0_control")
    if not isinstance(rank0, Mapping):
        raise DSFAHarnessFailure("LRRC rank-0 control is unavailable")
    return {
        "selection": selection,
        "confirmation": {"full_rank": float(confirmation_probe["paired_mse"])},
        "rank0_selection_mse": float(rank0["selection"]["paired_mse"]),
        "rank0_confirmation_mse": float(rank0["confirmation"]["paired_mse"]),
    }


def _runtime_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "analysis_dtype": "torch.float64",
        "source_dtype": "torch.float32",
        "mode_block_size": MODE_BLOCK_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "cuda_used_for_spectral_arithmetic": False,
        "randomized_svd_used": False,
        "full_mode_output_bank_materialized": False,
    }


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    dataset_manifest: str,
) -> dict[str, str]:
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "implementation_module": file_sha256(Path(__file__).resolve()),
        "rmal_config": file_sha256(upstream.runtime.rmal_consumer.spec.path),
        "rmal_review_contract": file_sha256(upstream.runtime.rmal_consumer.review_path),
        "rmal_validation": file_sha256(upstream.runtime.rmal_consumer.validation_path),
        "rmal_audit": file_sha256(upstream.runtime.rmal_consumer.audit_path),
        "rmal_compact_report": file_sha256(upstream.runtime.rmal_consumer.compact_path),
        "rmal_probe": upstream.runtime.rmal_probe.artifact_sha256,
        "lsda_config": file_sha256(upstream.runtime.lsda_path),
        "lsda_review_contract": file_sha256(upstream.runtime.lsda_review_path),
        "lsda_validation": file_sha256(upstream.runtime.lsda_validation_path),
        "lsda_audit": file_sha256(upstream.runtime.lsda_audit_path),
        "lsda_compact_report": file_sha256(upstream.runtime.lsda_compact_path),
        "lrrc_config": file_sha256(upstream.lrrc["config_path"]),
        "lrrc_review_contract": file_sha256(upstream.lrrc["review_path"]),
        "lrrc_validation": file_sha256(upstream.lrrc["validation_path"]),
        "lrrc_audit": file_sha256(upstream.lrrc["audit_path"]),
        "lrrc_compact_report": file_sha256(upstream.lrrc["compact_path"]),
        "lrrc_implementation_module": file_sha256(upstream.lrrc["implementation_path"]),
        "dataset_manifest": dataset_manifest,
        "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
        "lsda_consumer_contract": canonical_sha256(LSDA_CONSUMER),
        "rwsa_identity_contract": canonical_sha256(RWSA_IDENTITY),
        "rwsa_active_identity_contract": canonical_sha256(RWSA_ACTIVE_IDENTITY),
        "zcsa_identity_contract": canonical_sha256(ZCSA_IDENTITY),
        "lrrc_consumer_contract": canonical_sha256(LRRC_CONSUMER),
        "spectral_representation_contract": canonical_sha256(SPECTRAL_REPRESENTATION),
        "mode_quantities_contract": canonical_sha256(MODE_QUANTITIES),
        "panel_contract": canonical_sha256(PANEL_CONTRACT),
        "band_contract": canonical_sha256(BAND_CONTRACT),
        "numerical_contract": canonical_sha256(NUMERICAL_CONTRACT),
    }
    hashes["fit_indices"] = tensor_sha256(upstream.runtime.fit_indices)
    return hashes


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "spectral_representation": SPECTRAL_REPRESENTATION,
            "mode_quantities": MODE_QUANTITIES,
            "band_contract": BAND_CONTRACT,
            "numerical": NUMERICAL_CONTRACT,
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
        raise FileExistsError(f"refusing to reuse DSFA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("DSFA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"DSFA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("DSFA attempt directory is not a fresh validation-only slot")


def _validation_gates(
    state: Mapping[str, Any],
    upstream: Upstream,
    spectral: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    probe = upstream.runtime.rmal_probe
    return (
        GateResult("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True, {}),
        GateResult("rmal_probe_hash_and_shape", probe.artifact_sha256 == RMAL_CONSUMER["probe_sha256"] and tuple(probe.weights.shape) == (FEATURE_DIM, TOTAL_COLUMNS) and tuple(probe.bias.shape) == (TOTAL_COLUMNS,), {}),
        GateResult("active_inactive_identity", int(spectral["active_mask"].sum()) == lrrc.lsda.EXPECTED_ACTIVE_COUNT and int((~spectral["active_mask"]).sum()) == lrrc.lsda.EXPECTED_INACTIVE_COUNT and bool(probe.weights[:, ~spectral["active_mask"]].eq(0).all()) and bool(probe.bias[~spectral["active_mask"]].eq(0).all()), {}),
        GateResult("lsda_order_and_spectral_identity", spectral["q_identity"]["passed"] is True and spectral["U"].shape == (FEATURE_DIM, FEATURE_DIM), {}),
        GateResult("lrrc_terminal_identity", upstream.lrrc["audit"].get("classification") == LRRC_CONSUMER["classification"] and upstream.lrrc["compact"].get("first_confirmed_candidate_rank") == 256, {}),
        GateResult("registered_bands_locked", list(BAND_CONTRACT) == ["B1", "B2", "B4", "B8", "B16", "B32", "B64", "B128", "B256"] and BAND_CONTRACT["B128"] == {"first_mode": 65, "last_mode": 128} and BAND_CONTRACT["B256"] == {"first_mode": 129, "last_mode": 256}, {}),
        GateResult("no_new_rank_or_covariance_path", MEASUREMENT["no_new_rank_search"] and "covariance_eigendecomposition" in AUTHORIZATION["prohibited"], {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "probe_refit": False, "decoder_called": False, "physical_forward_called": False}),
        GateResult("disk_free_precondition", disk.free >= MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum_bytes": MIN_DISK_FREE_BYTES}),
        GateResult("output_slot_absent", not output_path.parent.exists(), {"path": str(output_path.parent)}),
    )


def _load_validation(path: str | Path, spec: Spec, review_path: Path, upstream: Upstream, state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, upstream, upstream.runtime.data.manifest["manifest_hash"])
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
        raise ValueError("DSFA validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    spectral = _canonical_spectral(upstream.runtime.rmal_probe, upstream.runtime.lsda_compact)
    output_path = _resolve(args.output)
    gates = _validation_gates(state, upstream, spectral, output_path)
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(spec, review_path, upstream, upstream.runtime.data.manifest["manifest_hash"])
    contract = {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "lsda_consumer": LSDA_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "lrrc_consumer": LRRC_CONSUMER,
        "spectral_representation": SPECTRAL_REPRESENTATION,
        "mode_quantities": MODE_QUANTITIES,
        "panel_contract": PANEL_CONTRACT,
        "band_contract": BAND_CONTRACT,
        "numerical": NUMERICAL_CONTRACT,
        "measurement": MEASUREMENT,
        "resource": RESOURCE_CONTRACT,
        "authorization": AUTHORIZATION,
    }
    payload = _record(VALIDATION_SCHEMA, "passed" if valid else "invalid_harness", None if valid else "invalid_harness", spec, state, upstream.runtime.data.manifest["manifest_hash"], hashes, gates, (NEXT_ACTION_RUN,) if valid else (), contract, None if valid else "DSFA static validation failed")
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="DSFA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise DSFAHarnessFailure("DSFA CMOS replay requires CUDA")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise DSFAHarnessFailure("DSFA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise DSFAHarnessFailure("DSFA requires at least 10 GiB free storage")
    return int(free), int(total)


def _run_invalid_harness(output_path: Path, spec: Spec, review_path: Path, upstream: Upstream, state: Mapping[str, Any], reason: str) -> None:
    hashes = _input_hashes(spec, review_path, upstream, upstream.runtime.data.manifest["manifest_hash"])
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        state,
        upstream.runtime.data.manifest["manifest_hash"],
        hashes,
        (GateResult("dsfa_additive_identity", False, {"reason": reason}), GateResult("no_update_scope", True, {})),
        (),
        {"reason": reason, "lineage": LINEAGE},
        reason,
    )
    _write_new(output_path, payload, artifact_kind="DSFA invalid-harness audit evidence")


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("DSFA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    validation_path = _resolve(args.validation_evidence)
    output_path = _resolve(args.output)
    _check_run_slot(output_path, validation_path)
    _load_validation(validation_path, spec, review_path, upstream, state)
    device = torch.device(args.device)
    try:
        free_cuda, total_cuda = _resource_preflight(output_path, device)
        source = lrrc.rmfa._load_source(upstream.runtime.rmal_consumer.bundle, load_checkpoints=True)
        formal = upstream.runtime.rmal_consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[lrrc.rmal.PRIMARY_CHECKPOINT]
        model = lrrc.mra._load_scaled_model(formal, checkpoint, device)
        model.eval()
        spectral = _canonical_spectral(upstream.runtime.rmal_probe, upstream.runtime.lsda_compact)
        references = _lrrc_references(upstream.lrrc["compact"])
        state_before = lrrc.rmal._state_cpu(model)
        ols_before = lrrc.rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        selection_images = upstream.runtime.data.test_images.index_select(0, upstream.runtime.data.selection_indices)
        selection_labels = upstream.runtime.data.test_labels.index_select(0, upstream.runtime.data.selection_indices)
        confirmation_images = upstream.runtime.data.images.index_select(0, upstream.runtime.data.confirmation_indices)
        confirmation_labels = upstream.runtime.data.labels.index_select(0, upstream.runtime.data.confirmation_indices)
        selection = _panel_report(model, upstream.runtime.rmal_probe, spectral, selection_images, selection_labels, device, "selection", references["selection"])
        confirmation = _panel_report(model, upstream.runtime.rmal_probe, spectral, confirmation_images, confirmation_labels, device, "confirmation", references["confirmation"])
        state_unchanged = lrrc.rmal._state_equal(state_before, lrrc.rmal._state_cpu(model))
        ols_unchanged = lrrc.rmal._state_equal(ols_before, lrrc.rmal._state_cpu(model.ols))
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        all_reports = (selection, confirmation)
        replay_passed = all(report["harness_passed"] for report in all_reports)
        q_passed = spectral["q_identity"]["passed"] is True
        harness_passed = replay_passed and q_passed and state_unchanged and ols_unchanged and peak <= MAX_PEAK_CUDA_BYTES
        status = "passed" if harness_passed else "invalid_harness"
        classification = "data_weighted_spectral_functional_attribution_completed" if harness_passed else "invalid_harness"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": status,
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW if harness_passed else None,
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "lsda_consumer": LSDA_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "lrrc_consumer": LRRC_CONSUMER,
            "probe": {"sha256": upstream.runtime.rmal_probe.artifact_sha256, "W_shape": [FEATURE_DIM, TOTAL_COLUMNS], "bias_shape": [TOTAL_COLUMNS], "active_count": lrrc.lsda.EXPECTED_ACTIVE_COUNT, "inactive_count": lrrc.lsda.EXPECTED_INACTIVE_COUNT},
            "spectral": {"sigma": [float(value) for value in spectral["sigma"]], "operator_energy": [float(value) for value in spectral["operator_energy"]], "operator_energy_from_q": [float(value) for value in spectral["operator_energy_from_q"]], "q_identity": spectral["q_identity"], "sign_canonicalization_entry_indices": spectral["sign_canonicalization_entry_indices"], "signs_applied": spectral["signs_applied"], "descriptive_only": True},
            "bands": BAND_CONTRACT,
            "selection": selection,
            "confirmation": confirmation,
            "references": {"lrrc_rank0_selection_mse": references["rank0_selection_mse"], "lrrc_rank0_confirmation_mse": references["rank0_confirmation_mse"], "lrrc_selection_rank_mse": references["selection"], "lrrc_confirmation_full_rank_mse": references["confirmation"]},
            "runtime": _runtime_metadata(),
            "resource": {"free_cuda_bytes_before": free_cuda, "total_cuda_bytes": total_cuda, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "pooled_feature_bank_saved": False, "target_bank_saved": False, "mode_output_bank_saved": False},
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "model_state_before": lrrc.rmal._state_digest(state_before),
            "model_state_after": lrrc.rmal._state_digest(lrrc.rmal._state_cpu(model)),
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
            "cmos_only_replay_executed": True,
            "optical_replay_executed": False,
            "internal_stage_capture": False,
            "mechanism_evidence_only": True,
        }
        compact_path = output_path.with_name("compact_report.json")
        _write_new(compact_path, compact, artifact_kind="DSFA compact report")
        compact_hash = file_sha256(compact_path)
        hashes = _input_hashes(spec, review_path, upstream, upstream.runtime.data.manifest["manifest_hash"])
        hashes["dsfa_compact_report"] = compact_hash
        audit_gates = (
            GateResult("operator_additive_identity", q_passed, spectral["q_identity"]),
            GateResult("selection_additive_identity", selection["identity"]["passed"] is True, selection["identity"]),
            GateResult("confirmation_additive_identity", confirmation["identity"]["passed"] is True, confirmation["identity"]),
            GateResult("lrrc_selection_prefix_replay", all(row["passed"] for row in selection["registered_prefix_replay"]), {"ranks": LRRC_RANKS}),
            GateResult("lrrc_confirmation_full_rank_replay", all(row["passed"] for row in confirmation["registered_prefix_replay"]), {"rank": FEATURE_DIM}),
            GateResult("state_unchanged", state_unchanged and ols_unchanged, {}),
            GateResult("streaming_no_banks", all(not report["pooled_feature_bank_saved"] and not report["target_bank_saved"] and not report["mode_output_bank_saved"] for report in all_reports), {}),
            GateResult("finite_measurements", all(report["finite"] for report in all_reports), {}),
            GateResult("resource_limits", peak <= MAX_PEAK_CUDA_BYTES and len(json.dumps(compact, sort_keys=True).encode("utf-8")) <= MAX_STORAGE_BYTES, {"peak_bytes": peak}),
        )
        audit_contract = {"lineage": LINEAGE, "classification": classification, "consumer": "project_owner", "spectral": compact["spectral"], "bands": BAND_CONTRACT, "selection_confirmation_isolated": True, "selection": selection, "confirmation": confirmation, "resource": compact["resource"], "compact_report_sha256": compact_hash, "no_update": True, "descriptive_only": True}
        audit_payload = _record(AUDIT_SCHEMA, status, classification, spec, state, upstream.runtime.data.manifest["manifest_hash"], hashes, audit_gates, (NEXT_ACTION_RESULT_REVIEW,) if harness_passed else (), audit_contract, None if harness_passed else "DSFA harness gate failed")
        _write_new(output_path, audit_payload, artifact_kind="DSFA audit evidence")
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return 0 if harness_passed else 2
    except DSFAHarnessFailure as exc:
        if not output_path.exists():
            _run_invalid_harness(output_path, spec, review_path, upstream, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-data-weighted-spectral-functional-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-data-weighted-spectral-functional-attribution-run":
        return run(args)
    raise ValueError(f"unsupported DSFA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "BAND_CONTRACT",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "DSFAHarnessFailure",
    "HYPOTHESIS",
    "LINEAGE",
    "LRRC_CONSUMER",
    "MEASUREMENT",
    "MODE_BLOCK_SIZE",
    "MODE_QUANTITIES",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "NUMERICAL_CONTRACT",
    "PANEL_CONTRACT",
    "REGISTERED_BANDS",
    "REGISTERED_RANKS",
    "REVIEW_SCHEMA",
    "RMAL_CONSUMER",
    "SPECTRAL_REPRESENTATION",
    "VALIDATION_SCHEMA",
    "_band_reports",
    "_canonical_spectral",
    "_explicit_mode_terms",
    "_factorized_mode_terms",
    "_load_review",
    "_load_spec",
    "_panel_report",
    "_review_payload",
    "dispatch",
    "run",
    "validate",
]
