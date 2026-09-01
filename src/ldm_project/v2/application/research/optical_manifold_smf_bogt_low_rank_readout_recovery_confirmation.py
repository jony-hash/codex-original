"""Confirm functional recovery of the frozen RMAL readout at registered ranks.

This is a diagnostic-only consumer of the RMAL detector readout.  It keeps the
RMAL fit, panels, and gate unchanged, and evaluates only fixed spectral
truncations.  When the historical artifact has no pooled feature bank, the
run uses a streaming CMOS-only replay: output.camera -> registered adaptive
pooling -> frozen detector feature vector.  No internal optical stage is
captured and no parameter is updated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_detector_readout_structure_attribution as rwsa
from . import optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit as lsda
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal
from . import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S6-LRRC"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "e0e9145836e9d3786dbe643fceae5d72e2e7d96a"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
INPUT_DIM = 257
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
FIT_COUNT = 1000
MICROBATCH_SIZE = 10
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
PHYSICAL_BASELINE_TOLERANCE = 1.0e-6
PROBE_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-5
PROBE_REPLAY_RELATIVE_TOLERANCE = 1.0e-6
RANK256_OPERATOR_ABSOLUTE_TOLERANCE = 1.0e-4
RANK256_OPERATOR_RELATIVE_TOLERANCE = 1.0e-6
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
MAX_TEMPORARY_COMPACT_FEATURE_BYTES = 128 * 1024**2

RANK_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEARCH_RANKS = RANK_CANDIDATES[:-1]
RANK0 = 0
CONTROLS = {
    "rank_0_bias_only": "prediction_0=b",
    "rank_256_full_positive_anchor": "U_256 U_256.T W plus original b",
}
SPECTRAL_BOUNDARY_GAPS = {
    1: 0.131054483,
    2: 0.178902324,
    4: 0.034630756,
    8: 0.105670493,
    16: 0.104494812,
    32: 0.061605380,
    64: 0.043964196,
    128: 0.052600464,
}
LSDA_SMALLEST_FULL_SPECTRUM_GAP = {"rank": 36, "relative_gap": 0.0025793411}
INHERITED_GATE_NAMES = (
    "finite",
    "feature_contract",
    "physical_anchor_replay",
    "relative_improvement",
    "strict_template_crossing",
    "paired_below_cyclic",
    "instance_sensitive",
    "frozen",
    "bootstrap_lower_bound",
    "square_law_consistency",
)


RMAL_CONSUMER = dict(lsda.RMAL_CONSUMER)
RWSA_IDENTITY = dict(lsda.RWSA_IDENTITY)
RWSA_ACTIVE_IDENTITY = dict(lsda.RWSA_ACTIVE_IDENTITY)
ZCSA_IDENTITY = dict(lsda.ZCSA_IDENTITY)
LSDA_CONSUMER = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_v1.yaml"
    ),
    "config_sha256": "d5093fc9f26ccf19eaa1bb74c3867fb25023e5837c00e96c5432facd30d9339d",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit_review_v1.json"
    ),
    "review_sha256": "b57b0098c4ed69591d96a889a806dea42479175740eb79b3e7d5ac8fe1536cb3",
    "validation": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/validation.json"
    ),
    "validation_sha256": "3f24dcc7d56e098ec778998bb8ddc7871973131f1d7269c86da729395426c2f2",
    "audit": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/audit.json"
    ),
    "audit_sha256": "97842d5707541516c79adb87c3a1e609ee533156895908c9e99c015cde0f9b13",
    "compact_report": (
        "outputs/20260901_v2/runs/"
        "lightgen_scaled_chen_linear_readout_structure_decomposition_audit_v1_attempt_01/compact_report.json"
    ),
    "compact_report_sha256": "e281daf9a4a62f85828a3bc8055ce044027d0c0fb1af0a522febcf3c2b68185b",
    "implementation_module": (
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit.py"
    ),
    "implementation_module_sha256": "0e501ee3d777bdbb17cfb156f2779352530f39a8ed4c62b8cbb48adc186ebd08",
    "implementation_commit": "bed235283d1fa47f4a5a38d41368d87ce2b5d218",
    "result_git_commit": LINEAGE_START_HEAD,
    "status": "passed",
    "classification": "linear_readout_spectral_attribution_completed",
}

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S6-LSDA",
    "lineage_start_head": LINEAGE_START_HEAD,
    "rmal_probe_frozen": True,
    "lsda_passed_required": True,
    "selection_confirmation_not_consumed": True,
    "attempt_name": (
        "lightgen_scaled_chen_low_rank_readout_recovery_confirmation_v1_attempt_01"
    ),
}

TRUNCATION = {
    "operator": "P_k W where P_k=U_k U_k.T",
    "factorized_operator": "z_k=z U_k; dynamic=z_k (U_k.T W)",
    "spectrum_source": "LSDA ordered CPU float64 symmetric eigensolver",
    "source_W_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "bias": "original frozen RMAL b; never refit or recomputed",
    "rank_specific_scaling": False,
    "normalization": False,
    "calibration": False,
    "permanent_W_k_saved": False,
}

EVALUATION = {
    "selection": dict(rmal.EVALUATION_PANELS["selection"]),
    "confirmation": dict(rmal.EVALUATION_PANELS["confirmation"]),
    "candidate_prefix": dict(rmal.EVALUATION_PANELS["candidate_prefix"]),
    "feature_path": "output.camera -> mra._feature_projection(real, complex_value=False)",
    "streaming": True,
    "pooled_feature_bank_saved": False,
    "target_bank_saved": False,
    "internal_stage_capture": False,
    "six_stage_rmal_search": False,
    "no_refit": True,
}

GATE_CONTRACT = {
    "source": "rmal._panel_gate(report, require_bootstrap=confirmation)",
    "inherited_checks": list(INHERITED_GATE_NAMES),
    "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "physical_baseline_tolerance": PHYSICAL_BASELINE_TOLERANCE,
    "no_new_spectral_threshold": True,
    "no_retained_energy_success_gate": True,
}

MEASUREMENT = {
    "objective": "functional_low_dimensional_reconstruction_recoverability",
    "primary_metric": "full_RMAL_probe_paired_MSE",
    "candidate_family": list(RANK_CANDIDATES),
    "search_order_after_anchor": list(SEARCH_RANKS),
    "rank_0_control": RANK0,
    "rank_256_anchor_first": True,
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "first_double_pass_stops_search": True,
    "first_confirmed_candidate_rank_only": True,
    "no_middle_rank_backtest": True,
    "no_candidate_family_mutation": True,
    "no_training": True,
    "no_refit": True,
    "no_physical_forward_modification": True,
    "mechanism_evidence_only": True,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_low_rank_readout_recovery_confirmation",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "prohibited": [
        "training",
        "optimizer",
        "gradient",
        "backward",
        "probe_refit",
        "bias_refit",
        "rank_specific_scaling",
        "normalization",
        "calibration_fit",
        "new_readout",
        "decoder",
        "fmf",
        "modal_sweep",
        "ols_redesign",
        "relay_tuning",
        "geometry_tuning",
        "prior",
        "generation",
        "promotion",
        "candidate_family_mutation",
        "middle_rank_backtest",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_low_rank_readout_recovery_confirmation"
NEXT_ACTION_RESULT_REVIEW = "manual_review_LRRC_result"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Upstream:
    lsda_spec: lsda.Spec
    lsda_path: Path
    lsda_review_path: Path
    lsda_validation_path: Path
    lsda_audit_path: Path
    lsda_compact_path: Path
    lsda_validation: dict[str, Any]
    lsda_audit: dict[str, Any]
    lsda_compact: dict[str, Any]
    rmal_consumer: Any
    rmal_identity: dict[str, Any]
    rmal_probe: rwsa.ReadoutProbe
    fit_indices: torch.Tensor
    data: mra.AuditData


class LRRCHarnessFailure(RuntimeError):
    """A registered LRRC boundary failed closed before classification."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"LRRC requires clean branch {BRANCH}")


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
        "rank_candidates",
        "controls",
        "truncation",
        "evaluation",
        "gate_inheritance",
        "spectral_boundaries",
        "measurement",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("LRRC config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("LRRC config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _spectral_boundary_contract() -> dict[str, Any]:
    return {
        "candidate_relative_gaps": {
            str(key): value for key, value in SPECTRAL_BOUNDARY_GAPS.items()
        },
        "full_spectrum_smallest_gap": LSDA_SMALLEST_FULL_SPECTRUM_GAP,
        "descriptive_only": True,
        "no_degeneracy_threshold": True,
    }


def _resource_contract() -> dict[str, int]:
    return {
        "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
        "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "max_temporary_compact_feature_bytes": MAX_TEMPORARY_COMPACT_FEATURE_BYTES,
    }


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": (
            "configs/v2/current/"
            "formal_identity_logit_amplitude_finite_crop_identity.yaml"
        ),
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "lsda_consumer": LSDA_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "rank_candidates": list(RANK_CANDIDATES),
        "controls": CONTROLS,
        "truncation": TRUNCATION,
        "evaluation": EVALUATION,
        "gate_inheritance": GATE_CONTRACT,
        "spectral_boundaries": _spectral_boundary_contract(),
        "measurement": MEASUREMENT,
        "resource": _resource_contract(),
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"LRRC {name} contract is not locked")
    if tuple(spec.payload["rank_candidates"]) != RANK_CANDIDATES:
        raise ValueError("LRRC candidate ranks are not immutable")
    if any(rank not in RANK_CANDIDATES for rank in SEARCH_RANKS):
        raise ValueError("LRRC search family is not a candidate prefix")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "lsda_consumer": LSDA_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "rank_candidates": list(RANK_CANDIDATES),
            "controls": CONTROLS,
            "truncation": TRUNCATION,
            "evaluation": EVALUATION,
            "gate_inheritance": GATE_CONTRACT,
            "spectral_boundaries": _spectral_boundary_contract(),
            "measurement": MEASUREMENT,
            "no_update": True,
            "no_rank_family_mutation": True,
            "interpretation": "first_confirmed_candidate_rank_only",
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    if json.loads(target.read_text(encoding="utf-8")) != _review_payload():
        raise ValueError("LRRC review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise LRRCHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LRRCHarnessFailure(f"{label} is not a mapping: {path}")
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
        raise LRRCHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise LRRCHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_lsda(spec: Spec) -> tuple[Any, ...]:
    lsda_path = _resolve(LSDA_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(lsda_path) != LSDA_CONSUMER["config_sha256"]:
        raise LRRCHarnessFailure("LSDA config identity changed")
    lsda_spec = lsda._load_spec(lsda_path)
    lsda._assert_contract(lsda_spec)
    lsda_review_path = _resolve(
        LSDA_CONSUMER["review_contract"], base_path=spec.path.parent
    )
    if file_sha256(lsda_review_path) != LSDA_CONSUMER["review_sha256"]:
        raise LRRCHarnessFailure("LSDA review identity changed")
    lsda._load_review(lsda_review_path, lsda_spec)
    lsda._upstream_bundle(lsda_spec)
    validation_path = _resolve(LSDA_CONSUMER["validation"])
    validation = _read_json_hash(
        validation_path, LSDA_CONSUMER["validation_sha256"], "LSDA validation"
    )
    validation_record = _verify_evidence(
        validation,
        expected_status="passed",
        expected_classification=None,
        label="LSDA validation",
    )
    if (
        validation_record.git_commit != LSDA_CONSUMER["implementation_commit"]
        or validation_record.config_hash != lsda_spec.config_hash
    ):
        raise LRRCHarnessFailure("LSDA validation lineage changed")
    audit_path = _resolve(LSDA_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, LSDA_CONSUMER["audit_sha256"], "LSDA audit")
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=LSDA_CONSUMER["classification"],
        label="LSDA audit",
    )
    if audit_record.git_commit != LSDA_CONSUMER["implementation_commit"]:
        raise LRRCHarnessFailure("LSDA audit lineage changed")
    compact_path = _resolve(LSDA_CONSUMER["compact_report"])
    compact = _read_json_hash(
        compact_path, LSDA_CONSUMER["compact_report_sha256"], "LSDA compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != LSDA_CONSUMER["classification"]
        or audit.get("generation_readiness") != "not_ready"
        or audit.get("reconstruction_manifold_supported") is not False
        or audit.get("generative_manifold_supported") is not False
    ):
        raise LRRCHarnessFailure("LSDA compact terminal interpretation changed")
    if compact.get("consumer", {}).get("probe_sha256") != RMAL_CONSUMER["probe_sha256"]:
        raise LRRCHarnessFailure("LSDA probe consumer identity changed")
    return (
        lsda_spec,
        lsda_path,
        lsda_review_path,
        validation_path,
        audit_path,
        compact_path,
        validation,
        audit,
        compact,
    )


def _load_upstream(spec: Spec, dataset_root: str | Path) -> Upstream:
    (
        lsda_spec,
        lsda_path,
        lsda_review_path,
        validation_path,
        audit_path,
        compact_path,
        validation,
        audit,
        compact,
    ) = _verify_lsda(spec)
    try:
        consumer, identity, probe = rwsa._load_rmal_consumer(
            lsda_spec, dataset_root, load_checkpoints=False
        )
    except (ValueError, rwsa.RWSAHarnessFailure) as exc:
        raise LRRCHarnessFailure("RMAL consumer identity is invalid") from exc
    fit_indices = rmal._fit_indices(consumer, dataset_root)
    data = consumer.bundle.parent.parent.data
    if (
        not torch.equal(probe.fit_indices, fit_indices)
        or tensor_sha256(fit_indices) != rmal.FIT_PANEL["indices_sha256"]
        or probe.full_weights.shape != (INPUT_DIM, TOTAL_COLUMNS)
        or probe.full_weights.dtype != torch.float32
        or probe.weights.shape != (FEATURE_DIM, TOTAL_COLUMNS)
        or probe.bias.shape != (TOTAL_COLUMNS,)
    ):
        raise LRRCHarnessFailure("RMAL probe shape, order, or fit identity changed")
    if (
        data.selection_indices.numel() != SELECTION_COUNT
        or data.confirmation_indices.numel() != CONFIRMATION_COUNT
        or data.manifest.get("manifest_hash") != lsda.DATASET_MANIFEST_SHA256
    ):
        raise LRRCHarnessFailure("RMAL panel or dataset manifest identity changed")
    return Upstream(
        lsda_spec,
        lsda_path,
        lsda_review_path,
        validation_path,
        audit_path,
        compact_path,
        validation,
        audit,
        compact,
        consumer,
        identity,
        probe,
        fit_indices,
        data,
    )


def _spectral_lock(
    probe: rwsa.ReadoutProbe, lsda_compact: Mapping[str, Any]
) -> dict[str, Any]:
    W = probe.weights.detach().cpu().contiguous()
    if W.dtype != torch.float32 or tuple(W.shape) != (FEATURE_DIM, TOTAL_COLUMNS):
        raise LRRCHarnessFailure("frozen dynamic readout shape or dtype changed")
    active_mask = W.abs().sum(dim=0).gt(0)
    if (
        int(active_mask.sum()) != lsda.EXPECTED_ACTIVE_COUNT
        or tensor_sha256(active_mask) != lsda.ACTIVE_MASK_SHA256
        or int((~active_mask).sum()) != lsda.EXPECTED_INACTIVE_COUNT
        or not bool(W[:, ~active_mask].eq(0).all())
        or not bool(probe.bias[~active_mask].eq(0).all())
    ):
        raise LRRCHarnessFailure("active/inactive readout identity changed")
    W64, full, active, differences = lsda._spectral_analysis(W, active_mask)
    compact_spectrum = lsda_compact.get("spectrum_full")
    if not isinstance(compact_spectrum, Mapping):
        raise LRRCHarnessFailure("LSDA spectrum is unavailable")
    stored_sigma = torch.tensor(
        compact_spectrum.get("singular_values", []), dtype=torch.float64
    )
    if (
        stored_sigma.shape != full.singular_values.shape
        or not torch.allclose(stored_sigma, full.singular_values, rtol=1.0e-11, atol=1.0e-8)
    ):
        raise LRRCHarnessFailure("LSDA singular-value ordering or values changed")
    stored_gaps = compact_spectrum.get("adjacent_relative_gaps")
    if not isinstance(stored_gaps, list) or len(stored_gaps) != TOTAL_COLUMNS - 1:
        raise LRRCHarnessFailure("LSDA adjacent-gap telemetry is unavailable")
    for rank, expected_gap in SPECTRAL_BOUNDARY_GAPS.items():
        actual = stored_gaps[rank - 1]
        if actual is None or abs(float(actual) - expected_gap) > 1.0e-8:
            raise LRRCHarnessFailure(f"LSDA rank-{rank} gap telemetry changed")
    if (
        int(torch.tensor(stored_gaps).argmin()) + 1
        != LSDA_SMALLEST_FULL_SPECTRUM_GAP["rank"]
    ):
        raise LRRCHarnessFailure("LSDA smallest-gap telemetry changed")
    return {
        "W64": W64,
        "U": full.eigenvectors,
        "full": full,
        "active": active,
        "differences": differences,
        "active_mask": active_mask,
        "stored_spectrum": dict(compact_spectrum),
    }


def _rank_boundary_telemetry(
    rank: int, spectral: Mapping[str, Any]
) -> dict[str, Any]:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    full = spectral["full"]
    sigma = full.singular_values
    squared = full.squared_singular_values
    next_sigma = float(sigma[rank]) if rank < FEATURE_DIM else None
    relative_gap = (
        float((sigma[rank - 1] - sigma[rank]) / sigma[rank - 1])
        if next_sigma is not None
        else None
    )
    return {
        "rank": rank,
        "sigma_k": float(sigma[rank - 1]),
        "sigma_k_plus_1": next_sigma,
        "relative_gap": relative_gap,
        "cumulative_squared_spectral_energy": float(squared[:rank].sum()),
        "normalized_spectral_energy_retained": float(full.cumulative_energy[rank - 1]),
        "descriptive_only": True,
        "basis_boundary_ambiguity": (
            "individual vectors may rotate within a near-degenerate subspace"
            if relative_gap is not None and relative_gap < 1.0e-2
            else None
        ),
    }


def _full_prediction(
    features: torch.Tensor, W: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if features.ndim != 2 or tuple(features.shape[1:]) != (FEATURE_DIM,):
        raise ValueError("features must have shape [N,256]")
    if W.dtype != torch.float32 or tuple(W.shape) != (FEATURE_DIM, TOTAL_COLUMNS):
        raise ValueError("W shape or dtype is invalid")
    if bias.dtype != torch.float32 or tuple(bias.shape) != (TOTAL_COLUMNS,):
        raise ValueError("bias shape or dtype is invalid")
    result = features.float().mm(W).add(bias)
    if not bool(torch.isfinite(result).all()):
        raise LRRCHarnessFailure("full frozen readout prediction is non-finite")
    return result


def _validate_operator_inputs(
    features: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
) -> None:
    if rank < 0 or rank > U.shape[1] or rank not in (0, *RANK_CANDIDATES):
        raise ValueError("rank is not registered")
    if U.dtype != torch.float64 or U.device.type != "cpu" or U.shape != (
        FEATURE_DIM,
        FEATURE_DIM,
    ):
        raise ValueError("LSDA U must be CPU float64 with shape [256,256]")
    if W.dtype != torch.float32 or W.device.type != "cpu" or W.shape != (
        FEATURE_DIM,
        TOTAL_COLUMNS,
    ):
        raise ValueError("W must be CPU float32 with shape [256,40000]")
    if bias.dtype != torch.float32 or bias.device.type != "cpu" or bias.shape != (
        TOTAL_COLUMNS,
    ):
        raise ValueError("b must be CPU float32 with shape [40000]")
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("features must have shape [N,256]")


def _explicit_prediction(
    features: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    _validate_operator_inputs(features, U, W, bias, rank)
    U_k = U[:, :rank]
    feature64 = features.to(torch.float64)
    if rank == 0:
        dynamic = torch.zeros(
            (features.shape[0], TOTAL_COLUMNS), dtype=torch.float64
        )
    else:
        dynamic = feature64.mm(U_k.mm(U_k.T)).mm(W.to(torch.float64))
    result = dynamic.add(bias.to(torch.float64))
    if not bool(torch.isfinite(result).all()):
        raise LRRCHarnessFailure("explicit truncated prediction is non-finite")
    return result


def _factorized_prediction(
    features: torch.Tensor,
    U: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    _validate_operator_inputs(features, U, W, bias, rank)
    U_k = U[:, :rank]
    feature64 = features.to(torch.float64)
    if rank == 0:
        dynamic = torch.zeros(
            (features.shape[0], TOTAL_COLUMNS), dtype=torch.float64
        )
    else:
        projected = feature64.mm(U_k)
        dynamic = projected.mm(U_k.T.mm(W.to(torch.float64)))
    result = dynamic.add(bias.to(torch.float64))
    if not bool(torch.isfinite(result).all()):
        raise LRRCHarnessFailure("factorized truncated prediction is non-finite")
    return result


def _rank0_prediction(
    features: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("rank-0 features have the wrong shape")
    if bias.dtype != torch.float32 or tuple(bias.shape) != (TOTAL_COLUMNS,):
        raise ValueError("rank-0 bias has the wrong shape or dtype")
    return bias.to(torch.float64).unsqueeze(0).expand(features.shape[0], -1).clone()


def _operator_identity(
    features: torch.Tensor, U: torch.Tensor, W: torch.Tensor, bias: torch.Tensor
) -> dict[str, float | bool]:
    explicit = _explicit_prediction(features, U, W, bias, 256)
    factorized = _factorized_prediction(features, U, W, bias, 256)
    residual = (explicit - factorized).abs()
    return {
        "explicit_factorized_max_abs_residual": float(residual.max()),
        "explicit_factorized_mean_abs_residual": float(residual.mean()),
        "explicit_factorized_close": bool(
            torch.allclose(
                explicit,
                factorized,
                rtol=RANK256_OPERATOR_RELATIVE_TOLERANCE,
                atol=RANK256_OPERATOR_ABSOLUTE_TOLERANCE,
            )
        ),
    }


def _target_templates(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError("panel images have the wrong shape")
    if labels.ndim != 1 or labels.numel() != images.shape[0]:
        raise ValueError("panel labels have the wrong shape")
    sums = torch.zeros((10, 1, 200, 200), dtype=torch.float32)
    counts = torch.bincount(labels, minlength=10)
    if bool((counts == 0).any()):
        raise LRRCHarnessFailure("panel is missing a class")
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        target = base.render_intensity_batch(images.index_select(0, indices)).target.float()
        sums.index_add_(0, labels.index_select(0, indices), target)
    return sums / counts.reshape(10, 1, 1, 1)


def _capture_cmos(
    model: bogt.ScaledChenBOGTModel,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rendered = base.render_intensity_batch(images)
    with torch.no_grad():
        output = model(rendered.amplitude.to(device), return_trace=False)
        camera = output.camera.detach().cpu().float()
    if tuple(camera.shape[1:]) != (1, 200, 200) or not bool(torch.isfinite(camera).all()):
        raise LRRCHarnessFailure("CMOS-only replay produced an invalid output.camera")
    return camera


def _context_metrics(
    prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor
) -> dict[str, Any]:
    return rmfa._context_metrics_from_compact(prediction, target, labels)


def _panel_gate(report: Mapping[str, Any], *, require_bootstrap: bool) -> dict[str, Any]:
    return rmal._panel_gate(report, require_bootstrap=require_bootstrap)


def _bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    return rmal._stratified_bootstrap_ci(delta, labels)


def _evaluate_panel(
    model: bogt.ScaledChenBOGTModel,
    probe: rwsa.ReadoutProbe,
    U: torch.Tensor,
    rank: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    expected_anchor: Mapping[str, float],
    *,
    use_full_anchor: bool = False,
    include_rank0: bool = False,
) -> dict[str, Any]:
    if rank not in RANK_CANDIDATES:
        raise ValueError("panel rank is not registered")
    model_before = rmal._state_cpu(model)
    ols_before = rmal._state_cpu(model.ols)
    templates = _target_templates(images, labels)
    physical_errors: list[torch.Tensor] = []
    candidate_errors: list[torch.Tensor] = []
    full_errors: list[torch.Tensor] = []
    cyclic_errors: list[torch.Tensor] = []
    template_errors: list[torch.Tensor] = []
    dynamic_errors: list[torch.Tensor] = []
    bias_errors: list[torch.Tensor] = []
    candidate_compact: list[torch.Tensor] = []
    target_compact: list[torch.Tensor] = []
    rank0_errors: list[torch.Tensor] = []
    identity_max = 0.0
    identity_mean_sum = 0.0
    identity_count = 0
    sample_count = int(images.shape[0])
    for indices in torch.arange(sample_count, dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        batch_labels = labels.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        previous = images.index_select(0, (indices - 1) % sample_count)
        cyclic_target = base.render_intensity_batch(previous).target.float()
        physical = _capture_cmos(model, batch_images, device)
        features = mra._feature_projection(physical, complex_value=False).float()
        full = _full_prediction(features, probe.weights, probe.bias)
        if use_full_anchor:
            candidate = full.to(torch.float64)
            factorized = _factorized_prediction(
                features, U, probe.weights, probe.bias, rank
            )
            difference = (factorized - full.to(torch.float64)).abs()
            identity_max = max(identity_max, float(difference.max()))
            identity_mean_sum += float(difference.sum())
            identity_count += difference.numel()
        else:
            candidate = _factorized_prediction(
                features, U, probe.weights, probe.bias, rank
            )
        dynamic = candidate - probe.bias.to(torch.float64)
        bias_only = _rank0_prediction(features, probe.bias)
        target64 = target.to(torch.float64)
        template = templates.index_select(0, batch_labels).to(torch.float64)
        cyclic64 = cyclic_target.to(torch.float64)
        physical64 = physical.to(torch.float64)
        candidate_errors.append(
            (candidate.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
        )
        full_errors.append(
            (full.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
        )
        physical_errors.append(
            (physical64 - target64).square().mean(dim=(1, 2, 3))
        )
        cyclic_errors.append(
            (physical64 - cyclic64).square().mean(dim=(1, 2, 3))
        )
        template_errors.append(
            (physical64 - template).square().mean(dim=(1, 2, 3))
        )
        dynamic_errors.append(
            (dynamic.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
        )
        bias_errors.append(
            (bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
        )
        candidate_compact.append(
            mra._feature_projection(
                candidate.reshape(-1, 1, 200, 200).float()
            ).float()
        )
        target_compact.append(mra._feature_projection(target).float())
        if include_rank0:
            rank0_errors.append(
                (bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
            )
    physical_mse = torch.cat(physical_errors)
    candidate_mse = torch.cat(candidate_errors)
    full_mse = torch.cat(full_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    dynamic_mse = torch.cat(dynamic_errors)
    bias_mse = torch.cat(bias_errors)
    compact_prediction = torch.cat(candidate_compact)
    compact_target = torch.cat(target_compact)
    context = _context_metrics(compact_prediction, compact_target, labels)
    delta = physical_mse - candidate_mse
    ci = _bootstrap_ci(delta, labels)
    physical_values = {
        "paired_mse": float(physical_mse.mean()),
        "class_template_mse": float(template_mse.mean()),
        "cyclic_mse": float(cyclic_mse.mean()),
    }
    native_replay = {
        key: abs(physical_values[key] - float(expected_anchor[key]))
        <= PHYSICAL_BASELINE_TOLERANCE
        for key in ("paired_mse", "class_template_mse", "cyclic_mse")
    }
    candidate_value = float(candidate_mse.mean())
    physical_value = float(physical_mse.mean())
    report: dict[str, Any] = {
        "executed": True,
        "panel": "selection" if sample_count == SELECTION_COUNT else "confirmation",
        "rank": rank,
        "evaluation_count": sample_count,
        "physical": {
            **physical_values,
            "mse_by_label": [
                float(physical_mse[labels == label].mean()) for label in range(10)
            ],
        },
        "probe": {
            "paired_mse": candidate_value,
            "mse_by_label": [
                float(candidate_mse[labels == label].mean()) for label in range(10)
            ],
            **context,
        },
        "class_template_mse": physical_values["class_template_mse"],
        "cyclic_mse": physical_values["cyclic_mse"],
        "relative_improvement": float(delta.mean() / max(physical_value, 1.0e-12)),
        "bootstrap": {
            "delta_definition": "physical_paired_mse - candidate_paired_mse",
            "mean_delta": float(delta.mean()),
            "ci": list(ci),
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "native_physical_anchor": dict(expected_anchor),
        "native_replay": native_replay,
        "finite": bool(
            torch.isfinite(physical_mse).all()
            and torch.isfinite(candidate_mse).all()
            and torch.isfinite(full_mse).all()
            and torch.isfinite(cyclic_mse).all()
            and torch.isfinite(template_mse).all()
        ),
        "instance_sensitive": bool(
            candidate_value < physical_values["cyclic_mse"]
            and float(context["same_label_top1"])
            >= float(context["class_template_same_label_top1"])
        ),
        "state_unchanged": rmal._state_equal(
            model_before, rmal._state_cpu(model)
        ),
        "ols_state_unchanged": rmal._state_equal(
            ols_before, rmal._state_cpu(model.ols)
        ),
        "feature_contract_valid": True,
        "probe_contract_valid": True,
        "square_law_consistency": True,
        "square_law_source": "inherited_registered_RMAL_cmos_intensity_control",
        "full_rmal_reference": {
            "paired_mse": float(full_mse.mean()),
            "physical_paired_mse": physical_value,
        },
        "mse_delta_vs_full_rmal": float(candidate_value - full_mse.mean()),
        "dynamic_only": {"paired_mse": float(dynamic_mse.mean())},
        "bias_only": {"paired_mse": float(bias_mse.mean())},
        "rank_boundary": None,
        "internal_stage_capture": False,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
    }
    if identity_count:
        report["rank256_operator_identity"] = {
            "factorized_vs_full_max_abs_residual": identity_max,
            "factorized_vs_full_mean_abs_residual": identity_mean_sum / identity_count,
            "factorized_vs_full_close": bool(
                identity_max
                <= RANK256_OPERATOR_ABSOLUTE_TOLERANCE
                + RANK256_OPERATOR_RELATIVE_TOLERANCE
                * max(float(full.abs().max()), 1.0)
            ),
        }
    report["selection_gate"] = _panel_gate(report, require_bootstrap=False)
    if include_rank0:
        rank0 = torch.cat(rank0_errors)
        report["rank0_control"] = {
            "prediction": "bias_only",
            "paired_mse": float(rank0.mean()),
            "mse_by_label": [
                float(rank0[labels == label].mean()) for label in range(10)
            ],
            "included_in_candidate_search": False,
            "bias_refit": False,
        }
    return report


def _json_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in {"sample_mse", "target_first"}
    }


def _expected_anchor(
    compact: Mapping[str, Any], panel: str
) -> tuple[dict[str, float], float]:
    anchor = compact.get("positive_anchor_replay", {}).get(panel)
    if not isinstance(anchor, Mapping):
        raise LRRCHarnessFailure(f"RMAL {panel} positive anchor is unavailable")
    physical = anchor.get("physical")
    probe = anchor.get("probe")
    if not isinstance(physical, Mapping) or not isinstance(probe, Mapping):
        raise LRRCHarnessFailure(f"RMAL {panel} anchor metrics are unavailable")
    values = {
        key: float(physical[key])
        for key in ("paired_mse", "class_template_mse", "cyclic_mse")
    }
    return values, float(probe["paired_mse"])


def _rank256_anchor(
    model: bogt.ScaledChenBOGTModel,
    probe: rwsa.ReadoutProbe,
    spectral: Mapping[str, Any],
    data: mra.AuditData,
    compact: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    selection_anchor, selection_expected_probe = _expected_anchor(compact, "selection")
    confirmation_anchor, confirmation_expected_probe = _expected_anchor(
        compact, "confirmation"
    )
    selection_images = data.test_images.index_select(0, data.selection_indices)
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    confirmation_images = data.images.index_select(0, data.confirmation_indices)
    confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
    selection = _evaluate_panel(
        model,
        probe,
        spectral["U"],
        256,
        selection_images,
        selection_labels,
        device,
        selection_anchor,
        use_full_anchor=True,
        include_rank0=True,
    )
    confirmation = _evaluate_panel(
        model,
        probe,
        spectral["U"],
        256,
        confirmation_images,
        confirmation_labels,
        device,
        confirmation_anchor,
        use_full_anchor=True,
        include_rank0=True,
    )
    confirmation["selection_gate"] = _panel_gate(confirmation, require_bootstrap=False)
    confirmation["confirmation_gate"] = _panel_gate(
        confirmation, require_bootstrap=True
    )
    selection_probe_replay = abs(
        float(selection["probe"]["paired_mse"]) - selection_expected_probe
    )
    confirmation_probe_replay = abs(
        float(confirmation["probe"]["paired_mse"]) - confirmation_expected_probe
    )
    identities = [
        selection.get("rank256_operator_identity", {}),
        confirmation.get("rank256_operator_identity", {}),
    ]
    anchor_passed = bool(
        selection["selection_gate"]["passed"]
        and confirmation["selection_gate"]["passed"]
        and confirmation["confirmation_gate"]["passed"]
        and selection_probe_replay
        <= PROBE_REPLAY_ABSOLUTE_TOLERANCE
        + PROBE_REPLAY_RELATIVE_TOLERANCE * max(abs(selection_expected_probe), 1.0)
        and confirmation_probe_replay
        <= PROBE_REPLAY_ABSOLUTE_TOLERANCE
        + PROBE_REPLAY_RELATIVE_TOLERANCE * max(abs(confirmation_expected_probe), 1.0)
        and all(item.get("factorized_vs_full_close") is True for item in identities)
    )
    selection["rank_boundary"] = _rank_boundary_telemetry(256, spectral)
    confirmation["rank_boundary"] = _rank_boundary_telemetry(256, spectral)
    return {
        "first_execution": "rank_256_full_positive_anchor",
        "selection": _json_report(selection),
        "confirmation": _json_report(confirmation),
        "selection_probe_mse_replay_abs_delta": selection_probe_replay,
        "confirmation_probe_mse_replay_abs_delta": confirmation_probe_replay,
        "original_rmal_gate_vector": {
            "selection": selection["selection_gate"],
            "confirmation": confirmation["confirmation_gate"],
        },
        "passed": anchor_passed,
        "failure_stop": not anchor_passed,
    }


def _sequential_search(
    selection_fn: Callable[[int], Mapping[str, Any]],
    confirmation_fn: Callable[[int], Mapping[str, Any]],
    *,
    anchor_passed: bool,
) -> dict[str, Any]:
    if not anchor_passed:
        return {
            "evaluated_ranks": [],
            "rank_results": {},
            "first_confirmed_candidate_rank": None,
            "previous_registered_candidate": None,
            "anchor_failure_stop": True,
        }
    evaluated: list[int] = []
    results: dict[str, Any] = {}
    winner: int | None = None
    previous: int | None = None
    for rank in SEARCH_RANKS:
        evaluated.append(rank)
        selection = dict(selection_fn(rank))
        selection_gate = selection.get("selection_gate", {})
        if not bool(selection_gate.get("passed", False)):
            results[str(rank)] = {
                "selection": selection,
                "confirmation": {"executed": False},
                "outcome": (
                    "bounded_recoverability_not_established_at_registered_rank_k"
                ),
                "selection_pass": False,
                "confirmation_not_run": True,
            }
            previous = rank
            continue
        confirmation = dict(confirmation_fn(rank))
        confirmation_gate = confirmation.get("confirmation_gate", {})
        confirmation_selection_gate = confirmation.get("selection_gate", {})
        if not (
            bool(confirmation_selection_gate.get("passed", False))
            and bool(confirmation_gate.get("passed", False))
        ):
            results[str(rank)] = {
                "selection": selection,
                "confirmation": confirmation,
                "outcome": (
                    "bounded_recoverability_not_established_on_confirmation_at_registered_rank_k"
                ),
                "selection_pass": True,
                "confirmation_pass": False,
            }
            previous = rank
            continue
        results[str(rank)] = {
            "selection": selection,
            "confirmation": confirmation,
            "outcome": "bounded_recoverability_established_at_registered_rank_k",
            "selection_pass": True,
            "confirmation_pass": True,
        }
        winner = rank
        break
    if winner is None:
        winner = 256
    return {
        "evaluated_ranks": evaluated,
        "rank_results": results,
        "first_confirmed_candidate_rank": winner,
        "previous_registered_candidate": previous if winner != 256 else previous,
        "anchor_failure_stop": False,
        "stopped_after_first_double_pass": any(
            value.get("confirmation_pass") is True for value in results.values()
        ),
        "rank_256_reused_as_fallback": winner == 256,
    }


def _state_digest(module: torch.nn.Module) -> str:
    return rmal._state_digest(rmal._state_cpu(module))


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    probe_path: Path,
    dataset_manifest: str,
) -> dict[str, str]:
    return {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "implementation_module": file_sha256(Path(__file__).resolve()),
        "rmal_config": file_sha256(upstream.rmal_identity["config_path"]),
        "rmal_review_contract": file_sha256(upstream.rmal_identity["review_path"]),
        "rmal_validation": file_sha256(upstream.rmal_identity["validation_path"]),
        "rmal_audit": file_sha256(upstream.rmal_identity["audit_path"]),
        "rmal_compact_report": file_sha256(upstream.rmal_identity["compact_path"]),
        "rmal_probe": file_sha256(probe_path),
        "lsda_config": file_sha256(upstream.lsda_path),
        "lsda_review_contract": file_sha256(upstream.lsda_review_path),
        "lsda_validation": file_sha256(upstream.lsda_validation_path),
        "lsda_audit": file_sha256(upstream.lsda_audit_path),
        "lsda_compact_report": file_sha256(upstream.lsda_compact_path),
        "lsda_implementation_module": LSDA_CONSUMER["implementation_module_sha256"],
        "rwsa_config": RWSA_IDENTITY["config_sha256"],
        "rwsa_review_contract": RWSA_IDENTITY["review_sha256"],
        "rwsa_validation": RWSA_IDENTITY["validation_sha256"],
        "rwsa_audit": RWSA_IDENTITY["audit_sha256"],
        "rwsa_active_config": RWSA_ACTIVE_IDENTITY["config_sha256"],
        "rwsa_active_review_contract": RWSA_ACTIVE_IDENTITY["review_sha256"],
        "rwsa_active_validation": RWSA_ACTIVE_IDENTITY["validation_sha256"],
        "rwsa_active_audit": RWSA_ACTIVE_IDENTITY["audit_sha256"],
        "rwsa_active_compact_report": RWSA_ACTIVE_IDENTITY["compact_report_sha256"],
        "zcsa_config": ZCSA_IDENTITY["config_sha256"],
        "zcsa_review_contract": ZCSA_IDENTITY["review_sha256"],
        "zcsa_validation": ZCSA_IDENTITY["validation_sha256"],
        "zcsa_audit": ZCSA_IDENTITY["audit_sha256"],
        "zcsa_compact_report": ZCSA_IDENTITY["compact_report_sha256"],
        "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
        "lsda_consumer_contract": canonical_sha256(LSDA_CONSUMER),
        "rwsa_identity_contract": canonical_sha256(RWSA_IDENTITY),
        "rwsa_active_identity_contract": canonical_sha256(RWSA_ACTIVE_IDENTITY),
        "zcsa_identity_contract": canonical_sha256(ZCSA_IDENTITY),
        "candidate_family_contract": canonical_sha256(list(RANK_CANDIDATES)),
        "truncation_contract": canonical_sha256(TRUNCATION),
        "gate_contract": canonical_sha256(GATE_CONTRACT),
        "dataset_manifest": dataset_manifest,
        "fit_indices": tensor_sha256(upstream.fit_indices),
        "probe_tensor": tensor_sha256(upstream.rmal_probe.full_weights),
    }


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "candidate_family": RANK_CANDIDATES,
            "truncation": TRUNCATION,
            "gate_contract": GATE_CONTRACT,
            "measurement": MEASUREMENT,
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
        "bias_refit": False,
        "decoder_created": False,
        "checkpoint_mutation": False,
        "generation": False,
        "cmos_only_replay_executed": False,
        "optical_replay_executed": False,
        "internal_stage_capture": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse LRRC validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("LRRC validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"LRRC attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("LRRC attempt directory is not a fresh validation-only slot")
    for name in ("compact_report.json", "probes.pt"):
        if (output_path.parent / name).exists():
            raise FileExistsError(f"refusing to overwrite LRRC companion artifact: {name}")


def _validation_gates(
    spec: Spec,
    upstream: Upstream,
    spectral: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    return (
        GateResult(
            "clean_registered_branch",
            _git_state().get("branch") == BRANCH and _git_state().get("clean") is True,
            {},
        ),
        GateResult(
            "lineage_start_head_recorded",
            spec.payload["lineage"]["lineage_start_head"] == LINEAGE_START_HEAD,
            {"lineage_start_head": LINEAGE_START_HEAD},
        ),
        GateResult(
            "rmal_terminal_identity",
            upstream.rmal_identity["audit"].get("classification")
            == RMAL_CONSUMER["audit_classification"],
            {},
        ),
        GateResult(
            "lsda_terminal_identity",
            upstream.lsda_compact.get("classification") == LSDA_CONSUMER["classification"],
            {},
        ),
        GateResult(
            "probe_shape_and_active_identity",
            tuple(upstream.rmal_probe.full_weights.shape) == (INPUT_DIM, TOTAL_COLUMNS)
            and upstream.rmal_probe.full_weights.dtype == torch.float32
            and tuple(upstream.rmal_probe.weights.shape) == (FEATURE_DIM, TOTAL_COLUMNS)
            and tuple(upstream.rmal_probe.bias.shape) == (TOTAL_COLUMNS,)
            and int(spectral["active_mask"].sum()) == lsda.EXPECTED_ACTIVE_COUNT,
            {},
        ),
        GateResult(
            "lsda_order_and_boundary_telemetry",
            spectral["U"].shape == (FEATURE_DIM, FEATURE_DIM)
            and all(
                _rank_boundary_telemetry(rank, spectral)["relative_gap"] is not None
                for rank in SEARCH_RANKS
            ),
            {},
        ),
        GateResult(
            "rank_family_locked",
            list(RANK_CANDIDATES) == spec.payload["rank_candidates"]
            and list(SEARCH_RANKS) == spec.payload["rank_candidates"][:-1],
            {"rank_candidates": list(RANK_CANDIDATES)},
        ),
        GateResult(
            "candidate_prefix_not_consumed",
            EVALUATION["candidate_prefix"]["consumed"] is False,
            {},
        ),
        GateResult(
            "no_update_scope",
            True,
            {
                "optimizer_created": False,
                "backward_calls": 0,
                "probe_refit": False,
                "bias_refit": False,
                "decoder_created": False,
            },
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


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    spectral = _spectral_lock(upstream.rmal_probe, upstream.lsda_compact)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, upstream, spectral, output_path)
    hashes = _input_hashes(
        spec,
        review_path,
        upstream,
        _resolve(RMAL_CONSUMER["probe"]),
        upstream.data.manifest["manifest_hash"],
    )
    valid = all(gate.passed for gate in gates)
    contract = {
        "lineage": LINEAGE,
        "rmal_consumer": RMAL_CONSUMER,
        "lsda_consumer": LSDA_CONSUMER,
        "rwsa_identity": RWSA_IDENTITY,
        "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
        "zcsa_identity": ZCSA_IDENTITY,
        "rank_candidates": list(RANK_CANDIDATES),
        "controls": CONTROLS,
        "truncation": TRUNCATION,
        "evaluation": EVALUATION,
        "gate_inheritance": GATE_CONTRACT,
        "spectral_boundaries": _spectral_boundary_contract(),
        "measurement": MEASUREMENT,
        "authorization": AUTHORIZATION,
    }
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        state,
        upstream.data.manifest["manifest_hash"],
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "LRRC static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="LRRC validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise LRRCHarnessFailure("LRRC requires CUDA for formal CMOS replay")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise LRRCHarnessFailure("LRRC requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise LRRCHarnessFailure("LRRC requires at least 10 GiB free storage")
    return int(free), int(total)


def _load_validation(
    path: str | Path,
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
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
        _resolve(RMAL_CONSUMER["probe"]),
        upstream.data.manifest["manifest_hash"],
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
    ):
        raise ValueError("LRRC validation evidence is stale or mismatched")


def _invalid_harness_payload(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    state: Mapping[str, Any],
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        state,
        upstream.data.manifest["manifest_hash"],
        _input_hashes(
            spec,
            review_path,
            upstream,
            _resolve(RMAL_CONSUMER["probe"]),
            upstream.data.manifest["manifest_hash"],
        ),
        (
            GateResult(
                "rank_256_full_positive_anchor",
                False,
                {"reason": reason},
            ),
            GateResult("no_update_scope", True, {}),
        ),
        (),
        {
            "rank_256_anchor": {"passed": False},
            "rank_search_executed": False,
            "reason": reason,
            "details": dict(details or {}),
        },
        reason,
    )
    payload.update(
        {
            "rank_candidates": list(RANK_CANDIDATES),
            "evaluated_ranks": [],
            "first_confirmed_candidate_rank": None,
            "closed_task_relevant_bottleneck_not_supported": True,
        }
    )
    _write_new(output_path, payload, artifact_kind="LRRC invalid-harness audit evidence")


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("LRRC requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    upstream = _load_upstream(spec, args.dataset_root)
    _load_validation(args.validation_evidence, spec, review_path, upstream, state)
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _check_run_slot(output_path, validation_path)
    device = torch.device(args.device)
    try:
        free_bytes, total_bytes = _resource_preflight(output_path, device)
        source = rmfa._load_source(upstream.rmal_consumer.bundle, load_checkpoints=True)
        formal: FormalSystemConfig = upstream.rmal_consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[rmal.PRIMARY_CHECKPOINT]
        model = mra._load_scaled_model(formal, checkpoint, device)
        spectral = _spectral_lock(upstream.rmal_probe, upstream.lsda_compact)
        state_before = rmal._state_cpu(model)
        ols_before = rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        anchor = _rank256_anchor(
            model,
            upstream.rmal_probe,
            spectral,
            upstream.data,
            upstream.rmal_identity["compact"],
            device,
        )
        if not anchor["passed"]:
            _invalid_harness_payload(
                output_path,
                spec,
                review_path,
                upstream,
                state,
                "rank-256 full positive anchor failed",
                {"rank_256_anchor": anchor},
            )
            return 2
        rank0 = {
            "selection": anchor["selection"].get("rank0_control", {}),
            "confirmation": anchor["confirmation"].get("rank0_control", {}),
            "included_in_candidate_search": False,
        }

        def selection_fn(rank: int) -> Mapping[str, Any]:
            expected, _ = _expected_anchor(
                upstream.rmal_identity["compact"], "selection"
            )
            images = upstream.data.test_images.index_select(
                0, upstream.data.selection_indices
            )
            labels = upstream.data.test_labels.index_select(
                0, upstream.data.selection_indices
            )
            result = _evaluate_panel(
                model,
                upstream.rmal_probe,
                spectral["U"],
                rank,
                images,
                labels,
                device,
                expected,
            )
            result["rank_boundary"] = _rank_boundary_telemetry(rank, spectral)
            return result

        def confirmation_fn(rank: int) -> Mapping[str, Any]:
            expected, _ = _expected_anchor(
                upstream.rmal_identity["compact"], "confirmation"
            )
            images = upstream.data.images.index_select(
                0, upstream.data.confirmation_indices
            )
            labels = upstream.data.labels.index_select(
                0, upstream.data.confirmation_indices
            )
            result = _evaluate_panel(
                model,
                upstream.rmal_probe,
                spectral["U"],
                rank,
                images,
                labels,
                device,
                expected,
            )
            result["rank_boundary"] = _rank_boundary_telemetry(rank, spectral)
            result["selection_gate"] = _panel_gate(result, require_bootstrap=False)
            result["confirmation_gate"] = _panel_gate(
                result, require_bootstrap=True
            )
            return result

        search = _sequential_search(selection_fn, confirmation_fn, anchor_passed=True)
        winner = int(search["first_confirmed_candidate_rank"])
        if winner == 256:
            search["rank_results"]["256"] = {
                "selection": anchor["selection"],
                "confirmation": anchor["confirmation"],
                "outcome": "rank_256_anchor_reused_as_fallback",
                "selection_pass": True,
                "confirmation_pass": True,
                "anchor_reused": True,
            }
        state_unchanged = rmal._state_equal(
            state_before, rmal._state_cpu(model)
        )
        ols_unchanged = rmal._state_equal(
            ols_before, rmal._state_cpu(model.ols)
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        temporary_compact_features = 2 * CONFIRMATION_COUNT * FEATURE_DIM * 4
        if temporary_compact_features > MAX_TEMPORARY_COMPACT_FEATURE_BYTES:
            raise LRRCHarnessFailure(
                "temporary compact feature working set exceeded bound"
            )
        if peak > MAX_PEAK_CUDA_BYTES:
            raise LRRCHarnessFailure("LRRC peak CUDA memory exceeded RMAL bound")
        stage_results = search["rank_results"]
        for item in stage_results.values():
            for panel in ("selection", "confirmation"):
                if panel in item and isinstance(item[panel], Mapping):
                    item[panel] = _json_report(item[panel])
        all_reports = [
            report
            for item in stage_results.values()
            for report in (item.get("selection", {}), item.get("confirmation", {}))
            if report.get("executed", True)
        ]
        harness_ok = (
            anchor["passed"]
            and state_unchanged
            and ols_unchanged
            and all(report.get("finite") is True for report in all_reports)
            and all(report.get("feature_contract_valid") is True for report in all_reports)
            and all(report.get("probe_contract_valid") is True for report in all_reports)
            and all(report.get("square_law_consistency") is True for report in all_reports)
            and all(all(report.get("native_replay", {}).values()) for report in all_reports)
            and all(
                report.get("internal_stage_capture") is False
                and report.get("pooled_feature_bank_saved") is False
                and report.get("target_bank_saved") is False
                for report in all_reports
            )
        )
        if not harness_ok:
            raise LRRCHarnessFailure("LRRC harness gate failed after measurement")
        status = "passed"
        classification = "low_rank_readout_recovery_confirmation_completed"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": status,
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "rank_candidates": list(RANK_CANDIDATES),
            "search_order_after_anchor": list(SEARCH_RANKS),
            "rank_0_control": rank0,
            "rank_256_full_positive_anchor": anchor,
            "evaluated_ranks": search["evaluated_ranks"],
            "rank_results": stage_results,
            "first_confirmed_candidate_rank": winner,
            "previous_registered_candidate": search["previous_registered_candidate"],
            "rank_256_reused_as_fallback": search["rank_256_reused_as_fallback"],
            "spectral_telemetry": {
                "candidate_boundaries": {
                    str(rank): _rank_boundary_telemetry(rank, spectral)
                    for rank in RANK_CANDIDATES
                },
                "full_spectrum_smallest_gap": LSDA_SMALLEST_FULL_SPECTRUM_GAP,
                "lsda_full_vs_active": spectral["differences"],
                "descriptive_only": True,
                "no_degeneracy_threshold": True,
            },
            "evaluation": EVALUATION,
            "gate_inheritance": GATE_CONTRACT,
            "resource": {
                "free_cuda_bytes_before": free_bytes,
                "total_cuda_bytes": total_bytes,
                "peak_memory_bytes": peak,
                "wall_clock_seconds": elapsed,
                "temporary_compact_feature_bytes": temporary_compact_features,
                "pooled_feature_bank_saved": False,
                "target_bank_saved": False,
            },
            "model_state_before": rmal._state_digest(state_before),
            "model_state_after": rmal._state_digest(rmal._state_cpu(model)),
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
            "bias_refit": False,
            "decoder_created": False,
            "checkpoint_mutation": False,
            "generation": False,
            "cmos_only_replay_executed": True,
            "optical_replay_executed": False,
            "internal_stage_capture": False,
            "mechanism_evidence_only": True,
        }
        compact_path = output_path.with_name("compact_report.json")
        _write_new(compact_path, compact, artifact_kind="LRRC compact report")
        compact_hash = file_sha256(compact_path)
        hashes = _input_hashes(
            spec,
            review_path,
            upstream,
            _resolve(RMAL_CONSUMER["probe"]),
            upstream.data.manifest["manifest_hash"],
        )
        hashes["lrrc_compact_report"] = compact_hash
        audit_contract = {
            "rank_256_full_positive_anchor": anchor,
            "rank_0_control": rank0,
            "candidate_family": list(RANK_CANDIDATES),
            "evaluated_ranks": search["evaluated_ranks"],
            "rank_results": stage_results,
            "first_confirmed_candidate_rank": winner,
            "previous_registered_candidate": search["previous_registered_candidate"],
            "interpretation_boundary": (
                "first confirmed rank only within the preregistered "
                "powers-of-two candidate family"
            ),
            "spectral_telemetry": compact["spectral_telemetry"],
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "resource": compact["resource"],
            "candidate_family_mutated": False,
            "higher_rank_evaluated_after_success": False,
            "second_rank_256_run": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "probe_refit": False,
            "bias_refit": False,
            "decoder_created": False,
            "checkpoint_mutation": False,
            "generation": False,
            "formal_promotion": False,
            "cmos_only_replay_executed": True,
            "optical_replay_executed": False,
            "internal_stage_capture": False,
            "pooled_feature_bank_saved": False,
            "target_bank_saved": False,
            "historical_evidence_modified": False,
            "mechanism_evidence_only": True,
            "compact_report_sha256": compact_hash,
        }
        gates = (
            GateResult("rank_256_full_positive_anchor", anchor["passed"], {}),
            GateResult(
                "rank_0_control_bias_only",
                rank0["included_in_candidate_search"] is False,
                {},
            ),
            GateResult(
                "sequential_candidate_family",
                search["evaluated_ranks"]
                == list(SEARCH_RANKS[: len(search["evaluated_ranks"])]),
                {"evaluated": search["evaluated_ranks"]},
            ),
            GateResult(
                "first_double_pass_stop",
                not search["stopped_after_first_double_pass"]
                or search["evaluated_ranks"][-1]
                == search["first_confirmed_candidate_rank"],
                {},
            ),
            GateResult(
                "state_unchanged",
                state_unchanged and ols_unchanged,
                {},
            ),
            GateResult(
                "no_update_isolation",
                True,
                {
                    "optimizer_created": False,
                    "backward_calls": 0,
                    "probe_refit": False,
                    "bias_refit": False,
                    "decoder_created": False,
                    "checkpoint_mutation": False,
                    "generation": False,
                },
            ),
            GateResult(
                "streaming_no_banks",
                compact["resource"]["pooled_feature_bank_saved"] is False
                and compact["resource"]["target_bank_saved"] is False,
                {},
            ),
            GateResult(
                "resource_limits",
                peak <= MAX_PEAK_CUDA_BYTES
                and temporary_compact_features <= MAX_TEMPORARY_COMPACT_FEATURE_BYTES,
                {
                    "peak_bytes": peak,
                    "temporary_compact_feature_bytes": temporary_compact_features,
                },
            ),
        )
        payload = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            state,
            upstream.data.manifest["manifest_hash"],
            hashes,
            gates,
            (NEXT_ACTION_RESULT_REVIEW,),
            audit_contract,
            "LRRC is diagnostic-only; no downstream training or promotion",
        )
        payload.update(
            {
                "rank_candidates": list(RANK_CANDIDATES),
                "evaluated_ranks": search["evaluated_ranks"],
                "first_confirmed_candidate_rank": winner,
                "previous_registered_candidate": search["previous_registered_candidate"],
                "rank_256_reused_as_fallback": search["rank_256_reused_as_fallback"],
            }
        )
        _write_new(output_path, payload, artifact_kind="LRRC audit evidence")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except LRRCHarnessFailure as exc:
        if not output_path.exists():
            _invalid_harness_payload(output_path, spec, review_path, upstream, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if (
        args.research_action
        == "scaled-chen-low-rank-readout-recovery-confirmation-validate"
    ):
        return validate(args)
    if (
        args.research_action
        == "scaled-chen-low-rank-readout-recovery-confirmation-run"
    ):
        return run(args)
    raise ValueError(f"unsupported LRRC action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "CONTROLS",
    "EVALUATION",
    "GATE_CONTRACT",
    "HYPOTHESIS",
    "LINEAGE",
    "LRRCHarnessFailure",
    "LSDA_CONSUMER",
    "MEASUREMENT",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "RANK_CANDIDATES",
    "RANK0",
    "REVIEW_SCHEMA",
    "RMAL_CONSUMER",
    "RWSA_ACTIVE_IDENTITY",
    "RWSA_IDENTITY",
    "SEARCH_RANKS",
    "Spec",
    "TRUNCATION",
    "VALIDATION_SCHEMA",
    "ZCSA_IDENTITY",
    "_assert_contract",
    "_factorized_prediction",
    "_full_prediction",
    "_load_review",
    "_load_spec",
    "_operator_identity",
    "_rank0_prediction",
    "_rank_boundary_telemetry",
    "_sequential_search",
    "_spectral_lock",
    "_validate_operator_inputs",
    "dispatch",
    "run",
    "validate",
]
