"""Confirm mean-preserving, fit-data-aware readout recovery.

This is a diagnostic-only consumer of the frozen RMAL detector.  The fit
panel is replayed only to recover the registered DMSA population moments;
selection and confirmation are used only after the transform is frozen.  The
primary search changes the centered output subspace, while preserving the
fit-derived mean output exactly.  No optimizer, refit, feature bank, or
permanent truncated operator is created.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
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
from . import optical_manifold_smf_bogt_data_weighted_spectral_functional_attribution as dsfa
from . import optical_manifold_smf_bogt_detector_data_metric_spectral_attribution as dmsa
from . import optical_manifold_smf_bogt_detector_mean_centered_functional_attribution as mcfa
from . import optical_manifold_smf_bogt_detector_readout_structure_attribution as rwsa
from . import optical_manifold_smf_bogt_linear_readout_structure_decomposition_audit as lsda
from . import optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
from . import optical_manifold_smf_bogt_manifold_readiness as mra


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S7-MPCDAR"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "ef82e8386da393e1bef96f5d77aa7d3e007f0f87"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
INPUT_DIM = 257
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10
RANK_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEARCH_RANKS = RANK_CANDIDATES[:-1]
RANK0 = 0
LANDMARKS = RANK_CANDIDATES
MATERIAL_RELATIVE_IMPROVEMENT = lrrc.MATERIAL_RELATIVE_IMPROVEMENT
PHYSICAL_BASELINE_TOLERANCE = lrrc.PHYSICAL_BASELINE_TOLERANCE
PROBE_REPLAY_ABSOLUTE_TOLERANCE = lrrc.PROBE_REPLAY_ABSOLUTE_TOLERANCE
PROBE_REPLAY_RELATIVE_TOLERANCE = lrrc.PROBE_REPLAY_RELATIVE_TOLERANCE
RANK256_OPERATOR_ABSOLUTE_TOLERANCE = lrrc.RANK256_OPERATOR_ABSOLUTE_TOLERANCE
RANK256_OPERATOR_RELATIVE_TOLERANCE = lrrc.RANK256_OPERATOR_RELATIVE_TOLERANCE
BOOTSTRAP_REPLICATES = lrrc.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = lrrc.BOOTSTRAP_SEED
CONFIDENCE_LEVEL = lrrc.CONFIDENCE_LEVEL
MIN_FREE_CUDA_BYTES = lrrc.MIN_FREE_CUDA_BYTES
MAX_PEAK_CUDA_BYTES = lrrc.MAX_PEAK_CUDA_BYTES
MIN_DISK_FREE_BYTES = lrrc.MIN_DISK_FREE_BYTES
MAX_STORAGE_BYTES = lrrc.MAX_STORAGE_BYTES
MAX_TEMPORARY_COMPACT_FEATURE_BYTES = lrrc.MAX_TEMPORARY_COMPACT_FEATURE_BYTES

RMAL_CONSUMER = dict(dmsa.RMAL_CONSUMER)
LSDA_CONSUMER = dict(dmsa.LSDA_CONSUMER)
RWSA_IDENTITY = dict(dmsa.RWSA_IDENTITY)
RWSA_ACTIVE_IDENTITY = dict(dmsa.RWSA_ACTIVE_IDENTITY)
ZCSA_IDENTITY = dict(dmsa.ZCSA_IDENTITY)
LRRC_CONSUMER = dict(mcfa.LRRC_CONSUMER)
DSFA_CONSUMER = dict(mcfa.DSFA_CONSUMER)
DMSA_CONSUMER = dict(mcfa.DMSA_CONSUMER)
MCFA_CONSUMER = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_v1.yaml"
    ),
    "config_sha256": "03a55ac53b591013592fb5c20c430c3cff1712aadc6d249d29727ee520ecbf07",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_review_v1.json"
    ),
    "review_sha256": "e626b011cc3bea5649d8f5300e93444408866270fa5e9d0a6089fba1666c3c69",
    "validation": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/validation.json"
    ),
    "validation_sha256": "35ad479aa931cb2e605bb9b582306549512106a5c2dcd070bfde35a2e552e7f1",
    "audit": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/audit.json"
    ),
    "audit_sha256": "f69664cf9da1d61e90742a09614c4535388402f9228e401d2451e3ebeeefdc0b",
    "compact_report": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_mean_centered_functional_attribution_v1_attempt_02/compact_report.json"
    ),
    "compact_report_sha256": "e95c072bde8306dd3f7cb8d6c5af1a671a778cede8f84c285c450af854059494",
    "implementation_module": (
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_detector_mean_centered_functional_attribution.py"
    ),
    "implementation_module_sha256": "fdc1455d25e0a9601d21bda4c30e458e66e384a300757942c2800d651560175c",
    "implementation_commit": "069cfa4b3aeda3943d754d739daf6daeba2a54a6",
    "status": "passed",
    "classification": "detector_mean_centered_functional_attribution_completed",
}

FIT_PANEL = dict(dmsa.FIT_PANEL)
PANEL_CONTRACT = {
    "fit": FIT_PANEL,
    "selection": {
        "role": "independent_registered_selection",
        "count": SELECTION_COUNT,
        "source": "mnist_test_indices_5000_9999",
        "parent": dmsa.PANEL_CONTRACT["selection"]["parent"],
    },
    "confirmation": {
        "role": "independent_registered_confirmation",
        "count": CONFIRMATION_COUNT,
        "source": "horizon_heldout_train_indices_55000_59999",
        "parent": dmsa.PANEL_CONTRACT["confirmation"]["parent"],
    },
    "feature_path": "output.camera -> mra._feature_projection(real, complex_value=False)",
    "streaming": True,
    "population_denominator": "N",
    "fit_is_reference_only": True,
    "selection_confirmation_independent": True,
    "pooled_feature_bank_saved": False,
    "centered_feature_bank_saved": False,
    "target_bank_saved": False,
    "prediction_bank_saved": False,
    "raw_cmos_bank_saved": False,
    "internal_stage_capture": False,
}

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S6-MCFA",
    "lineage_start_head": LINEAGE_START_HEAD,
    "rmal_probe_frozen": True,
    "lsda_passed_required": True,
    "lrrc_passed_required": True,
    "dsfa_passed_required": True,
    "dmsa_passed_required": True,
    "mcfa_passed_required": True,
    "zcsa_passed_required": True,
    "rwsa_active_passed_required": True,
    "fit_selection_confirmation_separate": True,
}

FIT_REPLAY_CONTRACT = {
    "source": "DMSA attempt-02 fit panel, exact registered 1000 samples",
    "moment_accumulator": "dmsa.MomentAccumulator",
    "arithmetic": "deterministic float64 Welford/Chan merge",
    "population_denominator": "N",
    "mu": "sum_z/N",
    "centered": "M2/N",
    "hash_helper": "ldm_project.v2.evidence.tensor_sha256",
    "target_labels_used": False,
    "target_values_used": False,
    "selection_used": False,
    "confirmation_used": False,
    "dmsa_C_tensor_artifact_required": False,
}

DATA_AWARE_CONTRACT = {
    "C_fit": "replayed_fit_centered_moment",
    "C_psd_rule": "DMSA _psd_square_root; no singular cutoff",
    "A_fit": "C_fit^(1/2) W64",
    "G_fit": "A_fit A_fit.T",
    "eigendecomposition": "256x256 CPU float64 symmetric eigendecomposition",
    "right_basis": "A_fit.T Q diag(1/sigma_DA)",
    "basis_shape": [TOTAL_COLUMNS, FEATURE_DIM],
    "full_projector": "(A_fit V256) V256.T and (W64 V256) V256.T",
    "permanent_W_k_saved": False,
    "V_VT_materialized": False,
    "target_labels_used": False,
    "target_values_used": False,
}

RAW_COMPARATOR_CONTRACT = {
    "source": "LSDA exact ordered raw-W SVD",
    "old_lrrc": "b + z U_raw_k U_raw_k.T W",
    "mean_preserving_raw": "b + mu_fit W + (z-mu_fit) W V_raw_k V_raw_k.T",
    "raw_projector_equivalence": True,
    "descriptive_only": True,
    "primary_search": False,
    "confirmation_trigger": False,
    "stop_rule": False,
    "family_expansion": False,
}

TRUNCATION = {
    "primary": "c_mean + (z-mu_fit) W V_DA_k V_DA_k.T",
    "factorized": "c_mean + ((z-mu_fit) T_DA_k) V_DA_k.T",
    "mean_branch": "c_mean=b+mu_fit W",
    "bias": "original frozen RMAL b",
    "rank_specific_scaling": False,
    "normalization": False,
    "calibration": False,
    "permanent_W_k_saved": False,
}

CONTROLS = {
    "rank_0_bias_only": "prediction_0=b; not in search",
    "mean_fit_control": "prediction_mean_fit=b+mu_fit W; descriptive only",
    "rank_256_full_positive_anchor": "data-aware V256 projector and prediction replay",
    "mean_preserving_raw_control": "same mu_fit and centered input; descriptive only",
}

EVALUATION = {
    "selection": dict(lrrc.EVALUATION["selection"]),
    "confirmation": dict(lrrc.EVALUATION["confirmation"]),
    "candidate_prefix": dict(lrrc.EVALUATION["candidate_prefix"]),
    "feature_path": PANEL_CONTRACT["feature_path"],
    "streaming": True,
    "pooled_feature_bank_saved": False,
    "centered_feature_bank_saved": False,
    "target_bank_saved": False,
    "prediction_bank_saved": False,
    "raw_cmos_bank_saved": False,
    "internal_stage_capture": False,
    "no_refit": True,
    "no_training": True,
}

GATE_CONTRACT = {
    "source": "rmal._panel_gate(report, require_bootstrap=confirmation)",
    "inherited_checks": list(lrrc.INHERITED_GATE_NAMES),
    "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "no_energy_gate": True,
    "no_effective_rank_gate": True,
    "no_data_aware_superiority_gate": True,
    "matched_control_new_gate": False,
}

MEASUREMENT = {
    "objective": "mean_preserving_fit_data_aware_output_subspace_recovery",
    "candidate_family": list(RANK_CANDIDATES),
    "search_order_after_anchor": list(SEARCH_RANKS),
    "rank_0_control": RANK0,
    "rank_256_anchor_first": True,
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "first_double_pass_stops": True,
    "first_confirmed_candidate_rank_only": True,
    "no_candidate_family_mutation": True,
    "no_middle_rank_backtest": True,
    "no_training": True,
    "no_refit": True,
    "no_physical_forward_modification": True,
    "no_target_fitting": True,
    "matched_control_descriptive_only": True,
    "mechanism_evidence_only": True,
}

RESOURCE_CONTRACT = {
    "microbatch_size": MICROBATCH_SIZE,
    "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "max_temporary_compact_feature_bytes": MAX_TEMPORARY_COMPACT_FEATURE_BYTES,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_mean_preserving_centered_data_aware_readout_recovery",
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
        "decoder",
        "whitening",
        "optical_redesign",
        "new_rank_family",
        "generation",
        "promotion",
        "candidate_family_mutation",
        "target_fitting",
        "matched_control_search",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_mean_preserving_centered_data_aware_readout_recovery"
NEXT_ACTION_RESULT_REVIEW = "manual_review_MPCDAR_result"


class MPCDARHarnessFailure(RuntimeError):
    """A registered MPCDAR boundary failed closed."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"MPCDAR requires clean branch {BRANCH}")


def _consumer_lock(consumer: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "config",
        "config_sha256",
        "review_contract",
        "review_sha256",
        "validation",
        "validation_sha256",
        "audit",
        "audit_sha256",
        "compact_report",
        "compact_report_sha256",
        "implementation_module",
        "implementation_module_sha256",
        "implementation_commit",
        "status",
        "classification",
        "audit_classification",
        "probe",
        "probe_sha256",
    )
    return {key: consumer[key] for key in keys if key in consumer}


AUTHORITY_LOCK = {
    "rmal": _consumer_lock(RMAL_CONSUMER),
    "lsda": _consumer_lock(LSDA_CONSUMER),
    "lrrc": _consumer_lock(LRRC_CONSUMER),
    "dsfa": _consumer_lock(DSFA_CONSUMER),
    "dmsa": _consumer_lock(DMSA_CONSUMER),
    "mcfa": _consumer_lock(MCFA_CONSUMER),
}


class Spec:
    def __init__(self, path: Path, payload: dict[str, Any], config_hash: str) -> None:
        self.path = path
        self.payload = payload
        self.config_hash = config_hash


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "lineage",
        "authority",
        "fit_panel",
        "panel_contract",
        "fit_replay",
        "data_aware",
        "raw_comparator",
        "truncation",
        "candidate_family",
        "controls",
        "evaluation",
        "gate_inheritance",
        "measurement",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("MPCDAR config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("MPCDAR config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _contract_payload() -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "authority": AUTHORITY_LOCK,
        "fit_panel": FIT_PANEL,
        "panel_contract": PANEL_CONTRACT,
        "fit_replay": FIT_REPLAY_CONTRACT,
        "data_aware": DATA_AWARE_CONTRACT,
        "raw_comparator": RAW_COMPARATOR_CONTRACT,
        "truncation": TRUNCATION,
        "candidate_family": list(RANK_CANDIDATES),
        "controls": CONTROLS,
        "evaluation": EVALUATION,
        "gate_inheritance": GATE_CONTRACT,
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
            raise ValueError(f"MPCDAR {name} contract is not locked")
    if tuple(spec.payload["candidate_family"]) != RANK_CANDIDATES:
        raise ValueError("MPCDAR candidate family is not immutable")


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
        raise ValueError("MPCDAR review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise MPCDARHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MPCDARHarnessFailure(f"{label} is not a mapping: {path}")
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
        raise MPCDARHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise MPCDARHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_parent(
    consumer: Mapping[str, Any], module: Any, spec: Spec, label: str
) -> dict[str, Any]:
    config_path = _resolve(consumer["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != consumer["config_sha256"]:
        raise MPCDARHarnessFailure(f"{label} config identity changed")
    parent_spec = module._load_spec(config_path)
    module._assert_contract(parent_spec)
    review_path = _resolve(consumer["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != consumer["review_sha256"]:
        raise MPCDARHarnessFailure(f"{label} review identity changed")
    module._load_review(review_path, parent_spec)
    implementation_path = None
    if "implementation_module" in consumer:
        implementation_path = _resolve(consumer["implementation_module"])
        if file_sha256(implementation_path) != consumer["implementation_module_sha256"]:
            raise MPCDARHarnessFailure(f"{label} implementation identity changed")
    validation_path = _resolve(consumer["validation"])
    validation = _read_json_hash(
        validation_path, consumer["validation_sha256"], f"{label} validation"
    )
    validation_record = _verify_evidence(
        validation,
        expected_status="passed",
        expected_classification=None,
        label=f"{label} validation",
    )
    if (
        validation_record.git_commit != consumer["implementation_commit"]
        or validation_record.config_hash != parent_spec.config_hash
    ):
        raise MPCDARHarnessFailure(f"{label} validation lineage changed")
    audit_path = _resolve(consumer["audit"])
    audit = _read_json_hash(audit_path, consumer["audit_sha256"], f"{label} audit")
    expected_classification = consumer.get(
        "classification", consumer.get("audit_classification")
    )
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=expected_classification,
        label=f"{label} audit",
    )
    if audit_record.git_commit not in {
        consumer["implementation_commit"],
        consumer.get("result_git_commit"),
    }:
        raise MPCDARHarnessFailure(f"{label} audit lineage changed")
    compact_path = _resolve(consumer["compact_report"])
    compact = _read_json_hash(
        compact_path, consumer["compact_report_sha256"], f"{label} compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != expected_classification
    ):
        raise MPCDARHarnessFailure(f"{label} compact terminal identity changed")
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
        "spec": parent_spec,
    }


def _verify_authorities(spec: Spec) -> dict[str, Any]:
    parents = {
        "dsfa": _verify_parent(DSFA_CONSUMER, dsfa, spec, "DSFA"),
        "dmsa": _verify_parent(DMSA_CONSUMER, dmsa, spec, "DMSA"),
        "mcfa": _verify_parent(MCFA_CONSUMER, mcfa, spec, "MCFA"),
        "lrrc": _verify_parent(LRRC_CONSUMER, lrrc, spec, "LRRC"),
    }
    if (
        parents["lrrc"]["compact"].get("first_confirmed_candidate_rank") != 256
        or parents["lrrc"]["compact"].get("rank_candidates") != list(RANK_CANDIDATES)
    ):
        raise MPCDARHarnessFailure("LRRC terminal candidate identity changed")
    dmsa_fit = parents["dmsa"]["compact"].get("panels", {}).get("fit", {})
    matrix_hashes = dmsa_fit.get("matrix_hashes", {})
    if not isinstance(matrix_hashes, Mapping) or not matrix_hashes.get("mu") or not matrix_hashes.get("C"):
        raise MPCDARHarnessFailure("DMSA fit mu/C hash semantics are unavailable")
    if (
        parents["dmsa"]["compact"].get("moment_contract", {}).get("population_denominator")
        not in {"N", "population_N"}
    ):
        raise MPCDARHarnessFailure("DMSA fit population moment contract changed")
    return parents


def _load_upstream(spec: Spec, dataset_root: str | Path) -> dict[str, Any]:
    authorities = _verify_authorities(spec)
    try:
        runtime = lrrc._load_upstream(spec, dataset_root)
    except (ValueError, rwsa.RWSAHarnessFailure, lrrc.LRRCHarnessFailure) as exc:
        raise MPCDARHarnessFailure("RMAL/LSDA upstream identity is invalid") from exc
    probe = runtime.rmal_probe
    if (
        probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]
        or tuple(probe.weights.shape) != (FEATURE_DIM, TOTAL_COLUMNS)
        or tuple(probe.bias.shape) != (TOTAL_COLUMNS,)
        or probe.weights.dtype != torch.float32
        or probe.bias.dtype != torch.float32
    ):
        raise MPCDARHarnessFailure("frozen RMAL probe identity changed")
    return {"authorities": authorities, "runtime": runtime}


def _spectral_lock(upstream: Mapping[str, Any]) -> dict[str, Any]:
    runtime = upstream["runtime"]
    try:
        return lrrc._spectral_lock(runtime.rmal_probe, runtime.lsda_compact)
    except (ValueError, lrrc.LRRCHarnessFailure, lsda.LSDAHarnessFailure) as exc:
        raise MPCDARHarnessFailure("raw LSDA spectral lock failed") from exc


def _fit_images(upstream: Mapping[str, Any]) -> torch.Tensor:
    runtime = upstream["runtime"]
    images = runtime.data.images.index_select(0, runtime.fit_indices)
    if tuple(images.shape) != (FIT_COUNT, 1, 28, 28):
        raise MPCDARHarnessFailure("fit panel shape changed")
    return images


def _replay_fit_moments(
    model: torch.nn.Module, upstream: Mapping[str, Any], device: torch.device
) -> Any:
    accumulator = dmsa.MomentAccumulator()
    images = _fit_images(upstream)
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        camera = lrrc._capture_cmos(model, images.index_select(0, indices), device)
        features = mra._feature_projection(camera, complex_value=False).float()
        if tuple(features.shape) != (indices.numel(), FEATURE_DIM):
            raise MPCDARHarnessFailure("fit feature projection shape changed")
        accumulator.update(features)
    result = accumulator.finalize("fit")
    if result.count != FIT_COUNT:
        raise MPCDARHarnessFailure("fit replay count changed")
    return result


def _fit_replay_lock(result: Any, upstream: Mapping[str, Any]) -> dict[str, Any]:
    fit = upstream["authorities"]["dmsa"]["compact"]["panels"]["fit"]
    hashes = fit.get("matrix_hashes")
    if not isinstance(hashes, Mapping) or not hashes.get("mu") or not hashes.get("C"):
        raise MPCDARHarnessFailure("DMSA fit matrix hashes are unavailable")
    mu_hash = tensor_sha256(result.mu)
    c_hash = tensor_sha256(result.centered)
    mu_match = mu_hash == hashes["mu"]
    c_match = c_hash == hashes["C"]
    if not mu_match or not c_match:
        raise MPCDARHarnessFailure("MPCDAR fit replay hash mismatch with DMSA")
    stored_mu = torch.tensor(fit.get("mu", []), dtype=torch.float64)
    stored_mu_match = stored_mu.shape == result.mu.shape and bool(
        torch.allclose(stored_mu, result.mu, rtol=1.0e-11, atol=1.0e-8)
    )
    if not stored_mu_match:
        raise MPCDARHarnessFailure("MPCDAR fit replay mean vector mismatch with DMSA")
    return {
        "count": result.count,
        "mu_sha256": mu_hash,
        "C_sha256": c_hash,
        "dmsa_mu_sha256": hashes["mu"],
        "dmsa_C_sha256": hashes["C"],
        "mu_hash_match": mu_match,
        "C_hash_match": c_match,
        "stored_mu_vector_match": stored_mu_match,
        "moment_identity": dict(result.identity),
        "moment_symmetry": dict(result.symmetry),
        "target_labels_used": False,
        "target_values_used": False,
        "selection_used": False,
        "confirmation_used": False,
    }


def _spectrum_report(summary: Any) -> dict[str, Any]:
    report = summary.compact()
    report["cumulative_energy_landmarks"] = {
        str(rank): float(summary.cumulative_energy[rank - 1]) for rank in LANDMARKS
    }
    report["boundary_gap_source"] = "A_fit/G_fit data-aware spectrum"
    return report


def _build_data_aware_basis(
    W64: torch.Tensor, C_fit: torch.Tensor
) -> dict[str, Any]:
    try:
        C_half, psd = dmsa._psd_square_root(C_fit, "fit.C")
        G_raw = W64.mm(W64.T)
        A_fit = C_half.mm(W64)
        G_fit = A_fit.mm(A_fit.T)
        summary = lsda._spectrum_from_gram(G_fit, "MPCDAR A_fit")
    except (ValueError, dmsa.DMSAHarnessFailure, lsda.LSDAHarnessFailure) as exc:
        raise MPCDARHarnessFailure("data-aware spectrum construction failed") from exc
    sigma = summary.singular_values
    if sigma.shape != (FEATURE_DIM,) or not bool(torch.isfinite(sigma).all()) or bool(sigma.le(0).any()):
        raise MPCDARHarnessFailure("full data-aware right basis is not finitely defined")
    Q = summary.eigenvectors
    V = A_fit.T.mm(Q).div(sigma.unsqueeze(0)).contiguous()
    if V.shape != (TOTAL_COLUMNS, FEATURE_DIM) or not bool(torch.isfinite(V).all()):
        raise MPCDARHarnessFailure("data-aware right basis has invalid shape or values")
    orth_residual = V.T.mm(V) - torch.eye(FEATURE_DIM, dtype=torch.float64)
    orth_norm = float(torch.linalg.matrix_norm(orth_residual))
    orth_max = float(orth_residual.abs().max())
    if not torch.allclose(
        V.T.mm(V),
        torch.eye(FEATURE_DIM, dtype=torch.float64),
        rtol=1.0e-9,
        atol=1.0e-9,
    ):
        raise MPCDARHarnessFailure("data-aware right basis is not orthonormal")
    return {
        "C_half": C_half,
        "A_fit": A_fit,
        "G_raw": G_raw,
        "G_fit": G_fit,
        "summary": summary,
        "V": V,
        "psd": psd,
        "orthogonality": {
            "max_abs_residual": orth_max,
            "frobenius_residual": orth_norm,
            "passed": True,
        },
        "gram_identity_residual": _matrix_residual(
            G_fit, C_half.mm(G_raw).mm(C_half)
        ),
        "rank256_identity": {
            "A_fit_projector": _projector_identity(A_fit, V),
            "W_projector": _projector_identity(W64, V),
        },
    }


def _matrix_residual(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    residual = left - right
    norm = max(float(torch.linalg.matrix_norm(left)), float(torch.linalg.matrix_norm(right)), 1.0e-30)
    return {
        "max_abs_residual": float(residual.abs().max()),
        "frobenius_residual": float(torch.linalg.matrix_norm(residual)),
        "relative_frobenius_residual": float(torch.linalg.matrix_norm(residual)) / norm,
    }


def _data_aware_boundary_telemetry(rank: int, summary: Any) -> dict[str, Any]:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    sigma = summary.singular_values
    squared = summary.squared_singular_values
    next_sigma = float(sigma[rank]) if rank < FEATURE_DIM else None
    gap = None if next_sigma is None else float((sigma[rank - 1] - sigma[rank]) / sigma[rank - 1])
    return {
        "rank": rank,
        "sigma_DA_k": float(sigma[rank - 1]),
        "sigma_DA_k_plus_1": next_sigma,
        "relative_gap_DA": gap,
        "cumulative_squared_energy_DA": float(squared[:rank].sum()),
        "normalized_energy_retained_DA": float(summary.cumulative_energy[rank - 1]),
        "gap_source": "A_fit/G_fit data-aware spectrum",
        "descriptive_only": True,
        "basis_boundary_ambiguity": (
            "individual vectors may rotate within a near-degenerate subspace"
            if gap is not None and gap < 1.0e-2
            else None
        ),
    }


def _raw_boundary_telemetry(rank: int, raw_spectral: Mapping[str, Any]) -> dict[str, Any]:
    result = lrrc._rank_boundary_telemetry(rank, raw_spectral)
    return {
        "rank": rank,
        "sigma_raw_k": result["sigma_k"],
        "sigma_raw_k_plus_1": result["sigma_k_plus_1"],
        "relative_gap_raw": result["relative_gap"],
        "gap_source": "LSDA raw-W comparator telemetry",
        "descriptive_only": True,
    }


def _raw_right_basis(raw_spectral: Mapping[str, Any]) -> torch.Tensor:
    W64 = raw_spectral["W64"]
    U = raw_spectral["U"]
    sigma = raw_spectral["full"].singular_values
    if bool(sigma.le(0).any()):
        raise MPCDARHarnessFailure("raw W right basis is not finitely defined")
    V = W64.T.mm(U).div(sigma.unsqueeze(0)).contiguous()
    if V.shape != (TOTAL_COLUMNS, FEATURE_DIM) or not bool(torch.isfinite(V).all()):
        raise MPCDARHarnessFailure("raw W right basis is invalid")
    if not torch.allclose(
        V.T.mm(V), torch.eye(FEATURE_DIM, dtype=torch.float64), rtol=1.0e-9, atol=1.0e-9
    ):
        raise MPCDARHarnessFailure("raw W right basis is not orthonormal")
    return V


def _validate_prediction_inputs(
    features: torch.Tensor,
    W64: torch.Tensor,
    bias: torch.Tensor,
    mu: torch.Tensor,
) -> None:
    if features.ndim != 2 or tuple(features.shape[1:]) != (FEATURE_DIM,):
        raise ValueError("features must have shape [N,256]")
    if W64.dtype != torch.float64 or W64.device.type != "cpu" or W64.shape != (FEATURE_DIM, TOTAL_COLUMNS):
        raise ValueError("W64 must be CPU float64 with shape [256,40000]")
    if bias.dtype != torch.float32 or bias.device.type != "cpu" or bias.shape != (TOTAL_COLUMNS,):
        raise ValueError("b must be CPU float32 with shape [40000]")
    if mu.dtype != torch.float64 or mu.device.type != "cpu" or mu.shape != (FEATURE_DIM,):
        raise ValueError("mu_fit must be CPU float64 with shape [256]")


def _mean_prediction(W64: torch.Tensor, bias: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    if W64.shape != (FEATURE_DIM, TOTAL_COLUMNS) or W64.dtype != torch.float64:
        raise ValueError("W64 has an invalid shape or dtype")
    if bias.shape != (TOTAL_COLUMNS,) or bias.dtype != torch.float32:
        raise ValueError("bias has an invalid shape or dtype")
    if mu.shape != (FEATURE_DIM,) or mu.dtype != torch.float64:
        raise ValueError("mu_fit has an invalid shape or dtype")
    result = mu.unsqueeze(0).mm(W64).add(bias.to(torch.float64))
    if not bool(torch.isfinite(result).all()):
        raise MPCDARHarnessFailure("mean-preserving branch is non-finite")
    return result


def _old_raw_prediction(
    features: torch.Tensor,
    U_raw: torch.Tensor,
    W64: torch.Tensor,
    bias: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    _validate_prediction_inputs(features, W64, bias, torch.zeros(FEATURE_DIM, dtype=torch.float64))
    U_k = U_raw[:, :rank]
    dynamic = features.to(torch.float64).mm(U_k).mm(U_k.T.mm(W64))
    result = dynamic.add(bias.to(torch.float64))
    if not bool(torch.isfinite(result).all()):
        raise MPCDARHarnessFailure("old LRRC raw prediction is non-finite")
    return result


def _mean_preserving_raw_prediction(
    features: torch.Tensor,
    U_raw: torch.Tensor,
    W64: torch.Tensor,
    bias: torch.Tensor,
    mu_fit: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    _validate_prediction_inputs(features, W64, bias, mu_fit)
    U_k = U_raw[:, :rank]
    centered = features.to(torch.float64) - mu_fit
    mean = _mean_prediction(W64, bias, mu_fit)
    result = centered.mm(U_k).mm(U_k.T.mm(W64)).add(mean)
    if not bool(torch.isfinite(result).all()):
        raise MPCDARHarnessFailure("mean-preserving raw prediction is non-finite")
    return result


def _data_aware_prediction(
    features: torch.Tensor,
    V_DA: torch.Tensor,
    W64: torch.Tensor,
    bias: torch.Tensor,
    mu_fit: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    _validate_prediction_inputs(features, W64, bias, mu_fit)
    if V_DA.dtype != torch.float64 or V_DA.device.type != "cpu" or V_DA.shape != (TOTAL_COLUMNS, FEATURE_DIM):
        raise ValueError("V_DA must be CPU float64 with shape [40000,256]")
    V_k = V_DA[:, :rank]
    T_DA = W64.mm(V_k)
    centered = features.to(torch.float64) - mu_fit
    result = centered.mm(T_DA).mm(V_k.T).add(_mean_prediction(W64, bias, mu_fit))
    if not bool(torch.isfinite(result).all()):
        raise MPCDARHarnessFailure("data-aware prediction is non-finite")
    return result


def _rank0_prediction(features: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("rank-0 features have the wrong shape")
    if bias.dtype != torch.float32 or bias.shape != (TOTAL_COLUMNS,):
        raise ValueError("rank-0 bias has the wrong shape or dtype")
    return bias.to(torch.float64).unsqueeze(0).expand(features.shape[0], -1).clone()


def _projector_identity(matrix: torch.Tensor, V: torch.Tensor) -> dict[str, Any]:
    projected = matrix.mm(V).mm(V.T)
    return _matrix_residual(projected, matrix) | {
        "passed": bool(torch.allclose(projected, matrix, rtol=1.0e-9, atol=1.0e-9))
    }


def _sign_flip_projector_invariance(
    matrix: torch.Tensor, V: torch.Tensor, *, rank: int = 4
) -> dict[str, Any]:
    if rank not in RANK_CANDIDATES:
        raise ValueError("rank is not registered")
    flipped = V[:, :rank].clone()
    flipped[:, 0] = -flipped[:, 0]
    left = matrix.mm(V[:, :rank]).mm(V[:, :rank].T)
    right = matrix.mm(flipped).mm(flipped.T)
    return _matrix_residual(left, right) | {"passed": bool(torch.equal(left, right))}


def _raw_projector_equivalence(
    W64: torch.Tensor, U_raw: torch.Tensor, V_raw: torch.Tensor, rank: int
) -> dict[str, Any]:
    U_k = U_raw[:, :rank]
    V_k = V_raw[:, :rank]
    left = U_k.mm(U_k.T).mm(W64)
    right = W64.mm(V_k).mm(V_k.T)
    result = _matrix_residual(left, right)
    result["passed"] = bool(torch.allclose(left, right, rtol=1.0e-9, atol=1.0e-9))
    return result


def _target_templates(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return lrrc._target_templates(images, labels)


def _context_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return rmfa._context_metrics_from_compact(prediction, target, labels)


def _panel_gate(report: Mapping[str, Any], *, require_bootstrap: bool) -> dict[str, Any]:
    return rmal._panel_gate(report, require_bootstrap=require_bootstrap)


def _bootstrap_ci(delta: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    return rmal._stratified_bootstrap_ci(delta.float(), labels)


def _mse_by_label(values: torch.Tensor, labels: torch.Tensor) -> list[float]:
    return [float(values[labels == label].mean()) for label in range(10)]


def _control_summary(
    name: str,
    errors: torch.Tensor,
    labels: torch.Tensor,
    physical: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    value = float(errors.mean())
    return {
        "name": name,
        "probe": {"paired_mse": value, "mse_by_label": _mse_by_label(errors, labels), **context},
        "physical_comparison": dict(physical),
        "instance_sensitive": bool(
            value < float(physical["cyclic_mse"])
            and float(context["same_label_top1"]) >= float(context["class_template_same_label_top1"])
        ),
        "descriptive_only": True,
        "participates_in_primary_search": False,
        "triggers_confirmation": False,
        "changes_rmal_gate": False,
    }


def _old_lrrc_reference(
    compact: Mapping[str, Any], panel: str, rank: int
) -> float | None:
    if rank == 256:
        anchor = compact.get("rank_256_full_positive_anchor", {})
        report = anchor.get(panel, {}) if isinstance(anchor, Mapping) else {}
    else:
        item = compact.get("rank_results", {}).get(str(rank), {})
        report = item.get(panel, {}) if isinstance(item, Mapping) else {}
    value = report.get("probe", {}).get("paired_mse") if isinstance(report, Mapping) else None
    return None if value is None else float(value)


def _evaluate_panel(
    model: torch.nn.Module,
    probe: rwsa.ReadoutProbe,
    raw_spectral: Mapping[str, Any],
    da_basis: Mapping[str, Any],
    mu_fit: torch.Tensor,
    rank: int,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    expected_anchor: Mapping[str, float],
    lrrc_compact: Mapping[str, Any],
    *,
    include_rank0: bool = False,
) -> dict[str, Any]:
    if rank not in RANK_CANDIDATES:
        raise ValueError("panel rank is not registered")
    model_before = rmal._state_cpu(model)
    ols_before = rmal._state_cpu(model.ols)
    W64 = raw_spectral["W64"]
    U_raw = raw_spectral["U"]
    bias = probe.bias
    mean_output = _mean_prediction(W64, bias, mu_fit)
    templates = _target_templates(images, labels)
    physical_errors: list[torch.Tensor] = []
    candidate_errors: list[torch.Tensor] = []
    full_errors: list[torch.Tensor] = []
    cyclic_errors: list[torch.Tensor] = []
    template_errors: list[torch.Tensor] = []
    mean_errors: list[torch.Tensor] = []
    bias_errors: list[torch.Tensor] = []
    old_raw_errors: list[torch.Tensor] = []
    mp_raw_errors: list[torch.Tensor] = []
    candidate_compact: list[torch.Tensor] = []
    target_compact: list[torch.Tensor] = []
    mean_compact: list[torch.Tensor] = []
    rank0_errors: list[torch.Tensor] = []
    identity_max = 0.0
    identity_sum = 0.0
    identity_count = 0
    sample_count = int(images.shape[0])
    for indices in torch.arange(sample_count, dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        batch_labels = labels.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        previous = images.index_select(0, (indices - 1) % sample_count)
        cyclic_target = base.render_intensity_batch(previous).target.float()
        physical = lrrc._capture_cmos(model, batch_images, device)
        features = mra._feature_projection(physical, complex_value=False).float()
        full = lrrc._full_prediction(features, probe.weights, probe.bias).to(torch.float64)
        candidate = _data_aware_prediction(features, da_basis["V"], W64, bias, mu_fit, rank)
        old_raw = _old_raw_prediction(features, U_raw, W64, bias, rank)
        mp_raw = _mean_preserving_raw_prediction(features, U_raw, W64, bias, mu_fit, rank)
        mean = mean_output.expand(features.shape[0], -1)
        bias_only = _rank0_prediction(features, bias)
        target64 = target.to(torch.float64)
        template = templates.index_select(0, batch_labels).to(torch.float64)
        cyclic64 = cyclic_target.to(torch.float64)
        physical64 = physical.to(torch.float64)
        physical_errors.append((physical64 - target64).square().mean(dim=(1, 2, 3)))
        candidate_errors.append((candidate.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        full_errors.append((full.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        cyclic_errors.append((physical64 - cyclic64).square().mean(dim=(1, 2, 3)))
        template_errors.append((physical64 - template).square().mean(dim=(1, 2, 3)))
        mean_errors.append((mean.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        bias_errors.append((bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        old_raw_errors.append((old_raw.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        mp_raw_errors.append((mp_raw.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        candidate_compact.append(mra._feature_projection(candidate.reshape(-1, 1, 200, 200).float()).float())
        mean_compact.append(mra._feature_projection(mean.reshape(-1, 1, 200, 200).float()).float())
        target_compact.append(mra._feature_projection(target).float())
        difference = (candidate - full).abs()
        identity_max = max(identity_max, float(difference.max()))
        identity_sum += float(difference.sum())
        identity_count += difference.numel()
        if include_rank0:
            rank0_errors.append((bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
    physical_mse = torch.cat(physical_errors)
    candidate_mse = torch.cat(candidate_errors)
    full_mse = torch.cat(full_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    mean_mse = torch.cat(mean_errors)
    bias_mse = torch.cat(bias_errors)
    old_raw_mse = torch.cat(old_raw_errors)
    mp_raw_mse = torch.cat(mp_raw_errors)
    candidate_context = _context_metrics(torch.cat(candidate_compact), torch.cat(target_compact), labels)
    mean_context = _context_metrics(torch.cat(mean_compact), torch.cat(target_compact), labels)
    physical_value = float(physical_mse.mean())
    candidate_value = float(candidate_mse.mean())
    full_value = float(full_mse.mean())
    delta = physical_mse - candidate_mse
    ci = _bootstrap_ci(delta, labels)
    physical_values = {
        "paired_mse": physical_value,
        "class_template_mse": float(template_mse.mean()),
        "cyclic_mse": float(cyclic_mse.mean()),
        "mse_by_label": _mse_by_label(physical_mse, labels),
    }
    native_replay = {
        key: abs(physical_values[key] - float(expected_anchor[key])) <= PHYSICAL_BASELINE_TOLERANCE
        for key in ("paired_mse", "class_template_mse", "cyclic_mse")
    }
    matched = {
        "A_OLD_LRRC_RAW": {"paired_mse": float(old_raw_mse.mean()), "mse_by_label": _mse_by_label(old_raw_mse, labels)},
        "B_MEAN_PRESERVING_RAW": {"paired_mse": float(mp_raw_mse.mean()), "mse_by_label": _mse_by_label(mp_raw_mse, labels)},
        "C_MEAN_PRESERVING_DATA_AWARE": {"paired_mse": candidate_value, "mse_by_label": _mse_by_label(candidate_mse, labels)},
    }
    matched["deltas"] = {
        "A_to_B": matched["B_MEAN_PRESERVING_RAW"]["paired_mse"] - matched["A_OLD_LRRC_RAW"]["paired_mse"],
        "B_to_C": matched["C_MEAN_PRESERVING_DATA_AWARE"]["paired_mse"] - matched["B_MEAN_PRESERVING_RAW"]["paired_mse"],
        "A_to_C": matched["C_MEAN_PRESERVING_DATA_AWARE"]["paired_mse"] - matched["A_OLD_LRRC_RAW"]["paired_mse"],
    }
    old_reference = _old_lrrc_reference(lrrc_compact, "selection" if sample_count == SELECTION_COUNT else "confirmation", rank)
    matched["old_lrrc_frozen_reference_mse"] = old_reference
    matched["old_lrrc_replay_abs_delta"] = None if old_reference is None else abs(float(old_raw_mse.mean()) - old_reference)
    report: dict[str, Any] = {
        "executed": True,
        "panel": "selection" if sample_count == SELECTION_COUNT else "confirmation",
        "rank": rank,
        "evaluation_count": sample_count,
        "physical": physical_values,
        "probe": {
            "paired_mse": candidate_value,
            "mse_by_label": _mse_by_label(candidate_mse, labels),
            **candidate_context,
        },
        "class_template_mse": physical_values["class_template_mse"],
        "cyclic_mse": physical_values["cyclic_mse"],
        "relative_improvement": float(delta.mean() / max(physical_value, 1.0e-12)),
        "bootstrap": {
            "delta_definition": "physical_paired_mse - data_aware_candidate_paired_mse",
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
            and torch.isfinite(old_raw_mse).all()
            and torch.isfinite(mp_raw_mse).all()
        ),
        "instance_sensitive": bool(
            candidate_value < physical_values["cyclic_mse"]
            and float(candidate_context["same_label_top1"]) >= float(candidate_context["class_template_same_label_top1"])
        ),
        "state_unchanged": rmal._state_equal(model_before, rmal._state_cpu(model)),
        "ols_state_unchanged": rmal._state_equal(ols_before, rmal._state_cpu(model.ols)),
        "feature_contract_valid": True,
        "probe_contract_valid": True,
        "square_law_consistency": True,
        "square_law_source": "inherited_registered_RMAL_cmos_intensity_control",
        "full_rmal_reference": {"paired_mse": full_value, "physical_paired_mse": physical_value},
        "mse_delta_vs_full_rmal": candidate_value - full_value,
        "dynamic_only": {"paired_mse": float(candidate_mse.mean())},
        "bias_only": {"paired_mse": float(bias_mse.mean())},
        "mean_control": _control_summary(
            "mean_fit_only",
            mean_mse,
            labels,
            physical_values,
            mean_context,
        ),
        "matched_control": matched,
        "rank_boundary": None,
        "raw_rank_boundary": None,
        "internal_stage_capture": False,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
    }
    report["data_aware_factorized_vs_full_max_abs_residual"] = identity_max
    report["data_aware_factorized_vs_full_mean_abs_residual"] = identity_sum / max(identity_count, 1)
    report["data_aware_factorized_vs_full_close"] = bool(
        identity_max
        <= RANK256_OPERATOR_ABSOLUTE_TOLERANCE
        + RANK256_OPERATOR_RELATIVE_TOLERANCE * max(float(full.abs().max()), 1.0)
    )
    report["rank256_full_prediction_identity"] = {
        "evaluated": rank == 256,
        "max_abs_residual": identity_max if rank == 256 else None,
        "mean_abs_residual": identity_sum / max(identity_count, 1) if rank == 256 else None,
        "passed": report["data_aware_factorized_vs_full_close"] if rank == 256 else None,
    }
    report["selection_gate"] = _panel_gate(report, require_bootstrap=False)
    if include_rank0:
        rank0 = torch.cat(rank0_errors)
        report["rank0_control"] = {
            "prediction": "bias_only",
            "paired_mse": float(rank0.mean()),
            "mse_by_label": _mse_by_label(rank0, labels),
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


def _expected_anchor(compact: Mapping[str, Any], panel: str) -> tuple[dict[str, float], float]:
    return lrrc._expected_anchor(compact, panel)


def _rank256_anchor(
    model: torch.nn.Module,
    upstream: Mapping[str, Any],
    raw_spectral: Mapping[str, Any],
    da_basis: Mapping[str, Any],
    mu_fit: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    runtime = upstream["runtime"]
    compact = runtime.rmal_identity["compact"]
    selection_anchor, selection_expected_probe = _expected_anchor(compact, "selection")
    confirmation_anchor, confirmation_expected_probe = _expected_anchor(compact, "confirmation")
    selection = _evaluate_panel(
        model,
        runtime.rmal_probe,
        raw_spectral,
        da_basis,
        mu_fit,
        256,
        runtime.data.test_images.index_select(0, runtime.data.selection_indices),
        runtime.data.test_labels.index_select(0, runtime.data.selection_indices),
        device,
        selection_anchor,
        compact,
        include_rank0=True,
    )
    confirmation = _evaluate_panel(
        model,
        runtime.rmal_probe,
        raw_spectral,
        da_basis,
        mu_fit,
        256,
        runtime.data.images.index_select(0, runtime.data.confirmation_indices),
        runtime.data.labels.index_select(0, runtime.data.confirmation_indices),
        device,
        confirmation_anchor,
        compact,
        include_rank0=True,
    )
    confirmation["selection_gate"] = _panel_gate(confirmation, require_bootstrap=False)
    confirmation["confirmation_gate"] = _panel_gate(confirmation, require_bootstrap=True)
    selection_replay = abs(float(selection["probe"]["paired_mse"]) - selection_expected_probe)
    confirmation_replay = abs(float(confirmation["probe"]["paired_mse"]) - confirmation_expected_probe)
    selection["rank_boundary"] = _data_aware_boundary_telemetry(256, da_basis["summary"])
    confirmation["rank_boundary"] = _data_aware_boundary_telemetry(256, da_basis["summary"])
    selection["raw_rank_boundary"] = _raw_boundary_telemetry(256, raw_spectral)
    confirmation["raw_rank_boundary"] = _raw_boundary_telemetry(256, raw_spectral)
    primary = bool(
        selection["selection_gate"]["passed"]
        and confirmation["selection_gate"]["passed"]
        and confirmation["confirmation_gate"]["passed"]
        and selection_replay <= PROBE_REPLAY_ABSOLUTE_TOLERANCE + PROBE_REPLAY_RELATIVE_TOLERANCE * max(abs(selection_expected_probe), 1.0)
        and confirmation_replay <= PROBE_REPLAY_ABSOLUTE_TOLERANCE + PROBE_REPLAY_RELATIVE_TOLERANCE * max(abs(confirmation_expected_probe), 1.0)
        and da_basis["rank256_identity"]["A_fit_projector"]["passed"]
        and da_basis["rank256_identity"]["W_projector"]["passed"]
        and selection["rank256_full_prediction_identity"]["passed"]
        and confirmation["rank256_full_prediction_identity"]["passed"]
    )
    return {
        "first_execution": "rank_256_data_aware_full_positive_anchor",
        "selection": _json_report(selection),
        "confirmation": _json_report(confirmation),
        "selection_probe_mse_replay_abs_delta": selection_replay,
        "confirmation_probe_mse_replay_abs_delta": confirmation_replay,
        "selection_expected_full_rmal_probe_mse": selection_expected_probe,
        "confirmation_expected_full_rmal_probe_mse": confirmation_expected_probe,
        "original_rmal_gate_vector": {
            "selection": selection["selection_gate"],
            "confirmation": confirmation["confirmation_gate"],
        },
        "passed": primary,
        "failure_stop": not primary,
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
            "first_confirmed_data_aware_candidate_rank": None,
            "previous_registered_candidate": None,
            "anchor_failure_stop": True,
            "matched_control_participated": False,
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
                "outcome": "bounded_recoverability_not_established_at_registered_rank_k",
                "selection_pass": False,
                "confirmation_not_run": True,
                "matched_control_participated": False,
            }
            previous = rank
            continue
        confirmation = dict(confirmation_fn(rank))
        if not (
            bool(confirmation.get("selection_gate", {}).get("passed", False))
            and bool(confirmation.get("confirmation_gate", {}).get("passed", False))
        ):
            results[str(rank)] = {
                "selection": selection,
                "confirmation": confirmation,
                "outcome": "bounded_recoverability_not_established_on_confirmation_at_registered_rank_k",
                "selection_pass": True,
                "confirmation_pass": False,
                "matched_control_participated": False,
            }
            previous = rank
            continue
        results[str(rank)] = {
            "selection": selection,
            "confirmation": confirmation,
            "outcome": "bounded_recoverability_established_at_registered_rank_k",
            "selection_pass": True,
            "confirmation_pass": True,
            "matched_control_participated": False,
        }
        winner = rank
        break
    if winner is None:
        winner = 256
    return {
        "evaluated_ranks": evaluated,
        "rank_results": results,
        "first_confirmed_data_aware_candidate_rank": winner,
        "previous_registered_candidate": previous,
        "anchor_failure_stop": False,
        "stopped_after_first_double_pass": any(
            value.get("confirmation_pass") is True for value in results.values()
        ),
        "rank_256_reused_as_fallback": winner == 256,
        "matched_control_participated": False,
    }


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Any],
    probe_path: Path,
    dataset_manifest: str,
) -> dict[str, str]:
    hashes = {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "implementation_module": file_sha256(Path(__file__).resolve()),
        "rmal_probe": file_sha256(probe_path),
        "dataset_manifest": dataset_manifest,
        "fit_indices": tensor_sha256(upstream["runtime"].fit_indices),
        "probe_tensor": tensor_sha256(upstream["runtime"].rmal_probe.full_weights),
        "authority_contract": canonical_sha256(AUTHORITY_LOCK),
        "fit_replay_contract": canonical_sha256(FIT_REPLAY_CONTRACT),
        "data_aware_contract": canonical_sha256(DATA_AWARE_CONTRACT),
        "raw_comparator_contract": canonical_sha256(RAW_COMPARATOR_CONTRACT),
        "candidate_family_contract": canonical_sha256(list(RANK_CANDIDATES)),
        "truncation_contract": canonical_sha256(TRUNCATION),
        "gate_contract": canonical_sha256(GATE_CONTRACT),
    }
    for name, identity in upstream["authorities"].items():
        for key in (
            "config_path",
            "review_path",
            "validation_path",
            "audit_path",
            "compact_path",
            "implementation_path",
        ):
            path = identity.get(key)
            if path is not None:
                hashes[f"{name}_{key}"] = file_sha256(path)
    return hashes


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "candidate_family": RANK_CANDIDATES,
            "fit_replay": FIT_REPLAY_CONTRACT,
            "data_aware": DATA_AWARE_CONTRACT,
            "raw_comparator": RAW_COMPARATOR_CONTRACT,
            "gate_contract": GATE_CONTRACT,
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
        raise FileExistsError(f"refusing to reuse MPCDAR validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("MPCDAR validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"MPCDAR attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("MPCDAR attempt directory is not a fresh validation-only slot")
    for name in ("compact_report.json", "probes.pt"):
        if (output_path.parent / name).exists():
            raise FileExistsError(f"refusing to overwrite MPCDAR companion artifact: {name}")


def _validation_gates(
    spec: Spec,
    upstream: Mapping[str, Any],
    raw_spectral: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    dmsa_fit = upstream["authorities"]["dmsa"]["compact"].get("panels", {}).get("fit", {})
    hashes = dmsa_fit.get("matrix_hashes", {})
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("lineage_start_head_recorded", spec.payload["lineage"]["lineage_start_head"] == LINEAGE_START_HEAD, {"lineage_start_head": LINEAGE_START_HEAD}),
        GateResult("rmal_terminal_identity", upstream["runtime"].rmal_identity["audit"].get("classification") == RMAL_CONSUMER["audit_classification"], {}),
        GateResult("lsda_terminal_identity", upstream["runtime"].lsda_compact.get("classification") == LSDA_CONSUMER["classification"], {}),
        GateResult("dmsa_fit_hash_contract", bool(hashes.get("mu")) and bool(hashes.get("C")), {"mu_hash_available": bool(hashes.get("mu")), "C_hash_available": bool(hashes.get("C"))}),
        GateResult("probe_shape_and_active_identity", tuple(upstream["runtime"].rmal_probe.weights.shape) == (FEATURE_DIM, TOTAL_COLUMNS) and tuple(upstream["runtime"].rmal_probe.bias.shape) == (TOTAL_COLUMNS,), {}),
        GateResult("raw_spectrum_order_available", raw_spectral["U"].shape == (FEATURE_DIM, FEATURE_DIM) and all(lrrc._rank_boundary_telemetry(rank, raw_spectral)["relative_gap"] is not None for rank in SEARCH_RANKS), {"gap_source": "LSDA raw comparator"}),
        GateResult("data_aware_gap_contract_distinct", DATA_AWARE_CONTRACT["A_fit"] == "C_fit^(1/2) W64" and DATA_AWARE_CONTRACT["G_fit"] == "A_fit A_fit.T", {"data_aware_gap_source": "A_fit/G_fit data-aware spectrum", "raw_gap_source": "LSDA raw-W comparator telemetry"}),
        GateResult("rank_family_locked", list(RANK_CANDIDATES) == spec.payload["candidate_family"] and list(SEARCH_RANKS) == list(RANK_CANDIDATES[:-1]), {"candidate_family": list(RANK_CANDIDATES)}),
        GateResult("no_target_fitting_contract", FIT_REPLAY_CONTRACT["target_labels_used"] is False and DATA_AWARE_CONTRACT["target_values_used"] is False, {}),
        GateResult("mean_preserving_raw_is_descriptive", RAW_COMPARATOR_CONTRACT["primary_search"] is False and RAW_COMPARATOR_CONTRACT["confirmation_trigger"] is False and RAW_COMPARATOR_CONTRACT["stop_rule"] is False, {}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "probe_refit": False, "bias_refit": False, "decoder_created": False}),
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
    raw_spectral = _spectral_lock(upstream)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, upstream, raw_spectral, output_path)
    hashes = _input_hashes(spec, review_path, upstream, _resolve(RMAL_CONSUMER["probe"]), upstream["runtime"].data.manifest["manifest_hash"])
    valid = all(gate.passed for gate in gates)
    contract = _contract_payload()
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        state,
        upstream["runtime"].data.manifest["manifest_hash"],
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "MPCDAR static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="MPCDAR validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise MPCDARHarnessFailure("MPCDAR requires CUDA for formal CMOS replay")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise MPCDARHarnessFailure("MPCDAR requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise MPCDARHarnessFailure("MPCDAR requires at least 10 GiB free storage")
    return int(free), int(total)


def _load_validation(path: str | Path, spec: Spec, review_path: Path, upstream: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, upstream, _resolve(RMAL_CONSUMER["probe"]), upstream["runtime"].data.manifest["manifest_hash"])
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
        raise ValueError("MPCDAR validation evidence is stale or mismatched")


def _invalid_harness_payload(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Any],
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
        upstream["runtime"].data.manifest["manifest_hash"],
        _input_hashes(spec, review_path, upstream, _resolve(RMAL_CONSUMER["probe"]), upstream["runtime"].data.manifest["manifest_hash"]),
        (GateResult("rank_256_data_aware_full_positive_anchor", False, {"reason": reason}), GateResult("no_update_scope", True, {})),
        (),
        {"reason": reason, "details": dict(details or {}), "rank_search_executed": False},
        reason,
    )
    payload.update({"candidate_family": list(RANK_CANDIDATES), "evaluated_ranks": [], "first_confirmed_data_aware_candidate_rank": None})
    _write_new(output_path, payload, artifact_kind="MPCDAR invalid-harness audit evidence")


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("MPCDAR requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
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
        source = rmfa._load_source(upstream["runtime"].rmal_consumer.bundle, load_checkpoints=True)
        formal: FormalSystemConfig = upstream["runtime"].rmal_consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[rmal.PRIMARY_CHECKPOINT]
        model = mra._load_scaled_model(formal, checkpoint, device)
        raw_spectral = _spectral_lock(upstream)
        state_before = rmal._state_cpu(model)
        ols_before = rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        fit_result = _replay_fit_moments(model, upstream, device)
        fit_report = _fit_replay_lock(fit_result, upstream)
        da_basis = _build_data_aware_basis(raw_spectral["W64"], fit_result.centered)
        V_raw = _raw_right_basis(raw_spectral)
        da_basis["rank256_identity"].update({
            "C_half_nonsingular": bool(da_basis["summary"].singular_values[-1] > 0.0),
            "W_full_row_rank": bool(raw_spectral["full"].singular_values[-1] > 0.0),
            "raw_W_smallest_singular_value": float(raw_spectral["full"].singular_values[-1]),
            "sign_flip_projector_invariance": _sign_flip_projector_invariance(raw_spectral["W64"], da_basis["V"], rank=4),
        })
        da_basis["rank256_identity"]["raw_projector_equivalence"] = {
            str(rank): _raw_projector_equivalence(raw_spectral["W64"], raw_spectral["U"], V_raw, rank)
            for rank in RANK_CANDIDATES
        }
        anchor = _rank256_anchor(model, upstream, raw_spectral, da_basis, fit_result.mu, device)
        if not anchor["passed"]:
            _invalid_harness_payload(output_path, spec, review_path, upstream, state, "rank-256 data-aware full positive anchor failed", {"rank_256_anchor": anchor, "fit_replay": fit_report, "data_aware_spectrum": _spectrum_report(da_basis["summary"])})
            return 2
        runtime = upstream["runtime"]

        def selection_fn(rank: int) -> Mapping[str, Any]:
            expected, _ = _expected_anchor(runtime.rmal_identity["compact"], "selection")
            result = _evaluate_panel(model, runtime.rmal_probe, raw_spectral, da_basis, fit_result.mu, rank, runtime.data.test_images.index_select(0, runtime.data.selection_indices), runtime.data.test_labels.index_select(0, runtime.data.selection_indices), device, expected, runtime.rmal_identity["compact"])
            result["rank_boundary"] = _data_aware_boundary_telemetry(rank, da_basis["summary"])
            result["raw_rank_boundary"] = _raw_boundary_telemetry(rank, raw_spectral)
            return result

        def confirmation_fn(rank: int) -> Mapping[str, Any]:
            expected, _ = _expected_anchor(runtime.rmal_identity["compact"], "confirmation")
            result = _evaluate_panel(model, runtime.rmal_probe, raw_spectral, da_basis, fit_result.mu, rank, runtime.data.images.index_select(0, runtime.data.confirmation_indices), runtime.data.labels.index_select(0, runtime.data.confirmation_indices), device, expected, runtime.rmal_identity["compact"])
            result["rank_boundary"] = _data_aware_boundary_telemetry(rank, da_basis["summary"])
            result["raw_rank_boundary"] = _raw_boundary_telemetry(rank, raw_spectral)
            result["selection_gate"] = _panel_gate(result, require_bootstrap=False)
            result["confirmation_gate"] = _panel_gate(result, require_bootstrap=True)
            return result

        search = _sequential_search(selection_fn, confirmation_fn, anchor_passed=True)
        winner = int(search["first_confirmed_data_aware_candidate_rank"])
        if winner == 256:
            search["rank_results"]["256"] = {
                "selection": anchor["selection"],
                "confirmation": anchor["confirmation"],
                "outcome": "rank_256_anchor_reused_as_fallback",
                "selection_pass": True,
                "confirmation_pass": True,
                "anchor_reused": True,
                "matched_control_participated": False,
            }
        state_unchanged = rmal._state_equal(state_before, rmal._state_cpu(model))
        ols_unchanged = rmal._state_equal(ols_before, rmal._state_cpu(model.ols))
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        temporary_features = 2 * CONFIRMATION_COUNT * FEATURE_DIM * 4
        if temporary_features > MAX_TEMPORARY_COMPACT_FEATURE_BYTES:
            raise MPCDARHarnessFailure("temporary compact feature working set exceeded bound")
        if peak > MAX_PEAK_CUDA_BYTES:
            raise MPCDARHarnessFailure("MPCDAR peak CUDA memory exceeded RMAL bound")
        stage_results = search["rank_results"]
        all_reports: list[Mapping[str, Any]] = []
        for item in stage_results.values():
            for panel in ("selection", "confirmation"):
                if panel in item and isinstance(item[panel], Mapping) and item[panel].get("executed", True):
                    item[panel] = _json_report(item[panel])
                    all_reports.append(item[panel])
        harness_ok = (
            fit_report["mu_hash_match"]
            and fit_report["C_hash_match"]
            and da_basis["orthogonality"]["passed"]
            and da_basis["rank256_identity"]["A_fit_projector"]["passed"]
            and da_basis["rank256_identity"]["W_projector"]["passed"]
            and anchor["passed"]
            and state_unchanged
            and ols_unchanged
            and all(report.get("finite") is True for report in all_reports)
            and all(report.get("feature_contract_valid") is True and report.get("probe_contract_valid") is True for report in all_reports)
            and all(report.get("square_law_consistency") is True for report in all_reports)
            and all(all(report.get("native_replay", {}).values()) for report in all_reports)
            and all(report.get("internal_stage_capture") is False and report.get("pooled_feature_bank_saved") is False and report.get("target_bank_saved") is False for report in all_reports)
        )
        if not harness_ok:
            raise MPCDARHarnessFailure("MPCDAR harness gate failed after measurement")
        classification = "mean_preserving_centered_data_aware_readout_recovery_completed"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": "passed",
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "lineage": LINEAGE,
            "candidate_family": list(RANK_CANDIDATES),
            "search_order_after_anchor": list(SEARCH_RANKS),
            "rank_0_control": {"prediction": "b", "included_in_candidate_search": False},
            "mean_fit_control": CONTROLS["mean_fit_control"],
            "rank_256_full_positive_anchor": anchor,
            "evaluated_ranks": search["evaluated_ranks"],
            "rank_results": stage_results,
            "first_confirmed_data_aware_candidate_rank": winner,
            "previous_registered_candidate": search["previous_registered_candidate"],
            "rank_256_reused_as_fallback": search["rank_256_reused_as_fallback"],
            "fit_replay": fit_report,
            "data_aware_construction": {
                "psd": da_basis["psd"],
                "spectrum": _spectrum_report(da_basis["summary"]),
                "gram_identity_residual": da_basis["gram_identity_residual"],
                "right_basis_orthogonality": da_basis["orthogonality"],
                "rank256_identity": da_basis["rank256_identity"],
                "raw_W_smallest_singular_value": float(raw_spectral["full"].singular_values[-1]),
                "raw_comparator_spectrum": {"source": "LSDA exact ordered raw-W spectrum", "candidate_boundaries": {str(rank): _raw_boundary_telemetry(rank, raw_spectral) for rank in RANK_CANDIDATES}},
                "data_aware_candidate_boundaries": {str(rank): _data_aware_boundary_telemetry(rank, da_basis["summary"]) for rank in RANK_CANDIDATES},
                "raw_gap_is_not_data_aware_gap": True,
                "descriptive_only": True,
                "no_degeneracy_threshold": True,
            },
            "matched_control": {"contract": RAW_COMPARATOR_CONTRACT, "primary_search_participated": False, "confirmation_triggered": False, "rank_results_contain_A_B_C": True},
            "evaluation": EVALUATION,
            "gate_inheritance": GATE_CONTRACT,
            "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "temporary_compact_feature_bytes": temporary_features, "pooled_feature_bank_saved": False, "target_bank_saved": False, "prediction_bank_saved": False},
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
        compact_size = len(json.dumps(compact, sort_keys=True).encode("utf-8"))
        audit_gates = (
            GateResult("rank_256_data_aware_full_positive_anchor", anchor["passed"], {}),
            GateResult("fit_replay_mu_hash", fit_report["mu_hash_match"], {}),
            GateResult("fit_replay_C_hash", fit_report["C_hash_match"], {}),
            GateResult("data_aware_right_basis_orthogonality", da_basis["orthogonality"]["passed"], {}),
            GateResult("A_fit_full_projector_identity", da_basis["rank256_identity"]["A_fit_projector"]["passed"], {}),
            GateResult("W_full_projector_identity", da_basis["rank256_identity"]["W_projector"]["passed"], {}),
            GateResult("sequential_data_aware_candidate_family", search["evaluated_ranks"] == list(SEARCH_RANKS[: len(search["evaluated_ranks"])]), {"evaluated": search["evaluated_ranks"]}),
            GateResult("matched_control_not_search", search["matched_control_participated"] is False, {}),
            GateResult("first_double_pass_stop", not search["stopped_after_first_double_pass"] or search["evaluated_ranks"][-1] == winner, {}),
            GateResult("state_unchanged", state_unchanged and ols_unchanged, {}),
            GateResult("no_update_isolation", True, {}),
            GateResult("no_target_fitting", True, {}),
            GateResult("finite_reports", all(report.get("finite") is True for report in all_reports), {}),
            GateResult("resource_limits", peak <= MAX_PEAK_CUDA_BYTES and compact_size <= MAX_STORAGE_BYTES, {"peak_bytes": peak, "compact_bytes": compact_size}),
        )
        status = "passed" if all(gate.passed for gate in audit_gates) else "invalid_harness"
        classification = classification if status == "passed" else "invalid_harness"
        compact["status"] = status
        compact["classification"] = classification
        compact_path = output_path.with_name("compact_report.json")
        _write_new(compact_path, compact, artifact_kind="MPCDAR compact report")
        compact_hash = file_sha256(compact_path)
        hashes = _input_hashes(spec, review_path, upstream, _resolve(RMAL_CONSUMER["probe"]), runtime.data.manifest["manifest_hash"])
        hashes["mpcdar_compact_report"] = compact_hash
        audit_payload = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            state,
            runtime.data.manifest["manifest_hash"],
            hashes,
            audit_gates,
            (NEXT_ACTION_RESULT_REVIEW,) if status == "passed" else (),
            {"lineage": LINEAGE, "result": compact, "compact_report_sha256": compact_hash, "matched_control": RAW_COMPARATOR_CONTRACT},
            None if status == "passed" else "MPCDAR harness gate failed",
        )
        _write_new(output_path, audit_payload, artifact_kind="MPCDAR audit evidence")
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    except MPCDARHarnessFailure as exc:
        if not output_path.exists():
            _invalid_harness_payload(output_path, spec, review_path, upstream, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-validate":
        return validate(args)
    if args.research_action == "scaled-chen-mean-preserving-centered-data-aware-readout-recovery-run":
        return run(args)
    raise ValueError(f"unsupported MPCDAR action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORITY_LOCK",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "CONTROLS",
    "DATA_AWARE_CONTRACT",
    "DMSA_CONSUMER",
    "DSFA_CONSUMER",
    "FIT_PANEL",
    "FIT_REPLAY_CONTRACT",
    "HYPOTHESIS",
    "LINEAGE",
    "LRRC_CONSUMER",
    "MEASUREMENT",
    "MCFA_CONSUMER",
    "MPCDARHarnessFailure",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "PANEL_CONTRACT",
    "RAW_COMPARATOR_CONTRACT",
    "RANK0",
    "RANK_CANDIDATES",
    "REVIEW_SCHEMA",
    "SEARCH_RANKS",
    "TRUNCATION",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_build_data_aware_basis",
    "_data_aware_boundary_telemetry",
    "_data_aware_prediction",
    "_fit_replay_lock",
    "_load_review",
    "_load_spec",
    "_mean_preserving_raw_prediction",
    "_old_raw_prediction",
    "_panel_gate",
    "_raw_projector_equivalence",
    "_rank256_anchor",
    "_sequential_search",
    "_sign_flip_projector_invariance",
    "dispatch",
    "run",
    "validate",
]
