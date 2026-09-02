"""Attribute frozen RMAL functional gain to panel mean and centered response.

This module is a descriptive, no-update consumer of the registered RMAL
detector readout.  It performs a deterministic two-pass CMOS-only replay for
each registered panel and keeps the panel population mean as an attribution
intermediate only.  No feature, target, prediction, or centered-feature bank
is persisted.
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
    optical_manifold_smf_bogt_detector_data_metric_spectral_attribution as dmsa,
)
from . import (
    optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc,
)


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_detector_mean_centered_functional_attribution_compact_report.v1"
)
HYPOTHESIS = "H-LOMA-CBOGT-S6-MCFA"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "3f1c01859a238986b263c0b040d21f96f531cb52"

TOTAL_COLUMNS = 40000
FEATURE_DIM = 256
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10

MOMENT_IDENTITY_ATOL = 1.0e-8
MOMENT_IDENTITY_RTOL = 1.0e-11
REPLAY_ATOL = lrrc.RANK256_OPERATOR_ABSOLUTE_TOLERANCE
REPLAY_RTOL = lrrc.RANK256_OPERATOR_RELATIVE_TOLERANCE
MIN_DISK_FREE_BYTES = lrrc.MIN_DISK_FREE_BYTES
MAX_STORAGE_BYTES = lrrc.MAX_STORAGE_BYTES
MIN_FREE_CUDA_BYTES = lrrc.MIN_FREE_CUDA_BYTES
MAX_PEAK_CUDA_BYTES = lrrc.MAX_PEAK_CUDA_BYTES

RMAL_CONSUMER = dict(dmsa.RMAL_CONSUMER)
LSDA_CONSUMER = dict(dmsa.LSDA_CONSUMER)
RWSA_IDENTITY = dict(dmsa.RWSA_IDENTITY)
RWSA_ACTIVE_IDENTITY = dict(dmsa.RWSA_ACTIVE_IDENTITY)
ZCSA_IDENTITY = dict(dmsa.ZCSA_IDENTITY)
LRRC_CONSUMER = dict(dmsa.LRRC_CONSUMER)
DSFA_CONSUMER = dict(dmsa.DSFA_CONSUMER)
DMSA_CONSUMER = {
    "config": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_v1.yaml"
    ),
    "config_sha256": "48e3e34d47a486fd50052f2d805d387c84fc216a9e9498775059c2de45560828",
    "review_contract": (
        "configs/research/"
        "lightgen_optical_manifold_smf_bogt_detector_data_metric_spectral_attribution_review_v1.json"
    ),
    "review_sha256": "2899715f961a19cdee83b53f709c9b77498ab618afe5963c6c63a5492ea2501a",
    "validation": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/validation.json"
    ),
    "validation_sha256": "f0eebcb978851b9ab58453e1b28e920eddc93e86a2b82133af0bd7ec8e8ce93d",
    "audit": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/audit.json"
    ),
    "audit_sha256": "c9cc448f1b58040d45b8b77c9cfaa9facd536cdcf40d0b4f152efe42e16411fe",
    "compact_report": (
        "outputs/20260902_v2/runs/"
        "lightgen_scaled_chen_detector_data_metric_spectral_attribution_v1_attempt_02/compact_report.json"
    ),
    "compact_report_sha256": "39564f0ce3973bd738d196191a06c7990407f9b7b6ab7e6e8a2eb3366dc2d0ac",
    "implementation_module": (
        "src/ldm_project/v2/application/research/"
        "optical_manifold_smf_bogt_detector_data_metric_spectral_attribution.py"
    ),
    "implementation_module_sha256": "88356728b165e7c1ff91fb825292a635901da2d02c60841da972b6434d36112f",
    "implementation_commit": "9818e87441eb2c86eef06c18c1aa646f3e57dea5",
    "status": "passed",
    "classification": "detector_data_metric_spectral_attribution_completed",
}

FIT_PANEL = dict(dmsa.FIT_PANEL)
PANEL_CONTRACT = {
    "fit": FIT_PANEL,
    "selection": dict(dmsa.PANEL_CONTRACT["selection"]),
    "confirmation": dict(dmsa.PANEL_CONTRACT["confirmation"]),
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
    "panel_mean_is_attribution_only": True,
    "deploy_time_mean_estimator": False,
}

LINEAGE = {
    "amendment": HYPOTHESIS,
    "parent_hypothesis": "H-LOMA-CBOGT-S6-DMSA",
    "lineage_start_head": LINEAGE_START_HEAD,
    "rmal_probe_frozen": True,
    "lsda_passed_required": True,
    "lrrc_passed_required": True,
    "dsfa_passed_required": True,
    "dmsa_passed_required": True,
    "zcsa_passed_required": True,
    "rwsa_active_passed_required": True,
    "fit_selection_confirmation_separate": True,
}

TWO_PASS_CONTRACT = {
    "pass_1": {
        "purpose": "exact CMOS-only no-update panel replay for sum_z",
        "feature_path": "output.camera -> mra._feature_projection(real, complex_value=False)",
        "accumulator": "CPU float64 sum_z",
        "mean": "mu=sum_z/N",
        "feature_bank_saved": False,
    },
    "pass_2": {
        "purpose": "same frozen replay using Pass-1 panel mu",
        "feature_bank_saved": False,
        "centered_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
        "mu_mutated": False,
    },
    "pass_order": ["pass_1_mean", "pass_2_functional"],
    "panel_passes": 2,
    "total_sample_visits": 14000,
    "microbatch_size": MICROBATCH_SIZE,
}

PREDICTION_CONTRACT = {
    "dynamic_mean": "mu W",
    "dynamic_centered": "(z-mu) W",
    "dynamic_full": "z W",
    "prediction_full": "b+z W",
    "identity": "dynamic_full=dynamic_mean+dynamic_centered",
    "analysis_dtype": "torch.float64",
    "source_feature_dtype": "torch.float32",
}

ENERGY_CONTRACT = {
    "E_mean": "||mu W||^2",
    "E_centered": "E[||(z-mu)W||^2]",
    "E_total": "E[||zW||^2]",
    "identity": "E_total=E_mean+E_centered",
    "mean_fraction": "E_mean/E_total when finite nonzero",
    "centered_fraction": "E_centered/E_total when finite nonzero",
    "no_epsilon": True,
    "no_renormalization": True,
}

FUNCTIONAL_CONTRACT = {
    "residual": "r=target-b",
    "mean_alignment_benefit": "2*<E[r],mu W>/40000",
    "mean_self_penalty": "||mu W||^2/40000",
    "mean_functional_gain": "mean_alignment_benefit-mean_self_penalty",
    "centered_alignment_benefit": "2*E[<r,(z-mu)W>]/40000",
    "centered_self_penalty": "E[||(z-mu)W||^2]/40000",
    "centered_functional_gain": "centered_alignment_benefit-centered_self_penalty",
    "total_functional_gain": "mean_functional_gain+centered_functional_gain",
    "signed_gain_shares": "raw gain / raw total gain when finite and nonzero",
    "signed_shares_not_probability": True,
    "no_abs_clamp_epsilon_or_renormalization": True,
}

MSE_CONTRACT = {
    "bias_mse": "MSE(b,target)",
    "bias_plus_mean_mse": "MSE(b+mu W,target)",
    "full_mse": "MSE(b+z W,target)",
    "mean_gain_identity": "mean_gain=bias_mse-bias_plus_mean_mse",
    "centered_gain_identity": "centered_gain=bias_plus_mean_mse-full_mse",
    "total_gain_identity": "total_gain=bias_mse-full_mse",
    "bias_plus_mean_attribution_only": True,
}

NUMERICAL_CONTRACT = {
    "source_dtype": "torch.float32",
    "analysis_dtype": "torch.float64",
    "moment_identity_atol": MOMENT_IDENTITY_ATOL,
    "moment_identity_rtol": MOMENT_IDENTITY_RTOL,
    "replay_atol": REPLAY_ATOL,
    "replay_rtol": REPLAY_RTOL,
    "population_denominator": "N",
    "no_N_minus_1": True,
    "no_epsilon": True,
    "no_scientific_threshold": True,
    "no_eigendecomposition": True,
    "no_psd_square_root": True,
    "no_covariance_spectrum": True,
}

MEASUREMENT = {
    "objective": "detector_mean_centered_functional_attribution",
    "primary_question": "panel_mean_response_muW_versus_centered_response_(z-mu)W_functional_MSE_gain",
    "full_output_columns": TOTAL_COLUMNS,
    "feature_dim": FEATURE_DIM,
    "panels": ["fit", "selection", "confirmation"],
    "fit_reference_only": True,
    "selection_confirmation_independent": True,
    "panel_mean_is_attribution_only": True,
    "deploy_time_mean_estimator": False,
    "no_training": True,
    "no_optimizer": True,
    "no_backward": True,
    "no_refit": True,
    "no_decoder": True,
    "no_whitening": True,
    "no_centering_deployment": True,
    "no_weighted_truncation": True,
    "no_rank_search": True,
    "no_generation": True,
    "no_physical_forward_modification": True,
    "descriptive_only": True,
}

RESOURCE_CONTRACT = {
    "microbatch_size": MICROBATCH_SIZE,
    "two_passes": True,
    "total_sample_visits": TWO_PASS_CONTRACT["total_sample_visits"],
    "w64_bytes_approx": 81920000,
    "feature_bank_saved": False,
    "centered_feature_bank_saved": False,
    "target_bank_saved": False,
    "prediction_bank_saved": False,
    "min_free_cuda_bytes_if_replay": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes_if_replay": MAX_PEAK_CUDA_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_detector_mean_centered_functional_attribution",
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
        "decoder",
        "whitening",
        "centering_deployment",
        "weighted_truncation",
        "rank_search",
        "generation",
        "promotion",
        "physical_forward_modification",
        "feature_bank_persistence",
        "centered_feature_bank_persistence",
        "target_bank_persistence",
        "prediction_bank_persistence",
        "scientific_classification",
        "mean_functionally_dominant_supported",
        "centered_functionally_dominant_supported",
        "mean_is_nuisance_supported",
        "centered_reconstruction_supported",
        "covariance_mismatch_supported",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_detector_mean_centered_functional_attribution"
NEXT_ACTION_RESULT_REVIEW = "manual_review_MCFA_result"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class Upstream:
    dmsa: dict[str, Any]
    lrrc: dict[str, Any]
    runtime: dmsa.Upstream


@dataclass(frozen=True)
class MeanResult:
    panel: str
    count: int
    mu: torch.Tensor


@dataclass
class MeanAccumulator:
    feature_dim: int = FEATURE_DIM
    count: int = 0
    sum_z: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.sum_z = torch.zeros(self.feature_dim, dtype=torch.float64)

    def update(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [N,feature_dim]")
        if features.shape[0] == 0:
            raise ValueError("features batch is empty")
        batch = features.detach().cpu().to(torch.float64).contiguous()
        if not bool(torch.isfinite(batch).all()):
            raise MCFAHarnessFailure("non-finite feature batch")
        self.sum_z += batch.sum(dim=0)
        self.count += int(batch.shape[0])

    def finalize(self, panel: str) -> MeanResult:
        if self.count <= 0 or self.sum_z is None:
            raise MCFAHarnessFailure(f"{panel} panel has no features")
        mu = self.sum_z / self.count
        if not bool(torch.isfinite(mu).all()):
            raise MCFAHarnessFailure(f"{panel} panel mean is non-finite")
        return MeanResult(panel, self.count, mu.contiguous())


@dataclass
class FunctionalAccumulator:
    feature_dim: int = FEATURE_DIM
    output_dim: int = TOTAL_COLUMNS
    count: int = 0
    sum_centered_feature: torch.Tensor | None = None
    sum_centered_dynamic: torch.Tensor | None = None
    centered_dynamic_energy_sum: float = 0.0
    full_dynamic_energy_sum: float = 0.0
    mean_residual_dot_sum: float = 0.0
    centered_residual_dot_sum: float = 0.0
    bias_error_sum: float = 0.0
    bias_plus_mean_error_sum: float = 0.0
    full_error_sum: float = 0.0
    decomposition_squared_sum: float = 0.0
    decomposition_abs_sum: float = 0.0
    decomposition_max_abs: float = 0.0
    finite: bool = True

    def __post_init__(self) -> None:
        self.sum_centered_feature = torch.zeros(self.feature_dim, dtype=torch.float64)
        self.sum_centered_dynamic = torch.zeros(self.output_dim, dtype=torch.float64)

    def update(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        W64: torch.Tensor,
        b64: torch.Tensor,
        mean_response: torch.Tensor,
    ) -> None:
        if features.ndim != 2 or tuple(features.shape[1:]) != (self.feature_dim,):
            raise ValueError("features must have shape [N,feature_dim]")
        if target.ndim != 2 or tuple(target.shape[1:]) != (self.output_dim,):
            raise ValueError("target must have shape [N,output_dim]")
        if features.shape[0] != target.shape[0] or features.shape[0] == 0:
            raise ValueError("feature and target batch counts are incompatible")
        batch = features.detach().cpu().to(torch.float64).contiguous()
        target64 = target.detach().cpu().to(torch.float64).contiguous()
        centered = batch - mu
        dynamic_centered = centered.mm(W64)
        dynamic_full = batch.mm(W64)
        decomposition = dynamic_full - mean_response - dynamic_centered
        residual = target64 - b64
        mean_residual = residual - mean_response
        full_residual = residual - dynamic_full
        self.sum_centered_feature += centered.sum(dim=0)
        self.sum_centered_dynamic += dynamic_centered.sum(dim=0)
        self.centered_dynamic_energy_sum += float(dynamic_centered.square().sum())
        self.full_dynamic_energy_sum += float(dynamic_full.square().sum())
        self.mean_residual_dot_sum += float((residual * mean_response).sum())
        self.centered_residual_dot_sum += float((residual * dynamic_centered).sum())
        self.bias_error_sum += float(residual.square().sum())
        self.bias_plus_mean_error_sum += float(mean_residual.square().sum())
        self.full_error_sum += float(full_residual.square().sum())
        self.decomposition_squared_sum += float(decomposition.square().sum())
        self.decomposition_abs_sum += float(decomposition.abs().sum())
        self.decomposition_max_abs = max(
            self.decomposition_max_abs, float(decomposition.abs().max())
        )
        self.count += int(batch.shape[0])
        self.finite = self.finite and bool(
            torch.isfinite(batch).all()
            and torch.isfinite(target64).all()
            and torch.isfinite(dynamic_centered).all()
            and torch.isfinite(dynamic_full).all()
            and torch.isfinite(decomposition).all()
        )

    def finalize(
        self,
        panel: str,
        mu: torch.Tensor,
        W64: torch.Tensor,
        mean_response: torch.Tensor,
    ) -> dict[str, Any]:
        if self.count <= 0 or self.sum_centered_feature is None or self.sum_centered_dynamic is None:
            raise MCFAHarnessFailure(f"{panel} panel has no functional samples")
        if not self.finite:
            raise MCFAHarnessFailure(f"{panel} functional replay is non-finite")
        mean_centered_feature = self.sum_centered_feature / self.count
        mean_centered_dynamic = self.sum_centered_dynamic / self.count
        E_mean = float(mean_response.square().sum())
        E_centered = self.centered_dynamic_energy_sum / self.count
        E_total = self.full_dynamic_energy_sum / self.count
        mean_cross = float(2.0 * mean_response.dot(mean_centered_dynamic))
        energy_residual = E_total - E_mean - E_centered
        bias_mse = self.bias_error_sum / self.count / self.output_dim
        bias_plus_mean_mse = (
            self.bias_plus_mean_error_sum / self.count / self.output_dim
        )
        full_mse = self.full_error_sum / self.count / self.output_dim
        mean_alignment = (
            2.0 * self.mean_residual_dot_sum / self.count / self.output_dim
        )
        mean_penalty = E_mean / self.output_dim
        centered_alignment = (
            2.0 * self.centered_residual_dot_sum / self.count / self.output_dim
        )
        centered_penalty = E_centered / self.output_dim
        mean_gain = mean_alignment - mean_penalty
        centered_gain = centered_alignment - centered_penalty
        total_gain = mean_gain + centered_gain
        mean_centered_feature_norm = float(torch.linalg.vector_norm(mean_centered_feature))
        mean_centered_dynamic_norm = float(torch.linalg.vector_norm(mean_centered_dynamic))
        decomposition_frobenius = math.sqrt(self.decomposition_squared_sum)
        return {
            "count": self.count,
            "mean_response": mean_response,
            "mean_centered_feature": mean_centered_feature,
            "mean_centered_dynamic": mean_centered_dynamic,
            "E_total": E_total,
            "E_mean": E_mean,
            "E_centered": E_centered,
            "mean_centered_output_cross": mean_cross,
            "energy_identity_residual": energy_residual,
            "dynamic_decomposition_max_abs_residual": self.decomposition_max_abs,
            "dynamic_decomposition_frobenius_residual": decomposition_frobenius,
            "dynamic_decomposition_mean_abs_residual": self.decomposition_abs_sum
            / self.count
            / self.output_dim,
            "bias_mse": bias_mse,
            "bias_plus_mean_mse": bias_plus_mean_mse,
            "full_mse": full_mse,
            "mean_alignment_benefit": mean_alignment,
            "mean_self_penalty": mean_penalty,
            "mean_functional_gain": mean_gain,
            "centered_alignment_benefit": centered_alignment,
            "centered_self_penalty": centered_penalty,
            "centered_functional_gain": centered_gain,
            "total_functional_gain": total_gain,
            "mean_gain_formula_residual": mean_gain - (mean_alignment - mean_penalty),
            "centered_gain_formula_residual": centered_gain
            - (centered_alignment - centered_penalty),
            "total_gain_decomposition_residual": total_gain - mean_gain - centered_gain,
            "centered_feature_mean_norm": mean_centered_feature_norm,
            "centered_dynamic_mean_norm": mean_centered_dynamic_norm,
            "dynamic_decomposition_passed": _close(
                decomposition_frobenius, 0.0,
                atol=REPLAY_ATOL,
                rtol=REPLAY_RTOL,
            ),
            "centered_feature_mean_passed": _close(
                mean_centered_feature_norm,
                0.0,
                atol=MOMENT_IDENTITY_ATOL,
                rtol=MOMENT_IDENTITY_RTOL,
            ),
            "centered_dynamic_mean_passed": _close(
                mean_centered_dynamic_norm,
                0.0,
                atol=MOMENT_IDENTITY_ATOL,
                rtol=MOMENT_IDENTITY_RTOL,
            ),
            "mean_centered_output_cross_passed": _close(
                abs(mean_cross),
                0.0,
                atol=MOMENT_IDENTITY_ATOL,
                rtol=MOMENT_IDENTITY_RTOL,
            ),
            "energy_identity_passed": _close(
                E_total,
                E_mean + E_centered,
                atol=MOMENT_IDENTITY_ATOL,
                rtol=MOMENT_IDENTITY_RTOL,
            ),
            "finite": self.finite
            and bool(
                torch.isfinite(mean_centered_feature).all()
                and torch.isfinite(mean_centered_dynamic).all()
                and torch.isfinite(mean_response).all()
                and torch.isfinite(W64).all()
                and torch.isfinite(mu).all()
            ),
        }


class MCFAHarnessFailure(RuntimeError):
    """A MCFA measurement or lineage boundary failed closed."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"MCFA requires clean branch {BRANCH}")


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
        "dmsa_consumer": DMSA_CONSUMER,
        "fit_panel": FIT_PANEL,
        "panel_contract": PANEL_CONTRACT,
        "two_pass": TWO_PASS_CONTRACT,
        "prediction": PREDICTION_CONTRACT,
        "energy": ENERGY_CONTRACT,
        "functional": FUNCTIONAL_CONTRACT,
        "mse": MSE_CONTRACT,
        "numerical": NUMERICAL_CONTRACT,
        "measurement": MEASUREMENT,
        "resource": RESOURCE_CONTRACT,
        "authorization": AUTHORIZATION,
    }


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
        "dmsa_consumer",
        "fit_panel",
        "panel_contract",
        "two_pass",
        "prediction",
        "energy",
        "functional",
        "mse",
        "numerical",
        "measurement",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("MCFA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("MCFA config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": (
            "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
        ),
        **_contract_payload(),
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"MCFA {name} contract is not locked")


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
        raise ValueError("MCFA review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise MCFAHarnessFailure(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MCFAHarnessFailure(f"{label} is not a mapping: {path}")
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
        raise MCFAHarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != expected_status
        or payload.get("classification") != expected_classification
        or record.status != expected_status
    ):
        raise MCFAHarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_lrrc(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(LRRC_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != LRRC_CONSUMER["config_sha256"]:
        raise MCFAHarnessFailure("LRRC config identity changed")
    lrrc_spec = lrrc._load_spec(config_path)
    lrrc._assert_contract(lrrc_spec)
    review_path = _resolve(LRRC_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != LRRC_CONSUMER["review_sha256"]:
        raise MCFAHarnessFailure("LRRC review identity changed")
    lrrc._load_review(review_path, lrrc_spec)
    implementation_path = _resolve(LRRC_CONSUMER["implementation_module"])
    if file_sha256(implementation_path) != LRRC_CONSUMER["implementation_module_sha256"]:
        raise MCFAHarnessFailure("LRRC implementation identity changed")
    validation_path = _resolve(LRRC_CONSUMER["validation"])
    validation = _read_json_hash(validation_path, LRRC_CONSUMER["validation_sha256"], "LRRC validation")
    validation_record = _verify_evidence(
        validation, expected_status="passed", expected_classification=None, label="LRRC validation"
    )
    if validation_record.git_commit != LRRC_CONSUMER["implementation_commit"] or validation_record.config_hash != lrrc_spec.config_hash:
        raise MCFAHarnessFailure("LRRC validation lineage changed")
    audit_path = _resolve(LRRC_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, LRRC_CONSUMER["audit_sha256"], "LRRC audit")
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=LRRC_CONSUMER["classification"],
        label="LRRC audit",
    )
    if audit_record.git_commit != LRRC_CONSUMER["implementation_commit"]:
        raise MCFAHarnessFailure("LRRC audit lineage changed")
    compact_path = _resolve(LRRC_CONSUMER["compact_report"])
    compact = _read_json_hash(
        compact_path, LRRC_CONSUMER["compact_report_sha256"], "LRRC compact report"
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != LRRC_CONSUMER["classification"]
        or compact.get("rank_candidates") != list(lrrc.RANK_CANDIDATES)
        or compact.get("first_confirmed_candidate_rank") != 256
    ):
        raise MCFAHarnessFailure("LRRC compact identity changed")
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


def _verify_dmsa(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(DMSA_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(config_path) != DMSA_CONSUMER["config_sha256"]:
        raise MCFAHarnessFailure("DMSA config identity changed")
    dmsa_spec = dmsa._load_spec(config_path)
    dmsa._assert_contract(dmsa_spec)
    review_path = _resolve(DMSA_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != DMSA_CONSUMER["review_sha256"]:
        raise MCFAHarnessFailure("DMSA review identity changed")
    dmsa._load_review(review_path, dmsa_spec)
    implementation_path = _resolve(DMSA_CONSUMER["implementation_module"])
    if file_sha256(implementation_path) != DMSA_CONSUMER["implementation_module_sha256"]:
        raise MCFAHarnessFailure("DMSA implementation identity changed")
    validation_path = _resolve(DMSA_CONSUMER["validation"])
    validation = _read_json_hash(
        validation_path, DMSA_CONSUMER["validation_sha256"], "DMSA validation"
    )
    validation_record = _verify_evidence(
        validation, expected_status="passed", expected_classification=None, label="DMSA validation"
    )
    if validation_record.git_commit != DMSA_CONSUMER["implementation_commit"] or validation_record.config_hash != dmsa_spec.config_hash:
        raise MCFAHarnessFailure("DMSA validation lineage changed")
    audit_path = _resolve(DMSA_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, DMSA_CONSUMER["audit_sha256"], "DMSA audit")
    audit_record = _verify_evidence(
        audit,
        expected_status="passed",
        expected_classification=DMSA_CONSUMER["classification"],
        label="DMSA audit",
    )
    if audit_record.git_commit != DMSA_CONSUMER["implementation_commit"]:
        raise MCFAHarnessFailure("DMSA audit lineage changed")
    compact_path = _resolve(DMSA_CONSUMER["compact_report"])
    compact = _read_json_hash(
        compact_path, DMSA_CONSUMER["compact_report_sha256"], "DMSA compact report"
    )
    panels = compact.get("panels")
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != DMSA_CONSUMER["classification"]
        or compact.get("probe", {}).get("sha256") != RMAL_CONSUMER["probe_sha256"]
        or compact.get("probe", {}).get("W_shape") != [FEATURE_DIM, TOTAL_COLUMNS]
        or compact.get("probe", {}).get("bias_shape") != [TOTAL_COLUMNS]
        or not isinstance(panels, Mapping)
    ):
        raise MCFAHarnessFailure("DMSA compact terminal identity changed")
    for panel in ("fit", "selection", "confirmation"):
        row = panels.get(panel)
        energy = row.get("energy_decomposition") if isinstance(row, Mapping) else None
        if not isinstance(energy, Mapping) or energy.get("identity_passed") is not True:
            raise MCFAHarnessFailure(f"DMSA {panel} energy identity changed")
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
    dmsa_identity = _verify_dmsa(spec)
    lrrc_identity = _verify_lrrc(spec)
    try:
        runtime = dmsa._load_upstream(spec, dataset_root)
    except (ValueError, dmsa.DMSAHarnessFailure, dsfa.DSFAHarnessFailure, lrrc.LRRCHarnessFailure) as exc:
        raise MCFAHarnessFailure("RMAL/LSDA/DSFA/DMSA upstream identity is invalid") from exc
    probe = runtime.runtime.runtime.rmal_probe
    if (
        probe.artifact_sha256 != RMAL_CONSUMER["probe_sha256"]
        or tuple(probe.weights.shape) != (FEATURE_DIM, TOTAL_COLUMNS)
        or tuple(probe.bias.shape) != (TOTAL_COLUMNS,)
        or probe.weights.dtype != torch.float32
        or probe.bias.dtype != torch.float32
    ):
        raise MCFAHarnessFailure("frozen RMAL probe identity changed")
    return Upstream(dmsa_identity, lrrc_identity, runtime)


def _verify_panel_identity(upstream: Upstream) -> dict[str, Any]:
    try:
        identity = dmsa._verify_panel_identity(upstream.runtime)
    except (ValueError, dmsa.DMSAHarnessFailure) as exc:
        raise MCFAHarnessFailure("fit/selection/confirmation panel identity changed") from exc
    return identity


def _panel_images(upstream: Upstream, panel: str) -> torch.Tensor:
    return dmsa._panel_images(upstream.runtime, panel)


def _rmal_runtime(upstream: Upstream) -> Any:
    return upstream.runtime.runtime.runtime


def _feature_batch(
    model: torch.nn.Module, batch_images: torch.Tensor, device: torch.device, panel: str
) -> torch.Tensor:
    camera = lrrc._capture_cmos(model, batch_images, device)
    features = lrrc.mra._feature_projection(camera, complex_value=False).float()
    if tuple(features.shape) != (batch_images.shape[0], FEATURE_DIM):
        raise MCFAHarnessFailure(f"{panel} feature projection shape changed")
    if not bool(torch.isfinite(features).all()):
        raise MCFAHarnessFailure(f"{panel} feature projection is non-finite")
    return features


def _replay_pass1(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    panel: str,
) -> MeanResult:
    expected_count = {"fit": FIT_COUNT, "selection": SELECTION_COUNT, "confirmation": CONFIRMATION_COUNT}[panel]
    accumulator = MeanAccumulator()
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        accumulator.update(_feature_batch(model, batch_images, device, panel))
    result = accumulator.finalize(panel)
    if result.count != expected_count:
        raise MCFAHarnessFailure(f"{panel} Pass-1 count changed")
    return result


def _replay_pass2(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    panel: str,
    mu: torch.Tensor,
    W64: torch.Tensor,
    b64: torch.Tensor,
) -> dict[str, Any]:
    expected_count = {"fit": FIT_COUNT, "selection": SELECTION_COUNT, "confirmation": CONFIRMATION_COUNT}[panel]
    if mu.dtype != torch.float64 or mu.device.type != "cpu" or tuple(mu.shape) != (FEATURE_DIM,):
        raise ValueError("Pass-1 mu must be CPU float64 with shape [256]")
    mean_response = mu.unsqueeze(0).mm(W64).reshape(TOTAL_COLUMNS).contiguous()
    accumulator = FunctionalAccumulator()
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        target_flat = target.reshape(indices.numel(), TOTAL_COLUMNS)
        features = _feature_batch(model, batch_images, device, panel)
        accumulator.update(features, target_flat, mu, W64, b64, mean_response)
    if accumulator.count != expected_count:
        raise MCFAHarnessFailure(f"{panel} Pass-2 count changed")
    report = accumulator.finalize(panel, mu, W64, mean_response)
    report["mean_response"] = mean_response
    report["pass_1_mean_frozen"] = True
    return report


def _close(actual: float, expected: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= atol + rtol * max(abs(expected), 1.0)


def _fraction(value: float, denominator: float) -> float | None:
    if not math.isfinite(value) or not math.isfinite(denominator) or denominator == 0.0:
        return None
    return value / denominator


def _identity_result(actual: float, expected: float, *, atol: float, rtol: float) -> dict[str, Any]:
    residual = actual - expected
    return {
        "actual": actual,
        "expected": expected,
        "absolute_residual": abs(residual),
        "relative_residual": abs(residual) / max(abs(expected), 1.0e-30),
        "passed": _close(actual, expected, atol=atol, rtol=rtol),
    }


def _gain_shares(mean_gain: float, centered_gain: float, total_gain: float) -> dict[str, Any]:
    if not math.isfinite(total_gain) or total_gain == 0.0:
        return {
            "mean_gain_share": "not_applicable",
            "centered_gain_share": "not_applicable",
            "share_sum": "not_applicable",
            "share_identity_passed": True,
        }
    mean_share = mean_gain / total_gain
    centered_share = centered_gain / total_gain
    share_sum = mean_share + centered_share
    return {
        "mean_gain_share": mean_share,
        "centered_gain_share": centered_share,
        "share_sum": share_sum,
        "share_identity_residual": share_sum - 1.0,
        "share_identity_passed": _close(
            share_sum, 1.0, atol=MOMENT_IDENTITY_ATOL, rtol=MOMENT_IDENTITY_RTOL
        ),
    }


def _energy_report(
    raw: Mapping[str, Any], dmsa_energy: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    E_total = float(raw["E_total"])
    E_mean = float(raw["E_mean"])
    E_centered = float(raw["E_centered"])
    residual = E_total - E_mean - E_centered
    energy = {
        "E_total": E_total,
        "E_mean": E_mean,
        "E_centered": E_centered,
        "mean_fraction": _fraction(E_mean, E_total),
        "centered_fraction": _fraction(E_centered, E_total),
        "identity_residual": residual,
        "identity_passed": _close(
            E_total, E_mean + E_centered,
            atol=MOMENT_IDENTITY_ATOL,
            rtol=MOMENT_IDENTITY_RTOL,
        ),
    }
    expected_fields = {
        "E_total": "E_total",
        "E_mean": "E_mean",
        "E_centered": "E_var",
        "mean_fraction": "mean_fraction",
        "centered_fraction": "variance_fraction",
    }
    replay = {
        name: _identity_result(
            float(energy[name]),
            float(dmsa_energy[expected_name]),
            atol=MOMENT_IDENTITY_ATOL,
            rtol=MOMENT_IDENTITY_RTOL,
        )
        for name, expected_name in expected_fields.items()
    }
    replay["passed"] = all(row["passed"] for row in replay.values())
    return energy, replay


def _mse_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    bias = float(raw["bias_mse"])
    bias_plus_mean = float(raw["bias_plus_mean_mse"])
    full = float(raw["full_mse"])
    mean_gain = float(raw["mean_functional_gain"])
    centered_gain = float(raw["centered_functional_gain"])
    total_gain = float(raw["total_functional_gain"])
    return {
        "bias_mse": bias,
        "bias_plus_mean_mse": bias_plus_mean,
        "full_probe_mse": full,
        "mean_gain_from_mse": bias - bias_plus_mean,
        "centered_gain_from_mse": bias_plus_mean - full,
        "total_gain_from_mse": bias - full,
        "mean_gain_identity": _identity_result(
            mean_gain, bias - bias_plus_mean, atol=REPLAY_ATOL, rtol=REPLAY_RTOL
        ),
        "centered_gain_identity": _identity_result(
            centered_gain, bias_plus_mean - full, atol=REPLAY_ATOL, rtol=REPLAY_RTOL
        ),
        "total_gain_identity": _identity_result(
            total_gain, bias - full, atol=REPLAY_ATOL, rtol=REPLAY_RTOL
        ),
        "bias_plus_mean_attribution_only": True,
    }


def _dmsa_replay(
    energy: Mapping[str, Any], dmsa_energy: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "E_total": "E_total",
        "E_mean": "E_mean",
        "E_centered": "E_var",
        "mean_fraction": "mean_fraction",
        "centered_fraction": "variance_fraction",
    }
    rows = {
        name: _identity_result(
            float(energy[name]),
            float(dmsa_energy[key]),
            atol=MOMENT_IDENTITY_ATOL,
            rtol=MOMENT_IDENTITY_RTOL,
        )
        for name, key in expected.items()
    }
    return {"reference": "DMSA attempt-02", "fields": rows, "passed": all(row["passed"] for row in rows.values())}


def _dsfa_replay(
    raw: Mapping[str, Any], dsfa_panel: Mapping[str, Any] | None
) -> dict[str, Any]:
    if dsfa_panel is None:
        return {"reference": "not_evaluated_fit_panel_no_DSFA_reference", "passed": True}
    expected_gain = float(dsfa_panel["total_functional_gain"])
    expected_mse = float(dsfa_panel["full_probe_mse"])
    gain = _identity_result(
        float(raw["total_functional_gain"]), expected_gain, atol=REPLAY_ATOL, rtol=REPLAY_RTOL
    )
    mse = _identity_result(
        float(raw["full_mse"]), expected_mse, atol=REPLAY_ATOL, rtol=REPLAY_RTOL
    )
    return {
        "reference": "DSFA attempt-01",
        "total_gain": gain,
        "full_probe_mse": mse,
        "passed": gain["passed"] and mse["passed"],
    }


def _lrrc_replay(
    raw: Mapping[str, Any], lrrc_compact: Mapping[str, Any], panel: str
) -> dict[str, Any]:
    if panel == "selection":
        row = lrrc_compact.get("rank_results", {}).get("256", {})
        reference = row.get("selection", {}).get("probe", {}).get("paired_mse")
    elif panel == "confirmation":
        reference = (
            lrrc_compact.get("rank_256_full_positive_anchor", {})
            .get("confirmation", {})
            .get("probe", {})
            .get("paired_mse")
        )
    else:
        return {"reference": "not_evaluated_fit_panel_no_LRRC_reference", "passed": True}
    if not isinstance(reference, (int, float)):
        raise MCFAHarnessFailure(f"LRRC {panel} full-rank scalar is unavailable")
    row = _identity_result(
        float(raw["full_mse"]), float(reference), atol=REPLAY_ATOL, rtol=REPLAY_RTOL
    )
    return {"reference": "LRRC full-rank scalar", "full_probe_mse": row, "passed": row["passed"]}


def _panel_report(
    panel: str,
    role: str,
    mean_result: MeanResult,
    raw: Mapping[str, Any],
    dmsa_panel: Mapping[str, Any],
    dsfa_panel: Mapping[str, Any] | None,
    lrrc_compact: Mapping[str, Any],
) -> dict[str, Any]:
    dmsa_energy = dmsa_panel["energy_decomposition"]
    energy, energy_replay = _energy_report(raw, dmsa_energy)
    dmsa_replay = _dmsa_replay(energy, dmsa_energy)
    functional = {
        "mean_alignment_benefit": float(raw["mean_alignment_benefit"]),
        "mean_self_penalty": float(raw["mean_self_penalty"]),
        "mean_functional_gain": float(raw["mean_functional_gain"]),
        "centered_alignment_benefit": float(raw["centered_alignment_benefit"]),
        "centered_self_penalty": float(raw["centered_self_penalty"]),
        "centered_functional_gain": float(raw["centered_functional_gain"]),
        "total_functional_gain": float(raw["total_functional_gain"]),
    }
    functional.update(
        _gain_shares(
            functional["mean_functional_gain"],
            functional["centered_functional_gain"],
            functional["total_functional_gain"],
        )
    )
    mse = _mse_report(raw)
    dsfa_replay = _dsfa_replay(raw, dsfa_panel)
    lrrc_replay = _lrrc_replay(raw, lrrc_compact, panel)
    mean_feature = raw["mean_centered_feature"]
    mean_dynamic = raw["mean_centered_dynamic"]
    dynamic_cross = float(raw["mean_centered_output_cross"])
    harness = {
        "dynamic_decomposition_max_abs_residual": float(raw["dynamic_decomposition_max_abs_residual"]),
        "dynamic_decomposition_frobenius_residual": float(raw["dynamic_decomposition_frobenius_residual"]),
        "dynamic_decomposition_mean_abs_residual": float(raw["dynamic_decomposition_mean_abs_residual"]),
        "centered_feature_mean_max_abs_residual": float(mean_feature.abs().max()),
        "centered_feature_mean_l2_residual": float(torch.linalg.vector_norm(mean_feature)),
        "centered_dynamic_mean_max_abs_residual": float(mean_dynamic.abs().max()),
        "centered_dynamic_mean_l2_residual": float(torch.linalg.vector_norm(mean_dynamic)),
        "centered_feature_mean_passed": bool(raw["centered_feature_mean_passed"]),
        "centered_dynamic_mean_passed": bool(raw["centered_dynamic_mean_passed"]),
        "mean_centered_output_cross": dynamic_cross,
        "mean_centered_output_cross_abs_residual": abs(dynamic_cross),
        "mean_centered_output_cross_passed": bool(raw["mean_centered_output_cross_passed"]),
        "energy_identity_residual": float(raw["energy_identity_residual"]),
        "energy_identity_passed": bool(raw["energy_identity_passed"]),
        "dynamic_decomposition_passed": bool(raw["dynamic_decomposition_passed"]),
        "mean_gain_formula_residual": float(raw["mean_gain_formula_residual"]),
        "centered_gain_formula_residual": float(raw["centered_gain_formula_residual"]),
        "total_gain_decomposition_residual": float(raw["total_gain_decomposition_residual"]),
        "finite": bool(raw["finite"]),
        "pass_1_mean_frozen": bool(raw["pass_1_mean_frozen"]),
    }
    panel_report = {
        "panel": panel,
        "role": role,
        "evaluation_count": mean_result.count,
        "feature_dim": FEATURE_DIM,
        "mean": {
            "mu_sha256": tensor_sha256(mean_result.mu),
            "population_denominator": "N",
            "values": mean_result.mu.tolist(),
            "attribution_only": True,
            "deploy_time_mean_estimator": False,
        },
        "output_energy": energy,
        "energy_decomposition": energy,
        "functional_gain": functional,
        "mse": mse,
        "harness": harness,
        "dmsa_energy_replay": dmsa_replay,
        "dsfa_cross_evidence_identity": dsfa_replay,
        "lrrc_scalar_replay": lrrc_replay,
        "streaming": True,
        "two_pass": True,
        "pass_1_mean_frozen": True,
        "pooled_feature_bank_saved": False,
        "centered_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
        "raw_cmos_bank_saved": False,
        "internal_stage_capture": False,
        "cmos_only_replay_executed": True,
        "optical_replay_executed": False,
        "finite": bool(raw["finite"]),
    }
    panel_report["harness_passed"] = bool(
        panel_report["finite"]
        and harness["pass_1_mean_frozen"]
        and harness["dynamic_decomposition_passed"]
        and harness["centered_feature_mean_passed"]
        and harness["centered_dynamic_mean_passed"]
        and harness["mean_centered_output_cross_passed"]
        and harness["energy_identity_passed"]
        and energy["identity_passed"]
        and energy_replay["passed"]
        and dmsa_replay["passed"]
        and mse["mean_gain_identity"]["passed"]
        and mse["centered_gain_identity"]["passed"]
        and mse["total_gain_identity"]["passed"]
        and dsfa_replay["passed"]
        and lrrc_replay["passed"]
        and functional["share_identity_passed"]
    )
    return panel_report


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
) -> dict[str, str]:
    runtime = _rmal_runtime(upstream)
    dsfa_identity = upstream.runtime.dsfa
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
        "lrrc_config": file_sha256(upstream.lrrc["config_path"]),
        "lrrc_review_contract": file_sha256(upstream.lrrc["review_path"]),
        "lrrc_validation": file_sha256(upstream.lrrc["validation_path"]),
        "lrrc_audit": file_sha256(upstream.lrrc["audit_path"]),
        "lrrc_compact_report": file_sha256(upstream.lrrc["compact_path"]),
        "lrrc_implementation_module": file_sha256(upstream.lrrc["implementation_path"]),
        "dsfa_config": file_sha256(dsfa_identity["config_path"]),
        "dsfa_review_contract": file_sha256(dsfa_identity["review_path"]),
        "dsfa_validation": file_sha256(dsfa_identity["validation_path"]),
        "dsfa_audit": file_sha256(dsfa_identity["audit_path"]),
        "dsfa_compact_report": file_sha256(dsfa_identity["compact_path"]),
        "dmsa_config": file_sha256(upstream.dmsa["config_path"]),
        "dmsa_review_contract": file_sha256(upstream.dmsa["review_path"]),
        "dmsa_validation": file_sha256(upstream.dmsa["validation_path"]),
        "dmsa_audit": file_sha256(upstream.dmsa["audit_path"]),
        "dmsa_compact_report": file_sha256(upstream.dmsa["compact_path"]),
        "dataset_manifest": runtime.data.manifest["manifest_hash"],
        "fit_indices": tensor_sha256(runtime.fit_indices),
        "rmal_consumer_contract": canonical_sha256(RMAL_CONSUMER),
        "lsda_consumer_contract": canonical_sha256(LSDA_CONSUMER),
        "lrrc_consumer_contract": canonical_sha256(LRRC_CONSUMER),
        "dsfa_consumer_contract": canonical_sha256(DSFA_CONSUMER),
        "dmsa_consumer_contract": canonical_sha256(DMSA_CONSUMER),
        "panel_contract": canonical_sha256(PANEL_CONTRACT),
        "two_pass_contract": canonical_sha256(TWO_PASS_CONTRACT),
        "prediction_contract": canonical_sha256(PREDICTION_CONTRACT),
        "energy_contract": canonical_sha256(ENERGY_CONTRACT),
        "functional_contract": canonical_sha256(FUNCTIONAL_CONTRACT),
        "mse_contract": canonical_sha256(MSE_CONTRACT),
        "numerical_contract": canonical_sha256(NUMERICAL_CONTRACT),
    }
    if panel_identity.get("fit", {}).get("indices_sha256") != FIT_PANEL["indices_sha256"]:
        raise MCFAHarnessFailure("fit panel identity changed during hash binding")
    return hashes


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "lineage": LINEAGE,
            "two_pass": TWO_PASS_CONTRACT,
            "prediction": PREDICTION_CONTRACT,
            "energy": ENERGY_CONTRACT,
            "functional": FUNCTIONAL_CONTRACT,
            "mse": MSE_CONTRACT,
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
        "closed_task_relevant_bottleneck_not_supported": True,
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
        "descriptive_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse MCFA validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("MCFA validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"MCFA validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name} or output_path.exists():
        raise FileExistsError("MCFA attempt directory is not a fresh validation-only slot")


def _validation_gates(
    state: Mapping[str, Any],
    upstream: Upstream,
    panel_identity: Mapping[str, Any],
    output_path: Path,
) -> tuple[GateResult, ...]:
    probe = _rmal_runtime(upstream).rmal_probe
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
        GateResult(
            "fit_selection_confirmation_identity",
            panel_identity["fit"]["indices_sha256"] == FIT_PANEL["indices_sha256"]
            and panel_identity["selection_confirmation_disjoint"]
            and panel_identity["fit_confirmation_disjoint"],
            {},
        ),
        GateResult("dmsa_terminal_identity", upstream.dmsa["compact"].get("classification") == DMSA_CONSUMER["classification"], {}),
        GateResult("panel_mean_attribution_boundary", PANEL_CONTRACT["panel_mean_is_attribution_only"] and not PANEL_CONTRACT["deploy_time_mean_estimator"], {}),
        GateResult("two_pass_contract", TWO_PASS_CONTRACT["pass_order"] == ["pass_1_mean", "pass_2_functional"] and TWO_PASS_CONTRACT["total_sample_visits"] == 14000, {}),
        GateResult("no_covariance_spectral_path", NUMERICAL_CONTRACT["no_eigendecomposition"] and NUMERICAL_CONTRACT["no_psd_square_root"] and NUMERICAL_CONTRACT["no_covariance_spectrum"], {}),
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
    output_path = _resolve(args.output)
    gates = _validation_gates(state, upstream, panel_identity, output_path)
    valid = all(gate.passed for gate in gates)
    hashes = _input_hashes(spec, review_path, upstream, panel_identity)
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        state,
        _rmal_runtime(upstream).data.manifest["manifest_hash"],
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        _contract_payload(),
        None if valid else "MCFA static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="MCFA validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise MCFAHarnessFailure("MCFA CMOS replay requires CUDA")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise MCFAHarnessFailure("MCFA requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise MCFAHarnessFailure("MCFA requires at least 10 GiB free storage")
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
        raise ValueError("MCFA validation evidence is stale or mismatched")


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
        _rmal_runtime(upstream).data.manifest["manifest_hash"],
        hashes,
        (GateResult("mcfa_harness", False, {"reason": reason}), GateResult("no_update_scope", True, {})),
        (),
        {"lineage": LINEAGE, "reason": reason},
        reason,
    )
    _write_new(output_path, payload, artifact_kind="MCFA invalid-harness audit evidence")


def _runtime_metadata() -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "analysis_dtype": "torch.float64",
        "source_dtype": "torch.float32",
        "microbatch_size": MICROBATCH_SIZE,
        "two_pass_replay": True,
        "total_sample_visits": TWO_PASS_CONTRACT["total_sample_visits"],
        "feature_bank_saved": False,
        "centered_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
        "eigendecomposition_used": False,
        "psd_square_root_used": False,
    }


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("MCFA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
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
        source = lrrc.rmfa._load_source(
            _rmal_runtime(upstream).rmal_consumer.bundle, load_checkpoints=True
        )
        formal = _rmal_runtime(upstream).rmal_consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[lrrc.rmal.PRIMARY_CHECKPOINT]
        model = lrrc.mra._load_scaled_model(formal, checkpoint, device)
        model.eval()
        probe = _rmal_runtime(upstream).rmal_probe
        W64 = probe.weights.detach().cpu().to(torch.float64).contiguous()
        b64 = probe.bias.detach().cpu().to(torch.float64).contiguous()
        if tuple(W64.shape) != (FEATURE_DIM, TOTAL_COLUMNS) or tuple(b64.shape) != (TOTAL_COLUMNS,):
            raise MCFAHarnessFailure("frozen RMAL operator shape changed at replay")
        state_before = lrrc.rmal._state_cpu(model)
        ols_before = lrrc.rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        reports: dict[str, dict[str, Any]] = {}
        dmsa_panels = upstream.dmsa["compact"]["panels"]
        dsfa_panels = upstream.runtime.dsfa["compact"]
        lrrc_compact = upstream.lrrc["compact"]
        for panel in ("fit", "selection", "confirmation"):
            images = _panel_images(upstream, panel)
            mean_result = _replay_pass1(model, images, device, panel)
            mu_before = mean_result.mu.clone()
            raw = _replay_pass2(model, images, device, panel, mean_result.mu, W64, b64)
            if not torch.equal(mu_before, mean_result.mu):
                raise MCFAHarnessFailure(f"{panel} Pass-1 mu was mutated during Pass-2")
            dsfa_panel = dsfa_panels.get(panel) if panel in ("selection", "confirmation") else None
            reports[panel] = _panel_report(
                panel,
                PANEL_CONTRACT[panel]["role"],
                mean_result,
                raw,
                dmsa_panels[panel],
                dsfa_panel,
                lrrc_compact,
            )
        state_unchanged = lrrc.rmal._state_equal(state_before, lrrc.rmal._state_cpu(model))
        ols_unchanged = lrrc.rmal._state_equal(ols_before, lrrc.rmal._state_cpu(model.ols))
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        audit_gates_without_resource = (
            GateResult("panel_replays", all(report["harness_passed"] for report in reports.values()), {}),
            GateResult("fit_output_energy_replay", reports["fit"]["dmsa_energy_replay"]["passed"], reports["fit"]["dmsa_energy_replay"]),
            GateResult("selection_output_energy_replay", reports["selection"]["dmsa_energy_replay"]["passed"], reports["selection"]["dmsa_energy_replay"]),
            GateResult("confirmation_output_energy_replay", reports["confirmation"]["dmsa_energy_replay"]["passed"], reports["confirmation"]["dmsa_energy_replay"]),
            GateResult("selection_dsfa_cross_identity", reports["selection"]["dsfa_cross_evidence_identity"]["passed"], reports["selection"]["dsfa_cross_evidence_identity"]),
            GateResult("confirmation_dsfa_cross_identity", reports["confirmation"]["dsfa_cross_evidence_identity"]["passed"], reports["confirmation"]["dsfa_cross_evidence_identity"]),
            GateResult("selection_lrrc_scalar_identity", reports["selection"]["lrrc_scalar_replay"]["passed"], reports["selection"]["lrrc_scalar_replay"]),
            GateResult("confirmation_lrrc_scalar_identity", reports["confirmation"]["lrrc_scalar_replay"]["passed"], reports["confirmation"]["lrrc_scalar_replay"]),
            GateResult("selection_confirmation_isolated", panel_identity["selection_confirmation_disjoint"] and panel_identity["fit_confirmation_disjoint"], {}),
            GateResult("state_unchanged", state_unchanged and ols_unchanged, {}),
            GateResult("streaming_no_banks", all(not report["pooled_feature_bank_saved"] and not report["centered_feature_bank_saved"] and not report["target_bank_saved"] and not report["prediction_bank_saved"] for report in reports.values()), {}),
            GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "probe_refit": False, "decoder_called": False, "physical_forward_called": False}),
            GateResult("finite_measurements", all(report["finite"] and report["harness_passed"] for report in reports.values()), {}),
            GateResult("attribution_boundary", all(report["mean"]["attribution_only"] and not report["mean"]["deploy_time_mean_estimator"] for report in reports.values()), {}),
        )
        compact = {
            "schema": COMPACT_SCHEMA,
            "hypothesis": HYPOTHESIS,
            "status": "passed",
            "classification": "detector_mean_centered_functional_attribution_completed",
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "lineage": LINEAGE,
            "rmal_consumer": RMAL_CONSUMER,
            "lsda_consumer": LSDA_CONSUMER,
            "rwsa_identity": RWSA_IDENTITY,
            "rwsa_active_identity": RWSA_ACTIVE_IDENTITY,
            "zcsa_identity": ZCSA_IDENTITY,
            "lrrc_consumer": LRRC_CONSUMER,
            "dsfa_consumer": DSFA_CONSUMER,
            "dmsa_consumer": DMSA_CONSUMER,
            "fit_panel": panel_identity["fit"],
            "panel_contract": PANEL_CONTRACT,
            "panel_identity": panel_identity,
            "probe": {"sha256": probe.artifact_sha256, "W_shape": [FEATURE_DIM, TOTAL_COLUMNS], "bias_shape": [TOTAL_COLUMNS]},
            "panels": reports,
            "two_pass_contract": TWO_PASS_CONTRACT,
            "prediction_contract": PREDICTION_CONTRACT,
            "energy_contract": ENERGY_CONTRACT,
            "functional_contract": FUNCTIONAL_CONTRACT,
            "mse_contract": MSE_CONTRACT,
            "numerical": NUMERICAL_CONTRACT,
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
            "centered_feature_bank_saved": False,
            "target_bank_saved": False,
            "prediction_bank_saved": False,
            "mechanism_evidence_only": True,
            "descriptive_only": True,
            "bias_plus_mean_attribution_only": True,
        }
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
        classification = "detector_mean_centered_functional_attribution_completed" if harness_passed else "invalid_harness"
        compact["status"] = status
        compact["classification"] = classification
        compact["next_action"] = NEXT_ACTION_RESULT_REVIEW if harness_passed else None
        compact_path = output_path.with_name("compact_report.json")
        _write_new(compact_path, compact, artifact_kind="MCFA compact report")
        compact_hash = file_sha256(compact_path)
        hashes = _input_hashes(spec, review_path, upstream, panel_identity)
        hashes["mcfa_compact_report"] = compact_hash
        audit_contract = {
            "lineage": LINEAGE,
            "classification": classification,
            "consumer": "project_owner",
            "fit": reports["fit"],
            "selection": reports["selection"],
            "confirmation": reports["confirmation"],
            "two_pass_contract": TWO_PASS_CONTRACT,
            "compact_report_sha256": compact_hash,
            "selection_confirmation_isolated": True,
            "no_update": True,
            "descriptive_only": True,
            "bias_plus_mean_attribution_only": True,
        }
        audit_payload = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            state,
            _rmal_runtime(upstream).data.manifest["manifest_hash"],
            hashes,
            audit_gates,
            (NEXT_ACTION_RESULT_REVIEW,) if harness_passed else (),
            audit_contract,
            None if harness_passed else "MCFA harness gate failed",
            cuda_used=True,
        )
        _write_new(output_path, audit_payload, artifact_kind="MCFA audit evidence")
        print(json.dumps(audit_payload, indent=2, sort_keys=True))
        return 0 if harness_passed else 2
    except MCFAHarnessFailure as exc:
        if not output_path.exists():
            _run_invalid_harness(output_path, spec, review_path, upstream, panel_identity, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-detector-mean-centered-functional-attribution-validate":
        return validate(args)
    if args.research_action == "scaled-chen-detector-mean-centered-functional-attribution-run":
        return run(args)
    raise ValueError(f"unsupported MCFA action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "DMSA_CONSUMER",
    "ENERGY_CONTRACT",
    "FUNCTIONAL_CONTRACT",
    "FIT_PANEL",
    "HYPOTHESIS",
    "LINEAGE",
    "MSE_CONTRACT",
    "MCFAHarnessFailure",
    "MeanAccumulator",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "NUMERICAL_CONTRACT",
    "PANEL_CONTRACT",
    "PREDICTION_CONTRACT",
    "REVIEW_SCHEMA",
    "RMAL_CONSUMER",
    "RESOURCE_CONTRACT",
    "TWO_PASS_CONTRACT",
    "VALIDATION_SCHEMA",
    "_assert_contract",
    "_contract_payload",
    "_energy_report",
    "_gain_shares",
    "_load_review",
    "_load_spec",
    "_mse_report",
    "_panel_report",
    "_replay_pass1",
    "_replay_pass2",
    "_review_payload",
    "_verify_panel_identity",
    "dispatch",
    "run",
    "validate",
]
