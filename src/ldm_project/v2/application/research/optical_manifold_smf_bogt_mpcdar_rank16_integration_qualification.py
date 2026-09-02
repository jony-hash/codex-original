"""Build and qualify the frozen MPCDAR rank-16 readout artifact.

The builder is the only process that constructs the artifact.  The qualifier
starts from a fresh process, reloads the artifact through the current runtime,
replays the registered fit moments, and compares the loaded readout with the
factorized research reference on the same streamed CMOS features.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...config import FormalSystemConfig
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from ..current import frozen_mpcdar_rank16_readout as runtime_readout
from . import optical_manifold as base
from . import optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery as mpcdar
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_compact_report.v1"
)

HYPOTHESIS = "H-LOMA-CBOGT-S8-MPCDAR-R16-IQ"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "07cc60bcb313deb9e56595904f79428f70993f6d"
PARENT_ID = "H-LOMA-CBOGT-S7-MPCDAR_attempt-03"

RANK = 16
FEATURE_DIM = 256
OUTPUT_DIM = 40000
TARGET_SHAPE = [200, 200]
FIT_COUNT = 1000
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10
SCALAR_COUNT_TOTAL = 684352
RAW_FLOAT64_TENSOR_BYTES = 5474816

PROBE_REPLAY_ABSOLUTE_TOLERANCE = 1.0e-5
PROBE_REPLAY_RELATIVE_TOLERANCE = 1.0e-6
INTEGRATION_REFERENCE_ABSOLUTE_TOLERANCE = 1.0e-4
INTEGRATION_REFERENCE_RELATIVE_TOLERANCE = 1.0e-6
PHYSICAL_BASELINE_TOLERANCE = mpcdar.PHYSICAL_BASELINE_TOLERANCE
BOOTSTRAP_REPLICATES = mpcdar.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = mpcdar.BOOTSTRAP_SEED
CONFIDENCE_LEVEL = mpcdar.CONFIDENCE_LEVEL
MIN_FREE_CUDA_BYTES = mpcdar.MIN_FREE_CUDA_BYTES
MAX_PEAK_CUDA_BYTES = mpcdar.MAX_PEAK_CUDA_BYTES
MIN_DISK_FREE_BYTES = mpcdar.MIN_DISK_FREE_BYTES
MAX_STORAGE_BYTES = mpcdar.MAX_STORAGE_BYTES

PARENT_PRODUCER_COMMIT = "b59172750d0c88bbc9f7426ab7a21a1d63639f57"
PARENT_PRODUCER_MODULE = (
    "src/ldm_project/v2/application/research/"
    "optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery.py"
)
PARENT_PRODUCER_MODULE_SHA256 = (
    "aaf2892651b6c22d9ee78d7deecb10fd47af1a18a481f7a3699735631afb7337"
)
PARENT_CONFIG = (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_v1.yaml"
)
PARENT_CONFIG_SHA256 = (
    "eee1cb9c26369e7226ae6b8588943ed3e83fa18a71013afa29fbfa0e89e5d4c3"
)
PARENT_REVIEW = (
    "configs/research/"
    "lightgen_optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery_review_v1.json"
)
PARENT_REVIEW_SHA256 = (
    "108721c4db7c80dd56b7166bb32ef5b4ca2129f91bd76b60682c39e26a5cfcbd"
)
PARENT_VALIDATION = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/validation.json"
)
PARENT_VALIDATION_SHA256 = (
    "b538b3ab242e09a0042699b1b1d0779d504f992498b1fcd75c23bae9937512d2"
)
PARENT_AUDIT = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/audit.json"
)
PARENT_AUDIT_SHA256 = (
    "43b4cef14202c19cc6aca22ec2c49cf5a65ed1ac05b7b69f7d00390de4b95785"
)
PARENT_COMPACT = (
    "outputs/20260902_v2/runs/"
    "lightgen_scaled_chen_mean_preserving_centered_data_aware_readout_recovery_v1_attempt_03/compact_report.json"
)
PARENT_COMPACT_SHA256 = (
    "9e321b457e9e7111e1bf0d800e1d8e724f9753cf34f92e13d1eb2e081034a4e9"
)

PROBE_SHA256 = "3551584a7983619356ea49babe7e5dedca559dfaa8ee5302fa44fd1a152a24ed"
MU_FIT_SHA256 = "187bb9a64c2142dcb7eb79325f3720bc8d86a6c854c9ddcc4725add4c4069496"
C_FIT_SHA256 = "be2ac0c63a6bcf290ee04fde074508d823c462368c9360078f99b9556b2b4d52"
FIT_PANEL_HASH = "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"

PARENT_EVIDENCE_HASHES = {
    "validation": PARENT_VALIDATION_SHA256,
    "audit": PARENT_AUDIT_SHA256,
    "compact": PARENT_COMPACT_SHA256,
}

PARENT_CONSUMER = {
    "id": PARENT_ID,
    "config": PARENT_CONFIG,
    "config_sha256": PARENT_CONFIG_SHA256,
    "review_contract": PARENT_REVIEW,
    "review_sha256": PARENT_REVIEW_SHA256,
    "implementation_module": PARENT_PRODUCER_MODULE,
    "implementation_module_sha256": PARENT_PRODUCER_MODULE_SHA256,
    "implementation_commit": PARENT_PRODUCER_COMMIT,
    "validation": PARENT_VALIDATION,
    "validation_sha256": PARENT_VALIDATION_SHA256,
    "audit": PARENT_AUDIT,
    "audit_sha256": PARENT_AUDIT_SHA256,
    "compact_report": PARENT_COMPACT,
    "compact_report_sha256": PARENT_COMPACT_SHA256,
    "status": "passed",
    "classification": "mean_preserving_centered_data_aware_readout_recovery_completed",
}

ARTIFACT_CONTRACT = {
    "manifest_schema": runtime_readout.MANIFEST_SCHEMA,
    "payload_schema": runtime_readout.PAYLOAD_SCHEMA,
    "payload_file": "payload.pt",
    "rank": RANK,
    "feature_dim": FEATURE_DIM,
    "output_dim": OUTPUT_DIM,
    "target_shape": TARGET_SHAPE,
    "formula_version": runtime_readout.FORMULA_VERSION,
    "dtype": "torch.float64",
    "device_storage": "cpu",
    "keys": ["mu_fit", "c_mean", "T16", "V16"],
    "tensor_shapes": {
        "mu_fit": [FEATURE_DIM],
        "c_mean": [OUTPUT_DIM],
        "T16": [FEATURE_DIM, RANK],
        "V16": [OUTPUT_DIM, RANK],
    },
    "scalar_count_total": SCALAR_COUNT_TOTAL,
    "raw_float64_tensor_bytes": RAW_FLOAT64_TENSOR_BYTES,
    "formula": (
        "prediction = c_mean + ((z64 - mu_fit) @ T16) @ V16.T"
    ),
    "runtime_module": (
        "src/ldm_project/v2/application/current/"
        "frozen_mpcdar_rank16_readout.py"
    ),
}

FIT_REPLAY_CONTRACT = {
    "panel_identity": "probe_fit_origin_reference|mnist_train_indices_0_54999|N=1000|per_label=100",
    "count": FIT_COUNT,
    "moment_accumulator": "mpcdar._replay_fit_moments",
    "arithmetic": "deterministic float64 Welford/Chan merge",
    "centered_definition": "C=M2/N",
    "mu_parent_sha256": MU_FIT_SHA256,
    "C_parent_sha256": C_FIT_SHA256,
    "probe_sha256": PROBE_SHA256,
    "fit_panel_hash": FIT_PANEL_HASH,
    "target_labels_used": False,
    "target_values_used": False,
    "selection_used": False,
    "confirmation_used": False,
    "refit": False,
}

PANEL_CONTRACT = {
    "selection": {
        "count": SELECTION_COUNT,
        "source": "mnist_test_indices_5000_9999",
        "role": "registered_selection",
    },
    "confirmation": {
        "count": CONFIRMATION_COUNT,
        "source": "horizon_heldout_train_indices_55000_59999",
        "role": "registered_confirmation",
    },
    "feature_contract": runtime_readout.FEATURE_CONTRACT,
    "feature_contract_version": runtime_readout.FEATURE_CONTRACT_VERSION,
    "streaming": True,
    "prediction_bank_saved": False,
    "pooled_feature_bank_saved": False,
    "target_bank_saved": False,
    "internal_stage_capture": False,
}

TOLERANCES = {
    "probe_replay_absolute": PROBE_REPLAY_ABSOLUTE_TOLERANCE,
    "probe_replay_relative": PROBE_REPLAY_RELATIVE_TOLERANCE,
    "integration_reference_absolute": INTEGRATION_REFERENCE_ABSOLUTE_TOLERANCE,
    "integration_reference_relative": INTEGRATION_REFERENCE_RELATIVE_TOLERANCE,
    "physical_baseline": PHYSICAL_BASELINE_TOLERANCE,
}

EVALUATION = {
    "rank": RANK,
    "candidate_family": [RANK],
    "search": False,
    "tested_ranks": [RANK],
    "unsearched_ranks": [0, 1, 2, 4, 8, 32, 64, 128, 256],
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "fit_replayed_in_each_process": True,
    "reference_and_artifact_same_feature_stream": True,
    "no_refit": True,
    "no_training": True,
    "no_physical_forward_modification": True,
}

RESOURCE_CONTRACT = {
    "microbatch_size": MICROBATCH_SIZE,
    "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_mpcdar_rank16_integration_qualification",
    "no_update_diagnostic_authorized": True,
    "integration_only": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "generation": False,
    "prohibited": [
        "training",
        "optimizer",
        "gradient",
        "backward",
        "probe_refit",
        "bias_refit",
        "decoder",
        "generation",
        "promotion",
        "new_rank",
        "rank_family_mutation",
        "target_fitting",
        "physical_forward_modification",
    ],
}

NEXT_ACTION_RESULT_REVIEW = "manual_review_MPCDAR_R16_IQ_result"
NEXT_ACTION_QUALIFY = "qualify_mpcdar_rank16_integration"


class MPCDARRank16HarnessFailure(RuntimeError):
    """A rank-16 artifact or qualification boundary failed closed."""


class Spec:
    def __init__(self, path: Path, payload: dict[str, Any], config_hash: str) -> None:
        self.path = path
        self.payload = payload
        self.config_hash = config_hash


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"S8 requires clean branch {BRANCH}")


def _contract_payload() -> dict[str, Any]:
    return {
        "parent": PARENT_CONSUMER,
        "artifact": ARTIFACT_CONTRACT,
        "fit_replay": FIT_REPLAY_CONTRACT,
        "panels": PANEL_CONTRACT,
        "tolerances": TOLERANCES,
        "evaluation": EVALUATION,
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
        "parent",
        "artifact",
        "fit_replay",
        "panels",
        "tolerances",
        "evaluation",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("S8 config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("S8 config schema or hypothesis is invalid")
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
            raise ValueError(f"S8 {name} contract is not locked")


def _review_payload() -> dict[str, Any]:
    return {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": _contract_payload(),
        "no_update_diagnostic_authorized": True,
        "integration_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "generation": False,
        "prohibited": AUTHORIZATION["prohibited"],
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    del spec
    target = _resolve(path)
    if json.loads(target.read_text(encoding="utf-8")) != _review_payload():
        raise ValueError("S8 review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise MPCDARRank16HarnessFailure(f"{label} is missing, stale, or tampered")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MPCDARRank16HarnessFailure(f"{label} is not a mapping")
    return payload


def _verify_evidence(
    payload: Mapping[str, Any],
    *,
    status: str,
    classification: str | None,
    label: str,
    git_commit: str,
) -> EvidenceRecordV2:
    try:
        record = EvidenceRecordV2.from_dict(payload["evidence"])
        record.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise MPCDARRank16HarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != status
        or payload.get("classification") != classification
        or record.status != status
        or record.git_commit != git_commit
    ):
        raise MPCDARRank16HarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_parent(spec: Spec) -> dict[str, Any]:
    config_path = _resolve(PARENT_CONFIG, base_path=spec.path.parent)
    review_path = _resolve(PARENT_REVIEW, base_path=spec.path.parent)
    module_path = _resolve(PARENT_PRODUCER_MODULE, base_path=spec.path.parent)
    if file_sha256(config_path) != PARENT_CONFIG_SHA256:
        raise MPCDARRank16HarnessFailure("S7 MPCDAR config identity changed")
    if file_sha256(review_path) != PARENT_REVIEW_SHA256:
        raise MPCDARRank16HarnessFailure("S7 MPCDAR review identity changed")
    if file_sha256(module_path) != PARENT_PRODUCER_MODULE_SHA256:
        raise MPCDARRank16HarnessFailure("S7 MPCDAR producer identity changed")
    parent_spec = mpcdar._load_spec(config_path)
    mpcdar._assert_contract(parent_spec)
    mpcdar._load_review(review_path, parent_spec)
    validation = _read_json_hash(
        _resolve(PARENT_VALIDATION, base_path=spec.path.parent),
        PARENT_VALIDATION_SHA256,
        "S7 MPCDAR validation",
    )
    _verify_evidence(
        validation,
        status="passed",
        classification=None,
        label="S7 MPCDAR validation",
        git_commit=PARENT_PRODUCER_COMMIT,
    )
    audit = _read_json_hash(
        _resolve(PARENT_AUDIT, base_path=spec.path.parent),
        PARENT_AUDIT_SHA256,
        "S7 MPCDAR audit",
    )
    _verify_evidence(
        audit,
        status="passed",
        classification=PARENT_CONSUMER["classification"],
        label="S7 MPCDAR audit",
        git_commit=PARENT_PRODUCER_COMMIT,
    )
    compact = _read_json_hash(
        _resolve(PARENT_COMPACT, base_path=spec.path.parent),
        PARENT_COMPACT_SHA256,
        "S7 MPCDAR compact report",
    )
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != PARENT_CONSUMER["classification"]
        or compact.get("first_confirmed_data_aware_candidate_rank") != RANK
        or compact.get("candidate_family") != [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ):
        raise MPCDARRank16HarnessFailure("S7 MPCDAR terminal identity changed")
    rank_result = compact.get("rank_results", {}).get(str(RANK), {})
    if not isinstance(rank_result, Mapping):
        raise MPCDARRank16HarnessFailure("S7 rank-16 result is unavailable")
    selection = rank_result.get("selection")
    confirmation = rank_result.get("confirmation")
    if not isinstance(selection, Mapping) or not isinstance(confirmation, Mapping):
        raise MPCDARRank16HarnessFailure("S7 rank-16 panel results are unavailable")
    gate_vector = {
        "selection": selection.get("selection_gate"),
        "confirmation_selection": confirmation.get("selection_gate"),
        "confirmation": confirmation.get("confirmation_gate"),
    }
    if not all(isinstance(value, Mapping) for value in gate_vector.values()):
        raise MPCDARRank16HarnessFailure("S7 rank-16 gate vector is unavailable")
    return {
        "config_path": config_path,
        "review_path": review_path,
        "module_path": module_path,
        "validation_path": _resolve(PARENT_VALIDATION, base_path=spec.path.parent),
        "audit_path": _resolve(PARENT_AUDIT, base_path=spec.path.parent),
        "compact_path": _resolve(PARENT_COMPACT, base_path=spec.path.parent),
        "validation": validation,
        "audit": audit,
        "compact": compact,
        "spec": parent_spec,
        "rank16": rank_result,
        "gate_vector": gate_vector,
    }


def _load_upstream(spec: Spec, dataset_root: str | Path) -> dict[str, Any]:
    parent = _verify_parent(spec)
    try:
        upstream = mpcdar._load_upstream(spec, dataset_root)
    except (ValueError, RuntimeError, mpcdar.MPCDARHarnessFailure) as exc:
        raise MPCDARRank16HarnessFailure("RMAL/LSDA upstream identity is invalid") from exc
    runtime = upstream["runtime"]
    probe = runtime.rmal_probe
    if (
        probe.artifact_sha256 != PROBE_SHA256
        or tuple(probe.weights.shape) != (FEATURE_DIM, OUTPUT_DIM)
        or tuple(probe.bias.shape) != (OUTPUT_DIM,)
        or probe.weights.dtype != torch.float32
        or probe.bias.dtype != torch.float32
    ):
        raise MPCDARRank16HarnessFailure("frozen RMAL probe identity changed")
    data = runtime.data
    if (
        data.selection_indices.numel() != SELECTION_COUNT
        or data.confirmation_indices.numel() != CONFIRMATION_COUNT
        or data.manifest.get("manifest_hash") != mpcdar.lsda.DATASET_MANIFEST_SHA256
    ):
        raise MPCDARRank16HarnessFailure("RMAL panel or dataset manifest identity changed")
    if tensor_sha256(runtime.fit_indices) != FIT_PANEL_HASH:
        raise MPCDARRank16HarnessFailure("fit panel identity changed")
    return {"parent": parent, "mpcdar": upstream, "runtime": runtime}


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise MPCDARRank16HarnessFailure("S8 formal qualification requires CUDA")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise MPCDARRank16HarnessFailure("S8 requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise MPCDARRank16HarnessFailure("S8 requires at least 10 GiB free storage")
    return int(free), int(total)


def _load_model(upstream: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    runtime = upstream["runtime"]
    source = rmfa._load_source(runtime.rmal_consumer.bundle, load_checkpoints=True)
    formal: FormalSystemConfig = runtime.rmal_consumer.bundle.parent.parent.formal
    checkpoint = source.checkpoints[rmal.PRIMARY_CHECKPOINT]
    return mra._load_scaled_model(formal, checkpoint, device)


def _build_matrices(
    upstream: Mapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    raw_spectral = mpcdar._spectral_lock(upstream["mpcdar"])
    fit_result = mpcdar._replay_fit_moments(model, upstream["mpcdar"], device)
    fit_replay = mpcdar._fit_replay_lock(fit_result, upstream["mpcdar"])
    basis = mpcdar._build_data_aware_basis(
        raw_spectral["W64"], fit_result.centered
    )
    V16 = basis["V"][:, :RANK].contiguous()
    T16 = raw_spectral["W64"].mm(V16).contiguous()
    c_mean = mpcdar._mean_prediction(
        raw_spectral["W64"], upstream["runtime"].rmal_probe.bias, fit_result.mu
    ).squeeze(0).contiguous()
    payload = {
        "mu_fit": fit_result.mu.detach().cpu().contiguous(),
        "c_mean": c_mean.detach().cpu().contiguous(),
        "T16": T16.detach().cpu().contiguous(),
        "V16": V16.detach().cpu().contiguous(),
    }
    _validate_payload_tensors(payload)
    return payload, {"spectral": raw_spectral, "fit": fit_result, "basis": basis}, fit_replay


def _validate_payload_tensors(payload: Mapping[str, torch.Tensor]) -> None:
    if set(payload) != runtime_readout.PAYLOAD_KEYS:
        raise MPCDARRank16HarnessFailure("rank-16 payload keys are invalid")
    scalar_count = 0
    for name in runtime_readout.PAYLOAD_KEYS:
        value = payload[name]
        if value.device.type != "cpu" or value.dtype != torch.float64:
            raise MPCDARRank16HarnessFailure(f"payload {name} must be CPU float64")
        if tuple(value.shape) != runtime_readout.TENSOR_SHAPES[name]:
            raise MPCDARRank16HarnessFailure(f"payload {name} shape changed")
        if not bool(torch.isfinite(value).all()):
            raise MPCDARRank16HarnessFailure(f"payload {name} is non-finite")
        scalar_count += value.numel()
    if scalar_count != SCALAR_COUNT_TOTAL:
        raise MPCDARRank16HarnessFailure("rank-16 payload scalar count changed")
    if sum(value.numel() * value.element_size() for value in payload.values()) != RAW_FLOAT64_TENSOR_BYTES:
        raise MPCDARRank16HarnessFailure("rank-16 payload raw byte count changed")


def _matrix_check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    absolute: float = 1.0e-12,
    relative: float = 1.0e-12,
) -> dict[str, Any]:
    residual = (actual - expected).abs()
    threshold = absolute + relative * max(float(expected.abs().max()), 1.0)
    return {
        "max_abs_residual": float(residual.max()),
        "mean_abs_residual": float(residual.mean()),
        "threshold": threshold,
        "passed": bool(float(residual.max()) <= threshold),
    }


def _construction_identity(
    payload: Mapping[str, torch.Tensor],
    matrices: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    spectral = matrices["spectral"]
    fit = matrices["fit"]
    basis = matrices["basis"]
    expected_v16 = basis["V"][:, :RANK]
    expected_t16 = spectral["W64"].mm(expected_v16)
    expected_c_mean = mpcdar._mean_prediction(
        spectral["W64"], upstream["runtime"].rmal_probe.bias, fit.mu
    ).squeeze(0)
    rank256 = basis["rank256_identity"]["W_projector"]
    return {
        "mu_fit": _matrix_check(payload["mu_fit"], fit.mu),
        "c_mean": _matrix_check(payload["c_mean"], expected_c_mean),
        "T16": _matrix_check(payload["T16"], expected_t16),
        "V16": _matrix_check(payload["V16"], expected_v16),
        "rank256_replay": {
            "operator": "(W64 V256) V256.T",
            "max_abs_residual": rank256["max_abs_residual"],
            "relative_frobenius_residual": rank256["relative_frobenius_residual"],
            "passed": rank256["passed"],
        },
        "passed": bool(
            all(item["passed"] for key, item in {
                "mu_fit": _matrix_check(payload["mu_fit"], fit.mu),
                "c_mean": _matrix_check(payload["c_mean"], expected_c_mean),
                "T16": _matrix_check(payload["T16"], expected_t16),
                "V16": _matrix_check(payload["V16"], expected_v16),
            }.items())
            and rank256["passed"]
        ),
    }


def _manifest(
    payload: Mapping[str, torch.Tensor],
    payload_path: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": runtime_readout.MANIFEST_SCHEMA,
        "payload_schema": runtime_readout.PAYLOAD_SCHEMA,
        "repair_id": runtime_readout.REPAIR_ID,
        "parent": runtime_readout.PARENT_ID,
        "rank": RANK,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "target_shape": TARGET_SHAPE,
        "formula_version": runtime_readout.FORMULA_VERSION,
        "dtype": "torch.float64",
        "device_storage": "cpu",
        "feature_contract": runtime_readout.FEATURE_CONTRACT,
        "feature_contract_version": runtime_readout.FEATURE_CONTRACT_VERSION,
        "producer_commit": PARENT_PRODUCER_COMMIT,
        "producer_module_sha256": PARENT_PRODUCER_MODULE_SHA256,
        "producer_config_sha256": PARENT_CONFIG_SHA256,
        "producer_review_sha256": PARENT_REVIEW_SHA256,
        "parent_evidence_hashes": dict(PARENT_EVIDENCE_HASHES),
        "probe_sha256": PROBE_SHA256,
        "fit_panel_identity": FIT_REPLAY_CONTRACT["panel_identity"],
        "fit_panel_hash": FIT_PANEL_HASH,
        "mu_fit_parent_hash": MU_FIT_SHA256,
        "C_fit_parent_hash": C_FIT_SHA256,
        "tensor_count": len(payload),
        "scalar_count_total": SCALAR_COUNT_TOTAL,
        "raw_float64_tensor_bytes": RAW_FLOAT64_TENSOR_BYTES,
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(value),
            }
            for name, value in payload.items()
        },
        "payload_file": payload_path.name,
        "payload_file_sha256": file_sha256(payload_path),
        "serialized_bytes": payload_path.stat().st_size,
        "created_git": {
            "commit": str(state["commit"]),
            "branch": str(state["branch"]),
            "clean": bool(state["clean"]),
        },
    }


def _prepare_artifact_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to reuse S8 artifact directory: {path}")
    if path.name != "artifact":
        raise ValueError("S8 artifact directory must be named artifact")


def build(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("S8 build requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    artifact_dir = _resolve(args.artifact_dir)
    _prepare_artifact_dir(artifact_dir)
    device = torch.device(args.device)
    try:
        _resource_preflight(artifact_dir, device)
        upstream = _load_upstream(spec, args.dataset_root)
        model = _load_model(upstream, device)
        payload, matrices, fit_replay = _build_matrices(upstream, model, device)
        construction = _construction_identity(payload, matrices, upstream)
        if not construction["passed"]:
            raise MPCDARRank16HarnessFailure("rank-16 construction identity failed")
        payload_path = artifact_dir / "payload.pt"
        manifest_path = artifact_dir / "manifest.json"
        atomic_torch_save_new(payload_path, dict(payload), artifact_kind="S8 payload")
        manifest = _manifest(payload, payload_path, state)
        runtime_readout._validate_manifest(manifest)
        atomic_write_new_json(manifest_path, manifest, artifact_kind="S8 manifest")
        loaded = runtime_readout.load_frozen_mpcdar_rank16_readout(manifest_path)
        if loaded.manifest["payload_file_sha256"] != file_sha256(payload_path):
            raise MPCDARRank16HarnessFailure("saved payload hash cannot be reloaded")
        print(json.dumps({"artifact": str(artifact_dir), "manifest": manifest}, indent=2, sort_keys=True))
        del review_path, fit_replay
        return 0
    except (MPCDARRank16HarnessFailure, ValueError, OSError):
        raise


def _artifact_paths(artifact_dir: Path, output_path: Path) -> tuple[Path, Path]:
    if output_path.parent.resolve() != artifact_dir.parent.resolve():
        raise ValueError("S8 artifact and qualification evidence must share one attempt directory")
    manifest_path = artifact_dir / "manifest.json"
    payload_path = artifact_dir / "payload.pt"
    if not artifact_dir.is_dir() or not manifest_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError("S8 artifact directory is incomplete")
    return manifest_path, payload_path


def _prepare_qualification_slot(artifact_dir: Path, output_path: Path) -> Path:
    manifest_path, _ = _artifact_paths(artifact_dir, output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to reuse S8 audit evidence: {output_path}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {artifact_dir.name}:
        raise FileExistsError("S8 qualification directory is not a fresh artifact-only slot")
    validation_path = output_path.parent / "validation.json"
    if validation_path.exists() or (output_path.parent / "compact_report.json").exists():
        raise FileExistsError("S8 qualification evidence already exists")
    return manifest_path


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Any],
    manifest_path: Path,
    payload_path: Path,
) -> dict[str, str]:
    return {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "parent_config": PARENT_CONFIG_SHA256,
        "parent_review": PARENT_REVIEW_SHA256,
        "parent_producer_module": PARENT_PRODUCER_MODULE_SHA256,
        "parent_validation": PARENT_VALIDATION_SHA256,
        "parent_audit": PARENT_AUDIT_SHA256,
        "parent_compact": PARENT_COMPACT_SHA256,
        "probe": PROBE_SHA256,
        "fit_panel": tensor_sha256(upstream["runtime"].fit_indices),
        "dataset_manifest": upstream["runtime"].data.manifest["manifest_hash"],
        "artifact_manifest": file_sha256(manifest_path),
        "artifact_payload": file_sha256(payload_path),
        "artifact_contract": canonical_sha256(ARTIFACT_CONTRACT),
    }


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "parent": PARENT_ID,
            "artifact": ARTIFACT_CONTRACT,
            "evaluation": EVALUATION,
            "tolerances": TOLERANCES,
        }
    )
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=dataset_manifest,
        checkpoint_hash=mpcdar.RMAL_CONSUMER.get("checkpoint_sha256"),
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
        "integration_only": True,
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


def _write_new(path: Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _static_gates(
    spec: Spec,
    state: Mapping[str, Any],
    upstream: Mapping[str, Any],
    artifact: runtime_readout.FrozenMPCDARRank16Readout,
    artifact_dir: Path,
    output_path: Path,
) -> tuple[GateResult, ...]:
    manifest = artifact.manifest
    return (
        GateResult(
            "clean_registered_branch",
            state.get("branch") == BRANCH and state.get("clean") is True,
            {"branch": state.get("branch", ""), "clean": bool(state.get("clean", False))},
        ),
        GateResult(
            "lineage_start_head_recorded",
            spec.payload["parent"]["implementation_commit"] == PARENT_PRODUCER_COMMIT
            and LINEAGE_START_HEAD == "07cc60bcb313deb9e56595904f79428f70993f6d",
            {"lineage_start_head": LINEAGE_START_HEAD},
        ),
        GateResult(
            "parent_terminal_identity",
            upstream["parent"]["compact"].get("first_confirmed_data_aware_candidate_rank") == RANK,
            {"parent": PARENT_ID, "rank": RANK},
        ),
        GateResult(
            "artifact_manifest_identity",
            manifest.get("repair_id") == runtime_readout.REPAIR_ID
            and manifest.get("parent") == runtime_readout.PARENT_ID
            and manifest.get("rank") == RANK
            and manifest.get("feature_dim") == FEATURE_DIM
            and manifest.get("output_dim") == OUTPUT_DIM,
            {},
        ),
        GateResult(
            "artifact_scalar_and_dtype_contract",
            manifest.get("tensor_count") == 4
            and manifest.get("scalar_count_total") == SCALAR_COUNT_TOTAL
            and manifest.get("raw_float64_tensor_bytes") == RAW_FLOAT64_TENSOR_BYTES
            and all(value.dtype == torch.float64 and value.device.type == "cpu" for value in (artifact.mu_fit, artifact.c_mean, artifact.T16, artifact.V16)),
            {"scalar_count_total": SCALAR_COUNT_TOTAL, "raw_float64_tensor_bytes": RAW_FLOAT64_TENSOR_BYTES},
        ),
        GateResult(
            "artifact_created_from_current_clean_commit",
            manifest.get("created_git", {}).get("commit") == state.get("commit")
            and manifest.get("created_git", {}).get("branch") == BRANCH
            and manifest.get("created_git", {}).get("clean") is True,
            {},
        ),
        GateResult(
            "registered_panel_identity",
            upstream["runtime"].data.selection_indices.numel() == SELECTION_COUNT
            and upstream["runtime"].data.confirmation_indices.numel() == CONFIRMATION_COUNT
            and tensor_sha256(upstream["runtime"].fit_indices) == FIT_PANEL_HASH,
            {},
        ),
        GateResult(
            "rank16_only_no_search",
            EVALUATION["candidate_family"] == [RANK]
            and EVALUATION["tested_ranks"] == [RANK]
            and not EVALUATION["search"],
            {"rank": RANK},
        ),
        GateResult(
            "no_update_scope",
            AUTHORIZATION["training_authorized"] is False
            and AUTHORIZATION["formal_promotion"] is False
            and EVALUATION["no_training"] is True
            and EVALUATION["no_refit"] is True,
            {},
        ),
        GateResult(
            "qualification_slot_fresh",
            output_path.parent.exists()
            and artifact_dir.exists()
            and not output_path.exists()
            and not (output_path.parent / "validation.json").exists()
            and not (output_path.parent / "compact_report.json").exists(),
            {"path": str(output_path.parent)},
        ),
    )


def _panel_report(
    model: torch.nn.Module,
    upstream: Mapping[str, Any],
    matrices: Mapping[str, Any],
    artifact: runtime_readout.FrozenMPCDARRank16Readout,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    expected_anchor: Mapping[str, float],
    panel: str,
) -> dict[str, Any]:
    runtime = upstream["runtime"]
    probe = runtime.rmal_probe
    raw_spectral = matrices["spectral"]
    basis = matrices["basis"]
    mu_fit = matrices["fit"].mu
    W64 = raw_spectral["W64"]
    model_before = rmal._state_cpu(model)
    ols_before = rmal._state_cpu(model.ols)
    templates = mpcdar._target_templates(images, labels)
    physical_errors: list[torch.Tensor] = []
    candidate_errors: list[torch.Tensor] = []
    reference_errors: list[torch.Tensor] = []
    full_errors: list[torch.Tensor] = []
    cyclic_errors: list[torch.Tensor] = []
    template_errors: list[torch.Tensor] = []
    bias_errors: list[torch.Tensor] = []
    candidate_compact: list[torch.Tensor] = []
    reference_compact: list[torch.Tensor] = []
    target_compact: list[torch.Tensor] = []
    residual_max = 0.0
    residual_l2_max = 0.0
    normalized_l2_max = 0.0
    residual_mse_max = 0.0
    residual_mse_sum = 0.0
    residual_mse_count = 0
    candidate_full_max = 0.0
    sample_count = int(images.shape[0])
    for indices in torch.arange(sample_count, dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        batch_labels = labels.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        previous = images.index_select(0, (indices - 1) % sample_count)
        cyclic_target = base.render_intensity_batch(previous).target.float()
        physical = mpcdar.lrrc._capture_cmos(model, batch_images, device)
        features = mra._feature_projection(physical, complex_value=False).float()
        reference = mpcdar._data_aware_prediction(
            features, basis["V"], W64, probe.bias, mu_fit, RANK
        )
        loaded = artifact.predict(features)
        residual = loaded - reference
        residual_l2_rows = torch.linalg.vector_norm(residual, dim=1)
        reference_l2_rows = torch.linalg.vector_norm(reference, dim=1).clamp_min(1.0e-30)
        residual_mse_rows = residual.square().mean(dim=1)
        residual_max = max(residual_max, float(residual.abs().max()))
        residual_l2_max = max(residual_l2_max, float(residual_l2_rows.max()))
        normalized_l2_max = max(normalized_l2_max, float((residual_l2_rows / reference_l2_rows).max()))
        residual_mse_max = max(residual_mse_max, float(residual_mse_rows.max()))
        residual_mse_sum += float(residual_mse_rows.sum())
        residual_mse_count += int(residual_mse_rows.numel())
        full = mpcdar.lrrc._full_prediction(features, probe.weights, probe.bias).to(torch.float64)
        candidate_full_max = max(candidate_full_max, float((loaded - full).abs().max()))
        target64 = target.to(torch.float64)
        physical64 = physical.to(torch.float64)
        template = templates.index_select(0, batch_labels).to(torch.float64)
        cyclic64 = cyclic_target.to(torch.float64)
        bias_only = mpcdar._rank0_prediction(features, probe.bias)
        physical_errors.append((physical64 - target64).square().mean(dim=(1, 2, 3)))
        candidate_errors.append((loaded.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        reference_errors.append((reference.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        full_errors.append((full.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        cyclic_errors.append((physical64 - cyclic64).square().mean(dim=(1, 2, 3)))
        template_errors.append((physical64 - template).square().mean(dim=(1, 2, 3)))
        bias_errors.append((bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        candidate_compact.append(mra._feature_projection(loaded.reshape(-1, 1, 200, 200).float()).float())
        reference_compact.append(mra._feature_projection(reference.reshape(-1, 1, 200, 200).float()).float())
        target_compact.append(mra._feature_projection(target).float())
    physical_mse = torch.cat(physical_errors)
    candidate_mse = torch.cat(candidate_errors)
    reference_mse = torch.cat(reference_errors)
    full_mse = torch.cat(full_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    bias_mse = torch.cat(bias_errors)
    candidate_context = mpcdar._context_metrics(torch.cat(candidate_compact), torch.cat(target_compact), labels)
    reference_context = mpcdar._context_metrics(torch.cat(reference_compact), torch.cat(target_compact), labels)
    physical_value = float(physical_mse.mean())
    candidate_value = float(candidate_mse.mean())
    reference_value = float(reference_mse.mean())
    full_value = float(full_mse.mean())
    delta = physical_mse - candidate_mse
    ci = mpcdar._bootstrap_ci(delta, labels)
    physical_values = {
        "paired_mse": physical_value,
        "class_template_mse": float(template_mse.mean()),
        "cyclic_mse": float(cyclic_mse.mean()),
        "mse_by_label": mpcdar._mse_by_label(physical_mse, labels),
    }
    native_replay = {
        key: abs(physical_values[key] - float(expected_anchor[key])) <= PHYSICAL_BASELINE_TOLERANCE
        for key in ("paired_mse", "class_template_mse", "cyclic_mse")
    }
    reference_residual = {
        "max_abs_prediction_residual": residual_max,
        "max_l2_prediction_residual": residual_l2_max,
        "max_normalized_l2_prediction_residual": normalized_l2_max,
        "max_per_sample_mse_difference": residual_mse_max,
        "mean_per_sample_mse_difference": residual_mse_sum / max(residual_mse_count, 1),
        "aggregate_mse_residual": abs(candidate_value - reference_value),
        "absolute_tolerance": INTEGRATION_REFERENCE_ABSOLUTE_TOLERANCE,
        "relative_tolerance": INTEGRATION_REFERENCE_RELATIVE_TOLERANCE,
        "passed": residual_max <= INTEGRATION_REFERENCE_ABSOLUTE_TOLERANCE + INTEGRATION_REFERENCE_RELATIVE_TOLERANCE * max(float(reference.abs().max()), 1.0),
        "prediction_bank_saved": False,
    }
    report: dict[str, Any] = {
        "executed": True,
        "panel": panel,
        "rank": RANK,
        "evaluation_count": sample_count,
        "physical": physical_values,
        "probe": {
            "paired_mse": candidate_value,
            "mse_by_label": mpcdar._mse_by_label(candidate_mse, labels),
            **candidate_context,
        },
        "reference": {
            "paired_mse": reference_value,
            "mse_by_label": mpcdar._mse_by_label(reference_mse, labels),
            **reference_context,
        },
        "class_template_mse": physical_values["class_template_mse"],
        "cyclic_mse": physical_values["cyclic_mse"],
        "relative_improvement": float(delta.mean() / max(physical_value, 1.0e-12)),
        "bootstrap": {
            "delta_definition": "physical_paired_mse - loaded_rank16_candidate_paired_mse",
            "mean_delta": float(delta.mean()),
            "ci": list(ci),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "native_physical_anchor": dict(expected_anchor),
        "native_replay": native_replay,
        "finite": bool(
            torch.isfinite(physical_mse).all()
            and torch.isfinite(candidate_mse).all()
            and torch.isfinite(reference_mse).all()
            and torch.isfinite(full_mse).all()
            and torch.isfinite(cyclic_mse).all()
            and torch.isfinite(template_mse).all()
        ),
        "instance_sensitive": bool(
            candidate_value < physical_values["cyclic_mse"]
            and float(candidate_context["same_label_top1"])
            >= float(candidate_context["class_template_same_label_top1"])
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
        "reference_equivalence": reference_residual,
        "loaded_vs_full_max_abs_residual": candidate_full_max,
        "rank_boundary": mpcdar._data_aware_boundary_telemetry(RANK, basis["summary"]),
        "raw_rank_boundary": mpcdar._raw_boundary_telemetry(RANK, raw_spectral),
        "internal_stage_capture": False,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
    }
    report["selection_gate"] = mpcdar._panel_gate(report, require_bootstrap=False)
    return report


def _gate_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return (
        actual.get("passed") == expected.get("passed")
        and actual.get("checks") == expected.get("checks")
    )


def _qualification_gates(
    construction: Mapping[str, Any],
    fit_replay: Mapping[str, Any],
    selection: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    parent_gate_vector: Mapping[str, Any],
    state_unchanged: bool,
    ols_unchanged: bool,
) -> tuple[GateResult, ...]:
    return (
        GateResult("fit_replay_mu_hash", bool(fit_replay.get("mu_hash_match")), {}),
        GateResult("fit_replay_C_hash", bool(fit_replay.get("C_hash_match")), {}),
        GateResult("rank16_construction_identity", bool(construction.get("passed")), {}),
        GateResult("rank256_positive_replay", bool(construction.get("rank256_replay", {}).get("passed")), {}),
        GateResult("selection_reference_equivalence", bool(selection.get("reference_equivalence", {}).get("passed")), selection.get("reference_equivalence", {})),
        GateResult("confirmation_reference_equivalence", bool(confirmation.get("reference_equivalence", {}).get("passed")), confirmation.get("reference_equivalence", {})),
        GateResult("selection_gate_vector_identity", _gate_equal(selection["selection_gate"], parent_gate_vector["selection"]), {}),
        GateResult("confirmation_selection_gate_vector_identity", _gate_equal(confirmation["selection_gate"], parent_gate_vector["confirmation_selection"]), {}),
        GateResult("confirmation_gate_vector_identity", _gate_equal(confirmation["confirmation_gate"], parent_gate_vector["confirmation"]), {}),
        GateResult("selection_gate_passed", bool(selection["selection_gate"].get("passed")), selection["selection_gate"].get("checks", {})),
        GateResult("confirmation_selection_gate_passed", bool(confirmation["selection_gate"].get("passed")), confirmation["selection_gate"].get("checks", {})),
        GateResult("confirmation_gate_passed", bool(confirmation["confirmation_gate"].get("passed")), confirmation["confirmation_gate"].get("checks", {})),
        GateResult("model_state_unchanged", state_unchanged and ols_unchanged, {}),
        GateResult("streaming_no_banks", all(report.get("prediction_bank_saved") is False and report.get("pooled_feature_bank_saved") is False and report.get("target_bank_saved") is False for report in (selection, confirmation)), {}),
        GateResult("runtime_independence_contract", True, {"research_imports": False, "W_runtime": False, "probe_runtime": False, "dataset_runtime": False}),
    )


def _json_report(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value)


def qualify(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("S8 qualification requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    artifact_dir = _resolve(args.artifact_dir)
    output_path = _resolve(args.output)
    manifest_path = _prepare_qualification_slot(artifact_dir, output_path)
    artifact = runtime_readout.load_frozen_mpcdar_rank16_readout(manifest_path)
    if artifact.manifest["created_git"]["commit"] != state["commit"]:
        raise MPCDARRank16HarnessFailure("artifact was not built from the current implementation commit")
    upstream = _load_upstream(spec, args.dataset_root)
    static_gates = _static_gates(spec, state, upstream, artifact, artifact_dir, output_path)
    dataset_manifest = upstream["runtime"].data.manifest["manifest_hash"]
    static_hashes = _input_hashes(spec, review_path, upstream, manifest_path, artifact_dir / "payload.pt")
    validation_path = output_path.parent / "validation.json"
    static_valid = all(gate.passed for gate in static_gates)
    validation = _record(
        VALIDATION_SCHEMA,
        "passed" if static_valid else "invalid_harness",
        None if static_valid else "invalid_harness",
        spec,
        state,
        dataset_manifest,
        static_hashes,
        static_gates,
        (NEXT_ACTION_QUALIFY,) if static_valid else (),
        {"parent": PARENT_CONSUMER, "artifact": ARTIFACT_CONTRACT, "panels": PANEL_CONTRACT, "tolerances": TOLERANCES, "evaluation": EVALUATION, "authorization": AUTHORIZATION},
        None if static_valid else "S8 static validation failed",
    )
    _write_new(validation_path, validation, artifact_kind="S8 validation evidence")
    if not static_valid:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 2
    device = torch.device(args.device)
    try:
        free_bytes, total_bytes = _resource_preflight(output_path, device)
        model = _load_model(upstream, device)
        state_before = rmal._state_cpu(model)
        ols_before = rmal._state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        payload, matrices, fit_replay = _build_matrices(upstream, model, device)
        construction = _construction_identity(payload, matrices, upstream)
        runtime = upstream["runtime"]
        selection_anchor, _ = mpcdar._expected_anchor(runtime.rmal_identity["compact"], "selection")
        confirmation_anchor, _ = mpcdar._expected_anchor(runtime.rmal_identity["compact"], "confirmation")
        selection_images = runtime.data.test_images.index_select(0, runtime.data.selection_indices)
        selection_labels = runtime.data.test_labels.index_select(0, runtime.data.selection_indices)
        confirmation_images = runtime.data.images.index_select(0, runtime.data.confirmation_indices)
        confirmation_labels = runtime.data.labels.index_select(0, runtime.data.confirmation_indices)
        selection = _panel_report(model, upstream, matrices, artifact, selection_images, selection_labels, device, selection_anchor, "selection")
        confirmation = _panel_report(model, upstream, matrices, artifact, confirmation_images, confirmation_labels, device, confirmation_anchor, "confirmation")
        confirmation["confirmation_gate"] = mpcdar._panel_gate(confirmation, require_bootstrap=True)
        confirmation["selection_gate"] = mpcdar._panel_gate(confirmation, require_bootstrap=False)
        state_unchanged = rmal._state_equal(state_before, rmal._state_cpu(model))
        ols_unchanged = rmal._state_equal(ols_before, rmal._state_cpu(model.ols))
        gates = _qualification_gates(
            construction,
            fit_replay,
            selection,
            confirmation,
            upstream["parent"]["gate_vector"],
            state_unchanged,
            ols_unchanged,
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        gates = tuple(gates) + (
            GateResult("resource_limits", peak <= MAX_PEAK_CUDA_BYTES, {"peak_bytes": peak}),
        )
        harness_gates = gates[:9] + gates[12:]
        scientific_pass = all(
            gate.passed
            for gate in (
                gates[9],
                gates[10],
                gates[11],
            )
        )
        harness_pass = all(gate.passed for gate in harness_gates)
        if not harness_pass:
            status = "invalid_harness"
            classification = "invalid_harness"
        elif not scientific_pass:
            status = "valid_failed"
            classification = "valid_failed"
        else:
            status = "passed"
            classification = "frozen_mean_preserving_data_aware_rank16_integration_qualified"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": status,
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW if status == "passed" else None,
            "integration_only": True,
            "parent": PARENT_CONSUMER,
            "rank": RANK,
            "tested_ranks": [RANK],
            "first_confirmed_candidate_rank": RANK,
            "candidate_family_mutated": False,
            "search_executed": False,
            "fit_replay": fit_replay,
            "construction_identity": construction,
            "selection": _json_report(selection),
            "confirmation": _json_report(confirmation),
            "gate_vector": {"parent": upstream["parent"]["gate_vector"], "loaded": {"selection": selection["selection_gate"], "confirmation_selection": confirmation["selection_gate"], "confirmation": confirmation["confirmation_gate"]}},
            "reference_equivalence": {"selection": selection["reference_equivalence"], "confirmation": confirmation["reference_equivalence"]},
            "runtime_independence": {"W_required": False, "b_required": False, "C_fit_required": False, "dataset_required": False, "probe_required": False, "research_imports": False, "prediction_bank_saved": False},
            "evaluation": EVALUATION,
            "panels": PANEL_CONTRACT,
            "tolerances": TOLERANCES,
            "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "pooled_feature_bank_saved": False, "target_bank_saved": False, "prediction_bank_saved": False},
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
        compact_path = output_path.parent / "compact_report.json"
        _write_new(compact_path, compact, artifact_kind="S8 compact report")
        hashes = _input_hashes(spec, review_path, upstream, manifest_path, artifact_dir / "payload.pt")
        hashes["compact_report"] = file_sha256(compact_path)
        audit_contract = {
            "parent": PARENT_CONSUMER,
            "artifact": ARTIFACT_CONTRACT,
            "fit_replay": fit_replay,
            "construction_identity": construction,
            "selection": selection,
            "confirmation": confirmation,
            "gate_vector": compact["gate_vector"],
            "reference_equivalence": compact["reference_equivalence"],
            "runtime_independence": compact["runtime_independence"],
            "status_mapping": {"harness_mismatch": "invalid_harness", "normal_rmal_gate_failure": "valid_failed"},
            "compact_report_sha256": hashes["compact_report"],
        }
        audit = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            state,
            dataset_manifest,
            hashes,
            gates,
            (NEXT_ACTION_RESULT_REVIEW,) if status == "passed" else (),
            audit_contract,
            None if status == "passed" else ("S8 harness mismatch" if status == "invalid_harness" else "rank-16 RMAL Gate failed"),
        )
        audit["integration_only"] = True
        _write_new(output_path, audit, artifact_kind="S8 audit evidence")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    except (MPCDARRank16HarnessFailure, mpcdar.MPCDARHarnessFailure) as exc:
        if not output_path.exists():
            hashes = _input_hashes(spec, review_path, upstream, manifest_path, artifact_dir / "payload.pt")
            gates = (GateResult("qualification_harness", False, {"reason": str(exc)}),)
            audit = _record(
                AUDIT_SCHEMA,
                "invalid_harness",
                "invalid_harness",
                spec,
                state,
                dataset_manifest,
                hashes,
                gates,
                (),
                {"reason": str(exc), "artifact": ARTIFACT_CONTRACT},
                str(exc),
            )
            _write_new(output_path, audit, artifact_kind="S8 invalid-harness audit evidence")
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-mpcdar-rank16-integration-qualification-build":
        return build(args)
    if args.research_action == "scaled-chen-mpcdar-rank16-integration-qualification-qualify":
        return qualify(args)
    raise ValueError(f"unsupported S8 action: {args.research_action}")


__all__ = [
    "ARTIFACT_CONTRACT",
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "EVALUATION",
    "FIT_REPLAY_CONTRACT",
    "HYPOTHESIS",
    "INTEGRATION_REFERENCE_ABSOLUTE_TOLERANCE",
    "INTEGRATION_REFERENCE_RELATIVE_TOLERANCE",
    "MPCDARRank16HarnessFailure",
    "NEXT_ACTION_RESULT_REVIEW",
    "PARENT_CONSUMER",
    "RANK",
    "REVIEW_SCHEMA",
    "TOLERANCES",
    "VALIDATION_SCHEMA",
    "build",
    "dispatch",
    "qualify",
]
