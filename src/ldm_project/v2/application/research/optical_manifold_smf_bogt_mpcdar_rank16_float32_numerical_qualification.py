"""Build and qualify the frozen MPCDAR rank-16 float32 consumer.

BUILD-F32 performs only a deterministic CPU cast of the immutable S8 artifact.
QUALIFY-F32 is a separate process that reloads both artifacts and compares the
S8 float64 consumer with the independent S9 float32 consumer on the same
streamed CMOS features and inherited RMAL panels.
"""

from __future__ import annotations

import argparse
import inspect
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
from ..current import frozen_mpcdar_rank16_float32_readout as candidate_runtime
from ..current import frozen_mpcdar_rank16_readout as reference_runtime
from . import optical_manifold as base
from . import optical_manifold_smf_bogt_low_rank_readout_recovery_confirmation as lrrc
from . import optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification as s8
from . import optical_manifold_smf_bogt_mean_preserving_centered_data_aware_readout_recovery as mpcdar
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa
from . import optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization as rmal


CONFIG_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_v1"
)
REVIEW_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_review.v1"
)
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_validation.v1"
)
AUDIT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification.v1"
)
COMPACT_SCHEMA = (
    "lightgen_optical_manifold_smf_bogt_mpcdar_rank16_float32_numerical_qualification_compact_report.v1"
)

HYPOTHESIS = "H-LOMA-CBOGT-S9-MPCDAR-R16-F32-NQ"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
LINEAGE_START_HEAD = "89545909f1cfb33f31b087a735cd5babdd960e8c"

RANK = 16
FEATURE_DIM = 256
OUTPUT_DIM = 40000
TARGET_SHAPE = [200, 200]
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
MICROBATCH_SIZE = 10
SCALAR_COUNT_TOTAL = 684352
RAW_FLOAT64_TENSOR_BYTES = 5474816
RAW_FLOAT32_TENSOR_BYTES = 2737408

FLOAT32_PREDICTION_ATOL = candidate_runtime.FLOAT32_PREDICTION_ATOL
FLOAT32_PREDICTION_RTOL = candidate_runtime.FLOAT32_PREDICTION_RTOL
FLOAT32_NORMALIZED_L2_MAX = candidate_runtime.FLOAT32_NORMALIZED_L2_MAX
FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX = candidate_runtime.FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX
FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX = candidate_runtime.FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
PHYSICAL_BASELINE_TOLERANCE = mpcdar.PHYSICAL_BASELINE_TOLERANCE

PARENT_CONFIG = "configs/research/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_v1.yaml"
PARENT_CONFIG_SHA256 = "bdad8d30fd7a774fbf82ceaf3d060a90fcf72f60da542ce3eba0377ff15b76ca"
PARENT_REVIEW = "configs/research/lightgen_optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification_review_v1.json"
PARENT_REVIEW_SHA256 = "5fa0ccf5ec74c0d3f5165fe70b684c8ed69952a760df9a575f66a03186dff4c6"
PARENT_PRODUCER_MODULE = "src/ldm_project/v2/application/research/optical_manifold_smf_bogt_mpcdar_rank16_integration_qualification.py"
PARENT_PRODUCER_MODULE_SHA256 = "71be8e24a47d6ea4a76e0f823e4ea8c282f72a0e4997dcceb710a256b6ca868d"
PARENT_IMPLEMENTATION_COMMIT = "c0140577fddb56af79d72a6d473cc49ef04f5ffc"
PARENT_VALIDATION = "outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/validation.json"
PARENT_VALIDATION_SHA256 = "844a9dc4187623c380655db63530e13e6ba1a9d4e62d6ef44dd95b29119457cb"
PARENT_AUDIT = "outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/audit.json"
PARENT_AUDIT_SHA256 = "3b98c0faa6816b11d2f6da8cc832e89d221344f4b01bd41eadb356968f1b39a4"
PARENT_COMPACT = "outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/compact_report.json"
PARENT_COMPACT_SHA256 = "12e342d75119ee143186046490bb14c064f449dc27520168a17e84ae758b8271"
PARENT_ARTIFACT_MANIFEST = "outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/artifact/manifest.json"
PARENT_ARTIFACT_MANIFEST_SHA256 = candidate_runtime.PARENT_MANIFEST_SHA256
PARENT_ARTIFACT_PAYLOAD = "outputs/20260902_v2/runs/lightgen_scaled_chen_mpcdar_rank16_integration_qualification_v1_attempt_03/artifact/payload.pt"
PARENT_ARTIFACT_PAYLOAD_SHA256 = candidate_runtime.PARENT_PAYLOAD_SHA256
PARENT_ARTIFACT_CREATED_COMMIT = candidate_runtime.PARENT_CREATED_COMMIT
PROBE_SHA256 = reference_runtime.PROBE_SHA256
FIT_PANEL_HASH = reference_runtime.FIT_PANEL_HASH

PARENT_CONSUMER = {
    "id": "H-LOMA-CBOGT-S8-MPCDAR-R16-IQ",
    "config": PARENT_CONFIG,
    "config_sha256": PARENT_CONFIG_SHA256,
    "review_contract": PARENT_REVIEW,
    "review_sha256": PARENT_REVIEW_SHA256,
    "implementation_module": PARENT_PRODUCER_MODULE,
    "implementation_module_sha256": PARENT_PRODUCER_MODULE_SHA256,
    "implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
    "validation": PARENT_VALIDATION,
    "validation_sha256": PARENT_VALIDATION_SHA256,
    "audit": PARENT_AUDIT,
    "audit_sha256": PARENT_AUDIT_SHA256,
    "compact_report": PARENT_COMPACT,
    "compact_report_sha256": PARENT_COMPACT_SHA256,
    "artifact_manifest": PARENT_ARTIFACT_MANIFEST,
    "artifact_manifest_sha256": PARENT_ARTIFACT_MANIFEST_SHA256,
    "artifact_payload": PARENT_ARTIFACT_PAYLOAD,
    "artifact_payload_sha256": PARENT_ARTIFACT_PAYLOAD_SHA256,
    "artifact_created_commit": PARENT_ARTIFACT_CREATED_COMMIT,
    "status": "passed",
    "classification": "frozen_mean_preserving_data_aware_rank16_integration_qualified",
}

ARTIFACT_CONTRACT = {
    "manifest_schema": candidate_runtime.MANIFEST_SCHEMA,
    "payload_schema": candidate_runtime.PAYLOAD_SCHEMA,
    "payload_file": "payload.pt",
    "rank": RANK,
    "feature_dim": FEATURE_DIM,
    "output_dim": OUTPUT_DIM,
    "target_shape": TARGET_SHAPE,
    "formula_version": candidate_runtime.FORMULA_VERSION,
    "source_dtype": "torch.float64",
    "target_dtype": "torch.float32",
    "dtype": "torch.float32",
    "device_storage": "cpu",
    "keys": ["mu_fit", "c_mean", "T16", "V16"],
    "tensor_shapes": {
        "mu_fit": [FEATURE_DIM],
        "c_mean": [OUTPUT_DIM],
        "T16": [FEATURE_DIM, RANK],
        "V16": [OUTPUT_DIM, RANK],
    },
    "scalar_count_total": SCALAR_COUNT_TOTAL,
    "raw_float32_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES,
    "cast_rule": candidate_runtime.CAST_RULE,
    "formula": "prediction = c_mean + ((z32 - mu_fit) @ T16) @ V16.T",
    "runtime_module": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_float32_readout.py",
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
    "feature_contract": candidate_runtime.FEATURE_CONTRACT,
    "feature_contract_version": candidate_runtime.FEATURE_CONTRACT_VERSION,
    "streaming": True,
    "microbatch_size": MICROBATCH_SIZE,
    "prediction_bank_saved": False,
    "pooled_feature_bank_saved": False,
    "target_bank_saved": False,
    "internal_stage_capture": False,
}

TOLERANCES = {
    "prediction_atol": FLOAT32_PREDICTION_ATOL,
    "prediction_rtol": FLOAT32_PREDICTION_RTOL,
    "normalized_l2_max": FLOAT32_NORMALIZED_L2_MAX,
    "probe_mse_absolute_drift_max": FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX,
    "probe_mse_relative_drift_max": FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX,
    "normalized_l2_denominator": "max(||prediction_f64||_2, 1.0e-30)",
    "per_sample_relative_mse_gate": False,
}

EVALUATION = {
    "rank": RANK,
    "candidate_family": [RANK],
    "search": False,
    "tested_ranks": [RANK],
    "no_rank_search": True,
    "no_refit": True,
    "no_training": True,
    "no_optimizer": True,
    "no_backward": True,
    "no_physical_forward_modification": True,
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "reference_runtime": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_readout.py",
    "candidate_runtime": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_float32_readout.py",
    "reference_dtype": "torch.float64",
    "candidate_dtype": "torch.float32",
    "readout_device": "cpu",
    "feature_stream": "existing_cmos_feature_producer",
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
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
    "decision": "approve_no_update_mpcdar_rank16_float32_numerical_qualification",
    "no_update_diagnostic_authorized": True,
    "integration_only": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "generation": False,
    "deployment_ready": False,
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
        "gpu_readout",
        "tf32",
        "amp",
        "mixed_precision",
        "float16",
        "bfloat16",
        "fallback_to_float64_candidate",
    ],
}

NEXT_ACTION_RESULT_REVIEW = "manual_review_MPCDAR_R16_F32_NQ_result"


class MPCDARRank16Float32HarnessFailure(RuntimeError):
    """A locked S8 input or S9 qualification boundary failed closed."""


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
        raise ValueError(f"S9 requires clean branch {BRANCH}")


def _contract_payload() -> dict[str, Any]:
    return {
        "parent": PARENT_CONSUMER,
        "artifact": ARTIFACT_CONTRACT,
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
        "panels",
        "tolerances",
        "evaluation",
        "resource",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("S9 config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("S9 config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    expected = {
        "formal_system_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        **_contract_payload(),
    }
    for name, value in expected.items():
        if spec.payload.get(name) != value:
            raise ValueError(f"S9 {name} contract is not locked")


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
        "deployment_ready": False,
    }


def _load_review(path: str | Path, spec: Spec) -> Path:
    del spec
    target = _resolve(path)
    if json.loads(target.read_text(encoding="utf-8")) != _review_payload():
        raise ValueError("S9 review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise MPCDARRank16Float32HarnessFailure(f"{label} is missing, stale, or tampered")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MPCDARRank16Float32HarnessFailure(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise MPCDARRank16Float32HarnessFailure(f"{label} is not a mapping")
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
        raise MPCDARRank16Float32HarnessFailure(f"{label} evidence record is invalid") from exc
    if (
        payload.get("status") != status
        or payload.get("classification") != classification
        or record.status != status
        or record.git_commit != git_commit
    ):
        raise MPCDARRank16Float32HarnessFailure(f"{label} terminal identity is stale")
    return record


def _verify_parent_evidence(spec: Spec) -> dict[str, Any]:
    parent = PARENT_CONSUMER
    config_path = _resolve(parent["config"], base_path=spec.path.parent)
    review_path = _resolve(parent["review_contract"], base_path=spec.path.parent)
    module_path = _resolve(parent["implementation_module"], base_path=spec.path.parent)
    if file_sha256(config_path) != parent["config_sha256"]:
        raise MPCDARRank16Float32HarnessFailure("S8 config identity changed")
    if file_sha256(review_path) != parent["review_sha256"]:
        raise MPCDARRank16Float32HarnessFailure("S8 review identity changed")
    if file_sha256(module_path) != parent["implementation_module_sha256"]:
        raise MPCDARRank16Float32HarnessFailure("S8 implementation identity changed")
    s8_spec = s8._load_spec(config_path)
    s8._assert_contract(s8_spec)
    s8._load_review(review_path, s8_spec)
    validation_path = _resolve(parent["validation"], base_path=spec.path.parent)
    audit_path = _resolve(parent["audit"], base_path=spec.path.parent)
    compact_path = _resolve(parent["compact_report"], base_path=spec.path.parent)
    validation = _read_json_hash(validation_path, parent["validation_sha256"], "S8 validation")
    _verify_evidence(
        validation,
        status="passed",
        classification=None,
        label="S8 validation",
        git_commit=PARENT_IMPLEMENTATION_COMMIT,
    )
    audit = _read_json_hash(audit_path, parent["audit_sha256"], "S8 audit")
    _verify_evidence(
        audit,
        status="passed",
        classification=parent["classification"],
        label="S8 audit",
        git_commit=PARENT_IMPLEMENTATION_COMMIT,
    )
    compact = _read_json_hash(compact_path, parent["compact_report_sha256"], "S8 compact report")
    if (
        compact.get("status") != "passed"
        or compact.get("classification") != parent["classification"]
        or compact.get("rank") != RANK
        or compact.get("tested_ranks") != [RANK]
        or compact.get("search_executed") is not False
    ):
        raise MPCDARRank16Float32HarnessFailure("S8 terminal identity changed")
    gate_vector = compact.get("gate_vector", {}).get("loaded")
    if not isinstance(gate_vector, Mapping):
        raise MPCDARRank16Float32HarnessFailure("S8 loaded gate vector is unavailable")
    manifest_path = _resolve(parent["artifact_manifest"], base_path=spec.path.parent)
    payload_path = _resolve(parent["artifact_payload"], base_path=spec.path.parent)
    if file_sha256(manifest_path) != parent["artifact_manifest_sha256"]:
        raise MPCDARRank16Float32HarnessFailure("S8 artifact manifest identity changed")
    if file_sha256(payload_path) != parent["artifact_payload_sha256"]:
        raise MPCDARRank16Float32HarnessFailure("S8 artifact payload identity changed")
    try:
        artifact = reference_runtime.load_frozen_mpcdar_rank16_readout(manifest_path)
    except (OSError, ValueError) as exc:
        raise MPCDARRank16Float32HarnessFailure("S8 artifact cannot be loaded") from exc
    if (
        artifact.manifest.get("created_git", {}).get("commit") != PARENT_ARTIFACT_CREATED_COMMIT
        or artifact.manifest.get("created_git", {}).get("branch") != BRANCH
        or artifact.manifest.get("created_git", {}).get("clean") is not True
    ):
        raise MPCDARRank16Float32HarnessFailure("S8 artifact creation identity changed")
    return {
        "spec": s8_spec,
        "config_path": config_path,
        "review_path": review_path,
        "module_path": module_path,
        "validation_path": validation_path,
        "audit_path": audit_path,
        "compact_path": compact_path,
        "validation": validation,
        "audit": audit,
        "compact": compact,
        "gate_vector": dict(gate_vector),
        "manifest_path": manifest_path,
        "payload_path": payload_path,
        "artifact": artifact,
    }


def _load_upstream(spec: Spec, dataset_root: str | Path) -> dict[str, Any]:
    parent = _verify_parent_evidence(spec)
    try:
        upstream = s8._load_upstream(parent["spec"], dataset_root)
    except (ValueError, RuntimeError, s8.MPCDARRank16HarnessFailure) as exc:
        raise MPCDARRank16Float32HarnessFailure("RMAL/LSDA upstream identity is invalid") from exc
    runtime = upstream["runtime"]
    probe = runtime.rmal_probe
    if (
        probe.artifact_sha256 != PROBE_SHA256
        or tuple(probe.weights.shape) != (FEATURE_DIM, OUTPUT_DIM)
        or tuple(probe.bias.shape) != (OUTPUT_DIM,)
        or probe.weights.dtype != torch.float32
        or probe.bias.dtype != torch.float32
    ):
        raise MPCDARRank16Float32HarnessFailure("frozen RMAL probe identity changed")
    data = runtime.data
    if (
        data.selection_indices.numel() != SELECTION_COUNT
        or data.confirmation_indices.numel() != CONFIRMATION_COUNT
        or data.manifest.get("manifest_hash") != mpcdar.lsda.DATASET_MANIFEST_SHA256
        or tensor_sha256(runtime.fit_indices) != FIT_PANEL_HASH
    ):
        raise MPCDARRank16Float32HarnessFailure("RMAL panel or dataset identity changed")
    return {"parent": parent, "runtime": runtime}


def _source_contract() -> dict[str, Any]:
    source = inspect.getsource(candidate_runtime)
    forbidden = {
        "explicit_float64_upcast": (".to(torch.float64)", ".double("),
        "research_dependency": ("application.research", "from ..research"),
        "linear_algebra_factorization": ("torch.linalg", "torch.svd", "torch.eigh"),
        "optimization_or_training": ("torch.optim", ".backward(", "optimizer", "training"),
        "full_operator_dependency": ("C_fit", "probes.pt", "W64", "load W"),
    }
    found = {
        name: [token for token in tokens if token in source]
        for name, tokens in forbidden.items()
    }
    return {
        "pure_float32_runtime_contract": all(not values for values in found.values()),
        "forbidden_tokens_found": found,
        "source_path": "src/ldm_project/v2/application/current/frozen_mpcdar_rank16_float32_readout.py",
    }


def _validate_float32_payload(payload: Mapping[str, torch.Tensor]) -> None:
    if set(payload) != candidate_runtime.PAYLOAD_KEYS:
        raise MPCDARRank16Float32HarnessFailure("S9 payload keys are invalid")
    scalar_count = 0
    raw_bytes = 0
    for name in candidate_runtime.PAYLOAD_KEYS:
        value = payload[name]
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise MPCDARRank16Float32HarnessFailure(f"S9 payload {name} must be CPU float32")
        if tuple(value.shape) != candidate_runtime.TENSOR_SHAPES[name]:
            raise MPCDARRank16Float32HarnessFailure(f"S9 payload {name} shape changed")
        if not bool(torch.isfinite(value).all()):
            raise MPCDARRank16Float32HarnessFailure(f"S9 payload {name} is non-finite")
        scalar_count += value.numel()
        raw_bytes += value.numel() * value.element_size()
    if scalar_count != SCALAR_COUNT_TOTAL or raw_bytes != RAW_FLOAT32_TENSOR_BYTES:
        raise MPCDARRank16Float32HarnessFailure("S9 payload scalar or raw-byte count changed")


def _exact_cast_identity(
    source: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    checks = {}
    for name in candidate_runtime.PAYLOAD_KEYS:
        expected = source[name].detach().cpu().contiguous().to(torch.float32)
        actual = candidate[name].detach().cpu().contiguous()
        checks[name] = {
            "source_dtype": str(source[name].dtype),
            "target_dtype": str(actual.dtype),
            "shape": list(actual.shape),
            "expected_tensor_sha256": tensor_sha256(expected),
            "actual_tensor_sha256": tensor_sha256(actual),
            "exact_equal": bool(torch.equal(actual, expected)),
        }
    return {"checks": checks, "passed": all(item["exact_equal"] for item in checks.values())}


def _manifest(
    payload: Mapping[str, torch.Tensor],
    payload_path: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": candidate_runtime.MANIFEST_SCHEMA,
        "payload_schema": candidate_runtime.PAYLOAD_SCHEMA,
        "repair_id": candidate_runtime.REPAIR_ID,
        "parent": candidate_runtime.PARENT_REPAIR_ID,
        "rank": RANK,
        "feature_dim": FEATURE_DIM,
        "output_dim": OUTPUT_DIM,
        "target_shape": TARGET_SHAPE,
        "formula_version": candidate_runtime.FORMULA_VERSION,
        "dtype": "torch.float32",
        "device_storage": "cpu",
        "feature_contract": candidate_runtime.FEATURE_CONTRACT,
        "feature_contract_version": candidate_runtime.FEATURE_CONTRACT_VERSION,
        "cast_rule": candidate_runtime.CAST_RULE,
        "source_dtype": "torch.float64",
        "target_dtype": "torch.float32",
        "parent_artifact": candidate_runtime._parent_contract(),
        "tensor_count": len(payload),
        "scalar_count_total": SCALAR_COUNT_TOTAL,
        "raw_float32_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES,
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


def _prepare_build_slot(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to reuse S9 artifact directory: {path}")
    if path.name != "artifact":
        raise ValueError("S9 artifact directory must be named artifact")


def _prepare_qualification_slot(artifact_dir: Path, output_path: Path) -> Path:
    if output_path.parent.resolve() != artifact_dir.parent.resolve():
        raise ValueError("S9 artifact and qualification evidence must share one attempt directory")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite S9 audit evidence: {output_path}")
    if not artifact_dir.is_dir():
        raise FileNotFoundError("S9 artifact directory is missing")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {artifact_dir.name}:
        raise FileExistsError("S9 qualification directory is not a fresh artifact-only slot")
    for name in ("validation.json", "compact_report.json"):
        if (output_path.parent / name).exists():
            raise FileExistsError("S9 qualification evidence already exists")
    manifest_path = artifact_dir / "manifest.json"
    payload_path = artifact_dir / "payload.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError("S9 artifact directory is incomplete")
    return manifest_path


def _input_hashes(
    spec: Spec,
    review_path: Path,
    upstream: Mapping[str, Any],
    parent: Mapping[str, Any],
    candidate_manifest: Path,
    candidate_payload: Path,
) -> dict[str, str]:
    return {
        "research_config": file_sha256(spec.path),
        "review_contract": file_sha256(review_path),
        "parent_config": PARENT_CONFIG_SHA256,
        "parent_review": PARENT_REVIEW_SHA256,
        "parent_implementation_module": PARENT_PRODUCER_MODULE_SHA256,
        "parent_validation": PARENT_VALIDATION_SHA256,
        "parent_audit": PARENT_AUDIT_SHA256,
        "parent_compact": PARENT_COMPACT_SHA256,
        "parent_artifact_manifest": file_sha256(parent["manifest_path"]),
        "parent_artifact_payload": file_sha256(parent["payload_path"]),
        "probe": PROBE_SHA256,
        "fit_panel": tensor_sha256(upstream["runtime"].fit_indices),
        "dataset_manifest": upstream["runtime"].data.manifest["manifest_hash"],
        "candidate_artifact_manifest": file_sha256(candidate_manifest),
        "candidate_artifact_payload": file_sha256(candidate_payload),
        "artifact_contract": canonical_sha256(ARTIFACT_CONTRACT),
        "tolerance_contract": canonical_sha256(TOLERANCES),
    }


def _context(spec: Spec, dataset_manifest: str) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "hypothesis": HYPOTHESIS,
            "parent": PARENT_CONSUMER,
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


def _build_payload(parent: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    payload = {
        name: parent[name].detach().cpu().contiguous().to(torch.float32)
        for name in candidate_runtime.PAYLOAD_KEYS
    }
    _validate_float32_payload(payload)
    return payload


def build(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    expected_parent_manifest = _resolve(PARENT_ARTIFACT_MANIFEST, base_path=spec.path.parent)
    requested_parent_manifest = _resolve(args.parent_manifest)
    if requested_parent_manifest != expected_parent_manifest:
        raise MPCDARRank16Float32HarnessFailure("S8 parent manifest path is not the locked path")
    parent = _verify_parent_evidence(spec)
    source = parent["artifact"]
    artifact_dir = _resolve(args.artifact_dir)
    _prepare_build_slot(artifact_dir)
    payload = _build_payload(
        {
            "mu_fit": source.mu_fit,
            "c_mean": source.c_mean,
            "T16": source.T16,
            "V16": source.V16,
        }
    )
    cast_identity = _exact_cast_identity(
        {
            "mu_fit": source.mu_fit,
            "c_mean": source.c_mean,
            "T16": source.T16,
            "V16": source.V16,
        },
        payload,
    )
    if not cast_identity["passed"]:
        raise MPCDARRank16Float32HarnessFailure("S9 deterministic cast identity failed")
    payload_path = artifact_dir / "payload.pt"
    manifest_path = artifact_dir / "manifest.json"
    atomic_torch_save_new(payload_path, payload, artifact_kind="S9 float32 payload")
    manifest = _manifest(payload, payload_path, state)
    candidate_runtime._validate_manifest(manifest)
    atomic_write_new_json(manifest_path, manifest, artifact_kind="S9 float32 manifest")
    loaded = candidate_runtime.load_frozen_mpcdar_rank16_float32_readout(manifest_path)
    loaded_identity = _exact_cast_identity(
        {
            "mu_fit": source.mu_fit,
            "c_mean": source.c_mean,
            "T16": source.T16,
            "V16": source.V16,
        },
        {
            "mu_fit": loaded.mu_fit,
            "c_mean": loaded.c_mean,
            "T16": loaded.T16,
            "V16": loaded.V16,
        },
    )
    if not loaded_identity["passed"]:
        raise MPCDARRank16Float32HarnessFailure("S9 fresh artifact reload cast identity failed")
    print(
        json.dumps(
            {
                "artifact": str(artifact_dir),
                "manifest": manifest,
                "cast_identity": cast_identity,
                "fresh_reload_identity": loaded_identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    del review_path
    return 0


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise MPCDARRank16Float32HarnessFailure("S9 formal qualification requires CUDA for CMOS replay")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise MPCDARRank16Float32HarnessFailure("S9 requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise MPCDARRank16Float32HarnessFailure("S9 requires at least 10 GiB free storage")
    return int(free), int(total)


def _load_model(upstream: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    runtime = upstream["runtime"]
    source = rmfa._load_source(runtime.rmal_consumer.bundle, load_checkpoints=True)
    formal: FormalSystemConfig = runtime.rmal_consumer.bundle.parent.parent.formal
    checkpoint = source.checkpoints[rmal.PRIMARY_CHECKPOINT]
    return mra._load_scaled_model(formal, checkpoint, device)


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().cpu().reshape(-1).to(torch.float64)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise MPCDARRank16Float32HarnessFailure("numerical summary received empty or non-finite values")
    return {
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "median": float(flat.median()),
        "p95": float(torch.quantile(flat, 0.95)),
        "p99": float(torch.quantile(flat, 0.99)),
    }


def _panel_report(
    model: torch.nn.Module,
    upstream: Mapping[str, Any],
    reference: reference_runtime.FrozenMPCDARRank16Readout,
    candidate: candidate_runtime.FrozenMPCDARRank16Float32Readout,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    expected_anchor: Mapping[str, float],
    panel: str,
) -> dict[str, Any]:
    runtime = upstream["runtime"]
    probe = runtime.rmal_probe
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
    per_sample_mse_diffs: list[torch.Tensor] = []
    max_abs = 0.0
    abs_sum = 0.0
    element_count = 0
    diff_square_sum = 0.0
    reference_square_sum = 0.0
    elementwise_passed = True
    max_elementwise_margin = float("-inf")
    sample_count = int(images.shape[0])
    for indices in torch.arange(sample_count, dtype=torch.long).split(MICROBATCH_SIZE):
        batch_images = images.index_select(0, indices)
        batch_labels = labels.index_select(0, indices)
        target = base.render_intensity_batch(batch_images).target.float()
        previous = images.index_select(0, (indices - 1) % sample_count)
        cyclic_target = base.render_intensity_batch(previous).target.float()
        physical = lrrc._capture_cmos(model, batch_images, device)
        features = mra._feature_projection(physical, complex_value=False).float()
        if features.device.type != "cpu" or features.dtype != torch.float32:
            raise MPCDARRank16Float32HarnessFailure("CMOS feature stream is not CPU float32 at readout boundary")
        prediction_f64 = reference.predict(features)
        prediction_f32 = candidate.predict(features)
        if prediction_f64.dtype != torch.float64 or prediction_f32.dtype != torch.float32:
            raise MPCDARRank16Float32HarnessFailure("readout dtype contract changed")
        difference = prediction_f32.to(torch.float64) - prediction_f64
        allowed = FLOAT32_PREDICTION_ATOL + FLOAT32_PREDICTION_RTOL * prediction_f64.abs()
        margin = difference.abs() - allowed
        batch_elementwise = bool((margin <= 0.0).all())
        elementwise_passed = elementwise_passed and batch_elementwise
        max_elementwise_margin = max(max_elementwise_margin, float(margin.max()))
        max_abs = max(max_abs, float(difference.abs().max()))
        abs_sum += float(difference.abs().sum())
        element_count += difference.numel()
        diff_square_sum += float(difference.square().sum())
        reference_square_sum += float(prediction_f64.square().sum())
        target64 = target.to(torch.float64)
        physical64 = physical.to(torch.float64)
        template = templates.index_select(0, batch_labels).to(torch.float64)
        cyclic64 = cyclic_target.to(torch.float64)
        candidate64 = prediction_f32.to(torch.float64)
        full = lrrc._full_prediction(features, probe.weights, probe.bias).to(torch.float64)
        bias_only = mpcdar._rank0_prediction(features, probe.bias)
        physical_errors.append((physical64 - target64).square().mean(dim=(1, 2, 3)))
        candidate_errors.append(
            (prediction_f32.reshape_as(target) - target).square().mean(dim=(1, 2, 3))
        )
        reference_errors.append(
            (prediction_f64.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3))
        )
        full_errors.append((full.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        cyclic_errors.append((physical64 - cyclic64).square().mean(dim=(1, 2, 3)))
        template_errors.append((physical64 - template).square().mean(dim=(1, 2, 3)))
        bias_errors.append((bias_only.reshape_as(target64) - target64).square().mean(dim=(1, 2, 3)))
        per_sample_mse_diffs.append(
            (candidate_errors[-1].to(torch.float64) - reference_errors[-1]).abs()
        )
        candidate_compact.append(
            mra._feature_projection(prediction_f32.reshape(-1, 1, 200, 200)).float()
        )
        reference_compact.append(
            mra._feature_projection(prediction_f64.reshape(-1, 1, 200, 200).float()).float()
        )
        target_compact.append(mra._feature_projection(target).float())
        del candidate64
    physical_mse = torch.cat(physical_errors)
    candidate_mse = torch.cat(candidate_errors)
    reference_mse = torch.cat(reference_errors)
    full_mse = torch.cat(full_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    bias_mse = torch.cat(bias_errors)
    mse_diffs = torch.cat(per_sample_mse_diffs)
    physical_value = float(physical_mse.mean())
    candidate_value = float(candidate_mse.to(torch.float64).mean())
    reference_value = float(reference_mse.mean())
    full_value = float(full_mse.mean())
    absolute_mse_drift = abs(candidate_value - reference_value)
    relative_mse_drift = absolute_mse_drift / max(abs(reference_value), 1.0e-30)
    normalized_l2 = (diff_square_sum**0.5) / max(reference_square_sum**0.5, 1.0e-30)
    delta = physical_mse - candidate_mse.to(torch.float64)
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
    candidate_context = mpcdar._context_metrics(
        torch.cat(candidate_compact), torch.cat(target_compact), labels
    )
    reference_context = mpcdar._context_metrics(
        torch.cat(reference_compact), torch.cat(target_compact), labels
    )
    numerical = {
        "max_absolute_prediction_difference": max_abs,
        "mean_absolute_prediction_difference": abs_sum / max(element_count, 1),
        "rms_prediction_difference": (diff_square_sum / max(element_count, 1)) ** 0.5,
        "normalized_l2": normalized_l2,
        "float64_aggregate_mse": reference_value,
        "float32_aggregate_mse": candidate_value,
        "absolute_aggregate_mse_drift": absolute_mse_drift,
        "relative_aggregate_mse_drift": relative_mse_drift,
        "elementwise": {
            "atol": FLOAT32_PREDICTION_ATOL,
            "rtol": FLOAT32_PREDICTION_RTOL,
            "max_margin": max_elementwise_margin,
            "passed": elementwise_passed,
        },
        "normalized_l2_gate": {
            "maximum": FLOAT32_NORMALIZED_L2_MAX,
            "passed": normalized_l2 <= FLOAT32_NORMALIZED_L2_MAX,
        },
        "aggregate_mse_absolute_gate": {
            "maximum": FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX,
            "passed": absolute_mse_drift <= FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX,
        },
        "aggregate_mse_relative_gate": {
            "maximum": FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX,
            "passed": relative_mse_drift <= FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX,
        },
        "per_sample_mse_absolute_difference": _summary(mse_diffs),
        "passed": bool(
            elementwise_passed
            and normalized_l2 <= FLOAT32_NORMALIZED_L2_MAX
            and absolute_mse_drift <= FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX
            and relative_mse_drift <= FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX
        ),
        "prediction_bank_saved": False,
    }
    report: dict[str, Any] = {
        "executed": True,
        "panel": panel,
        "rank": RANK,
        "evaluation_count": sample_count,
        "readout_device": "cpu",
        "readout_dtypes": {"feature": "torch.float32", "reference": "torch.float64", "candidate": "torch.float32"},
        "physical": physical_values,
        "probe": {
            "paired_mse": candidate_value,
            "mse_by_label": mpcdar._mse_by_label(candidate_mse.to(torch.float64), labels),
            **candidate_context,
        },
        "reference_probe": {
            "paired_mse": reference_value,
            "mse_by_label": mpcdar._mse_by_label(reference_mse, labels),
            **reference_context,
        },
        "class_template_mse": physical_values["class_template_mse"],
        "cyclic_mse": physical_values["cyclic_mse"],
        "relative_improvement": float(delta.mean() / max(physical_value, 1.0e-12)),
        "bootstrap": {
            "delta_definition": "physical_paired_mse - candidate_float32_paired_mse",
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
        "dynamic_only": {"paired_mse": candidate_value},
        "bias_only": {"paired_mse": float(bias_mse.mean())},
        "numerical": numerical,
        "fresh_filesystem_reload": True,
        "same_feature_stream": True,
        "pooled_feature_bank_saved": False,
        "target_bank_saved": False,
        "prediction_bank_saved": False,
    }
    report["selection_gate"] = rmal._panel_gate(report, require_bootstrap=False)
    if panel == "confirmation":
        report["confirmation_gate"] = rmal._panel_gate(report, require_bootstrap=True)
    return report


def _gate_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return actual.get("passed") == expected.get("passed") and actual.get("checks") == expected.get("checks")


def _static_gates(
    spec: Spec,
    review_path: Path,
    state: Mapping[str, Any],
    upstream: Mapping[str, Any],
    parent: Mapping[str, Any],
    artifact: candidate_runtime.FrozenMPCDARRank16Float32Readout,
    artifact_dir: Path,
    output_path: Path,
) -> tuple[GateResult, ...]:
    manifest = artifact.manifest
    source = parent["artifact"]
    cast = _exact_cast_identity(
        {"mu_fit": source.mu_fit, "c_mean": source.c_mean, "T16": source.T16, "V16": source.V16},
        {"mu_fit": artifact.mu_fit, "c_mean": artifact.c_mean, "T16": artifact.T16, "V16": artifact.V16},
    )
    source_contract = _source_contract()
    return (
        GateResult("clean_registered_branch", state.get("branch") == BRANCH and state.get("clean") is True, {"branch": state.get("branch"), "clean": state.get("clean")}),
        GateResult("lineage_start_head_recorded", LINEAGE_START_HEAD == "89545909f1cfb33f31b087a735cd5babdd960e8c", {"lineage_start_head": LINEAGE_START_HEAD}),
        GateResult("parent_s8_terminal_identity", parent["compact"].get("status") == "passed" and parent["compact"].get("rank") == RANK, {}),
        GateResult("parent_s8_artifact_identity", file_sha256(parent["manifest_path"]) == PARENT_ARTIFACT_MANIFEST_SHA256 and file_sha256(parent["payload_path"]) == PARENT_ARTIFACT_PAYLOAD_SHA256, {}),
        GateResult("s9_manifest_identity", manifest.get("repair_id") == candidate_runtime.REPAIR_ID and manifest.get("parent") == candidate_runtime.PARENT_REPAIR_ID and manifest.get("rank") == RANK and manifest.get("dtype") == "torch.float32", {}),
        GateResult("s9_artifact_scalar_and_dtype_contract", manifest.get("tensor_count") == 4 and manifest.get("scalar_count_total") == SCALAR_COUNT_TOTAL and manifest.get("raw_float32_tensor_bytes") == RAW_FLOAT32_TENSOR_BYTES and all(value.dtype == torch.float32 and value.device.type == "cpu" for value in (artifact.mu_fit, artifact.c_mean, artifact.T16, artifact.V16)), {"scalar_count_total": SCALAR_COUNT_TOTAL, "raw_float32_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES}),
        GateResult("exact_deterministic_cast_identity", cast["passed"], cast),
        GateResult("artifact_created_from_current_clean_commit", manifest.get("created_git", {}).get("commit") == state.get("commit") and manifest.get("created_git", {}).get("branch") == BRANCH and manifest.get("created_git", {}).get("clean") is True, {}),
        GateResult("registered_panel_identity", upstream["runtime"].data.selection_indices.numel() == SELECTION_COUNT and upstream["runtime"].data.confirmation_indices.numel() == CONFIRMATION_COUNT and tensor_sha256(upstream["runtime"].fit_indices) == FIT_PANEL_HASH and upstream["runtime"].rmal_probe.artifact_sha256 == PROBE_SHA256, {}),
        GateResult("rank16_only_no_search", EVALUATION["candidate_family"] == [RANK] and EVALUATION["tested_ranks"] == [RANK] and EVALUATION["search"] is False, {"rank": RANK}),
        GateResult("fresh_filesystem_reload", artifact.manifest_path.is_file() and artifact.manifest_path.parent == artifact_dir.resolve(), {"manifest_path": str(artifact.manifest_path)}),
        GateResult("cpu_readout_boundary", EVALUATION["readout_device"] == "cpu" and artifact.mu_fit.device.type == "cpu", {}),
        GateResult("pure_float32_runtime_contract", source_contract["pure_float32_runtime_contract"], source_contract),
        GateResult("no_update_scope", AUTHORIZATION["training_authorized"] is False and EVALUATION["no_training"] is True and EVALUATION["no_refit"] is True and EVALUATION["no_physical_forward_modification"] is True, {}),
        GateResult("qualification_slot_fresh", output_path.parent.exists() and output_path.parent.resolve() == artifact_dir.parent.resolve() and not output_path.exists(), {"path": str(output_path.parent), "review_sha256": file_sha256(review_path)}),
    )


def _qualification_gates(
    selection: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    parent_gate_vector: Mapping[str, Any],
    state_unchanged: bool,
    ols_unchanged: bool,
    resource_passed: bool,
) -> tuple[GateResult, ...]:
    selection_num = selection["numerical"]
    confirmation_num = confirmation["numerical"]
    return (
        GateResult("selection_elementwise_drift", selection_num["elementwise"]["passed"], selection_num["elementwise"]),
        GateResult("selection_normalized_l2", selection_num["normalized_l2_gate"]["passed"], selection_num["normalized_l2_gate"]),
        GateResult("selection_mse_absolute_drift", selection_num["aggregate_mse_absolute_gate"]["passed"], selection_num["aggregate_mse_absolute_gate"]),
        GateResult("selection_mse_relative_drift", selection_num["aggregate_mse_relative_gate"]["passed"], selection_num["aggregate_mse_relative_gate"]),
        GateResult("confirmation_elementwise_drift", confirmation_num["elementwise"]["passed"], confirmation_num["elementwise"]),
        GateResult("confirmation_normalized_l2", confirmation_num["normalized_l2_gate"]["passed"], confirmation_num["normalized_l2_gate"]),
        GateResult("confirmation_mse_absolute_drift", confirmation_num["aggregate_mse_absolute_gate"]["passed"], confirmation_num["aggregate_mse_absolute_gate"]),
        GateResult("confirmation_mse_relative_drift", confirmation_num["aggregate_mse_relative_gate"]["passed"], confirmation_num["aggregate_mse_relative_gate"]),
        GateResult("selection_gate_vector_identity", _gate_equal(selection["selection_gate"], parent_gate_vector["selection"]), {}),
        GateResult("confirmation_selection_gate_vector_identity", _gate_equal(confirmation["selection_gate"], parent_gate_vector["confirmation_selection"]), {}),
        GateResult("confirmation_gate_vector_identity", _gate_equal(confirmation["confirmation_gate"], parent_gate_vector["confirmation"]), {}),
        GateResult("selection_gate_passed", bool(selection["selection_gate"].get("passed")), selection["selection_gate"].get("checks", {})),
        GateResult("confirmation_selection_gate_passed", bool(confirmation["selection_gate"].get("passed")), confirmation["selection_gate"].get("checks", {})),
        GateResult("confirmation_gate_passed", bool(confirmation["confirmation_gate"].get("passed")), confirmation["confirmation_gate"].get("checks", {})),
        GateResult("confirmation_bootstrap_passed", bool(confirmation["confirmation_gate"].get("checks", {}).get("bootstrap_lower_bound")), {}),
        GateResult("model_state_unchanged", state_unchanged and ols_unchanged, {}),
        GateResult("streaming_no_prediction_bank", all(report.get("prediction_bank_saved") is False and report.get("pooled_feature_bank_saved") is False and report.get("target_bank_saved") is False for report in (selection, confirmation)), {}),
        GateResult("cpu_readout_execution", selection.get("readout_device") == "cpu" and confirmation.get("readout_device") == "cpu" and selection.get("readout_dtypes", {}).get("candidate") == "torch.float32" and confirmation.get("readout_dtypes", {}).get("reference") == "torch.float64", {}),
        GateResult("runtime_independence_contract", True, {"W_required": False, "b_required": False, "C_fit_required": False, "dataset_runtime": False, "probe_runtime": False, "research_imports": False}),
        GateResult("resource_limits", resource_passed, {}),
    )


def _json_report(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(value)


def qualify(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("S9 qualification requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    artifact_dir = _resolve(args.artifact_dir)
    output_path = _resolve(args.output)
    manifest_path = _prepare_qualification_slot(artifact_dir, output_path)
    parent = _verify_parent_evidence(spec)
    try:
        artifact = candidate_runtime.load_frozen_mpcdar_rank16_float32_readout(manifest_path)
    except (OSError, ValueError) as exc:
        raise MPCDARRank16Float32HarnessFailure("S9 candidate artifact cannot be freshly reloaded") from exc
    if artifact.manifest["created_git"]["commit"] != state["commit"]:
        raise MPCDARRank16Float32HarnessFailure("S9 artifact was not built from the current implementation commit")
    upstream = _load_upstream(spec, args.dataset_root)
    dataset_manifest = upstream["runtime"].data.manifest["manifest_hash"]
    static_gates = _static_gates(spec, review_path, state, upstream, parent, artifact, artifact_dir, output_path)
    static_valid = all(gate.passed for gate in static_gates)
    hashes = _input_hashes(spec, review_path, upstream, parent, manifest_path, artifact_dir / "payload.pt")
    validation_path = output_path.parent / "validation.json"
    validation = _record(
        VALIDATION_SCHEMA,
        "passed" if static_valid else "invalid_harness",
        None if static_valid else "invalid_harness",
        spec,
        state,
        dataset_manifest,
        hashes,
        static_gates,
        (NEXT_ACTION_RESULT_REVIEW,) if static_valid else (),
        {"parent": PARENT_CONSUMER, "artifact": ARTIFACT_CONTRACT, "panels": PANEL_CONTRACT, "tolerances": TOLERANCES, "evaluation": EVALUATION, "resource": RESOURCE_CONTRACT, "authorization": AUTHORIZATION},
        None if static_valid else "S9 static validation failed",
    )
    _write_new(validation_path, validation, artifact_kind="S9 validation evidence")
    if not static_valid:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 2
    device = torch.device(args.device)
    free_bytes, total_bytes = _resource_preflight(output_path, device)
    model = _load_model(upstream, device)
    state_before = rmal._state_cpu(model)
    ols_before = rmal._state_cpu(model.ols)
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        runtime = upstream["runtime"]
        selection_anchor, _ = mpcdar._expected_anchor(runtime.rmal_identity["compact"], "selection")
        confirmation_anchor, _ = mpcdar._expected_anchor(runtime.rmal_identity["compact"], "confirmation")
        selection_images = runtime.data.test_images.index_select(0, runtime.data.selection_indices)
        selection_labels = runtime.data.test_labels.index_select(0, runtime.data.selection_indices)
        confirmation_images = runtime.data.images.index_select(0, runtime.data.confirmation_indices)
        confirmation_labels = runtime.data.labels.index_select(0, runtime.data.confirmation_indices)
        reference = parent["artifact"]
        candidate = artifact
        selection = _panel_report(model, upstream, reference, candidate, selection_images, selection_labels, device, selection_anchor, "selection")
        confirmation = _panel_report(model, upstream, reference, candidate, confirmation_images, confirmation_labels, device, confirmation_anchor, "confirmation")
        confirmation["selection_gate"] = rmal._panel_gate(confirmation, require_bootstrap=False)
        confirmation["confirmation_gate"] = rmal._panel_gate(confirmation, require_bootstrap=True)
        state_unchanged = rmal._state_equal(state_before, rmal._state_cpu(model))
        ols_unchanged = rmal._state_equal(ols_before, rmal._state_cpu(model.ols))
        gates = _qualification_gates(selection, confirmation, parent["gate_vector"], state_unchanged, ols_unchanged, int(torch.cuda.max_memory_allocated(device)) <= MAX_PEAK_CUDA_BYTES)
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        numerical_pass = all(gate.passed for gate in gates[:8])
        scientific_pass = all(gate.passed for gate in gates[8:15])
        harness_pass = all(gate.passed for gate in gates[15:]) and all(gate.passed for gate in static_gates)
        if not harness_pass:
            status = "invalid_harness"
            classification = "invalid_harness"
        elif not numerical_pass or not scientific_pass:
            status = "valid_failed"
            classification = "valid_failed"
        else:
            status = "passed"
            classification = "frozen_mean_preserving_data_aware_rank16_float32_numerically_qualified"
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": status,
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW if status == "passed" else None,
            "integration_only": True,
            "parent": PARENT_CONSUMER,
            "rank": RANK,
            "tested_ranks": [RANK],
            "candidate_family": [RANK],
            "search_executed": False,
            "candidate_family_mutated": False,
            "selection": _json_report(selection),
            "confirmation": _json_report(confirmation),
            "gate_vector": {"s8_float64": parent["gate_vector"], "s9_float32": {"selection": selection["selection_gate"], "confirmation_selection": confirmation["selection_gate"], "confirmation": confirmation["confirmation_gate"]}},
            "gate_results": [gate.to_dict() for gate in gates],
            "static_gate_results": [gate.to_dict() for gate in static_gates],
            "artifact": {"manifest_sha256": file_sha256(manifest_path), "payload_sha256": file_sha256(artifact_dir / "payload.pt"), "raw_float32_tensor_bytes": RAW_FLOAT32_TENSOR_BYTES, "scalar_count_total": SCALAR_COUNT_TOTAL},
            "cast_identity": _exact_cast_identity({"mu_fit": reference.mu_fit, "c_mean": reference.c_mean, "T16": reference.T16, "V16": reference.V16}, {"mu_fit": candidate.mu_fit, "c_mean": candidate.c_mean, "T16": candidate.T16, "V16": candidate.V16}),
            "runtime_contract": _source_contract(),
            "evaluation": EVALUATION,
            "panels": PANEL_CONTRACT,
            "tolerances": TOLERANCES,
            "resource": {"free_cuda_bytes_before": free_bytes, "total_cuda_bytes": total_bytes, "peak_memory_bytes": peak, "wall_clock_seconds": elapsed, "readout_device": "cpu", "prediction_bank_saved": False, "pooled_feature_bank_saved": False, "target_bank_saved": False},
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
        _write_new(compact_path, compact, artifact_kind="S9 compact report")
        hashes["compact_report"] = file_sha256(compact_path)
        audit_contract = {
            "parent": PARENT_CONSUMER,
            "artifact": ARTIFACT_CONTRACT,
            "cast_identity": compact["cast_identity"],
            "runtime_contract": compact["runtime_contract"],
            "selection": selection,
            "confirmation": confirmation,
            "gate_vector": compact["gate_vector"],
            "gate_results": compact["gate_results"],
            "status_mapping": {"parent_or_artifact_mismatch": "invalid_harness", "numerical_or_rmal_gate_failure": "valid_failed"},
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
            tuple(static_gates) + tuple(gates),
            (NEXT_ACTION_RESULT_REVIEW,) if status == "passed" else (),
            audit_contract,
            None if status == "passed" else ("S9 harness mismatch" if status == "invalid_harness" else "S9 numerical or inherited RMAL Gate failed"),
        )
        audit["integration_only"] = True
        audit["cmos_only_replay_executed"] = True
        _write_new(output_path, audit, artifact_kind="S9 audit evidence")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if status == "passed" else 2
    except (MPCDARRank16Float32HarnessFailure, mpcdar.MPCDARHarnessFailure) as exc:
        if not output_path.exists():
            hashes["failure"] = canonical_sha256(str(exc))
            audit = _record(
                AUDIT_SCHEMA,
                "invalid_harness",
                "invalid_harness",
                spec,
                state,
                dataset_manifest,
                hashes,
                (GateResult("qualification_harness", False, {"reason": str(exc)}),),
                (),
                {"parent": PARENT_CONSUMER, "artifact": ARTIFACT_CONTRACT},
                str(exc),
            )
            _write_new(output_path, audit, artifact_kind="S9 invalid-harness audit evidence")
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-mpcdar-rank16-float32-numerical-qualification-build":
        return build(args)
    if args.research_action == "scaled-chen-mpcdar-rank16-float32-numerical-qualification-qualify":
        return qualify(args)
    raise ValueError(f"unsupported S9 action: {args.research_action}")


__all__ = [
    "ARTIFACT_CONTRACT",
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "COMPACT_SCHEMA",
    "CONFIG_SCHEMA",
    "EVALUATION",
    "FLOAT32_NORMALIZED_L2_MAX",
    "FLOAT32_PREDICTION_ATOL",
    "FLOAT32_PREDICTION_RTOL",
    "FLOAT32_PROBE_MSE_ABSOLUTE_DRIFT_MAX",
    "FLOAT32_PROBE_MSE_RELATIVE_DRIFT_MAX",
    "HYPOTHESIS",
    "MPCDARRank16Float32HarnessFailure",
    "NEXT_ACTION_RESULT_REVIEW",
    "PARENT_CONSUMER",
    "PANEL_CONTRACT",
    "RANK",
    "REVIEW_SCHEMA",
    "TOLERANCES",
    "VALIDATION_SCHEMA",
    "build",
    "dispatch",
    "qualify",
]
