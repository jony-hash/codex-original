"""Run the frozen S3 controller-to-OLS writability audit.

The audit is a no-update diagnostic consumer.  It loads the registered
controller generator, sends a fixed label-major code panel through the frozen
Scaled-Chen encoder prefix, and stops at the complex OLS coefficients.  The
decoder, relay, camera, OSERA evaluator, optimizer, training, and checkpoint
writers are intentionally outside this boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt as bogt,
)
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_manifold_readiness as mra,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.checkpoints import (
    state_content_hash,
    state_signature,
)
from ldm_project.v2.student.identity_generator import (
    IdentityAnchoredConditionalAmplitudeGenerator,
)
from scripts.research import audit_scaled_chen_ols_semantic_representation as s1
from scripts.research import replay_scaled_chen_repaired_coordinate_mra as rcr


TASK_ID = "H-LOMA-GEN-S3-COWA"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_controller_to_ols_writability_audit_v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_controller_to_ols_writability_audit.v1"
CONTROLLER_CONTRACT_SCHEMA = "lightgen_scaled_chen_controller_contract.v1"
PANEL_SCHEMA = "lightgen_scaled_chen_controller_panel_identity.v1"
INTERFACE_SCHEMA = "lightgen_scaled_chen_controller_ols_interface.v1"
TARGETING_SCHEMA = "lightgen_scaled_chen_label_targeting_metrics.v1"
REACHABILITY_SCHEMA = "lightgen_scaled_chen_absolute_reachability_metrics.v1"
CODE_EFFECT_SCHEMA = "lightgen_scaled_chen_code_effect_metrics.v1"
JACOBIAN_SCHEMA = "lightgen_scaled_chen_jacobian_metrics.v1"
PHASE_SCHEMA = "lightgen_scaled_chen_phase_diagnostics.v1"
ISOLATION_SCHEMA = "lightgen_scaled_chen_s3_state_isolation.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_controller_ols_compact_report.v1"

CONTROLLER_AUTHORITY_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_controller_state_authority_v1_attempt_01/"
    "controller_state_authority.json"
)
CONTROLLER_AUTHORITY_SHA256 = (
    "57942dcd4a425960b637bff99299dd6e1260198a22456fa5ef1ca133600302a8"
)
CONTROLLER_AUTHORITY_SELF_HASH = (
    "82a7d7d843f0928daf4f172c632ed3fe5fb79f4cadcc452dd4ebf8a2584f2f0d"
)
MIGRATED_CONTROLLER_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_controller_lineage_migration_v1_attempt_01/"
    "conditional_optical_generator_current_lineage_migrated.pt"
)
MIGRATED_CONTROLLER_SHA256 = (
    "572116dbb88464d0999a0391e7e94675cacd64ad5ae303bc2e0efb16e1f63ba2"
)
CONTROLLER_STATE_HASH = (
    "d36165ee1d5e9c2a66734ffdea0f8291cb8699e86df667e6a66ac169cd75e2fc"
)
CONTROLLER_STATE_SIGNATURE = (
    "88d31ec03d88a7e471d651b49c6eb20770e0fd6db5afb71798486499f62068fa"
)
GENERATOR_CONTRACT_HASH = (
    "3c9855ff473de172f704d83b00637c09a05da230be4d50cd6807d86539f49e11"
)
CONTROLLER_CONFIG_PATH = Path(
    "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml"
)
CONTROLLER_CONFIG_SHA256 = (
    "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
)
CONTROLLER_CONFIG_CANONICAL_SHA256 = (
    "edeefb801b563a4657cd621c86c725ed428bbd71678cddafb93fb629eb6ee030"
)

CHECKPOINT_PATH = Path(
    "outputs/20260829_v2/runs/"
    "lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/"
    "checkpoints/scaled_chen_horizon_step_5000.pt"
)
CHECKPOINT_SHA256 = (
    "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
)
MODE_BANK_SHA256 = (
    "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
)
CALIBRATION_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/"
    "repaired_calibration.pt"
)
CALIBRATION_SHA256 = (
    "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
)
CALIBRATION_MANIFEST_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/"
    "calibration_manifest.json"
)
CALIBRATION_MANIFEST_SHA256 = (
    "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
)

S1_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml"
)
S1_CONFIG_SHA256 = (
    "60d57cd134a567c17602775ec16f11fff05f56ffe900ac730774abde3432c521"
)
S1_CONFIG_CANONICAL_SHA256 = (
    "719526b8d6e82da20bdc9040314084b1192ce376d413eecdf0db07420ad2ab13"
)
S1_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01"
)
S1_MANIFEST_SHA256 = (
    "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
)
S1_IDENTITY_SHA256 = (
    "522ff5b00770ee34fe7b61b1f414775abe3310f398cc0fdf55049b30be2925e2"
)
S1_CONTRACT_SHA256 = (
    "f99c2ced713005fe3f4a5a0fa91a2cd7648f5647e2360f122bc68e77f225f10d"
)
S1_EMBEDDINGS_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
S1_DATASET_MANIFEST_SHA256 = (
    "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
)
S1_FIT_LABELS_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
S1_SELECTION_LABELS_SHA256 = (
    "31fce460255bda46afc2bd6b6777793873e2e6ee33e23a681def3ab8af6693cb"
)
S1_CONFIRMATION_LABELS_SHA256 = (
    "b05112caa3014d6c2081423b371361eefe1ec7a9f531e54db016b89bfbb77b18"
)

PORT_COUNT = 100
CODE_DIM = 8
CLASS_COUNT = 10
PANEL_CODES = 100
PANEL_PER_CLASS = 100
PANEL_SELECTION_PER_CLASS = 50
PANEL_CONFIRMATION_PER_CLASS = 50
PANEL_COUNT = CLASS_COUNT * PANEL_PER_CLASS
PANEL_SEED = 1234
NULL_SEED = 2053
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
EFFECTIVE_RANK_RELATIVE_TOLERANCE = 1.0e-6
DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_controller_to_ols_writability_audit_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_writability_audit_v1_attempt_01"
)

R1_POWER = "R1_POWER"
REQUIRED_AUTHORITY_GATE = "CONTROLLER_STATE_AUTHORITY_PASS"
PASS_CLASSIFICATION = "S3_CONTROL_TO_OLS_WRITABILITY_SUPPORTED"
TARGETING_ONLY_CLASSIFICATION = "S3_LABEL_DIRECTION_SUPPORTED_BUT_OLS_MANIFOLD_NOT_REACHED"
REACHABILITY_ONLY_CLASSIFICATION = "S3_OLS_REACHABLE_BUT_LABEL_SEMANTICS_NOT_CONTROLLED"
COLLAPSED_CLASSIFICATION = "S3_LABEL_WRITABILITY_SUPPORTED_CODE8_COLLAPSED"
NOT_SUPPORTED_CLASSIFICATION = "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED"
INTERFACE_UNRESOLVED_CLASSIFICATION = "S3_CONTROLLER_TO_SEMANTIC_OLS_INTERFACE_UNRESOLVED"
UNDERDETERMINED_CLASSIFICATION = "S3_UNDERDETERMINED"


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal SHA-256") from exc
    return value.lower()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return payload


def validate_amplitude_output(amplitude: torch.Tensor) -> torch.Tensor:
    """Validate the public controller-to-SLM amplitude boundary."""
    if (
        not torch.is_tensor(amplitude)
        or amplitude.ndim != 4
        or tuple(amplitude.shape[1:]) != (1, 200, 200)
        or torch.is_complex(amplitude)
        or not torch.is_floating_point(amplitude)
        or not torch.isfinite(amplitude).all()
        or bool(amplitude.min() < 0.0)
        or bool(amplitude.max() > 1.0)
    ):
        raise ValueError("amplitude output must be finite real [N,1,200,200] in [0,1]")
    return amplitude


def validate_power_observer(
    coefficients: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply the registered S1 R1 power observer without a new mapping."""
    if (
        not torch.is_tensor(coefficients)
        or coefficients.ndim != 2
        or coefficients.shape[1] != PORT_COUNT
        or not torch.is_complex(coefficients)
        or not torch.isfinite(coefficients).all()
    ):
        raise ValueError("coefficients must be finite complex [N,100]")
    if (
        not torch.is_tensor(scale)
        or scale.ndim != 1
        or scale.shape[0] != PORT_COUNT
        or not torch.is_floating_point(scale)
        or not torch.isfinite(scale).all()
        or bool((scale <= 0).any())
    ):
        raise ValueError("scale must be finite positive floating [100]")
    scale_work = scale.to(coefficients.device, coefficients.real.dtype)
    return coefficients.abs().square() / scale_work.unsqueeze(0)


def build_controller_panel() -> dict[str, torch.Tensor]:
    """Materialize the preregistered 100-code, label-major S3 panel."""
    generator = torch.Generator(device="cpu").manual_seed(PANEL_SEED)
    base_codes = torch.rand(
        (PANEL_CODES, CODE_DIM), generator=generator, dtype=torch.float32
    ).mul(2.0).sub(1.0)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(
        PANEL_PER_CLASS
    )
    codes = base_codes.repeat(CLASS_COUNT, 1)
    split = torch.cat(
        [
            torch.full((PANEL_SELECTION_PER_CLASS,), 0, dtype=torch.long),
            torch.full((PANEL_CONFIRMATION_PER_CLASS,), 1, dtype=torch.long),
        ]
    ).repeat(CLASS_COUNT)
    if labels.shape != (PANEL_COUNT,) or codes.shape != (PANEL_COUNT, CODE_DIM):
        raise ValueError("controller panel shape is invalid")
    if not torch.equal(split[:50], torch.zeros(50, dtype=torch.long)):
        raise ValueError("controller panel split is not class-major")
    return {
        "base_codes": base_codes.contiguous(),
        "codes": codes.contiguous(),
        "labels": labels.contiguous(),
        "split": split.contiguous(),
    }


def panel_identity(panel: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    required = {"base_codes", "codes", "labels", "split"}
    if set(panel) != required:
        raise ValueError("controller panel fields are invalid")
    for name in required:
        if not torch.is_tensor(panel[name]):
            raise ValueError(f"controller panel field is not a tensor: {name}")
    identity = {
        "schema": PANEL_SCHEMA,
        "seed": PANEL_SEED,
        "distribution": "uniform[-1,1]^8",
        "base_code_count": PANEL_CODES,
        "code_dim": CODE_DIM,
        "class_count": CLASS_COUNT,
        "states_per_class": PANEL_PER_CLASS,
        "selection_per_class": PANEL_SELECTION_PER_CLASS,
        "confirmation_per_class": PANEL_CONFIRMATION_PER_CLASS,
        "total_state_count": PANEL_COUNT,
        "order": "label_major",
        "same_base_codes_across_labels": True,
        "base_codes": {
            "shape": list(panel["base_codes"].shape),
            "dtype": str(panel["base_codes"].dtype),
            "sha256": tensor_sha256(panel["base_codes"]),
        },
        "codes": {
            "shape": list(panel["codes"].shape),
            "dtype": str(panel["codes"].dtype),
            "sha256": tensor_sha256(panel["codes"]),
        },
        "labels": {
            "shape": list(panel["labels"].shape),
            "dtype": str(panel["labels"].dtype),
            "sha256": tensor_sha256(panel["labels"]),
        },
        "split": {
            "shape": list(panel["split"].shape),
            "dtype": str(panel["split"].dtype),
            "sha256": tensor_sha256(panel["split"]),
        },
    }
    identity["panel_identity_sha256"] = canonical_sha256(identity)
    return identity


def fixed_label_permutation(
    labels: torch.Tensor, *, seed: int = NULL_SEED
) -> torch.Tensor:
    if (
        labels.ndim != 1
        or labels.numel() == 0
        or labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
    ):
        raise ValueError("null labels must be a non-empty integer vector")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return labels.index_select(0, torch.randperm(labels.numel(), generator=generator))


def classify_rank(singular_values: torch.Tensor) -> str:
    if (
        not torch.is_tensor(singular_values)
        or singular_values.ndim != 1
        or singular_values.numel() == 0
        or not torch.is_floating_point(singular_values)
        or not torch.isfinite(singular_values).all()
        or bool((singular_values < 0).any())
    ):
        raise ValueError("singular values must be finite nonnegative vector")
    leading = float(singular_values.max())
    if leading <= 0.0:
        return "CODE8_LOCAL_CONTROL_DEGENERATE"
    rank = int(
        (singular_values / leading >= EFFECTIVE_RANK_RELATIVE_TOLERANCE).sum()
    )
    if rank >= CODE_DIM:
        return "CODE8_LOCAL_CONTROL_FULL_RANK"
    return "CODE8_LOCAL_CONTROL_PARTIAL_RANK"


def classify_s3(
    *,
    authority: bool,
    interface: bool,
    targeting: bool,
    no_collapse: bool,
    reachability: bool,
    code_effect: bool,
    isolation: bool,
    finite: bool,
) -> str:
    if not interface:
        return INTERFACE_UNRESOLVED_CLASSIFICATION
    if not authority or not isolation or not finite:
        return UNDERDETERMINED_CLASSIFICATION
    if targeting and reachability and code_effect and no_collapse:
        return PASS_CLASSIFICATION
    if targeting and not reachability:
        return TARGETING_ONLY_CLASSIFICATION
    if reachability and not targeting:
        return REACHABILITY_ONLY_CLASSIFICATION
    if targeting and no_collapse and not code_effect:
        return COLLAPSED_CLASSIFICATION
    return NOT_SUPPORTED_CLASSIFICATION


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "controller_authority",
        "optical_authority",
        "s1_reference",
        "panel",
        "null_control",
        "observer",
        "targeting",
        "reachability",
        "code_effect",
        "jacobian",
        "phase",
        "execution",
        "authorization",
        "governance",
    }
    if set(payload) != required:
        raise ValueError("S3 config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("S3 config schema is invalid")
    if payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("S3 config task identity is invalid")
    controller = _mapping(payload["controller_authority"], "controller_authority")
    if dict(controller) != {
        "authority": _relative(CONTROLLER_AUTHORITY_PATH),
        "authority_sha256": CONTROLLER_AUTHORITY_SHA256,
        "authority_self_hash": CONTROLLER_AUTHORITY_SELF_HASH,
        "migrated_checkpoint": _relative(MIGRATED_CONTROLLER_PATH),
        "migrated_checkpoint_sha256": MIGRATED_CONTROLLER_SHA256,
        "state_content_hash": CONTROLLER_STATE_HASH,
        "state_signature": CONTROLLER_STATE_SIGNATURE,
        "class": "IdentityAnchoredConditionalAmplitudeGenerator",
        "architecture": "conditional_identity_logit_amplitude_v2",
        "code_dim": CODE_DIM,
        "hidden_dim": 64,
        "command_gain": 0.5,
    }:
        raise ValueError("S3 controller authority binding is not locked")
    optical = _mapping(payload["optical_authority"], "optical_authority")
    if dict(optical) != {
        "formal_config": _relative(CONTROLLER_CONFIG_PATH),
        "checkpoint": _relative(CHECKPOINT_PATH),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_step": 5000,
        "mode_bank_sha256": MODE_BANK_SHA256,
        "calibration": _relative(CALIBRATION_PATH),
        "calibration_sha256": CALIBRATION_SHA256,
        "calibration_manifest": _relative(CALIBRATION_MANIFEST_PATH),
        "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
        "prefix": "slm_to_front_to_encoder_to_pre_ols_to_complex_coefficients",
        "decoder_entered": False,
    }:
        raise ValueError("S3 optical authority binding is not locked")
    s1_ref = _mapping(payload["s1_reference"], "s1_reference")
    if dict(s1_ref) != {
        "config": _relative(S1_CONFIG_PATH),
        "config_sha256": S1_CONFIG_SHA256,
        "config_canonical_sha256": S1_CONFIG_CANONICAL_SHA256,
        "output_directory": _relative(S1_OUTPUT_PATH),
        "semantic_manifest_sha256": S1_MANIFEST_SHA256,
        "identity_manifest_sha256": S1_IDENTITY_SHA256,
        "contract_sha256": S1_CONTRACT_SHA256,
        "embeddings_sha256": S1_EMBEDDINGS_SHA256,
        "dataset_manifest_sha256": S1_DATASET_MANIFEST_SHA256,
        "representation": R1_POWER,
        "minimal_representation": "S1_POWER_SEMANTICS_SUPPORTED",
        "phase_attribution": "PHASE_SEMANTIC_INCREMENT_SUPPORTED",
    }:
        raise ValueError("S3 S1 reference binding is not locked")
    panel = _mapping(payload["panel"], "panel")
    if dict(panel) != {
        "seed": PANEL_SEED,
        "distribution": "uniform[-1,1]^8",
        "base_code_count": PANEL_CODES,
        "states_per_class": PANEL_PER_CLASS,
        "selection_per_class": PANEL_SELECTION_PER_CLASS,
        "confirmation_per_class": PANEL_CONFIRMATION_PER_CLASS,
        "same_base_codes_across_labels": True,
        "order": "label_major",
    }:
        raise ValueError("S3 panel contract is not locked")
    if dict(_mapping(payload["null_control"], "null_control")) != {
        "kind": "fixed_intended_label_permutation",
        "seed": NULL_SEED,
        "states_unchanged": True,
    }:
        raise ValueError("S3 null contract is not locked")
    if dict(_mapping(payload["observer"], "observer")) != {
        "representation": R1_POWER,
        "formula": "z_j=abs(c_j)^2/s_j",
        "complex_state_preserved": True,
        "s1_geometry": "euclidean",
        "primary_k": 1,
    }:
        raise ValueError("S3 observer contract is not locked")
    if dict(_mapping(payload["targeting"], "targeting")) != {
        "target": "min_distance_to_intended_class_reference",
        "other": "min_distance_to_non_intended_class_reference",
        "margin": "d_other_minus_d_target",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "require_per_class_direction": True,
    }:
        raise ValueError("S3 targeting contract is not locked")
    if dict(_mapping(payload["reachability"], "reachability")) != {
        "reference": "S1_confirmation_heldout_real_semantic_neighborhood",
        "quantiles": [0.50, 0.95],
        "delta": "controller_minus_real",
        "hard_criterion": "all_classes_upper95_q50_and_q95_le_zero",
        "selection_direction_consistent": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }:
        raise ValueError("S3 reachability contract is not locked")
    if dict(_mapping(payload["code_effect"], "code_effect")) != {
        "fixed_label": True,
        "metrics": [
            "within_label_pairwise_R1_distance",
            "code_effect_variance",
            "per_dimension_sensitivity",
            "duplicate_collision",
        ],
        "same_code_cross_label": True,
        "collapse_rule": "exact_state_duplicate_or_zero_effect_is_not_nondegenerate",
    }:
        raise ValueError("S3 code-effect contract is not locked")
    if dict(_mapping(payload["jacobian"], "jacobian")) != {
        "method": "deterministic_autograd",
        "quantity": "J_power=dz_du",
        "shape": [PORT_COUNT, CODE_DIM],
        "effective_rank_threshold": EFFECTIVE_RANK_RELATIVE_TOLERANCE,
        "parameter_update": False,
        "full_rank_required_for_s3": False,
    }:
        raise ValueError("S3 Jacobian contract is not locked")
    if dict(_mapping(payload["phase"], "phase")) != {
        "kind": "descriptive_relative_complex_phase",
        "reference": "max_mean_magnitude_port_per_class",
        "authoritative_for_gate": False,
    }:
        raise ValueError("S3 phase contract is not locked")
    execution = _mapping(payload["execution"], "execution")
    if dict(execution) != {
        "device": "cuda",
        "batch_size": 10,
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": 536870912,
        "require_cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("S3 execution contract is not locked")
    if dict(_mapping(payload["authorization"], "authorization")) != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_created": False,
        "decoder_entered": False,
        "evaluator_called": False,
        "prohibited": [
            "decoder",
            "relay",
            "camera",
            "OSERA",
            "S2_rerun",
            "training",
            "optimizer",
            "backward",
            "checkpoint_update",
            "prior_generation",
            "new_objective",
            "new_representation",
            "formal_promotion",
        ],
    }:
        raise ValueError("S3 authorization boundary is not locked")
    if dict(_mapping(payload["governance"], "governance")) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("S3 governance boundary is not locked")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S3 config must be a YAML mapping")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _expected_governance() -> dict[str, Any]:
    return {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }


def validate_controller_authority_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed at the CSAR-to-S3 authority boundary."""
    if payload.get("schema") != "controller_state_authority.v1":
        raise ValueError("controller authority schema is invalid")
    if payload.get("task_id") != "H-LOMA-GEN-S3-CSAR":
        raise ValueError("controller authority task identity is invalid")
    if payload.get("gate") != REQUIRED_AUTHORITY_GATE:
        raise ValueError("controller authority gate is not PASS")
    if payload.get("classification") != "CURRENT_FORMAL_CONTROLLER_STATE_AUTHORITY_REGISTERED":
        raise ValueError("controller authority classification is invalid")
    recorded_self_hash = payload.get("authority_hash")
    if recorded_self_hash != CONTROLLER_AUTHORITY_SELF_HASH:
        raise ValueError("controller authority self-hash is invalid")
    without_hash = dict(payload)
    without_hash.pop("authority_hash", None)
    if canonical_sha256(without_hash) != recorded_self_hash:
        raise ValueError("controller authority self-hash does not verify")
    expected_controller = {
        "class": "IdentityAnchoredConditionalAmplitudeGenerator",
        "architecture": "conditional_identity_logit_amplitude_v2",
        "code_dim": CODE_DIM,
        "hidden_dim": 64,
        "command_gain": 0.5,
        "contract_hash": GENERATOR_CONTRACT_HASH,
    }
    if dict(_mapping(payload.get("controller"), "controller")) != expected_controller:
        raise ValueError("controller authority contract is stale")
    expected_migrated = {
        "path": _relative(MIGRATED_CONTROLLER_PATH),
        "sha256": MIGRATED_CONTROLLER_SHA256,
        "state_content_hash": CONTROLLER_STATE_HASH,
        "state_signature": CONTROLLER_STATE_SIGNATURE,
    }
    if dict(_mapping(payload.get("migrated_checkpoint"), "migrated_checkpoint")) != expected_migrated:
        raise ValueError("migrated controller binding is stale")
    state = _mapping(payload.get("controller_state"), "controller_state")
    if state.get("state_content_hash") != CONTROLLER_STATE_HASH or state.get("state_signature") != CONTROLLER_STATE_SIGNATURE:
        raise ValueError("controller state identity is stale")
    if payload.get("current_config") != {
        "canonical_hash": CONTROLLER_CONFIG_CANONICAL_SHA256,
        "file_sha256": CONTROLLER_CONFIG_SHA256,
        "path": _relative(CONTROLLER_CONFIG_PATH),
    }:
        raise ValueError("controller runtime config binding is stale")
    if payload.get("governance") != _expected_governance():
        raise ValueError("controller authority governance changed")
    if payload.get("next_action") != "IMPLEMENT_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT":
        raise ValueError("controller authority next action is invalid")


def _validate_controller_authority(
    config: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], Path]:
    binding = _mapping(config["controller_authority"], "controller_authority")
    authority_path = _resolve(binding["authority"])
    if file_sha256(authority_path) != binding["authority_sha256"] or file_sha256(authority_path) != CONTROLLER_AUTHORITY_SHA256:
        raise ValueError("controller authority file hash is stale")
    authority = _read_json(authority_path)
    validate_controller_authority_payload(authority)
    migrated_path = _resolve(binding["migrated_checkpoint"])
    envelope = safe_torch_load(
        migrated_path,
        expected_sha256=binding["migrated_checkpoint_sha256"],
        map_location="cpu",
    )
    if not isinstance(envelope, Mapping) or not isinstance(envelope.get("manifest"), Mapping) or not isinstance(envelope.get("state_dict"), Mapping):
        raise ValueError("migrated controller envelope is invalid")
    manifest = _mapping(envelope["manifest"], "migrated controller manifest")
    state = envelope["state_dict"]
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("migrated controller state is not tensor-only")
    if manifest.get("architecture_kind") != "conditional_identity_logit_amplitude_v2" or manifest.get("state_content_hash") != CONTROLLER_STATE_HASH or manifest.get("state_signature") != CONTROLLER_STATE_SIGNATURE or manifest.get("generator_contract_hash") != GENERATOR_CONTRACT_HASH:
        raise ValueError("migrated controller manifest identity is stale")
    if state_content_hash(state) != CONTROLLER_STATE_HASH or state_signature(state) != CONTROLLER_STATE_SIGNATURE:
        raise ValueError("migrated controller state content or signature changed")
    expected_keys = {
        "generator.class_carrier_logits",
        "generator.class_heads",
        "generator.code_projection.weight",
        "generator.global_modes",
        "system.decoder.phase",
        "system.encoder.phase",
    }
    if set(state) != expected_keys:
        raise ValueError("migrated controller state key set changed")
    return authority, state, migrated_path


def _validate_s1_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["s1_reference"], "s1_reference")
    config_path = _resolve(binding["config"])
    if file_sha256(config_path) != binding["config_sha256"] or file_sha256(config_path) != S1_CONFIG_SHA256:
        raise ValueError("S1 config hash is stale")
    _, _, _, canonical = s1.load_config(config_path)
    if canonical != binding["config_canonical_sha256"] or canonical != S1_CONFIG_CANONICAL_SHA256:
        raise ValueError("S1 config canonical hash is stale")
    output = _resolve(binding["output_directory"])
    manifest_path = output / "semantic_manifest.json"
    identity_path = output / "identity_manifest.json"
    contract_path = output / "ols_representation_contract.json"
    embeddings_path = output / "ols_embeddings.pt"
    expected_files = (
        (manifest_path, binding["semantic_manifest_sha256"]),
        (identity_path, binding["identity_manifest_sha256"]),
        (contract_path, binding["contract_sha256"]),
        (embeddings_path, binding["embeddings_sha256"]),
    )
    for path, expected in expected_files:
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"S1 reference artifact is missing or changed: {path.name}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "lightgen_scaled_chen_ols_semantic_representation_audit.v1" or manifest.get("task_id") != "H-LOMA-GEN-S1-OLS-SRA" or manifest.get("status") != "valid_completed" or manifest.get("gate") != "S1_POWER_SEMANTICS_SUPPORTED" or manifest.get("minimal_representation") != "S1_POWER_SEMANTICS_SUPPORTED" or manifest.get("phase_attribution") != "PHASE_SEMANTIC_INCREMENT_SUPPORTED" or manifest.get("next_action") != "PLAN_S2_DIRECT_OLS_SEMANTIC_DECODE_CANARY":
        raise ValueError("S1 semantic manifest is not the registered authority")
    if manifest.get("data_identity", {}).get("reselection") is not False or manifest.get("data_identity", {}).get("self_leakage") is not False:
        raise ValueError("S1 reference panel was reselected or leaked")
    contract = _read_json(contract_path)
    if contract.get("minimal_representation") != R1_POWER and contract.get("representation") not in (None, R1_POWER):
        raise ValueError("S1 representation contract is not R1 power")
    embeddings = safe_torch_load(embeddings_path, expected_sha256=S1_EMBEDDINGS_SHA256, map_location="cpu")
    if not isinstance(embeddings, Mapping) or embeddings.get("schema") != "lightgen_scaled_chen_ols_semantic_embeddings.v1":
        raise ValueError("S1 embedding schema is invalid")
    representations = _mapping(embeddings.get("representations"), "S1 representations")
    labels = _mapping(embeddings.get("labels"), "S1 labels")
    indices = _mapping(embeddings.get("indices"), "S1 indices")
    result: dict[str, Any] = {"manifest": manifest, "contract": contract, "embeddings_path": embeddings_path}
    expected = {
        "fit": (1000, S1_FIT_LABELS_SHA256),
        "selection": (1000, S1_SELECTION_LABELS_SHA256),
        "confirmation": (5000, S1_CONFIRMATION_LABELS_SHA256),
    }
    for split, (count, labels_hash) in expected.items():
        split_labels = labels.get(split)
        split_indices = indices.get(split)
        split_representation = _mapping(representations.get(split), f"S1 {split} representations")
        if not torch.is_tensor(split_labels) or split_labels.shape != (count,) or tensor_sha256(split_labels) != labels_hash:
            raise ValueError(f"S1 {split} labels changed")
        if not torch.is_tensor(split_indices) or split_indices.shape != (count,):
            raise ValueError(f"S1 {split} indices changed")
        power = split_representation.get(R1_POWER)
        if not torch.is_tensor(power) or power.shape != (count, PORT_COUNT) or not torch.is_floating_point(power) or not torch.isfinite(power).all() or bool((power < 0).any()):
            raise ValueError(f"S1 {split} R1 power reference is invalid")
        result[split] = {"power": power.contiguous(), "labels": split_labels.to(torch.long).contiguous(), "indices": split_indices.to(torch.long).contiguous()}
    if not torch.equal(torch.bincount(result["fit"]["labels"], minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), 100, dtype=torch.long)):
        raise ValueError("S1 fit reference is not balanced")
    return result


def _load_formal_system() -> tuple[Any, Path]:
    config_path = _resolve(CONTROLLER_CONFIG_PATH)
    if file_sha256(config_path) != CONTROLLER_CONFIG_SHA256:
        raise ValueError("formal controller config file hash changed")
    config = base.load_config(config_path)
    if config.architecture != "conditional_identity_logit_amplitude_v2" or not config.system.is_formal or config.system.kind != "formal_amplitude_finite_crop_identity":
        raise ValueError("S3 requires the current formal Identity geometry")
    formal = config.system
    expected = {
        "wavelength_m": 532.0e-9,
        "slm_shape": (200, 200),
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": (100, 100),
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "layer_spacing_m": 1.5e-3,
    }
    for name, expected_value in expected.items():
        actual = getattr(formal, name)
        if isinstance(expected_value, float):
            if abs(actual - expected_value) > max(abs(expected_value) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch: {name}")
        elif actual != expected_value:
            raise ValueError(f"formal geometry mismatch: {name}")
    return formal, config_path


def _load_optical_authority(
    config: Mapping[str, Any], device: torch.device
) -> tuple[bogt.ScaledChenBOGTModel, bogt.UnitAwareCalibration, dict[str, Any], Path]:
    binding = _mapping(config["optical_authority"], "optical_authority")
    formal, formal_path = _load_formal_system()
    checkpoint_path = _resolve(binding["checkpoint"])
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=binding["checkpoint_sha256"], map_location="cpu")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(checkpoint.get("global_step", -1)) != 5000 or not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError("step-5000 optical checkpoint identity is invalid")
    model = mra._load_scaled_model(formal, checkpoint, device)
    if bogt.mode_bank_sha256(model) != binding["mode_bank_sha256"]:
        raise ValueError("fixed optical mode bank hash changed")
    calibration_manifest_path = _resolve(binding["calibration_manifest"])
    if file_sha256(calibration_manifest_path) != binding["calibration_manifest_sha256"]:
        raise ValueError("repaired calibration manifest hash changed")
    calibration_manifest = _read_json(calibration_manifest_path)
    if calibration_manifest.get("schema") != "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1" or calibration_manifest.get("gate") != "R2_CALIBRATION_REPAIR_PASS":
        raise ValueError("repaired calibration manifest is not the registered authority")
    calibration_path = _resolve(binding["calibration"])
    payload = safe_torch_load(calibration_path, expected_sha256=binding["calibration_sha256"], map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("repaired calibration payload is invalid")
    rcr.validate_repaired_calibration_identity(
        payload,
        expected_schema="scaled_chen_bogt_calibration.v2",
        expected_contract="unit_aware_raw_mean_v2",
        expected_calibration_indices_sha256="369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543",
        expected_dataset_manifest_hash="741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9",
        expected_mode_bank_sha256=MODE_BANK_SHA256,
    )
    calibration = bogt.UnitAwareCalibration(
        raw_port_mean=payload["raw_port_mean"],
        port_scale=payload["port_scale"],
        plane_reference=payload["plane_reference"],
        sample_count=int(payload["sample_count"]),
        plane_lower=payload.get("plane_lower"),
        plane_upper=payload.get("plane_upper"),
    )
    calibration.validate()
    return model, calibration, dict(checkpoint), formal_path


def _load_controller_generator(
    state: Mapping[str, torch.Tensor], device: torch.device
) -> IdentityAnchoredConditionalAmplitudeGenerator:
    generator = IdentityAnchoredConditionalAmplitudeGenerator().to(device)
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in state.items()
        if key.startswith("generator.")
    }
    expected = set(generator.state_dict())
    if set(generator_state) != expected:
        raise ValueError("controller generator state boundary is incomplete")
    missing, unexpected = generator.load_state_dict(generator_state, strict=True)
    if missing or unexpected:
        raise ValueError("controller generator strict state load failed")
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> dict[str, int]:
    execution = _mapping(config["execution"], "execution")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["require_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_S3_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    device = torch.device(str(execution["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free, total = torch.cuda.mem_get_info(device)
    if free < int(execution["min_free_cuda_bytes"]):
        raise RuntimeError("BLOCKED_S3_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < int(execution["min_disk_free_bytes"]):
        raise RuntimeError("BLOCKED_S3_RESOURCE: less than 10 GiB free disk")
    if output_path.exists():
        raise FileExistsError(f"refusing to reuse write-once S3 output: {_relative(output_path)}")
    return {"free_cuda_bytes_before": int(free), "total_cuda_bytes": int(total), "free_disk_bytes_before": int(disk.free)}


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().to(torch.float64).flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("summary input must be finite and non-empty")
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(values.max()),
        "finite": True,
    }


def _bootstrap_scalar_ci(
    values: torch.Tensor,
    *,
    statistic: str = "mean",
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = values.detach().to(torch.float64).flatten()
    if values.numel() == 0 or not torch.isfinite(values).all() or reps < 100:
        raise ValueError("bootstrap values are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randint(values.numel(), (reps, values.numel()), generator=generator)
    sampled = values.index_select(0, draws.reshape(-1)).reshape(reps, -1)
    if statistic == "mean":
        estimates = sampled.mean(dim=1)
    elif statistic == "q50":
        estimates = torch.quantile(sampled, 0.50, dim=1)
    elif statistic == "q95":
        estimates = torch.quantile(sampled, 0.95, dim=1)
    else:
        raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    interval = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    point = {
        "mean": values.mean(),
        "q50": torch.quantile(values, 0.50),
        "q95": torch.quantile(values, 0.95),
    }[statistic]
    return {
        "point": float(point),
        "lower95": float(interval[0]),
        "upper95": float(interval[1]),
        "replicates": reps,
        "seed": seed,
        "statistic": statistic,
    }


def _stratified_bootstrap_delta(
    real: torch.Tensor,
    null: torch.Tensor,
    labels: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | bool]:
    real = real.detach().to(torch.float64).flatten()
    null = null.detach().to(torch.float64).flatten()
    labels = labels.detach().to(torch.long).flatten()
    if real.shape != null.shape or real.shape != labels.shape or real.numel() == 0 or not torch.isfinite(real).all() or not torch.isfinite(null).all():
        raise ValueError("stratified bootstrap inputs are invalid")
    if labels.unique().numel() < 2:
        raise ValueError("stratified bootstrap needs at least two classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    estimates = torch.zeros(reps, dtype=torch.float64)
    total = float(labels.numel())
    for label in labels.unique(sorted=True):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        draws = torch.randint(positions.numel(), (reps, positions.numel()), generator=generator)
        sampled = positions.index_select(0, draws.reshape(-1)).reshape(reps, -1)
        delta = (real - null).index_select(0, sampled.reshape(-1)).reshape(reps, -1)
        estimates += delta.mean(dim=1) * (positions.numel() / total)
    interval = torch.quantile(estimates, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "point_delta": float((real - null).mean()),
        "lower95": float(interval[0]),
        "upper95": float(interval[1]),
        "replicates": reps,
        "seed": seed,
        "stratified": True,
        "finite": True,
    }


def _bootstrap_quantile_delta_ci(
    controller: torch.Tensor,
    real: torch.Tensor,
    *,
    quantile: float,
    seed: int,
    reps: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    controller = controller.detach().to(torch.float64).flatten()
    real = real.detach().to(torch.float64).flatten()
    if controller.numel() == 0 or real.numel() == 0 or not torch.isfinite(controller).all() or not torch.isfinite(real).all() or not 0.0 < quantile < 1.0:
        raise ValueError("quantile bootstrap inputs are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    controller_draws = torch.randint(controller.numel(), (reps, controller.numel()), generator=generator)
    real_draws = torch.randint(real.numel(), (reps, real.numel()), generator=generator)
    controller_samples = controller.index_select(0, controller_draws.reshape(-1)).reshape(reps, -1)
    real_samples = real.index_select(0, real_draws.reshape(-1)).reshape(reps, -1)
    deltas = torch.quantile(controller_samples, quantile, dim=1) - torch.quantile(real_samples, quantile, dim=1)
    interval = torch.quantile(deltas, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "quantile": quantile,
        "controller_point": float(torch.quantile(controller, quantile)),
        "real_point": float(torch.quantile(real, quantile)),
        "point_delta": float(torch.quantile(controller, quantile) - torch.quantile(real, quantile)),
        "lower95": float(interval[0]),
        "upper95": float(interval[1]),
        "replicates": reps,
        "seed": seed,
    }


def _distance_components(
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    support: torch.Tensor,
    support_labels: torch.Tensor,
) -> dict[str, torch.Tensor | list[list[int]]]:
    if queries.ndim != 2 or support.ndim != 2 or queries.shape[1] != support.shape[1] or query_labels.shape != (queries.shape[0],) or support_labels.shape != (support.shape[0],):
        raise ValueError("semantic distance inputs are not aligned")
    distances = torch.cdist(queries.to(torch.float64), support.to(torch.float64), p=2).square()
    target = torch.empty(queries.shape[0], dtype=torch.float64)
    other = torch.empty_like(target)
    nearest = distances.argmin(dim=1)
    for label in range(CLASS_COUNT):
        query_mask = query_labels == label
        target_mask = support_labels == label
        other_mask = support_labels != label
        if not target_mask.any() or not other_mask.any():
            raise ValueError("semantic reference lacks a class")
        target[query_mask] = distances[query_mask][:, target_mask].min(dim=1).values
        other[query_mask] = distances[query_mask][:, other_mask].min(dim=1).values
    predictions = support_labels.index_select(0, nearest)
    confusion = [
        [int(((query_labels == true_label) & (predictions == predicted_label)).sum()) for predicted_label in range(CLASS_COUNT)]
        for true_label in range(CLASS_COUNT)
    ]
    return {
        "target": target,
        "other": other,
        "margin": other - target,
        "predictions": predictions,
        "confusion": confusion,
        "nearest_distances": distances.min(dim=1).values,
    }


def _target_split_report(
    split: str,
    values: Mapping[str, torch.Tensor | list[list[int]]],
    null_values: Mapping[str, torch.Tensor | list[list[int]]],
    labels: torch.Tensor,
) -> dict[str, Any]:
    target = values["target"]
    other = values["other"]
    margin = values["margin"]
    null_target = null_values["target"]
    null_margin = null_values["margin"]
    predictions = values["predictions"]
    null_predictions = null_values["predictions"]
    if not all(torch.is_tensor(item) for item in (target, other, margin, null_target, null_margin, predictions, null_predictions)):
        raise ValueError("target report values are invalid")
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        mask = labels == label
        per_class[str(label)] = {
            "count": int(mask.sum()),
            "target": _summary(target[mask]),
            "other": _summary(other[mask]),
            "margin": _summary(margin[mask]),
            "null_target": _summary(null_target[mask]),
            "null_margin": _summary(null_margin[mask]),
            "margin_delta": _bootstrap_scalar_ci((margin - null_margin)[mask], seed=BOOTSTRAP_SEED + label),
            "target_distance_improvement": _bootstrap_scalar_ci((null_target - target)[mask], seed=BOOTSTRAP_SEED + 100 + label),
            "retrieval_accuracy": float((predictions[mask] == labels[mask]).to(torch.float64).mean()),
            "null_retrieval_accuracy": float((null_predictions[mask] == labels[mask]).to(torch.float64).mean()),
        }
    margin_delta = _stratified_bootstrap_delta(margin, null_margin, labels)
    target_improvement = _stratified_bootstrap_delta(null_target, target, labels)
    retrieval = (predictions == labels).to(torch.float64)
    null_retrieval = (null_predictions == labels).to(torch.float64)
    retrieval_delta = _stratified_bootstrap_delta(retrieval, null_retrieval, labels)
    return {
        "split": split,
        "sample_count": int(labels.numel()),
        "real": {
            "target_distance": _summary(target),
            "other_distance": _summary(other),
            "semantic_margin": _summary(margin),
            "retrieval_accuracy": float(retrieval.mean()),
            "confusion": values["confusion"],
            "class_order": list(range(CLASS_COUNT)),
        },
        "fixed_shuffled_null": {
            "target_distance": _summary(null_target),
            "semantic_margin": _summary(null_margin),
            "retrieval_accuracy": float(null_retrieval.mean()),
            "confusion": null_values["confusion"],
            "class_order": list(range(CLASS_COUNT)),
        },
        "real_minus_null_margin": margin_delta,
        "null_minus_real_target_distance": target_improvement,
        "real_minus_null_retrieval": retrieval_delta,
        "per_class": per_class,
        "all_classes_present_in_retrieval": bool(torch.unique(predictions).numel() == CLASS_COUNT),
        "finite": True,
    }


def _targeting_metrics(
    power: torch.Tensor,
    labels: torch.Tensor,
    split: torch.Tensor,
    reference_power: torch.Tensor,
    reference_labels: torch.Tensor,
    null_labels: torch.Tensor,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    direction = True
    for split_id, split_name in ((0, "selection"), (1, "confirmation")):
        mask = split == split_id
        values = _distance_components(power[mask], labels[mask], reference_power, reference_labels)
        null_values = _distance_components(power[mask], null_labels[mask], reference_power, reference_labels)
        report = _target_split_report(split_name, values, null_values, labels[mask])
        reports[split_name] = report
        direction = direction and report["real_minus_null_margin"]["lower95"] > 0.0 and report["null_minus_real_target_distance"]["lower95"] > 0.0 and report["real_minus_null_retrieval"]["lower95"] > 0.0
        direction = direction and all(report["per_class"][str(label)]["margin"]["mean"] > 0.0 for label in range(CLASS_COUNT))
    reports["gate"] = "S3_LABEL_TARGETING_PASS" if direction else "S3_LABEL_TARGETING_FAIL"
    reports["registered_null"] = {
        "kind": "fixed_intended_label_permutation",
        "seed": NULL_SEED,
        "states_unchanged": True,
        "null_labels_sha256": tensor_sha256(null_labels),
    }
    reports["overall_direction_consistent"] = direction
    reports["finite"] = True
    return reports


def _reachability_metrics(
    power: torch.Tensor,
    labels: torch.Tensor,
    split: torch.Tensor,
    reference_power: torch.Tensor,
    reference_labels: torch.Tensor,
    real_selection_power: torch.Tensor,
    real_selection_labels: torch.Tensor,
    real_confirmation_power: torch.Tensor,
    real_confirmation_labels: torch.Tensor,
) -> dict[str, Any]:
    """Compare controller neighborhoods with the frozen S1 real neighborhoods."""
    controller_distances = _distance_components(
        power, labels, reference_power, reference_labels
    )["target"]
    if not torch.is_tensor(controller_distances):
        raise ValueError("controller target distances are invalid")
    real_selection_distances = _distance_components(
        real_selection_power,
        real_selection_labels,
        reference_power,
        reference_labels,
    )["target"]
    real_confirmation_distances = _distance_components(
        real_confirmation_power,
        real_confirmation_labels,
        reference_power,
        reference_labels,
    )["target"]
    by_split: dict[str, Any] = {}
    confirmation_pass = True
    selection_direction = True
    for split_id, split_name, real_distances in (
        (0, "selection", real_selection_distances),
        (1, "confirmation", real_confirmation_distances),
    ):
        per_class: dict[str, Any] = {}
        split_pass = True
        for label in range(CLASS_COUNT):
            controller_class = controller_distances[(labels == label) & (split == split_id)]
            real_class = real_distances[real_selection_labels == label] if split_id == 0 else real_distances[real_confirmation_labels == label]
            q50 = _bootstrap_quantile_delta_ci(
                controller_class,
                real_class,
                quantile=0.50,
                seed=BOOTSTRAP_SEED + 200 + split_id * 20 + label,
            )
            q95 = _bootstrap_quantile_delta_ci(
                controller_class,
                real_class,
                quantile=0.95,
                seed=BOOTSTRAP_SEED + 400 + split_id * 20 + label,
            )
            per_class[str(label)] = {
                "controller": _summary(controller_class),
                "real": _summary(real_class),
                "delta_q50": q50,
                "delta_q95": q95,
            }
            if split_id == 1:
                split_pass = split_pass and q50["upper95"] <= 0.0 and q95["upper95"] <= 0.0
            else:
                selection_direction = selection_direction and q50["point_delta"] <= 0.0 and q95["point_delta"] <= 0.0
        by_split[split_name] = {
            "controller_split": split_name,
            "real_reference_split": split_name,
            "per_class": per_class,
            "hard_criterion_pass": split_pass if split_id == 1 else None,
            "direction_consistent": split_pass if split_id == 1 else all(
                item["delta_q50"]["point_delta"] <= 0.0 and item["delta_q95"]["point_delta"] <= 0.0
                for item in per_class.values()
            ),
            "finite": True,
        }
        if split_id == 1:
            confirmation_pass = split_pass
    gate = confirmation_pass and selection_direction
    return {
        "schema": REACHABILITY_SCHEMA,
        "reference": "S1_confirmation_heldout_real_semantic_neighborhood",
        "delta_definition": "controller_minus_real",
        "hard_criterion": "all_classes_upper95_q50_and_q95_le_zero",
        "selection_direction_consistent": selection_direction,
        "confirmation_hard_criterion_pass": confirmation_pass,
        "gate": "S3_ABSOLUTE_REACHABILITY_PASS" if gate else "S3_ABSOLUTE_REACHABILITY_FAIL",
        "splits": by_split,
        "finite": True,
    }


def _code_effect_metrics(
    power: torch.Tensor,
    coefficients: torch.Tensor,
    base_codes: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    if power.shape != (PANEL_COUNT, PORT_COUNT) or coefficients.shape != (PANEL_COUNT, PORT_COUNT):
        raise ValueError("code-effect panel shape is invalid")
    if base_codes.shape != (PANEL_CODES, CODE_DIM):
        raise ValueError("code-effect base-code shape is invalid")
    per_class: dict[str, Any] = {}
    all_pairwise_means: list[float] = []
    all_variances: list[float] = []
    sensitivity = torch.zeros(CODE_DIM, dtype=torch.float64)
    code_std = base_codes.to(torch.float64).std(dim=0, unbiased=False).clamp_min(1.0e-12)
    for label in range(CLASS_COUNT):
        class_power = power[labels == label]
        class_codes = base_codes
        distances = torch.cdist(class_power.to(torch.float64), class_power.to(torch.float64), p=2).square()
        upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
        variance = class_power.to(torch.float64).var(dim=0, unbiased=False)
        class_sensitivity = class_power.to(torch.float64).std(dim=0, unbiased=False) / code_std
        sensitivity += class_sensitivity
        pairwise_mean = float(upper.mean())
        variance_mean = float(variance.mean())
        all_pairwise_means.append(pairwise_mean)
        all_variances.append(variance_mean)
        per_class[str(label)] = {
            "within_label_pairwise_R1_distance": _summary(upper),
            "code_effect_variance": _summary(variance),
            "per_dimension_sensitivity": [float(value) for value in class_sensitivity],
            "nonzero_pairwise_mean": pairwise_mean > 0.0,
            "nonzero_variance_mean": variance_mean > 0.0,
            "code_count": int(class_codes.shape[0]),
        }
    sensitivity /= CLASS_COUNT
    same_code_cross_label: dict[str, Any] = {}
    reference = power[:PANEL_PER_CLASS]
    for label in range(1, CLASS_COUNT):
        matched = power[label * PANEL_PER_CLASS : (label + 1) * PANEL_PER_CLASS]
        diagonal = (matched - reference).to(torch.float64)
        distances = diagonal.square().sum(dim=1)
        same_code_cross_label[str(label)] = _summary(distances)
    unique_power = torch.unique(power, dim=0).shape[0]
    unique_complex = torch.unique(torch.view_as_real(coefficients.to(torch.complex128)), dim=0).shape[0]
    class_means = torch.stack([power[labels == label].to(torch.float64).mean(dim=0) for label in range(CLASS_COUNT)])
    class_mean_distances = torch.cdist(class_means, class_means, p=2).square()
    class_upper = class_mean_distances[torch.triu(torch.ones_like(class_mean_distances, dtype=torch.bool), diagonal=1)]
    duplicate_count = PANEL_COUNT - int(unique_power)
    complex_duplicate_count = PANEL_COUNT - int(unique_complex)
    nondegenerate = bool(
        min(all_pairwise_means) > 0.0
        and min(all_variances) > 0.0
        and duplicate_count == 0
        and complex_duplicate_count == 0
    )
    no_class_collapse = bool(float(class_upper.min()) > 0.0 and int(torch.unique(class_means, dim=0).shape[0]) == CLASS_COUNT)
    return {
        "schema": CODE_EFFECT_SCHEMA,
        "fixed_label_code_effect": True,
        "per_class": per_class,
        "same_code_cross_label_R1_distance_to_label_0": same_code_cross_label,
        "aggregate_per_dimension_sensitivity": [float(value) for value in sensitivity],
        "exact_unique_R1_states": int(unique_power),
        "exact_R1_state_duplicate_count": duplicate_count,
        "exact_unique_complex_states": int(unique_complex),
        "exact_complex_state_duplicate_count": complex_duplicate_count,
        "class_mean_pairwise_R1_distance": _summary(class_upper),
        "nondegenerate": nondegenerate,
        "no_class_collapse": no_class_collapse,
        "finite": True,
    }


def _differentiable_power(
    code: torch.Tensor,
    label: int,
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    model: bogt.ScaledChenBOGTModel,
    scale: torch.Tensor,
) -> torch.Tensor:
    labels = torch.full((1,), label, dtype=torch.long, device=code.device)
    amplitude = generator(code.unsqueeze(0), labels)
    validate_amplitude_output(amplitude)
    coefficients = rcr._encoded_complex_prefix(model, amplitude)[0]
    scale_work = scale.to(coefficients.device, coefficients.real.dtype)
    return (coefficients.real.square() + coefficients.imag.square()) / scale_work


def _jacobian_metrics(
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    model: bogt.ScaledChenBOGTModel,
    scale: torch.Tensor,
    base_code: torch.Tensor,
) -> dict[str, Any]:
    if base_code.shape != (CODE_DIM,):
        raise ValueError("Jacobian base code has the wrong shape")
    generator.eval()
    model.eval()
    per_class: dict[str, Any] = {}
    all_jacobians: list[torch.Tensor] = []
    for label in range(CLASS_COUNT):
        code = base_code.detach().clone().to(next(generator.parameters()).device).requires_grad_(True)

        def function(value: torch.Tensor) -> torch.Tensor:
            return _differentiable_power(value, label, generator, model, scale)

        jacobian = torch.autograd.functional.jacobian(
            function,
            code,
            create_graph=False,
            strict=False,
            vectorize=False,
        ).detach().to(torch.float64)
        if jacobian.shape != (PORT_COUNT, CODE_DIM) or not torch.isfinite(jacobian).all():
            raise ValueError(f"Jacobian for label {label} is invalid")
        singular_values = torch.linalg.svdvals(jacobian)
        classification = classify_rank(singular_values)
        effective_rank = int(
            (singular_values / singular_values.max().clamp_min(1.0e-30) >= EFFECTIVE_RANK_RELATIVE_TOLERANCE).sum()
        ) if float(singular_values.max()) > 0.0 else 0
        all_jacobians.append(jacobian.cpu())
        per_class[str(label)] = {
            "shape": list(jacobian.shape),
            "singular_values": [float(value) for value in singular_values.cpu()],
            "effective_rank": effective_rank,
            "classification": classification,
            "mean_abs_sensitivity_by_code_dimension": [float(value) for value in jacobian.abs().mean(dim=0).cpu()],
            "max_abs_sensitivity": float(jacobian.abs().max()),
            "finite": True,
        }
        del code, jacobian, singular_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    stacked = torch.cat(all_jacobians, dim=0)
    aggregate_singular_values = torch.linalg.svdvals(stacked)
    return {
        "schema": JACOBIAN_SCHEMA,
        "method": "deterministic_autograd",
        "quantity": "J_power=dz_du",
        "shape_per_class": [PORT_COUNT, CODE_DIM],
        "effective_rank_threshold": EFFECTIVE_RANK_RELATIVE_TOLERANCE,
        "base_code": [float(value) for value in base_code],
        "per_class": per_class,
        "aggregate_stack_shape": list(stacked.shape),
        "aggregate_singular_values": [float(value) for value in aggregate_singular_values],
        "aggregate_classification": classify_rank(aggregate_singular_values),
        "parameter_update": False,
        "optimizer": False,
        "training": False,
        "finite": True,
    }


def _phase_diagnostics(coefficients: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if coefficients.shape != (PANEL_COUNT, PORT_COUNT) or not torch.is_complex(coefficients):
        raise ValueError("phase diagnostics require complex [1000,100] coefficients")
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        class_coefficients = coefficients[labels == label].to(torch.complex128)
        mean_magnitude = class_coefficients.abs().mean(dim=0)
        anchor = int(mean_magnitude.argmax())
        relative = torch.angle(class_coefficients * class_coefficients[:, anchor].conj().unsqueeze(1))
        concentration = torch.sqrt(relative.cos().mean(dim=0).square() + relative.sin().mean(dim=0).square())
        per_class[str(label)] = {
            "anchor_port": anchor,
            "anchor_mean_magnitude": float(mean_magnitude[anchor]),
            "relative_phase_mean_abs": [float(value) for value in relative.abs().mean(dim=0)],
            "relative_phase_circular_concentration": [float(value) for value in concentration],
            "finite": True,
        }
    return {
        "schema": PHASE_SCHEMA,
        "kind": "descriptive_relative_complex_phase",
        "reference": "max_mean_magnitude_port_per_class",
        "authoritative_for_gate": False,
        "per_class": per_class,
        "finite": True,
    }


def _interface_report(
    amplitudes: torch.Tensor,
    coefficients: torch.Tensor,
    power: torch.Tensor,
) -> dict[str, Any]:
    validate_amplitude_output(amplitudes)
    validate_power_observer(coefficients, torch.ones(PORT_COUNT))
    if power.shape != (PANEL_COUNT, PORT_COUNT) or not torch.isfinite(power).all() or bool((power < 0).any()):
        raise ValueError("S3 power observer output is invalid")
    return {
        "schema": INTERFACE_SCHEMA,
        "gate": "S3_CONTROLLER_TO_SEMANTIC_OLS_INTERFACE_PASS",
        "controller_input": {"shape": [PANEL_COUNT, CODE_DIM], "kind": "label_plus_8D_code"},
        "amplitude_output": {"shape": list(amplitudes.shape), "dtype": str(amplitudes.dtype), "range": [float(amplitudes.min()), float(amplitudes.max())]},
        "complex_ols_output": {"shape": list(coefficients.shape), "dtype": str(coefficients.dtype)},
        "normalized_power_output": {"shape": list(power.shape), "dtype": str(power.dtype), "formula": "abs(c)^2/s"},
        "mapping": "label_plus_u8_to_controller_to_amplitude_to_frozen_frontend_to_fixed_ols",
        "adaptor": False,
        "resize": False,
        "sqrt_conversion": False,
        "logit_conversion": False,
        "decoder_entered": False,
        "finite": True,
    }


def _state_isolation(
    authority_path: Path,
    migrated_path: Path,
    checkpoint_path: Path,
    calibration_path: Path,
    model: bogt.ScaledChenBOGTModel,
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    controller_state: Mapping[str, torch.Tensor],
    before: Mapping[str, Any],
    *,
    peak_cuda_bytes: int,
) -> dict[str, Any]:
    after_controller_hash = state_content_hash(controller_state)
    after_signature = state_signature(controller_state)
    after_model_digest = rcr.state_digest(model)
    after_generator_hash = canonical_sha256({name: tensor_sha256(value) for name, value in sorted(generator.state_dict().items())})
    authority_after = file_sha256(authority_path)
    migrated_after = file_sha256(migrated_path)
    checkpoint_after = file_sha256(checkpoint_path)
    calibration_after = file_sha256(calibration_path)
    mode_after = bogt.mode_bank_sha256(model)
    unchanged = bool(
        before["controller_state_hash"] == after_controller_hash
        and before["controller_state_signature"] == after_signature
        and before["model_digest"] == after_model_digest
        and before["generator_hash"] == after_generator_hash
        and before["authority_sha256"] == authority_after
        and before["migrated_sha256"] == migrated_after
        and before["checkpoint_sha256"] == checkpoint_after
        and before["calibration_sha256"] == calibration_after
        and before["mode_bank_sha256"] == mode_after
    )
    return {
        "schema": ISOLATION_SCHEMA,
        "gate": "S3_STATE_ISOLATION_PASS" if unchanged else "S3_STATE_ISOLATION_FAIL",
        "controller_authority_sha256_before": before["authority_sha256"],
        "controller_authority_sha256_after": authority_after,
        "controller_state_hash_before": before["controller_state_hash"],
        "controller_state_hash_after": after_controller_hash,
        "controller_state_signature_before": before["controller_state_signature"],
        "controller_state_signature_after": after_signature,
        "migrated_checkpoint_sha256_before": before["migrated_sha256"],
        "migrated_checkpoint_sha256_after": migrated_after,
        "optical_checkpoint_sha256_before": before["checkpoint_sha256"],
        "optical_checkpoint_sha256_after": checkpoint_after,
        "calibration_sha256_before": before["calibration_sha256"],
        "calibration_sha256_after": calibration_after,
        "optical_model_digest_before": before["model_digest"],
        "optical_model_digest_after": after_model_digest,
        "generator_hash_before": before["generator_hash"],
        "generator_hash_after": after_generator_hash,
        "mode_bank_sha256_before": before["mode_bank_sha256"],
        "mode_bank_sha256_after": mode_after,
        "decoder_entered": False,
        "relay_called": False,
        "camera_called": False,
        "evaluator_called": False,
        "optimizer_created": False,
        "training_steps": 0,
        "training_backward_calls": 0,
        "parameter_update": False,
        "checkpoint_writes": 0,
        "peak_cuda_bytes": peak_cuda_bytes,
        "state_unchanged": unchanged,
        "finite": True,
    }


def _generator_state_hash(generator: IdentityAnchoredConditionalAmplitudeGenerator) -> str:
    return canonical_sha256(
        {
            name: tensor_sha256(value)
            for name, value in sorted(generator.state_dict().items())
        }
    )


def _capture_controller_states(
    model: bogt.ScaledChenBOGTModel,
    generator: IdentityAnchoredConditionalAmplitudeGenerator,
    panel: Mapping[str, torch.Tensor],
    scale: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amplitudes: list[torch.Tensor] = []
    coefficients: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    codes = panel["codes"]
    labels = panel["labels"]
    with torch.inference_mode():
        for positions in torch.arange(PANEL_COUNT).split(batch_size):
            batch_codes = codes.index_select(0, positions).to(device)
            batch_labels = labels.index_select(0, positions).to(device)
            amplitude = generator(batch_codes, batch_labels)
            validate_amplitude_output(amplitude)
            batch_coefficients = rcr._encoded_complex_prefix(model, amplitude)
            batch_power = validate_power_observer(batch_coefficients, scale)
            amplitudes.append(amplitude.detach().cpu().contiguous())
            coefficients.append(batch_coefficients.detach().cpu().contiguous())
            powers.append(batch_power.detach().cpu().contiguous())
            del batch_codes, batch_labels, amplitude, batch_coefficients, batch_power
    amplitude_bank = torch.cat(amplitudes, dim=0)
    coefficient_bank = torch.cat(coefficients, dim=0)
    power_bank = torch.cat(powers, dim=0)
    validate_amplitude_output(amplitude_bank)
    validate_power_observer(coefficient_bank, scale)
    if power_bank.shape != (PANEL_COUNT, PORT_COUNT) or not torch.isfinite(power_bank).all():
        raise ValueError("controller power bank is invalid")
    return amplitude_bank, coefficient_bank, power_bank


def _controller_contract(
    authority: Mapping[str, Any],
    controller_state: Mapping[str, torch.Tensor],
    model: bogt.ScaledChenBOGTModel,
) -> dict[str, Any]:
    return {
        "schema": CONTROLLER_CONTRACT_SCHEMA,
        "authority_gate": authority["gate"],
        "authority_classification": authority["classification"],
        "controller": dict(authority["controller"]),
        "migrated_checkpoint": dict(authority["migrated_checkpoint"]),
        "state_content_hash": state_content_hash(controller_state),
        "state_signature": state_signature(controller_state),
        "loaded_generator_keys": sorted(
            key for key in controller_state if key.startswith("generator.")
        ),
        "excluded_controller_system_keys": sorted(
            key for key in controller_state if key.startswith("system.")
        ),
        "optical_binding": {
            "checkpoint": _relative(CHECKPOINT_PATH),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "mode_bank_sha256": bogt.mode_bank_sha256(model),
            "system_phase_source": "fixed_step_5000_scaled_chen_optical_checkpoint",
            "controller_system_phase_used": False,
            "reason": "CSAR envelope system phases are not the S3 optical authority; the registered step-5000 optical checkpoint supplies the frozen encoder prefix",
        },
        "interface": "label_plus_u8_to_current_controller_to_amplitude_slm_to_frozen_frontend_to_fixed_ols",
        "no_adaptor": True,
        "no_resize": True,
        "no_sqrt_or_logit_conversion": True,
        "decoder_entered": False,
        "training": False,
    }


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    return {
        target.relative_to(output_path).as_posix(): {
            "path": target.relative_to(output_path).as_posix(),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        for target in sorted(output_path.iterdir(), key=lambda item: item.name)
        if target.is_file() and target.name != "s3_manifest.json"
    }


def _assert_output_tree(output_path: Path) -> None:
    expected = {
        "s3_manifest.json",
        "controller_contract.json",
        "controller_panel_identity.json",
        "controller_ols_interface.json",
        "semantic_reference_contract.json",
        "label_targeting_metrics.json",
        "absolute_reachability_metrics.json",
        "code_effect_metrics.json",
        "jacobian_metrics.json",
        "phase_diagnostics.json",
        "state_isolation.json",
        "compact_report.json",
        "controller_ols_states.pt",
    }
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != expected:
        raise ValueError(f"S3 output tree is not exact: {sorted(actual)}")
    if any(
        Path(name).name.startswith(".")
        or ".tmp" in name
        or name.lower().endswith((".tif", ".tiff"))
        for name in actual
    ):
        raise ValueError("S3 output contains hidden, temporary, or TIFF files")


def _git_preflight(*, require_clean: bool) -> dict[str, Any]:
    state = base._git_state()
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    allowed = {
        "scripts/research/run_scaled_chen_controller_to_ols_writability_audit.py",
        "configs/research/lightgen_scaled_chen_controller_to_ols_writability_audit_v1.yaml",
        "tests/v2/test_scaled_chen_controller_to_ols_writability.py",
        "docs/CURRENT_STATUS.md",
        "docs/migration_windows_native_20260830.md",
    }
    unexpected = sorted(
        str(line[3:]).replace("\\", "/")
        for line in state.get("status", [])
        if str(line[3:]).replace("\\", "/") not in allowed
    )
    if unexpected:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: {unexpected}")
    if require_clean and not state.get("clean"):
        raise RuntimeError("BLOCKED_S3_DIRTY_WORKTREE: formal audit requires clean HEAD")
    return state


def _semantic_reference_contract(reference: Mapping[str, Any]) -> dict[str, Any]:
    manifest = reference["manifest"]
    return {
        "schema": "lightgen_scaled_chen_s3_semantic_reference_contract.v1",
        "source_task": "H-LOMA-GEN-S1-OLS-SRA",
        "source_directory": _relative(S1_OUTPUT_PATH),
        "source_manifest_sha256": S1_MANIFEST_SHA256,
        "representation": R1_POWER,
        "definition": "z_j=abs(c_j)^2/s_j",
        "metric": "euclidean_squared",
        "fit_reference_count": 1000,
        "fit_reference_per_class": 100,
        "selection_real_count": 1000,
        "confirmation_real_count": 5000,
        "minimal_representation": manifest["minimal_representation"],
        "phase_attribution": manifest["phase_attribution"],
        "global_phase": manifest["phase_gauge"]["classification"],
        "reselection": manifest["data_identity"]["reselection"],
        "self_leakage": manifest["data_identity"]["self_leakage"],
        "finite": True,
    }


def _preflight(
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    state = _git_preflight(require_clean=False)
    _, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority, controller_state, migrated_path = _validate_controller_authority(config, base._repo_root())
    reference = _validate_s1_reference(config)
    _, formal_path = _load_formal_system()
    output = _resolve(output_path)
    resource = _resource_preflight(config, output)
    checkpoint_path = _resolve(config["optical_authority"]["checkpoint"])
    calibration_path = _resolve(config["optical_authority"]["calibration"])
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=CHECKPOINT_SHA256, map_location="cpu")
    calibration = safe_torch_load(calibration_path, expected_sha256=CALIBRATION_SHA256, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("S3 authority payload preflight failed")
    if state_content_hash(controller_state) != CONTROLLER_STATE_HASH:
        raise ValueError("S3 controller state preflight failed")
    return {
        "task_id": TASK_ID,
        "gate": "S3_STATIC_AUTHORITY_PREFLIGHT_PASS",
        "branch": state["branch"],
        "head": state["commit"],
        "config_sha256": config_sha256,
        "config_canonical_sha256": config_canonical_sha256,
        "controller_authority_sha256": file_sha256(_resolve(config["controller_authority"]["authority"])),
        "migrated_controller_sha256": file_sha256(migrated_path),
        "s1_manifest_sha256": file_sha256(_resolve(config["s1_reference"]["output_directory"]) / "semantic_manifest.json"),
        "formal_config": _relative(formal_path),
        "s1_reference": reference["manifest"]["minimal_representation"],
        "controller_state_hash": CONTROLLER_STATE_HASH,
        "controller_state_signature": CONTROLLER_STATE_SIGNATURE,
        "authority_gate": authority["gate"],
        "resource": resource,
        "output_path": _relative(output),
        "measurement_started": False,
        "decoder_entered": False,
        "optimizer_created": False,
        "training": False,
    }


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    state_before = _git_preflight(require_clean=True)
    _, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority, controller_state, migrated_path = _validate_controller_authority(config, base._repo_root())
    reference = _validate_s1_reference(config)
    output = _resolve(output_path)
    resource = _resource_preflight(config, output)
    panel = build_controller_panel()
    panel_id = panel_identity(panel)
    if panel_id["base_codes"]["sha256"] != "2c83f8fca3d1fde7d5c694f69b602dbc3d43b4e41cfb8159d1af1914efef8c86":
        raise ValueError("registered S3 controller panel seed materialized unexpectedly")
    null_labels = fixed_label_permutation(panel["labels"], seed=NULL_SEED)
    device = torch.device(str(config["execution"]["device"]))
    model, calibration, checkpoint, formal_path = _load_optical_authority(config, device)
    generator = _load_controller_generator(controller_state, device)
    scale = calibration.port_scale.detach().cpu().contiguous()
    if not torch.equal(scale, calibration.raw_port_mean.detach().cpu()):
        raise ValueError("registered calibration scale is not raw raw_port_mean")
    model_state_before = rcr.state_digest(model)
    generator_hash_before = _generator_state_hash(generator)
    mode_bank_before = bogt.mode_bank_sha256(model)
    before = {
        "authority_sha256": file_sha256(_resolve(config["controller_authority"]["authority"])),
        "migrated_sha256": file_sha256(migrated_path),
        "checkpoint_sha256": file_sha256(_resolve(config["optical_authority"]["checkpoint"])),
        "calibration_sha256": file_sha256(_resolve(config["optical_authority"]["calibration"])),
        "controller_state_hash": state_content_hash(controller_state),
        "controller_state_signature": state_signature(controller_state),
        "model_digest": model_state_before,
        "generator_hash": generator_hash_before,
        "mode_bank_sha256": mode_bank_before,
    }
    torch.cuda.reset_peak_memory_stats(device)
    amplitudes, coefficients, power = _capture_controller_states(
        model,
        generator,
        panel,
        scale,
        device=device,
        batch_size=int(config["execution"]["batch_size"]),
    )
    interface = _interface_report(amplitudes, coefficients, power)
    fit_power = reference["fit"]["power"]
    fit_labels = reference["fit"]["labels"]
    targeting = _targeting_metrics(
        power,
        panel["labels"],
        panel["split"],
        fit_power,
        fit_labels,
        null_labels,
    )
    reachability = _reachability_metrics(
        power,
        panel["labels"],
        panel["split"],
        fit_power,
        fit_labels,
        reference["selection"]["power"],
        reference["selection"]["labels"],
        reference["confirmation"]["power"],
        reference["confirmation"]["labels"],
    )
    code_effect = _code_effect_metrics(
        power,
        coefficients,
        panel["base_codes"],
        panel["labels"],
    )
    jacobian = _jacobian_metrics(
        generator,
        model,
        scale,
        panel["base_codes"][0],
    )
    phase = _phase_diagnostics(coefficients, panel["labels"])
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    isolation = _state_isolation(
        _resolve(config["controller_authority"]["authority"]),
        migrated_path,
        _resolve(config["optical_authority"]["checkpoint"]),
        _resolve(config["optical_authority"]["calibration"]),
        model,
        generator,
        controller_state,
        before,
        peak_cuda_bytes=peak_cuda_bytes,
    )
    finite = bool(
        torch.isfinite(amplitudes).all()
        and torch.isfinite(coefficients).all()
        and torch.isfinite(power).all()
        and targeting["finite"]
        and reachability["finite"]
        and code_effect["finite"]
        and jacobian["finite"]
        and phase["finite"]
    )
    interface_pass = interface["gate"] == "S3_CONTROLLER_TO_SEMANTIC_OLS_INTERFACE_PASS"
    targeting_pass = targeting["gate"] == "S3_LABEL_TARGETING_PASS"
    reachability_pass = reachability["gate"] == "S3_ABSOLUTE_REACHABILITY_PASS"
    code_effect_pass = bool(code_effect["nondegenerate"])
    no_collapse = bool(code_effect["no_class_collapse"])
    isolation_pass = isolation["gate"] == "S3_STATE_ISOLATION_PASS"
    classification = classify_s3(
        authority=authority["gate"] == REQUIRED_AUTHORITY_GATE,
        interface=interface_pass,
        targeting=targeting_pass,
        no_collapse=no_collapse,
        reachability=reachability_pass,
        code_effect=code_effect_pass,
        isolation=isolation_pass,
        finite=finite,
    )
    primary_gate = "CONTROL_TO_OLS_WRITABILITY_SUPPORTED" if classification == PASS_CLASSIFICATION else "CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED"
    if classification == PASS_CLASSIFICATION:
        next_action = "PLAN_S4_LABEL_CONDITIONAL_SEMANTIC_CONTROL"
    elif classification == TARGETING_ONLY_CLASSIFICATION:
        next_action = "REVIEW_CONTROLLER_TO_OLS_REACHABILITY_GAP"
    elif classification == REACHABILITY_ONLY_CLASSIFICATION:
        next_action = "REVIEW_LABEL_TO_OLS_SEMANTIC_MAPPING"
    elif classification == COLLAPSED_CLASSIFICATION:
        next_action = "REVIEW_CODE8_CONTROLLER_EFFECT"
    else:
        next_action = "REVIEW_S3_CONTROLLER_TO_OLS_WRITABILITY_AUDIT"
    compact_report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "primary_gate": primary_gate,
        "classification": classification,
        "controller_authority_gate": authority["gate"],
        "interface_gate": interface["gate"],
        "targeting_gate": targeting["gate"],
        "reachability_gate": reachability["gate"],
        "code_effect_nondegenerate": code_effect_pass,
        "no_class_collapse": no_collapse,
        "state_isolation_gate": isolation["gate"],
        "finite": finite,
        "decoder_entered": False,
        "training": False,
        "generation_readiness": "not_ready",
        "next_action": next_action,
    }
    controller_contract = _controller_contract(authority, controller_state, model)
    semantic_reference = _semantic_reference_contract(reference)
    states_payload = {
        "schema": "lightgen_scaled_chen_controller_ols_states.v1",
        "task_id": TASK_ID,
        "panel_identity_sha256": panel_id["panel_identity_sha256"],
        "labels": panel["labels"],
        "codes": panel["codes"],
        "split": panel["split"],
        "slm_amplitude": amplitudes,
        "coefficients": coefficients,
        "normalized_power": power,
        "null_labels": null_labels,
    }
    artifact_payloads: dict[str, Mapping[str, Any]] = {
        "controller_contract.json": controller_contract,
        "controller_panel_identity.json": panel_id,
        "controller_ols_interface.json": interface,
        "semantic_reference_contract.json": semantic_reference,
        "label_targeting_metrics.json": {"schema": TARGETING_SCHEMA, **targeting},
        "absolute_reachability_metrics.json": reachability,
        "code_effect_metrics.json": code_effect,
        "jacobian_metrics.json": jacobian,
        "phase_diagnostics.json": phase,
        "state_isolation.json": isolation,
        "compact_report.json": compact_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary_name:
        staging = Path(temporary_name)
        for name, payload in artifact_payloads.items():
            atomic_write_new_json(staging / name, payload, artifact_kind=f"S3 {name}")
        atomic_torch_save_new(staging / "controller_ols_states.pt", states_payload, artifact_kind="S3 controller states")
        artifact_refs = _artifact_refs(staging)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "hypothesis": HYPOTHESIS,
            "status": "valid_completed",
            "primary_gate": primary_gate,
            "classification": classification,
            "next_action": next_action,
            "configuration": {
                "path": _relative(config_path),
                "sha256": config_sha256,
                "canonical_sha256": config_canonical_sha256,
            },
            "git": {
                "branch": state_before["branch"],
                "starting_head": state_before["commit"],
                "ending_head": base._git_state()["commit"],
                "unexpected_dirty_paths": [],
            },
            "controller_authority": {
                "path": _relative(config["controller_authority"]["authority"]),
                "sha256": file_sha256(_resolve(config["controller_authority"]["authority"])),
                "self_hash": CONTROLLER_AUTHORITY_SELF_HASH,
                "migrated_checkpoint": _relative(migrated_path),
                "migrated_checkpoint_sha256": file_sha256(migrated_path),
                "state_content_hash": CONTROLLER_STATE_HASH,
                "state_signature": CONTROLLER_STATE_SIGNATURE,
                "gate": authority["gate"],
            },
            "optical_authority": {
                "formal_config": _relative(formal_path),
                "checkpoint": _relative(CHECKPOINT_PATH),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "global_step": int(checkpoint["global_step"]),
                "mode_bank_sha256": bogt.mode_bank_sha256(model),
                "calibration": _relative(CALIBRATION_PATH),
                "calibration_sha256": CALIBRATION_SHA256,
                "prefix": "slm_to_front_to_encoder_to_pre_ols_to_complex_coefficients",
                "decoder_entered": False,
            },
            "s1_reference": {
                "directory": _relative(S1_OUTPUT_PATH),
                "semantic_manifest_sha256": S1_MANIFEST_SHA256,
                "representation": R1_POWER,
                "gate": reference["manifest"]["gate"],
                "minimal_representation": reference["manifest"]["minimal_representation"],
            },
            "panel": panel_id,
            "null_control": {
                "kind": "fixed_intended_label_permutation",
                "seed": NULL_SEED,
                "states_unchanged": True,
                "labels_sha256": tensor_sha256(null_labels),
            },
            "interface": interface,
            "targeting": {
                "gate": targeting["gate"],
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "reachability": {
                "gate": reachability["gate"],
                "hard_criterion": reachability["hard_criterion"],
                "confirmation_hard_criterion_pass": reachability["confirmation_hard_criterion_pass"],
                "selection_direction_consistent": reachability["selection_direction_consistent"],
            },
            "code_effect": {
                "nondegenerate": code_effect["nondegenerate"],
                "no_class_collapse": code_effect["no_class_collapse"],
                "exact_R1_state_duplicate_count": code_effect["exact_R1_state_duplicate_count"],
            },
            "jacobian": {
                "method": jacobian["method"],
                "quantity": jacobian["quantity"],
                "effective_rank_threshold": EFFECTIVE_RANK_RELATIVE_TOLERANCE,
                "aggregate_classification": jacobian["aggregate_classification"],
            },
            "phase": {"schema": phase["schema"], "authoritative_for_gate": False},
            "state_isolation": isolation,
            "resource": {**resource, "peak_cuda_bytes": peak_cuda_bytes, "wall_clock_seconds": time.perf_counter() - started},
            "governance": _expected_governance(),
            "authorization": {
                "diagnostic_only": True,
                "training_authorized": False,
                "formal_promotion": False,
                "decoder_entered": False,
                "evaluator_called": False,
            },
            "layout": {
                "artifacts": "relative_paths_with_sha256_and_size_bytes",
                "manifest_self_hash_excluded": True,
                "write_once": True,
                "artifact_count_excluding_manifest": len(artifact_refs),
            },
            "artifacts": artifact_refs,
        }
        atomic_write_new_json(staging / "s3_manifest.json", manifest, artifact_kind="S3 manifest")
        _assert_output_tree(staging)
        staging.replace(output)
    _assert_output_tree(output)
    final_manifest = _read_json(output / "s3_manifest.json")
    if final_manifest.get("artifacts") != _artifact_refs(output):
        raise ValueError("S3 manifest artifact bindings do not verify")
    return {
        "task_id": TASK_ID,
        "output_dir": _relative(output),
        "status": final_manifest["status"],
        "primary_gate": final_manifest["primary_gate"],
        "classification": final_manifest["classification"],
        "next_action": final_manifest["next_action"],
        "controller_authority_gate": authority["gate"],
        "interface_gate": interface["gate"],
        "targeting_gate": targeting["gate"],
        "reachability_gate": reachability["gate"],
        "code_effect_nondegenerate": code_effect["nondegenerate"],
        "jacobian_classification": jacobian["aggregate_classification"],
        "state_isolation_gate": isolation["gate"],
        "peak_cuda_bytes": peak_cuda_bytes,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), default="run")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    result = _preflight(arguments.config, arguments.output_dir) if arguments.action == "preflight" else run(arguments.config, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
