"""Register one repaired-coordinate BOGT weight from frozen RCWV evidence.

This consumer performs only hash-bound artifact validation and Decimal-based
analytic replay.  It does not load a model for execution, create an optimizer,
measure gradients, or write a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)


TASK_ID = "H-LOMA-GEN-T1-RCAWR"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1"
REGISTRATION_MANIFEST_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_manifest.v1"
REGISTRATION_METRICS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_metrics.v1"
PANEL_DETAILS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_panel_details.v1"
COMPARISON_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_old_new_comparison.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_compact_report.v1"

REGISTRATION_GATE = "RCAWR_REGISTRATION_PASS"
REGISTRATION_INVALID = "RCAWR_REGISTRATION_INVALID"
RATIO_TOO_STRONG = "RCAWR_RATIO_TOO_STRONG"
RATIO_TOO_WEAK = "RCAWR_RATIO_TOO_WEAK"
RATIO_UNDETERMINED = "RCAWR_RATIO_UNDETERMINED"
INVALID_HARNESS = "invalid_harness"
NEXT_ACTION = "IMPLEMENT_GENERATIVE_ALIGNMENT_CANARY"

PORT_COUNT = 100
PANEL_COUNT = 3
BATCH_SIZE = 10
MICROBATCH_SIZE = 1
SEED = 1234
PARENT_STEP = 5000
ALPHA_OLD = 0.002996375650268832
OLD_GAMMA = 0.02969130038500517
TARGET_RATIO = 0.10
MIN_RATIO = 0.05
MAX_RATIO = 0.40
REGISTRATION_ALPHA = 0.00017905883237862847
EXPECTED_ALPHA_DECIMAL = Decimal("0.000179058832378628469663887774247392581098542509215070071798828")
DECIMAL_PRECISION = 60

NEW_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v2"
NEW_CALIBRATION_CONTRACT = "unit_aware_raw_mean_v2"
CHECKPOINT_SCHEMA = "scaled_chen_bogt_horizon_checkpoint.v1"
HISTORICAL_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v1"

OLD_SOURCE_COMMIT = "cefaa19a257b836f22ad04594ec3748c027b0eec"
OLD_PARENT_CONFIG_SHA256 = "e3d1b536aa081a8298b21b0b608f232463444a2863e138f941765adf96d8a701"
OLD_PREFLIGHT_SHA256 = "2570df7cf4fe7a2b8fbca5e8bddce3cba259984e038347c74b5e3855b6106d9b"
OLD_PILOT_REVIEW_SHA256 = "50a4bd770a7287d8993f06d64ed925450b2d09e009ed0cacdadf4d7742ad6095"
OLD_PANEL_ARTIFACT_SHA256 = "b12d616cbb34297aed5fa7b3d941964a4ac02cb2de6c1fce1b6cf1bb5de0f7b1"
OLD_CALIBRATION_SHA256 = "d871a6319e4a4efcc7746378305b3bdc678e93c876b9af9d4e875a7e59a2ed11"
OLD_REVIEW_SHA256 = "4978686212b164f281cf1cb5c070119929fa111ea34e92a77d4b1d5ae2ce10f6"
OLD_FORMAL_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
OLD_VALIDATION_SHA256 = "ff946fbbb13c691283aa927ccc570f14f1aa3be76b9b31f673f90f39e208ccc0"
OLD_MANIFEST_SHA256 = "e3f9c39b9f2fc6278f93f688e6ad7735cfad9a4dfbb9e74ba32d36ac10da33c2"
SCHEDULE_SHA256 = "73b77c379267fd4535188d96075ec14db9cfc2a53e9bdcb49fc0a0e3c4cd3145"
IMAGES_SHA256 = "cc86acbde125e1596a9d290165c4345cff2294032e1bb5cf4aeaf8c6467322f0"
LABELS_SHA256 = "5b15b6a6d7cf9f282941280ffeb569aca7640d0d7e78be939a9b1723e4f83549"
CALIBRATION_INDICES_SHA256 = "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543"
PANEL_INDICES_SHA256 = (
    "822948f09ebdad5f3fb36da36f45ae1dc0b9d65b6ff79675405aeec763f55861",
    "635817ff273289883881c3447115ff0db05a20aa08448e0e2095e4c420df836f",
    "2340114e34809126e683aca958f4590e195f3911aa42697ef94d6ab4940a0202",
)

R2_CONFIG_SHA256 = "b8c1efab6fd49c0e68696bc7a466cf6eaeeb499a842c6efa66fe080254b867a9"
R2_CONFIG_CANONICAL_SHA256 = "c9de9505a6cf64432e4f2c350bfc5dfa188bc9563cdf8f89c274193123045deb"
R2_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
R2_PAYLOAD_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
R2_DATASET_MANIFEST_SHA256 = "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9"
R2_MODE_BANK_SHA256 = "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
PARENT_V1_CALIBRATION_SHA256 = "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959"
PARENT_CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"

RCWV_MANIFEST_SHA256 = "56b791471be83d473986e45d4d2665fec569baecf59cacb38504a8224fa5ea7c"
RCWV_METRICS_SHA256 = "f8ef2bc6c1f18f045773bb518431105a84700f9b878566bcda24a3aa7ca0f8ad"
RCWV_COMPARISON_SHA256 = "56be2703667a1b9d9e7fcbdfb8e9c9361b6123da840656efd55d8f693f6bff3f"
RCWV_REPORT_SHA256 = "4b297b35994a4301452f7e36aec07f9f5ff26d7a5c2941264bfe3ed0db526ff7"
RCWV_PRODUCER_COMMIT = "47b9aa0896cae88516b3872e6f3355ff12434054"

DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_repaired_coordinate_bogt_weight_registration_v1_attempt_01"
)


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value, base=base._repo_root())


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching the disk."""
    requested = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", requested.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _json_file(path: str | Path, *, expected_sha256: str, name: str) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"{name} is missing: {_relative_path(target)}")
    actual = file_sha256(target)
    if actual != expected_sha256:
        raise ValueError(f"{name} hash mismatch")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload, actual


def _file_hash(path: str | Path, *, expected_sha256: str, name: str) -> tuple[Path, str]:
    target = _resolve(path)
    if not target.is_file():
        raise ValueError(f"{name} is missing: {_relative_path(target)}")
    actual = file_sha256(target)
    if actual != expected_sha256:
        raise ValueError(f"{name} hash mismatch")
    return target, actual


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "historical_registration",
        "repaired_calibration",
        "checkpoint",
        "dataset",
        "rcwv",
        "registration",
        "authorization",
        "governance",
        "output",
    }
    if set(payload) != required or payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RCAWR config fields or schema are invalid")
    if payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RCAWR task identity is invalid")

    historical = _require_mapping(payload["historical_registration"], "historical_registration")
    expected_historical = {
        "config_sha256": OLD_PARENT_CONFIG_SHA256,
        "source_commit": OLD_SOURCE_COMMIT,
        "source_branch": BRANCH,
        "preflight_sha256": OLD_PREFLIGHT_SHA256,
        "pilot_review_sha256": OLD_PILOT_REVIEW_SHA256,
        "gradient_panels_sha256": OLD_PANEL_ARTIFACT_SHA256,
        "calibration_sha256": OLD_CALIBRATION_SHA256,
        "review_contract_sha256": OLD_REVIEW_SHA256,
        "formal_config_sha256": OLD_FORMAL_CONFIG_SHA256,
        "dataset_manifest_sha256": OLD_MANIFEST_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "old_alpha": ALPHA_OLD,
        "old_gamma": OLD_GAMMA,
        "target_ratio": TARGET_RATIO,
        "acceptance_interval": [MIN_RATIO, MAX_RATIO],
        "ratio_formula": "alpha_times_bogt_gradient_norm_divided_by_mse_gradient_norm",
        "registration_formula": "target_ratio_times_mse_gradient_norm_divided_by_bogt_gradient_norm",
        "aggregation": "mean_of_panel_specific_analytic_alpha",
        "authority_set": "B_registered_preflight_global_alpha_replay",
        "set_a_classification": "A_IS_REPLAY_CONTEXT_ONLY",
        "set_b_classification": "B_IS_REGISTRATION_AUTHORITY",
    }
    for key, expected in expected_historical.items():
        if historical.get(key) != expected:
            raise ValueError(f"historical registration binding is invalid: {key}")
    if tuple(historical.get("panel_indices_sha256", ())) != PANEL_INDICES_SHA256:
        raise ValueError("historical panel identity is invalid")

    repaired = _require_mapping(payload["repaired_calibration"], "repaired_calibration")
    expected_repaired = {
        "config_sha256": R2_CONFIG_SHA256,
        "config_canonical_sha256": R2_CONFIG_CANONICAL_SHA256,
        "manifest_sha256": R2_MANIFEST_SHA256,
        "payload_sha256": R2_PAYLOAD_SHA256,
        "schema": NEW_CALIBRATION_SCHEMA,
        "contract_version": NEW_CALIBRATION_CONTRACT,
        "dataset_manifest_sha256": R2_DATASET_MANIFEST_SHA256,
        "calibration_indices_sha256": CALIBRATION_INDICES_SHA256,
        "mode_bank_sha256": R2_MODE_BANK_SHA256,
        "parent_v1_sha256": PARENT_V1_CALIBRATION_SHA256,
    }
    for key, expected in expected_repaired.items():
        if repaired.get(key) != expected:
            raise ValueError(f"repaired calibration identity is invalid: {key}")

    checkpoint = _require_mapping(payload["checkpoint"], "checkpoint")
    if dict(checkpoint) != {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": PARENT_CHECKPOINT_SHA256,
        "schema": CHECKPOINT_SCHEMA,
        "global_step": PARENT_STEP,
    }:
        raise ValueError("checkpoint identity is invalid")

    dataset = _require_mapping(payload["dataset"], "dataset")
    expected_dataset = {
        "images_sha256": IMAGES_SHA256,
        "labels_sha256": LABELS_SHA256,
        "calibration_indices_sha256": CALIBRATION_INDICES_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "repaired_manifest_sha256": R2_DATASET_MANIFEST_SHA256,
        "panel_count": PANEL_COUNT,
        "batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(f"dataset identity is invalid: {key}")
    if tuple(dataset.get("panel_indices_sha256", ())) != PANEL_INDICES_SHA256:
        raise ValueError("dataset panel identity is invalid")

    rcwv = _require_mapping(payload["rcwv"], "rcwv")
    expected_rcwv = {
        "manifest_sha256": RCWV_MANIFEST_SHA256,
        "gradient_panel_metrics_sha256": RCWV_METRICS_SHA256,
        "old_new_comparison_sha256": RCWV_COMPARISON_SHA256,
        "compact_report_sha256": RCWV_REPORT_SHA256,
        "task_id": "H-LOMA-GEN-T1-RCWV",
        "gate": "RC_ALPHA_TOO_STRONG",
        "gamma": 0.0,
    }
    for key, expected in expected_rcwv.items():
        if rcwv.get(key) != expected:
            raise ValueError(f"RCWV identity is invalid: {key}")
    if tuple(rcwv.get("parameter_subset", ())) != (
        "system.encoder.phase",
        "system.decoder.phase",
    ):
        raise ValueError("RCWV parameter subset is invalid")

    registration = _require_mapping(payload["registration"], "registration")
    expected_alpha = Decimal(str(registration.get("alpha")))
    if registration.get("name") != "repaired_coordinate_bogt_alpha_v1":
        raise ValueError("new weight identity is invalid")
    if registration.get("coordinate") != NEW_CALIBRATION_SCHEMA:
        raise ValueError("new weight coordinate is invalid")
    if registration.get("old_alpha") != ALPHA_OLD:
        raise ValueError("old alpha is not immutable")
    if expected_alpha != Decimal(str(REGISTRATION_ALPHA)):
        raise ValueError("registered alpha is invalid")
    if registration.get("target_ratio") != TARGET_RATIO or registration.get("acceptance_interval") != [MIN_RATIO, MAX_RATIO]:
        raise ValueError("new weight target or boundary is invalid")
    if registration.get("candidate_count") != 1 or registration.get("selection") != "analytic_only_no_sweep":
        raise ValueError("new weight selection policy is invalid")
    if registration.get("rule") != "mean_of_panel_specific_analytic_alpha":
        raise ValueError("new weight aggregation rule is invalid")
    if registration.get("source_gradient_authority") != "RCWV_attempt_01":
        raise ValueError("new weight source authority is invalid")

    if payload["authorization"] != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "training": False,
        "optimizer_created": False,
        "new_cuda_measurement": False,
    }:
        raise ValueError("RCAWR authorization boundary is invalid")
    if payload["governance"] != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("RCAWR governance boundary is invalid")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RCAWR config must be a YAML mapping")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _close(left: float, right: float, *, rel_tol: float = 1.0e-12, abs_tol: float = 1.0e-15) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _validate_panel_records(records: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or len(records) != PANEL_COUNT:
        raise ValueError(f"{name} must contain exactly three panels")
    result: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        row = _require_mapping(record, f"{name}[{index}]")
        if row.get("panel") != index or row.get("indices_sha256") != PANEL_INDICES_SHA256[index]:
            raise ValueError(f"{name}[{index}] panel identity is invalid")
        result.append(row)
    return result


def _validate_state_isolation(record: Mapping[str, Any], name: str) -> None:
    boolean_false = (
        "optimizer_created",
        "optimizer_step_called",
        "optimizer_state_mutated",
        "checkpoint_created",
        "phase_updated",
        "refit",
        "training",
        "prior_generation",
        "decoder_generation",
    )
    if any(record.get(key) is not False for key in boolean_false):
        raise ValueError(f"{name} contains a forbidden state mutation")
    if record.get("state_unchanged") is not True:
        raise ValueError(f"{name} learned state is not unchanged")
    if record.get("state_digest_before") != record.get("state_digest_after"):
        raise ValueError(f"{name} state digests do not match")
    if record.get("checkpoint_unchanged") is not True:
        raise ValueError(f"{name} checkpoint was changed")
    if record.get("optimizer_state_unchanged") is not True:
        raise ValueError(f"{name} optimizer state was changed")


def _load_historical_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _require_mapping(config["historical_registration"], "historical_registration")
    old_config_path, _ = _file_hash(
        section["config"], expected_sha256=OLD_PARENT_CONFIG_SHA256, name="historical BOGT config"
    )
    old_config = yaml.safe_load(old_config_path.read_text(encoding="utf-8"))
    if not isinstance(old_config, dict) or old_config.get("schema") != "lightgen_optical_manifold_smf_bogt_v1" or old_config.get("hypothesis") != "H-LOMA-CBOGT-S0":
        raise ValueError("historical BOGT config schema is invalid")
    dataset = _require_mapping(old_config.get("dataset"), "historical dataset")
    if dataset.get("logical_batch_size") != BATCH_SIZE or dataset.get("microbatch_size") != MICROBATCH_SIZE or dataset.get("schedule_seed") != SEED:
        raise ValueError("historical BOGT batch or seed contract is invalid")
    gates = _require_mapping(old_config.get("gates"), "historical gates")
    if gates.get("gradient_target_ratio") != TARGET_RATIO or gates.get("gradient_ratio_range") != [MIN_RATIO, MAX_RATIO]:
        raise ValueError("historical target or boundary is invalid")
    training = _require_mapping(old_config.get("training"), "historical training")
    if training.get("optimizer") != "Adam" or training.get("learning_rate") != 1.0e-3 or training.get("zero_phase_seed") != SEED:
        raise ValueError("historical optimizer continuation contract is invalid")

    preflight_path = _resolve(section["preflight"])
    preflight, preflight_hash = _json_file(
        preflight_path, expected_sha256=OLD_PREFLIGHT_SHA256, name="historical preflight"
    )
    evidence = _require_mapping(preflight.get("evidence"), "historical preflight evidence")
    if preflight.get("schema") != "lightgen_optical_manifold_smf_bogt_preflight.v1" or preflight.get("hypothesis") != "H-LOMA-CBOGT-S0" or evidence.get("status") != "passed" or preflight.get("training_authorized") is not False or preflight.get("formal_promotion") is not False:
        raise ValueError("historical preflight is not a passed registration record")
    contract = _require_mapping(preflight.get("contract"), "historical preflight contract")
    manifest = _require_mapping(contract.get("dataset_manifest"), "historical dataset manifest")
    if contract.get("alpha") != ALPHA_OLD or contract.get("gamma") != OLD_GAMMA or manifest.get("manifest_hash") != OLD_MANIFEST_SHA256:
        raise ValueError("historical preflight alpha or dataset binding is invalid")
    if manifest.get("panel_indices_sha256") != list(PANEL_INDICES_SHA256) or manifest.get("schedule_sha256") != SCHEDULE_SHA256:
        raise ValueError("historical preflight panel binding is invalid")
    panel_norms = _validate_panel_records(
        [
            {
                "panel": index,
                "indices_sha256": PANEL_INDICES_SHA256[index],
                **record,
            }
            for index, record in enumerate(contract.get("panel_norms", []))
        ],
        "historical panel norms",
    )

    review_path, review_hash = _file_hash(
        section["review_contract"], expected_sha256=OLD_REVIEW_SHA256, name="historical review contract"
    )
    formal_path, formal_hash = _file_hash(
        section["formal_config"], expected_sha256=OLD_FORMAL_CONFIG_SHA256, name="historical formal config"
    )
    pilot_path = _resolve(section["pilot_review"])
    pilot, pilot_hash = _json_file(
        pilot_path, expected_sha256=OLD_PILOT_REVIEW_SHA256, name="historical pilot review"
    )
    approved = _require_mapping(pilot.get("approved_contract"), "historical approved contract")
    expected_approved = {
        "arms": ["mse_only", "mse_op", "mse_kl", "mse_kl_op"],
        "steps_per_arm": 100,
        "logical_batch_size": BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "zero_phase_seed": SEED,
        "posthoc_phase_quantization": [0.0, 0.82, 1.63, 2.45, 3.26, 4.08, 4.89, 5.71],
        "post_pass_next_action": "manual_review_scaled_chen_bogt_1000_step_three_seed_continuation_contract",
    }
    if pilot.get("schema") != "lightgen_optical_manifold_smf_bogt_pilot_review.v1" or pilot.get("hypothesis") != "H-LOMA-CBOGT-S0" or pilot.get("reviewer_role") != "project_owner" or pilot.get("decision") != "approve_bounded_scaled_chen_bogt_pilot" or pilot.get("bounded_diagnostic_training_authorized") is not True or pilot.get("training_authorized") is not False or pilot.get("formal_promotion") is not False or dict(approved) != expected_approved:
        raise ValueError("historical pilot review contract is invalid")
    bindings = _require_mapping(pilot.get("source_bindings"), "historical pilot bindings")
    expected_bindings = {
        "source_commit": OLD_SOURCE_COMMIT,
        "source_branch": BRANCH,
        "source_config_sha256": OLD_PARENT_CONFIG_SHA256,
        "source_formal_config_sha256": OLD_FORMAL_CONFIG_SHA256,
        "source_lifecycle_review_sha256": OLD_REVIEW_SHA256,
        "source_validation_sha256": OLD_VALIDATION_SHA256,
        "source_preflight_sha256": OLD_PREFLIGHT_SHA256,
        "source_dataset_manifest_hash": OLD_MANIFEST_SHA256,
        "source_schedule_sha256": SCHEDULE_SHA256,
        "source_calibration_sha256": OLD_CALIBRATION_SHA256,
        "source_gradient_panels_sha256": OLD_PANEL_ARTIFACT_SHA256,
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("historical pilot source bindings are invalid")
    measured = _require_mapping(pilot.get("measured_preflight"), "historical measured preflight")
    if measured.get("alpha") != ALPHA_OLD or measured.get("gamma") != OLD_GAMMA or measured.get("gradient_panels") != contract.get("panel_norms"):
        raise ValueError("historical measured preflight is invalid")

    panel_path = _resolve(section["gradient_panels"])
    panel_file_hash = file_sha256(panel_path)
    if panel_file_hash != OLD_PANEL_ARTIFACT_SHA256:
        raise ValueError("historical gradient panel artifact hash mismatch")
    panel_artifact = safe_torch_load(panel_path, map_location="cpu")
    if not isinstance(panel_artifact, Mapping) or panel_artifact.get("schema") != "scaled_chen_bogt_gradient_panels.v1" or panel_artifact.get("dataset_manifest_hash") != OLD_MANIFEST_SHA256:
        raise ValueError("historical gradient panel artifact identity is invalid")
    artifact_panels = _validate_panel_records(panel_artifact.get("panels"), "historical gradient panel artifact")
    registered: list[dict[str, Any]] = []
    for index, (preflight_panel, artifact_panel) in enumerate(zip(panel_norms, artifact_panels, strict=True)):
        norms = _require_mapping(artifact_panel.get("gradient_norms"), f"historical panel {index} norms")
        for field in ("mse", "kl", "op"):
            if not _close(float(norms[field]), float(preflight_panel[field]), rel_tol=0.0, abs_tol=1.0e-18):
                raise ValueError(f"historical panel {index} gradient norm mismatch")
        mse = float(preflight_panel["mse"])
        bogt = float(preflight_panel["kl"])
        registered.append(
            {
                "panel": index,
                "indices_sha256": PANEL_INDICES_SHA256[index],
                "raw_mse_grad_norm": mse,
                "raw_bogt_grad_norm": bogt,
                "raw_op_grad_norm": float(preflight_panel["op"]),
                "effective_bogt_ratio": float(ALPHA_OLD * bogt / mse),
            }
        )
    if not all(MIN_RATIO <= row["effective_bogt_ratio"] <= MAX_RATIO for row in registered):
        raise ValueError("historical registered Set B is outside the inclusive boundary")

    calibration_path = _resolve(section["calibration"])
    calibration_hash = file_sha256(calibration_path)
    if calibration_hash != OLD_CALIBRATION_SHA256:
        raise ValueError("historical calibration hash mismatch")
    calibration_payload = safe_torch_load(calibration_path, map_location="cpu")
    if not isinstance(calibration_payload, Mapping) or calibration_payload.get("schema") != HISTORICAL_CALIBRATION_SCHEMA:
        raise ValueError("historical calibration schema is invalid")

    return {
        "config_path": old_config_path,
        "config_sha256": OLD_PARENT_CONFIG_SHA256,
        "preflight_path": preflight_path,
        "preflight_sha256": preflight_hash,
        "pilot_path": pilot_path,
        "pilot_review_sha256": pilot_hash,
        "panel_path": panel_path,
        "panel_sha256": panel_file_hash,
        "calibration_path": calibration_path,
        "calibration_sha256": calibration_hash,
        "review_path": review_path,
        "review_sha256": review_hash,
        "formal_path": formal_path,
        "formal_sha256": formal_hash,
        "panel_norms": panel_norms,
        "registered_panels": registered,
    }


def _load_repaired_calibration(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _require_mapping(config["repaired_calibration"], "repaired_calibration")
    r2_config_path = _resolve(section["config"])
    r2_config, r2_config_hash = _json_file(
        r2_config_path, expected_sha256=R2_CONFIG_SHA256, name="R2 calibration config"
    )
    if canonical_sha256(r2_config) != R2_CONFIG_CANONICAL_SHA256 or r2_config.get("schema") != "lightgen_scaled_chen_bogt_calibration_contract_repair_v2" or r2_config.get("task_id") != "H-LOMA-GEN-P1-R2-UACR" or r2_config.get("repaired_schema") != NEW_CALIBRATION_SCHEMA or r2_config.get("contract_version") != NEW_CALIBRATION_CONTRACT:
        raise ValueError("calibration identity is invalid")

    manifest_path = _resolve(section["manifest"])
    manifest, manifest_hash = _json_file(
        manifest_path, expected_sha256=R2_MANIFEST_SHA256, name="R2 calibration manifest"
    )
    if manifest.get("schema") != "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1" or manifest.get("task_id") != "H-LOMA-GEN-P1-R2-UACR" or manifest.get("status") != "completed" or manifest.get("gate") != "R2_CALIBRATION_REPAIR_PASS":
        raise ValueError("calibration identity is invalid")
    if manifest.get("cuda_replay") is not False or manifest.get("downstream_replay") is not False or manifest.get("optical_data_changed") is not False:
        raise ValueError("calibration identity is invalid")
    contract = _require_mapping(manifest.get("calibration_contract"), "R2 calibration contract")
    if contract.get("schema") != NEW_CALIBRATION_SCHEMA or contract.get("contract_version") != NEW_CALIBRATION_CONTRACT or contract.get("raw_scale_field") != "raw_port_mean" or contract.get("registered_scale_field") != "port_scale" or contract.get("registered_scale_equals_raw") is not True:
        raise ValueError("calibration identity is invalid")
    parent_v1 = _require_mapping(manifest.get("parent_v1"), "R2 parent calibration")
    if parent_v1.get("schema") != HISTORICAL_CALIBRATION_SCHEMA or parent_v1.get("sha256") != PARENT_V1_CALIBRATION_SHA256 or parent_v1.get("sha256_unchanged_after_repair") is not True:
        raise ValueError("calibration identity is invalid")
    artifact_refs = _require_mapping(manifest.get("artifacts"), "R2 artifact references")
    payload_ref = _require_mapping(artifact_refs.get("repaired_calibration.pt"), "R2 payload reference")
    payload_path = _resolve(section["payload"])
    if payload_ref.get("path") != "repaired_calibration.pt" or payload_ref.get("sha256") != R2_PAYLOAD_SHA256 or payload_ref.get("size_bytes") != payload_path.stat().st_size:
        raise ValueError("calibration identity is invalid")
    payload_hash = file_sha256(payload_path)
    if payload_hash != R2_PAYLOAD_SHA256:
        raise ValueError("calibration identity is invalid")
    payload = safe_torch_load(payload_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("calibration identity is invalid")
    expected_fields = {
        "schema": NEW_CALIBRATION_SCHEMA,
        "contract_version": NEW_CALIBRATION_CONTRACT,
        "task_id": "H-LOMA-GEN-P1-R2-UACR",
        "dataset_manifest_hash": R2_DATASET_MANIFEST_SHA256,
        "calibration_indices_sha256": CALIBRATION_INDICES_SHA256,
        "mode_bank_sha256": R2_MODE_BANK_SHA256,
        "parent_v1_artifact_sha256": PARENT_V1_CALIBRATION_SHA256,
        "parent_v1_schema": HISTORICAL_CALIBRATION_SCHEMA,
        "zero_phase_reference": "independent_train_only_zero_phase",
        "floor_policy": "none_for_positive_finite_raw_means",
        "plane_identity": "pre_ols_plane_power_reference_v1",
        "optical_forward_dtype": "torch.float32/torch.complex64",
        "sample_count": 1000,
        "port_count": PORT_COUNT,
        "raw_scale_dtype": "torch.float32",
        "registered_scale_dtype": "torch.float32",
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise ValueError("calibration identity is invalid")
    raw = payload.get("raw_port_mean")
    registered = payload.get("port_scale")
    if not torch.is_tensor(raw) or not torch.is_tensor(registered) or tuple(raw.shape) != (PORT_COUNT,) or tuple(registered.shape) != (PORT_COUNT,) or raw.dtype != torch.float32 or registered.dtype != torch.float32 or not torch.isfinite(raw).all() or not torch.isfinite(registered).all() or not torch.equal(raw, registered):
        raise ValueError("calibration identity is invalid")
    return {
        "config_path": r2_config_path,
        "config_sha256": r2_config_hash,
        "config_canonical_sha256": canonical_sha256(r2_config),
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "payload_path": payload_path,
        "payload_sha256": payload_hash,
        "schema": NEW_CALIBRATION_SCHEMA,
        "contract_version": NEW_CALIBRATION_CONTRACT,
    }


def _load_checkpoint(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _require_mapping(config["checkpoint"], "checkpoint")
    path, observed_hash = _file_hash(
        section["path"], expected_sha256=PARENT_CHECKPOINT_SHA256, name="step-5000 parent checkpoint"
    )
    return {
        "path": path,
        "sha256": observed_hash,
        "schema": section["schema"],
        "global_step": section["global_step"],
    }


def _validate_rcwv_state(record: Mapping[str, Any], name: str) -> None:
    _validate_state_isolation(record, name)
    if record.get("checkpoint_file_hash_before") != PARENT_CHECKPOINT_SHA256 or record.get("checkpoint_file_hash_after") != PARENT_CHECKPOINT_SHA256:
        raise ValueError(f"{name} checkpoint binding is invalid")


def _load_rcwv_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    section = _require_mapping(config["rcwv"], "rcwv")
    directory = _resolve(section["directory"])
    expected_tree = {
        str(section["manifest"]),
        str(section["gradient_panel_metrics"]),
        str(section["old_new_comparison"]),
        str(section["compact_report"]),
    }
    actual_tree = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_tree != expected_tree:
        raise ValueError("RCWV artifact tree is not exact")

    manifest_path = directory / str(section["manifest"])
    manifest, manifest_hash = _json_file(
        manifest_path, expected_sha256=RCWV_MANIFEST_SHA256, name="RCWV manifest"
    )
    if manifest.get("schema") != "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_manifest.v1" or manifest.get("task_id") != "H-LOMA-GEN-T1-RCWV" or manifest.get("status") != "completed" or manifest.get("gate") != "RC_ALPHA_TOO_STRONG" or manifest.get("classification") != "RC_ALPHA_TOO_STRONG":
        raise ValueError("RCWV artifact identity is invalid")
    producer = _require_mapping(manifest.get("producer_git"), "RCWV producer git")
    if producer.get("branch") != BRANCH or producer.get("commit") != RCWV_PRODUCER_COMMIT or producer.get("worktree_clean_at_execution") is not False:
        raise ValueError("RCWV producer identity is invalid")
    source_bindings = _require_mapping(manifest.get("source_bindings"), "RCWV source bindings")
    if source_bindings.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256 or source_bindings.get("repaired_calibration_payload_sha256") != R2_PAYLOAD_SHA256:
        raise ValueError("RCWV source binding is invalid")
    if manifest.get("checkpoint", {}).get("global_step") != PARENT_STEP or manifest.get("checkpoint", {}).get("sha256") != PARENT_CHECKPOINT_SHA256:
        raise ValueError("RCWV checkpoint identity is invalid")

    artifact_refs = _require_mapping(manifest.get("artifacts"), "RCWV artifact references")

    def load_ref(name: str, hash_key: str, label: str) -> tuple[Path, dict[str, Any], str]:
        reference = _require_mapping(artifact_refs.get(name), f"RCWV {label} reference")
        path = directory / name
        if reference.get("path") != name or reference.get("sha256") != section[hash_key] or reference.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"RCWV {label} reference is invalid")
        payload, actual = _json_file(path, expected_sha256=str(section[hash_key]), name=f"RCWV {label}")
        if actual != reference.get("sha256"):
            raise ValueError(f"RCWV {label} reference is invalid")
        return path, payload, actual

    metrics_path, metrics, metrics_hash = load_ref(
        str(section["gradient_panel_metrics"]), "gradient_panel_metrics_sha256", "gradient panel metrics"
    )
    comparison_path, comparison, comparison_hash = load_ref(
        str(section["old_new_comparison"]), "old_new_comparison_sha256", "old-new comparison"
    )
    report_path, report, report_hash = load_ref(
        str(section["compact_report"]), "compact_report_sha256", "compact report"
    )
    if metrics.get("schema") != "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_gradient_panel_metrics.v1" or metrics.get("task_id") != "H-LOMA-GEN-T1-RCWV" or metrics.get("alpha") != ALPHA_OLD or metrics.get("gamma") != 0.0 or metrics.get("panel_count") != PANEL_COUNT or metrics.get("batch_size") != BATCH_SIZE or metrics.get("microbatch_size") != MICROBATCH_SIZE or tuple(metrics.get("parameter_subset", ())) != ("system.encoder.phase", "system.decoder.phase"):
        raise ValueError("RCWV gradient contract is invalid")
    dataset = _require_mapping(metrics.get("dataset"), "RCWV dataset")
    if dataset.get("registered_old_manifest_sha256") != OLD_MANIFEST_SHA256 or dataset.get("registered_repaired_manifest_sha256") != R2_DATASET_MANIFEST_SHA256 or tuple(dataset.get("panel_indices_sha256", ())) != PANEL_INDICES_SHA256:
        raise ValueError("RCWV panel identity is invalid")
    registered = _validate_panel_records(metrics.get("registered_old_provenance"), "RCWV registered Set B")
    parent = _require_mapping(metrics.get("parent_step_5000"), "RCWV parent replay")
    set_a = _validate_panel_records(parent.get("old_v1"), "RCWV Set A")
    repaired = _validate_panel_records(parent.get("repaired_v2"), "RCWV repaired panels")
    repaired_gate = _require_mapping(parent.get("repaired_effective_ratio_gate"), "RCWV repaired gate")
    if repaired_gate.get("classification") != "RC_ALPHA_TOO_STRONG" or repaired_gate.get("lower") != MIN_RATIO or repaired_gate.get("upper") != MAX_RATIO or repaired_gate.get("all_inclusive") is not False:
        raise ValueError("RCWV repaired gate is invalid")
    for index, row in enumerate(registered):
        mse = float(row["raw_mse_grad_norm"])
        bogt = float(row["raw_bogt_grad_norm"])
        if not _close(float(row["effective_bogt_ratio"]), ALPHA_OLD * bogt / mse):
            raise ValueError(f"RCWV Set B ratio {index} is not reproducible")
    for index, row in enumerate(repaired):
        if not all(math.isfinite(float(row[field])) and float(row[field]) > 0.0 for field in ("raw_mse_grad_norm", "raw_bogt_grad_norm", "effective_bogt_ratio")):
            raise ValueError(f"RCWV repaired panel {index} is non-finite")
        if not _close(float(row["effective_bogt_ratio"]), ALPHA_OLD * float(row["raw_bogt_grad_norm"]) / float(row["raw_mse_grad_norm"]), rel_tol=1.0e-11):
            raise ValueError(f"RCWV repaired ratio {index} is not reproducible")
    if repaired_gate.get("panel_ratios") != [
        float(row["effective_bogt_ratio"]) for row in repaired
    ]:
        raise ValueError("RCWV repaired panel replay is inconsistent")
    if comparison.get("schema") != "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_comparison.v1" or comparison.get("task_id") != "H-LOMA-GEN-T1-RCWV" or comparison.get("same_parent_checkpoint") is not True or comparison.get("same_panel_order_and_indices") is not True or comparison.get("same_forward_contract") is not True:
        raise ValueError("RCWV comparison identity is invalid")
    if report.get("schema") != "lightgen_scaled_chen_repaired_coordinate_bogt_weight_validity_compact_report.v1" or report.get("task_id") != "H-LOMA-GEN-T1-RCWV" or report.get("gate") != "RC_ALPHA_TOO_STRONG" or report.get("training_authorized") is not False or report.get("formal_promotion") is not False:
        raise ValueError("RCWV report identity is invalid")
    report_state = _require_mapping(report.get("state_isolation"), "RCWV report state isolation")
    _validate_rcwv_state(report_state, "RCWV report state isolation")
    manifest_state = _require_mapping(manifest.get("state_isolation"), "RCWV manifest state isolation")
    _validate_rcwv_state(manifest_state, "RCWV manifest state isolation")
    if report_state != manifest_state:
        raise ValueError("RCWV state isolation records diverge")

    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "metrics_path": metrics_path,
        "metrics_sha256": metrics_hash,
        "comparison_path": comparison_path,
        "comparison_sha256": comparison_hash,
        "report_path": report_path,
        "report_sha256": report_hash,
        "registered_old_panels": [dict(row) for row in registered],
        "set_a_old_panels": [dict(row) for row in set_a],
        "repaired_panels": [dict(row) for row in repaired],
        "manifest": manifest,
        "metrics": metrics,
        "comparison": comparison,
        "report": report,
        "state_isolation": dict(report_state),
    }


def load_rcwv_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the hash-bound RCWV evidence without executing its measurement."""
    return _load_rcwv_authority(config)


def classify_set_provenance(authority: Mapping[str, Any]) -> dict[str, str]:
    set_a = _validate_panel_records(authority.get("set_a_old_panels"), "Set A")
    set_b = _validate_panel_records(authority.get("registered_old_panels"), "Set B")
    same_panels = all(left.get("indices_sha256") == right.get("indices_sha256") for left, right in zip(set_a, set_b, strict=True))
    different_state_ratios = any(not _close(float(left["effective_bogt_ratio"]), float(right["effective_bogt_ratio"]), rel_tol=0.0, abs_tol=1.0e-12) for left, right in zip(set_a, set_b, strict=True))
    if not same_panels or not different_state_ratios:
        raise ValueError("Set A and Set B provenance is inconsistent")
    return {
        "set_a": "A_IS_REPLAY_CONTEXT_ONLY",
        "set_b": "B_IS_REGISTRATION_AUTHORITY",
    }


def classify_registration_ratios(
    ratios: Sequence[float], *, lower: float = MIN_RATIO, upper: float = MAX_RATIO
) -> str:
    if len(ratios) != PANEL_COUNT or not all(math.isfinite(float(value)) for value in ratios):
        return INVALID_HARNESS
    below = any(float(value) < lower for value in ratios)
    above = any(float(value) > upper for value in ratios)
    if below and above:
        return RATIO_UNDETERMINED
    if above:
        return RATIO_TOO_STRONG
    if below:
        return RATIO_TOO_WEAK
    return REGISTRATION_GATE


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def derive_registration(
    repaired_panels: Sequence[Mapping[str, Any]],
    *,
    old_alpha: float | Decimal,
    target_ratio: float | Decimal,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Rebuild one alpha from raw frozen G_BOGT/G_MSE ratios."""
    old_alpha_decimal = _decimal(old_alpha, "old alpha")
    if old_alpha_decimal != Decimal(str(ALPHA_OLD)):
        raise ValueError("old alpha is not the registered immutable value")
    target_decimal = _decimal(target_ratio, "target ratio")
    if target_decimal != Decimal("0.1"):
        raise ValueError("target ratio is not the historical value")
    if lower != MIN_RATIO or upper != MAX_RATIO:
        raise ValueError("registration boundary is not the historical inclusive interval")
    panels = _validate_panel_records(list(repaired_panels), "repaired panels")
    k_values: list[Decimal] = []
    panel_alphas: list[Decimal] = []
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        for index, panel in enumerate(panels):
            mse = _decimal(panel.get("raw_mse_grad_norm"), f"panel {index} MSE gradient norm")
            bogt = _decimal(panel.get("raw_bogt_grad_norm"), f"panel {index} BOGT gradient norm")
            recorded_ratio = _decimal(panel.get("effective_bogt_ratio"), f"panel {index} recorded ratio")
            raw_k_value = bogt / mse
            if not _close(float(recorded_ratio), float(old_alpha_decimal * raw_k_value), rel_tol=1.0e-11):
                raise ValueError(f"panel {index} frozen ratio does not reconstruct K_i")
            # Preserve the frozen artifact's serialized effective ratio as
            # the decimal source. Raw norms are still required to reproduce
            # that ratio above; this avoids inventing precision beyond the
            # authority that crossed the measurement boundary.
            k_value = recorded_ratio / old_alpha_decimal
            k_values.append(k_value)
            panel_alphas.append(target_decimal * old_alpha_decimal / recorded_ratio)
        alpha_decimal = sum(panel_alphas, Decimal(0)) / Decimal(PANEL_COUNT)
        predicted_decimal = [alpha_decimal * value for value in k_values]
    predicted = [float(value) for value in predicted_decimal]
    gate = classify_registration_ratios(predicted, lower=lower, upper=upper)
    gate_details = {
        "classification": gate,
        "rule": "all_three_panels_inclusive_registered_interval",
        "lower": lower,
        "upper": upper,
        "panel_ratios": predicted,
        "all_inclusive": gate == REGISTRATION_GATE,
        "minimum": min(predicted),
        "median": sorted(predicted)[1],
        "maximum": max(predicted),
    }
    return {
        "k_values": [float(value) for value in k_values],
        "k_values_decimal": [_decimal_text(value) for value in k_values],
        "panel_alphas": [float(value) for value in panel_alphas],
        "panel_alphas_decimal": [_decimal_text(value) for value in panel_alphas],
        "alpha": float(alpha_decimal),
        "alpha_decimal": _decimal_text(alpha_decimal),
        "alpha_new_over_old": float(alpha_decimal / old_alpha_decimal),
        "alpha_old_over_new": float(old_alpha_decimal / alpha_decimal),
        "predicted_ratios": predicted,
        "predicted_ratios_decimal": [_decimal_text(value) for value in predicted_decimal],
        "gate": gate,
        "gate_details": gate_details,
    }


def _compare_panel_values(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], name: str) -> None:
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        for key in ("panel", "indices_sha256"):
            if lhs.get(key) != rhs.get(key):
                raise ValueError(f"{name} panel {index} identity mismatch")
        for key in ("raw_mse_grad_norm", "raw_bogt_grad_norm", "effective_bogt_ratio"):
            if not _close(float(lhs[key]), float(rhs[key]), rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(f"{name} panel {index} value mismatch")


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "weight_registration_manifest.json":
            refs[target.name] = {
                "path": target.name,
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path, *, include_manifest: bool = True) -> None:
    expected = {
        "weight_registration_metrics.json",
        "panel_registration_details.json",
        "old_new_alpha_comparison.json",
        "compact_report.json",
    }
    if include_manifest:
        expected.add("weight_registration_manifest.json")
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"RCAWR output tree is not exact: {sorted(actual)}")
    if any(Path(name).name.startswith(".") or ".tmp" in name for name in actual):
        raise RuntimeError("RCAWR output contains hidden or temporary files")


def _source_bindings(
    config_path: Path,
    config_sha256: str,
    historical: Mapping[str, Any],
    repaired: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    rcwv: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "registration_config": {
            "path": _relative_path(config_path),
            "sha256": config_sha256,
        },
        "historical_registration": {
            "config": _relative_path(historical["config_path"]),
            "config_sha256": historical["config_sha256"],
            "preflight": _relative_path(historical["preflight_path"]),
            "preflight_sha256": historical["preflight_sha256"],
            "pilot_review": _relative_path(historical["pilot_path"]),
            "pilot_review_sha256": historical["pilot_review_sha256"],
            "gradient_panels": _relative_path(historical["panel_path"]),
            "gradient_panels_sha256": historical["panel_sha256"],
            "calibration": _relative_path(historical["calibration_path"]),
            "calibration_sha256": historical["calibration_sha256"],
            "source_commit": OLD_SOURCE_COMMIT,
            "source_branch": BRANCH,
        },
        "repaired_calibration": {
            "config": _relative_path(repaired["config_path"]),
            "config_sha256": repaired["config_sha256"],
            "config_canonical_sha256": repaired["config_canonical_sha256"],
            "manifest": _relative_path(repaired["manifest_path"]),
            "manifest_sha256": repaired["manifest_sha256"],
            "payload": _relative_path(repaired["payload_path"]),
            "payload_sha256": repaired["payload_sha256"],
            "schema": repaired["schema"],
            "contract_version": repaired["contract_version"],
        },
        "parent_checkpoint": {
            "path": _relative_path(checkpoint["path"]),
            "sha256": checkpoint["sha256"],
            "schema": checkpoint["schema"],
            "global_step": checkpoint["global_step"],
        },
        "rcwv": {
            "directory": _relative_path(rcwv["directory"]),
            "manifest": _relative_path(rcwv["manifest_path"]),
            "manifest_sha256": rcwv["manifest_sha256"],
            "gradient_panel_metrics": _relative_path(rcwv["metrics_path"]),
            "gradient_panel_metrics_sha256": rcwv["metrics_sha256"],
            "old_new_comparison": _relative_path(rcwv["comparison_path"]),
            "old_new_comparison_sha256": rcwv["comparison_sha256"],
            "compact_report": _relative_path(rcwv["report_path"]),
            "compact_report_sha256": rcwv["report_sha256"],
        },
    }


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config_target, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    state = base._git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise RuntimeError("BLOCKED_PREVIOUS_TRACK_NOT_CLOSED: RCAWR requires a clean registered branch")
    historical = _load_historical_authority(config)
    repaired = _load_repaired_calibration(config)
    checkpoint = _load_checkpoint(config)
    rcwv = _load_rcwv_authority(config)
    _compare_panel_values(historical["registered_panels"], rcwv["registered_old_panels"], "historical Set B vs RCWV Set B")
    provenance = classify_set_provenance(rcwv)
    if provenance != {
        "set_a": "A_IS_REPLAY_CONTEXT_ONLY",
        "set_b": "B_IS_REGISTRATION_AUTHORITY",
    }:
        raise ValueError("historical Set A/Set B provenance classification is invalid")
    registration = derive_registration(
        rcwv["repaired_panels"],
        old_alpha=ALPHA_OLD,
        target_ratio=TARGET_RATIO,
        lower=MIN_RATIO,
        upper=MAX_RATIO,
    )
    if Decimal(registration["alpha_decimal"]) != EXPECTED_ALPHA_DECIMAL:
        raise RuntimeError("RCAWR_REGISTRATION_INVALID: analytic alpha replay does not match the registered value")
    if registration["gate"] != REGISTRATION_GATE:
        raise RuntimeError(f"RCAWR_REGISTRATION_INVALID: predicted ratio Gate is {registration['gate']}")

    requested_output = _resolve(output_path if output_path is not None else config["output"]["base"])
    actual_output = next_write_once_output(requested_output)
    bindings = _source_bindings(config_target, config_sha256, historical, repaired, checkpoint, rcwv)
    state_isolation = {
        "scientific_state_mutated": False,
        "model_loaded_for_execution": False,
        "state_unchanged": True,
        "state_digest_before": rcwv["state_isolation"]["state_digest_before"],
        "state_digest_after": rcwv["state_isolation"]["state_digest_after"],
        "optimizer_created": False,
        "optimizer_step_called": False,
        "optimizer_state_mutated": False,
        "optimizer_state_unchanged": True,
        "optimizer_state_digest_before": rcwv["state_isolation"]["optimizer_state_digest_before"],
        "optimizer_state_digest_after": rcwv["state_isolation"]["optimizer_state_digest_after"],
        "checkpoint_created": False,
        "checkpoint_unchanged": True,
        "checkpoint_file_hash_before": rcwv["state_isolation"]["checkpoint_file_hash_before"],
        "checkpoint_file_hash_after": rcwv["state_isolation"]["checkpoint_file_hash_after"],
        "phase_updated": False,
        "gradient_measurement": False,
        "forward_measurement": False,
        "refit": False,
        "training": False,
        "prior_generation": False,
        "decoder_generation": False,
    }
    panel_details = {
        "schema": PANEL_DETAILS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "source_gradient_authority": "RCWV_attempt_01",
        "coordinate": NEW_CALIBRATION_SCHEMA,
        "panels": [
            {
                "panel": index,
                "indices_sha256": PANEL_INDICES_SHA256[index],
                "raw_mse_gradient_norm": float(panel["raw_mse_grad_norm"]),
                "raw_bogt_gradient_norm": float(panel["raw_bogt_grad_norm"]),
                "k_i": registration["k_values"][index],
                "k_i_decimal": registration["k_values_decimal"][index],
                "old_registered_effective_ratio": float(rcwv["registered_old_panels"][index]["effective_bogt_ratio"]),
                "step_5000_replay_effective_ratio": float(rcwv["set_a_old_panels"][index]["effective_bogt_ratio"]),
                "repaired_old_alpha_effective_ratio": float(panel["effective_bogt_ratio"]),
                "alpha_i_new": registration["panel_alphas"][index],
                "alpha_i_new_decimal": registration["panel_alphas_decimal"][index],
                "predicted_effective_ratio": registration["predicted_ratios"][index],
                "predicted_effective_ratio_decimal": registration["predicted_ratios_decimal"][index],
            }
            for index, panel in enumerate(rcwv["repaired_panels"])
        ],
    }
    metrics = {
        "schema": REGISTRATION_METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": REGISTRATION_GATE,
        "classification": REGISTRATION_GATE,
        "registration_name": "repaired_coordinate_bogt_alpha_v1",
        "coordinate": NEW_CALIBRATION_SCHEMA,
        "target_ratio": TARGET_RATIO,
        "acceptance_interval": [MIN_RATIO, MAX_RATIO],
        "historical_rule": "alpha_i = target_ratio * G_MSE_i / G_BOGT_i; alpha_registered = mean(alpha_i)",
        "old_alpha": ALPHA_OLD,
        "alpha": registration["alpha"],
        "alpha_decimal": registration["alpha_decimal"],
        "alpha_new_over_old": registration["alpha_new_over_old"],
        "alpha_old_over_new": registration["alpha_old_over_new"],
        "k_values": registration["k_values"],
        "k_values_decimal": registration["k_values_decimal"],
        "panel_alphas": registration["panel_alphas"],
        "panel_alphas_decimal": registration["panel_alphas_decimal"],
        "predicted_ratios": registration["predicted_ratios"],
        "predicted_ratios_decimal": registration["predicted_ratios_decimal"],
        "predicted_ratio_gate": registration["gate_details"],
        "source_bindings": bindings,
        "state_isolation": state_isolation,
        "authorization": config["authorization"],
        "governance": config["governance"],
        "next_action": NEXT_ACTION,
    }
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "historical_set_a": {
            "classification": "A_IS_REPLAY_CONTEXT_ONLY",
            "state": "step_5000_parent_replay",
            "panels": rcwv["set_a_old_panels"],
        },
        "historical_set_b": {
            "classification": "B_IS_REGISTRATION_AUTHORITY",
            "state": "historical_preflight_registered_global_alpha_replay",
            "panels": rcwv["registered_old_panels"],
        },
        "alpha_old": ALPHA_OLD,
        "alpha_new": registration["alpha"],
        "alpha_new_over_old": registration["alpha_new_over_old"],
        "alpha_old_over_new": registration["alpha_old_over_new"],
        "reduction_factor_is_descriptive_only": True,
        "same_panel_indices": True,
        "same_historical_alpha": True,
        "source_gradient_authority": "RCWV_attempt_01",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": REGISTRATION_GATE,
        "classification": REGISTRATION_GATE,
        "registration_name": "repaired_coordinate_bogt_alpha_v1",
        "alpha": registration["alpha"],
        "alpha_decimal": registration["alpha_decimal"],
        "predicted_effective_ratio_summary": registration["gate_details"],
        "state_isolation": state_isolation,
        "training_authorized": False,
        "formal_promotion": False,
        "next_action": NEXT_ACTION,
    }
    manifest = {
        "schema": REGISTRATION_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": REGISTRATION_GATE,
        "classification": REGISTRATION_GATE,
        "producer_git": {
            "branch": state["branch"],
            "commit": state["commit"],
            "worktree_clean_at_execution": bool(state["clean"]),
        },
        "source_bindings": bindings,
        "historical_registration": {
            "target_ratio": TARGET_RATIO,
            "acceptance_interval": [MIN_RATIO, MAX_RATIO],
            "ratio_formula": "alpha_times_bogt_gradient_norm_divided_by_mse_gradient_norm",
            "registration_formula": "target_ratio_times_mse_gradient_norm_divided_by_bogt_gradient_norm",
            "aggregation": "mean_of_panel_specific_analytic_alpha",
            "authority_set": "B_registered_preflight_global_alpha_replay",
            "set_a_classification": "A_IS_REPLAY_CONTEXT_ONLY",
            "set_b_classification": "B_IS_REGISTRATION_AUTHORITY",
            "old_alpha": ALPHA_OLD,
            "old_gamma_reference_only": OLD_GAMMA,
        },
        "repaired_coordinate": {
            "schema": NEW_CALIBRATION_SCHEMA,
            "contract_version": NEW_CALIBRATION_CONTRACT,
            "registration_name": "repaired_coordinate_bogt_alpha_v1",
            "alpha": registration["alpha"],
            "alpha_decimal": registration["alpha_decimal"],
            "source_gradient_authority": "RCWV_attempt_01",
            "k_values": registration["k_values"],
            "k_values_decimal": registration["k_values_decimal"],
            "panel_alphas": registration["panel_alphas"],
            "panel_alphas_decimal": registration["panel_alphas_decimal"],
            "predicted_ratios": registration["predicted_ratios"],
            "predicted_ratios_decimal": registration["predicted_ratios_decimal"],
        },
        "registered_gate": registration["gate_details"],
        "optimizer_continuation": {
            "status": "AVAILABLE_AND_BOUND",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "seed": SEED,
            "parent_step": PARENT_STEP,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "consumer_must_bind_new_weight": True,
            "training_started": False,
        },
        "state_isolation": state_isolation,
        "authorization": config["authorization"],
        "governance": config["governance"],
        "artifact_layout": {
            "manifest_self_hash_excluded": True,
            "paths_are_relative_to_output": True,
            "contains_images": False,
        },
        "artifacts": {},
        "next_action": NEXT_ACTION,
    }

    actual_output.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(actual_output / "weight_registration_metrics.json", metrics, artifact_kind="RCAWR metrics")
    atomic_write_new_json(actual_output / "panel_registration_details.json", panel_details, artifact_kind="RCAWR panel details")
    atomic_write_new_json(actual_output / "old_new_alpha_comparison.json", comparison, artifact_kind="RCAWR comparison")
    atomic_write_new_json(actual_output / "compact_report.json", report, artifact_kind="RCAWR report")
    _assert_output_tree(actual_output, include_manifest=False)
    manifest["artifacts"] = _artifact_refs(actual_output)
    atomic_write_new_json(actual_output / "weight_registration_manifest.json", manifest, artifact_kind="RCAWR manifest")
    _assert_output_tree(actual_output)
    storage_bytes = sum(ref["size_bytes"] for ref in manifest["artifacts"].values())
    if storage_bytes > int(config["output"]["max_storage_bytes"]):
        raise RuntimeError("RCAWR output exceeds the diagnostic storage contract")
    return {
        "output_path": str(actual_output),
        "manifest": manifest,
        "report": report,
        "registration": registration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
