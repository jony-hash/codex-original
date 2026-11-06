"""Train and independently measure one full-population G1 projected-box mapper."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUN_SCIENTIFIC_ACTIVE,
    RUN_VALID_FAILED,
)
from scripts.research.lightgen import execute_lightgen_g1_fit20_box_b_convergence_v1 as convergence
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_PROJECTED_BOX_FULLPOP_V1"
SCHEMA = "lightgen_g1_projected_box_fullpop.v1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_PROJECTED_BOX_FULLPOP_20261027"
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_projected_box_fullpop_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261027_lightgen_g1_projected_box_fullpop_v1/runs"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
TRAIN_ROWS = 10_000
VALIDATION_ROWS = 2_000
DEV_ROWS = 1_980
PROBE_ROWS = 200
HOLDOUT_ROWS = 2_000
TARGET_COUNT = 20
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
GRIDS = (2048, 4096)
METRIC_KEYS = fit20.METRIC_KEYS
GATE_LIMIT = fit20.GATE_LIMIT
RECOVERY_TOLERANCE = fit20.RECOVERY_TOLERANCE
UPDATE_BUDGET = 5_000
BOX_LAMBDA = 5.0
SOURCE_MAPPER_UPDATES = 10_000
LINEAGE_MAPPER_UPDATES = 15_000
OBSERVATION_STEPS = (0, 1000, 2500, 5000)
FIXED_CHECKPOINT_STEPS = (0, 1000, 2500, 5000)


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fit20._atomic_json(path, value)


def _atomic_torch(path: Path, value: Any) -> None:
    fit20._atomic_torch(path, value)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(),
             "full-population task uses its fixed reviewed configuration", ValueError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "full-population configuration must be a mapping", ValueError)
    population, training, measurement, output = (
        config["population"], config["training"], config["measurement"], config["output"]
    )
    _require(config.get("schema") == SCHEMA and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only"
             and population.get("final_holdout_per_class") == 200
             and population.get("final_holdout_seed_base") == 2026102850
             and population.get("schedule_seed") == 2026102871
             and training.get("new_updates") == UPDATE_BUDGET
             and training.get("rng_seed") == 2026102870
             and training.get("box_lambda") == BOX_LAMBDA
             and training.get("observation_steps") == list(OBSERVATION_STEPS)
             and measurement.get("grids") == list(GRIDS)
             and measurement.get("per_shard") == 10
             and measurement.get("relative_error_max") == GATE_LIMIT
             and _resolve(output["experiment_root"]) == RUNS_DIR.resolve()
             and _resolve(output["attempt_root"]) == ATTEMPT_DIR.resolve()
             and _resolve(output["preflight_root"]) == PREFLIGHT_DIR.resolve(),
             "full-population configuration changed its fixed source, population, budget, or paths",
             ValueError)
    expected_sources = {
        "projected_mapping_config": "configs/research/lightgen/g1_slm_projected_mapping_v2.yaml",
        "v2_terminal_checkpoint": "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/terminal_checkpoint.pt",
        "v2_fixed_response_projector_evidence": "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/fixed_response_and_projector.pt",
        "prior_fit20_b_convergence_config": "configs/research/lightgen/g1_fit20_box_b_convergence_v1.yaml",
        "prior_fit20_b_cloud_attempt": "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/attempt_01",
        "prior_fit20_b_cloud_review": "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/review_repair_01",
        "prior_fit20_b_windows_backup": "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/autodl_backup_attempt_01",
        "fit20_b_repair_windows_receipt": "outputs/20261027_lightgen_g1_fit20_box_b_convergence_v1/runs/autodl_backup_review_repair_01/windows_review_receipt.json",
        "fit20_cloud_witness_measurements": "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs/attempt_01/preflight_evidence/witness_qualification_measurements.pt",
        "fit20_witness_measurements": "outputs/20261027_lightgen_g1_projected_fit20_box_ab_v1/runs/autodl_backup_attempt_01/preflight_evidence/witness_qualification_measurements.pt",
    }
    _require(config.get("source") == expected_sources
             and config["population"] == {
                 "train_role": "fixed_original_g1_train_10000",
                 "validation_role": "original_validation_excluding_fit20_rows_1980",
                 "training_probe_per_class": 20, "known_reachable_control_count": 20,
                 "final_holdout_per_class": 200, "final_holdout_seed_base": 2026102850,
                 "max_proposals_per_class": 1_000_000, "proposal_batch_size": 256,
                 "rejection": "original_total_modal_power_only", "schedule_seed": 2026102871,
             }
             and config["training"] == {
                 "new_updates": 5000, "epochs": 5, "logical_batch_size": 10,
                 "samples_per_class_per_batch": 1, "optimizer": "Adam", "learning_rate": 1.0e-4,
                 "betas": [0.9, 0.999], "epsilon": 1.0e-8, "weight_decay": 0.0,
                 "box_lambda": 5.0, "mapper_precision": "float32",
                 "fixed_projection_precision": "float64", "optical_precision": "float64_complex128",
                 "amp": False, "tf32": False, "rng_seed": 2026102870,
                 "progress_interval_steps": 100, "checkpoint_interval_steps": 500,
                 "observation_steps": [0, 1000, 2500, 5000],
                 "select_best_validation_checkpoint": False,
             }
             and config["measurement"] == {
                 "grids": [2048, 4096], "per_shard": 10, "relative_error_max": 1.0e-4,
                 "coefficient_alignment": "one_global_phase_per_sample",
                 "native_cmos_unit": "raw_w_per_m2", "cmos_normalization": "none",
                 "cmos_power_matching": False, "cmos_gain_adjustment": False,
                 "candidate": "b_terminal_only", "frozen_v2_baseline": "paired_description_only",
             }
             and config["output"] == {
                 "experiment_root": "outputs/20261027_lightgen_g1_projected_box_fullpop_v1/runs",
                 "preflight_root": "outputs/20261027_lightgen_g1_projected_box_fullpop_v1/runs/preflight_01",
                 "attempt_root": "outputs/20261027_lightgen_g1_projected_box_fullpop_v1/runs/attempt_01",
                 "windows_backup_root": "outputs/20261027_lightgen_g1_projected_box_fullpop_v1/runs/autodl_backup_attempt_01",
                 "final_consumer": "recompute_exact_mapper_commands_full_optics_native_cmos_and_all_six_errors",
             }, "full-population input, population, fixed budget, or measurement contract changed", ValueError)
    _require(_resolve(config["source"]["prior_fit20_b_cloud_attempt"])
             == convergence.ATTEMPT_DIR.resolve()
             and _resolve(config["source"]["prior_fit20_b_cloud_review"])
             == (convergence.RUNS_DIR / "review_repair_01").resolve()
             and _resolve(config["source"]["v2_fixed_response_projector_evidence"])
             == (_resolve(config["source"]["v2_terminal_checkpoint"]).parent
                 / "preflight_evidence" / "fixed_response_and_projector.pt").resolve()
             and _resolve(config["source"]["fit20_cloud_witness_measurements"])
             == (fit20.RUNS_DIR / "attempt_01" / "preflight_evidence" /
                 "witness_qualification_measurements.pt").resolve(),
             "full-population configuration changed the explicit cloud B evidence paths", ValueError)
    return config


def _set_numerics() -> None:
    projected._set_numerics()


def _load_fixed_projector_evidence(
    path: Path, expected_response: torch.Tensor, expected_scale: torch.Tensor,
    expected_matrix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _require(path.is_file(), f"explicit fixed response/projector evidence is missing: {path}",
             FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping)
             and payload.get("schema") == "lightgen_g1_slm_fixed_response_projector.v2"
             and payload.get("task_id") == projected.TASK_ID
             and payload.get("real_imag_order") == "per_port_interleaved_real_imag",
             "fixed response/projector evidence identity or real-imag order changed")
    response = payload.get("response_matrix")
    matrix = payload.get("normalized_real_matrix")
    inverse = payload.get("full_right_inverse")
    scale = payload.get("scale")
    expected_response = expected_response.detach().cpu().to(torch.complex128)
    expected_scale = expected_scale.detach().cpu().to(torch.float64)
    expected_matrix = expected_matrix.detach().cpu().to(torch.float64)
    _require(torch.is_tensor(response) and response.dtype == torch.complex128
             and tuple(response.shape) == tuple(expected_response.shape)
             and torch.is_tensor(matrix) and matrix.dtype == torch.float64
             and tuple(matrix.shape) == tuple(expected_matrix.shape)
             and torch.is_tensor(inverse) and inverse.dtype == torch.float64
             and tuple(inverse.shape) == (expected_matrix.shape[1], expected_matrix.shape[0])
             and torch.is_tensor(scale) and scale.dtype == torch.float64
             and tuple(scale.shape) == tuple(expected_scale.shape),
             "fixed response/projector evidence tensor types or shapes changed")
    _require(torch.equal(response, expected_response) and torch.equal(scale, expected_scale)
             and torch.isfinite(response.real).all().item()
             and torch.isfinite(response.imag).all().item()
             and torch.isfinite(scale).all().item() and torch.all(scale > 0.0).item()
             and torch.isfinite(matrix).all().item() and torch.isfinite(inverse).all().item(),
             "fixed response/projector evidence differs from finite frozen source inputs")
    matrix_error = float((matrix - expected_matrix).abs().max().item())
    _require(matrix_error <= 1.0e-14,
             f"saved real-imag response matrix differs from the frozen response by {matrix_error:.3g}")
    return {"response_matrix": response.contiguous(), "matrix": matrix.contiguous(),
            "inverse": inverse.contiguous(), "scale": scale.contiguous(),
            "matrix_max_abs_error": torch.tensor(matrix_error, dtype=torch.float64)}


def _verify_fixed_projector(matrix: torch.Tensor, inverse: torch.Tensor) -> dict[str, float]:
    _require(matrix.dtype == torch.float64 and inverse.dtype == torch.float64
             and matrix.ndim == 2 and tuple(inverse.shape) == (matrix.shape[1], matrix.shape[0])
             and torch.isfinite(matrix).all().item() and torch.isfinite(inverse).all().item(),
             "fixed projector matrices are malformed or nonfinite")
    device = matrix.device
    identity = torch.eye(matrix.shape[0], dtype=torch.float64, device=device)
    identity_error = float((matrix @ inverse - identity).abs().max().item())
    gram = matrix @ matrix.T
    moore_penrose = matrix.T @ torch.linalg.solve(gram, identity)
    inverse_error = float((inverse - moore_penrose).abs().max().item())
    _require(identity_error <= 1.0e-10 and inverse_error <= 1.0e-10,
             "fixed right inverse fails A@P=I or Moore-Penrose verification")
    return {"max_abs_AP_minus_I": identity_error,
            "max_abs_inverse_minus_moore_penrose": inverse_error,
            "right_inverse_verified": True}


def _require_projector_preflight(summary: Mapping[str, Any],
                                 identity: ResearchRunIdentity) -> Mapping[str, Any]:
    projector = summary.get("fixed_projector_verification", {})
    expected_path = str(identity.config_identity["source"][
        "v2_fixed_response_projector_evidence"]).replace("\\", "/").lstrip("./")
    reported_path = str(projector.get("source_path", "")).replace("\\", "/").rstrip("/")
    source_matches = ((reported_path == expected_path
                       or reported_path.endswith("/" + expected_path))
                      and ".." not in reported_path.split("/"))
    _require(summary.get("status") == "READY"
             and source_matches
             and projector.get("right_inverse_verified") is True
             and projector.get("max_abs_AP_minus_I", math.inf) <= 1.0e-10
             and projector.get("max_abs_inverse_minus_moore_penrose", math.inf) <= 1.0e-10
             and projector.get("normalized_matrix_max_abs_error", math.inf) <= 1.0e-14,
             "full-population READY evidence lacks a verified fixed response projector")
    return projector


def _run_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    semantics = json.dumps({"task_id": TASK_ID, "authority": AUTHORITY, "config": config,
                            "source_task": projected.TASK_ID,
                            "source_mapper_updates": SOURCE_MAPPER_UPDATES,
                            "new_updates": UPDATE_BUDGET,
                            "data_roles": ["fixed_g1_train_10000", "development_1980",
                                           "training_probe_200", "new_final_holdout_2000"],
                            "objective": "full_optics_phase_invariant_coefficient_loss+5x_preclip_box_loss"},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:projected_box_fullpop",
        semantic_authority_id=AUTHORITY, data_role="G1_FULL_TRAIN_AND_NEW_FINAL_HOLDOUT",
        config_identity=dict(config),
        model_identity={"frozen_optics": "C_seed_2026101501_step_45000",
                        "source_mapper": "projected_mapping_v2_terminal_step_5000",
                        "mapper": "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid+fixed_projection",
                        "objective": "full_optics_Lc_plus_5Lbox"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4,
                            "betas": [0.9, 0.999], "epsilon": 1.0e-8,
                            "weight_decay": 0.0, "new_updates": UPDATE_BUDGET,
                            "source_optimizer_inherited": False},
        sampler_identity={"training_rows": TRAIN_ROWS, "schedule_seed": 2026102871,
                          "rng_seed": 2026102870, "epochs": 5,
                          "new_holdout_seed_base": 2026102850},
        execution_source_semantics_id=semantics,
    )


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    source_config = projected._load_config(_resolve(config["source"]["projected_mapping_config"]))
    inputs = projected._load_inputs(source_config)
    checkpoint_path = _resolve(config["source"]["v2_terminal_checkpoint"])
    _require(checkpoint_path.is_file(), f"explicit V2 terminal checkpoint is missing: {checkpoint_path}",
             FileNotFoundError)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adam_steps = [int(torch.as_tensor(value["step"]).item())
                  for value in checkpoint.get("optimizer_state", {}).get("state", {}).values()
                  if isinstance(value, Mapping) and "step" in value]
    _require(checkpoint.get("task_id") == projected.TASK_ID
             and checkpoint.get("schema") == "lightgen_g1_slm_projected_mapping_checkpoint.v2"
             and checkpoint.get("completed_updates") == 5000
             and checkpoint.get("checkpoint_step") == 5000
             and checkpoint.get("source_mapper_updates") == 5000
             and checkpoint.get("next_batch_indices") is None
             and checkpoint.get("history_count") == 5000
             and checkpoint.get("optimizer_deepcopied") is True
             and len(adam_steps) == 4 and adam_steps == [5000] * 4
             and set(checkpoint.get("mapper_state", {})) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"},
             "explicit source mapper is not the closed V2 5,000-update terminal")
    projector_path = _resolve(config["source"]["v2_fixed_response_projector_evidence"])
    expected_matrix = projected._normalized_response(inputs["response_matrix"], inputs["scale"])
    projector = _load_fixed_projector_evidence(
        projector_path, inputs["response_matrix"], inputs["scale"], expected_matrix)
    inputs["source_mapper_path"] = checkpoint_path
    inputs["source_mapper"] = checkpoint
    inputs["terminal_mapper"] = checkpoint
    inputs["v2_terminal_checkpoint"] = checkpoint
    inputs["response"] = inputs["response_matrix"]
    inputs["matrix"] = projector["matrix"]
    inputs["inverse"] = projector["inverse"]
    inputs["projector_evidence_path"] = projector_path
    inputs["projector_matrix_max_abs_error"] = float(projector["matrix_max_abs_error"].item())
    return inputs


def _verify_cloud_review_manifest(review_dir: Path, expected_run_id: str) -> dict[str, Any]:
    manifest = _json(review_dir / "transfer_manifest.json")
    _require(manifest.get("schema") == "lightgen_g1_fit20_box_b_review_transfer.v1"
             and manifest.get("task_id") == convergence.TASK_ID
             and manifest.get("run_id") == expected_run_id
             and isinstance(manifest.get("files"), list)
             and manifest.get("file_count") == len(manifest["files"])
             and manifest.get("bytes_total") == sum(int(item["bytes"]) for item in manifest["files"]),
             "cloud B repair manifest is invalid")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "cloud B repair manifest has an unsafe or duplicate path")
        expected[key] = int(item["bytes"])
    checked = []
    for relative, manifest_size in sorted(expected.items()):
        path = review_dir / Path(relative)
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and current == manifest_size,
                 f"cloud B repair file differs from its manifest: {relative}")
        checked.append({"path": relative, "manifest_bytes": manifest_size,
                        "current_bytes": current})
    transfer_manifest_bytes = (review_dir / "transfer_manifest.json").stat().st_size
    return {"status": "REMOTE_REPAIR_MANIFEST_MATCHED", "file_count": len(checked) + 1,
            "bytes_total": sum(item["current_bytes"] for item in checked) + transfer_manifest_bytes,
            "files": checked, "transfer_manifest_bytes": transfer_manifest_bytes}


def _validate_prior_b_repair(config: Mapping[str, Any], inputs: Mapping[str, Any], *,
                             host: str) -> dict[str, Any]:
    _require(host in {"cloud", "windows"}, "prior B evidence host must be explicit")
    if host == "cloud":
        evidence_dir = _resolve(config["source"]["prior_fit20_b_cloud_attempt"])
        _require(evidence_dir.is_dir(), f"cloud B1500 attempt is missing: {evidence_dir}", FileNotFoundError)
        transfer = convergence._verify_remote_manifest(evidence_dir)
        repair_dir = _resolve(config["source"]["prior_fit20_b_cloud_review"])
    else:
        evidence_dir = _resolve(config["source"]["prior_fit20_b_windows_backup"])
        _require(evidence_dir.is_dir(), f"B1500 Windows backup is missing: {evidence_dir}", FileNotFoundError)
        transfer = convergence._verify_transfer(evidence_dir)
        repair_dir = _resolve(config["source"]["fit20_b_repair_windows_receipt"]).parent
    identity = convergence._assert_attempt(evidence_dir)
    state = _json(evidence_dir / "task_state.json")
    decision = _json(evidence_dir / "final_decision.json")
    _require(state.get("state") == RUN_COMPLETED
             and decision.get("lifecycle_state") == RUN_COMPLETED
             and decision.get("run_id") == identity.run_id
             and decision.get("new_scientific_updates") == 1000
             and decision.get("branch_terminal_step") == 1500
             and decision.get("summary", {}).get("passed_rows") == TARGET_COUNT
             and decision.get("summary", {}).get("denominator") == TARGET_COUNT,
             "closed B1500 fit20 result no longer passes its 20-row six-metric decision")

    prior_config = convergence._load_config(
        _resolve(config["source"]["prior_fit20_b_convergence_config"]))
    prior_inputs = convergence._load_inputs(prior_config)
    prior_source = convergence._load_source(prior_config, prior_inputs, host)
    prior_review = convergence._validate_windows_review(prior_config, prior_source)
    _require(prior_source["identity"].run_id == decision.get("source_run_id"),
             "B1500 attempt is not bound to the reviewed fit20 A/B predecessor")

    receipt_path = _resolve(config["source"]["fit20_b_repair_windows_receipt"])
    _require(receipt_path.is_file(),
             f"Windows-reconsumed B1500 evidence repair receipt is missing: {receipt_path}",
             FileNotFoundError)
    receipt = _json(receipt_path)
    recovery = receipt.get("recovery_raw_states_recomputed", {})
    repair_transfer = receipt.get("repair_file_size_reconciliation", {})
    if host == "cloud":
        cloud_repair = _verify_cloud_review_manifest(repair_dir, identity.run_id)
        reported_files = {str(item.get("path")): item for item in repair_transfer.get("files", [])}
        expected_files = {item["path"]: item["manifest_bytes"] for item in cloud_repair["files"]}
        expected_files["transfer_manifest.json"] = cloud_repair["transfer_manifest_bytes"]
        _require(set(reported_files) == set(expected_files)
                 and all(item.get("remote_bytes") == expected
                         and item.get("receipt_local_bytes") == expected
                         and item.get("current_bytes") == expected
                         for path, expected in expected_files.items()
                         for item in (reported_files[path],)),
                 "uploaded Windows B repair receipt does not match the cloud repair manifest")
    else:
        cloud_repair = convergence._verify_review_transfer(repair_dir, identity.run_id)
    _require(receipt.get("schema") == "lightgen_g1_fit20_box_b_convergence_windows_reconsumption.v1"
             and receipt.get("task_id") == convergence.TASK_ID
             and receipt.get("run_id") == identity.run_id
             and receipt.get("status") == "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED"
             and receipt.get("scientific_status") == RUN_COMPLETED
             and receipt.get("all_20_rows_recomputed_at_steps") == [0, 500, 1000]
             and receipt.get("terminal_all_six_metrics_recomputed") is True
             and receipt.get("exact_cuda_command_replay_consumed") is True
             and receipt.get("scientific_updates_during_repair") == 0
             and repair_transfer.get("status") == "COMPLETE"
             and repair_transfer.get("file_count", 0) > 0
             and recovery.get("all_raw_recovery_states_verified") is True
             and recovery.get("full_step4_history_saved") is False
             and repair_transfer.get("status") == "COMPLETE"
             and repair_transfer.get("file_count") == cloud_repair.get("file_count")
             and repair_transfer.get("bytes_total") == cloud_repair.get("bytes_total")
             and prior_review.get("status") == "WINDOWS_REVIEW_RECONSUMPTION_PASSED",
             "full-population task requires the exact successful Windows B1500 repair receipt")
    return {"run_id": identity.run_id, "windows_repair_receipt": str(receipt_path),
            "windows_repair_status": receipt["status"],
            "b1500_six_metric_rows": TARGET_COUNT,
            "exact_cuda_command_replay_consumed": True,
            "recovery_fourth_step_history_absent_and_disclosed": True,
            "historical_result_transfer": transfer,
            "predecessor_fit20_run_id": prior_source["identity"].run_id,
            "predecessor_fit20_review_status": prior_review["status"],
            "cloud_repair_manifest": cloud_repair}


def _population(config: Mapping[str, Any], inputs: Mapping[str, Any], *,
                host: str) -> tuple[dict[str, Any], torch.Tensor]:
    _require(host in {"cloud", "windows"}, "full-population source host must be explicit")
    population, schedule = projected._build_population(config, inputs)
    validation = population["validation"]
    validation["source_validation_indices"] = torch.arange(VALIDATION_ROWS, dtype=torch.long)
    excluded = _fit20_validation_indices()
    dev_indices = _development_indices()
    dev = {key: value[dev_indices].clone() for key, value in validation.items()
           if torch.is_tensor(value) and value.shape[:1] == (VALIDATION_ROWS,)}
    dev["source_role"] = "original_validation_excluding_fit20_rows_1980"
    probe_indices = torch.cat([torch.arange(label * 1000, label * 1000 + 20)
                               for label in range(CLASS_COUNT)])
    probe = {key: value[probe_indices].clone() for key, value in population["train"].items()
             if torch.is_tensor(value) and value.shape[:1] == (TRAIN_ROWS,)}
    probe["source_role"] = "first_20_saved_training_coefficients_per_class"
    _require(tuple(population["train"]["coefficients"].shape) == (TRAIN_ROWS, PORT_COUNT)
             and torch.equal(population["train"]["labels"], torch.arange(10).repeat_interleave(1000))
             and tuple(dev["coefficients"].shape) == (DEV_ROWS, PORT_COUNT)
             and tuple(probe["coefficients"].shape) == (PROBE_ROWS, PORT_COUNT)
             and tuple(population["holdout"]["coefficients"].shape) == (HOLDOUT_ROWS, PORT_COUNT)
             and torch.equal(population["holdout"]["labels"], torch.arange(10).repeat_interleave(200))
             and tuple(schedule.shape) == (5, 1000, CLASS_COUNT),
             "fixed full-population roles, sizes, or five-epoch schedule are malformed")
    unique_coefficients = _assert_population_independence(population)
    _require(torch.equal(schedule.reshape(UPDATE_BUDGET, CLASS_COUNT) // 1000,
                         torch.arange(CLASS_COUNT).expand(UPDATE_BUDGET, CLASS_COUNT)),
             "schedule does not select exactly one sample from each class per update")
    fit20_path_key = ("prior_fit20_b_cloud_attempt" if host == "cloud"
                      else "prior_fit20_b_windows_backup")
    fit20_path = _resolve(config["source"][fit20_path_key])
    fit20_fixed = torch.load(fit20_path / "fixed_population_and_schedule.pt",
                             map_location="cpu", weights_only=False)
    known = {key: fit20_fixed[key].clone() for key in
             ("coefficients", "labels", "proposal_ids", "source_validation_indices")}
    expected_indices = excluded
    _require(tuple(known["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
             and torch.equal(known["source_validation_indices"], expected_indices)
             and torch.equal(known["coefficients"], validation["coefficients"][expected_indices]),
             "known-reachable positive-control population differs from the original fit20 rows")
    witness_path_key = ("fit20_cloud_witness_measurements" if host == "cloud"
                        else "fit20_witness_measurements")
    witness_path = _resolve(config["source"][witness_path_key])
    _require(witness_path.is_file(), f"fit20 positive-control witness is missing: {witness_path}",
             FileNotFoundError)
    witness = torch.load(witness_path, map_location="cpu", weights_only=False)
    _require(witness.get("population_role") == "known_reachable_fit20"
             and torch.equal(witness.get("target_coefficients"), known["coefficients"])
             and torch.equal(witness.get("source_validation_indices"), known["source_validation_indices"])
             and torch.is_tensor(witness.get("amplitudes"))
             and tuple(witness["amplitudes"].shape) == (TARGET_COUNT, 1, 200, 200),
             "saved positive-control witness is not bound to the fixed fit20 targets")
    population["development"] = dev
    population["training_probe"] = probe
    population["known_reachable"] = known
    population["known_reachable_witness"] = witness
    population["excluded_fit20_validation_indices"] = expected_indices
    population["unique_coefficient_rows"] = unique_coefficients
    population["role_contract"] = {
        "train": "only_population_entering_gradient_updates",
        "development": "1980_original_validation_rows_excluding_known_fit20_rows; observation_only",
        "training_probe": "first_20_rows_per_training_class; observation_only",
        "known_reachable": "20_saved_fit20_targets_and_witness; positive_control_only",
        "holdout": "new_2000_coefficients; not consumed until terminal measurement",
    }
    return population, schedule


def _fit20_validation_indices() -> torch.Tensor:
    return torch.tensor([200 * label + offset for label in range(CLASS_COUNT)
                         for offset in (0, 1)], dtype=torch.long)


def _development_indices() -> torch.Tensor:
    excluded = _fit20_validation_indices()
    keep = torch.ones(VALIDATION_ROWS, dtype=torch.bool)
    keep[excluded] = False
    return torch.arange(VALIDATION_ROWS)[keep]


def _assert_population_independence(population: Mapping[str, Any]) -> int:
    cohorts = (population["train"], population["validation"], population["holdout"])
    coefficients = torch.cat([item["coefficients"].detach().cpu() for item in cohorts])
    _require(torch.is_complex(coefficients) and torch.isfinite(coefficients.real).all().item()
             and torch.isfinite(coefficients.imag).all().item(),
             "population coefficients must be finite complex values")
    rows = torch.view_as_real(coefficients).reshape(coefficients.shape[0], -1).numpy()
    unique_count = int(np.unique(rows, axis=0).shape[0])
    _require(unique_count == coefficients.shape[0],
             "training, validation, or new final holdout contains duplicate coefficient rows")
    holdout = population["holdout"]
    for label in range(CLASS_COUNT):
        labels = holdout["labels"]
        ids = holdout["proposal_ids"][labels == label]
        _require(torch.unique(ids).numel() == ids.numel(),
                 f"new final holdout class-{label} contains duplicate proposal identities")
    return unique_count


def _rng_snapshot() -> dict[str, Any]:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}


def _restore_rng(rng: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(rng["cpu"].cpu())
    torch.cuda.set_rng_state_all(rng["cuda"])


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (torch.equal(left["cpu"].cpu(), right["cpu"].cpu())
            and len(left["cuda"]) == len(right["cuda"])
            and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(left["cuda"], right["cuda"], strict=True)))


def _optimizer(mapper: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Adam:
    training = config["training"]
    return torch.optim.Adam(mapper.parameters(), lr=float(training["learning_rate"]),
                            betas=tuple(float(value) for value in training["betas"]),
                            eps=float(training["epsilon"]), weight_decay=float(training["weight_decay"]))


def _optimizer_steps(state: Mapping[str, Any]) -> list[int]:
    return [int(torch.as_tensor(value["step"]).item())
            for value in state.get("state", {}).values()
            if isinstance(value, Mapping) and "step" in value]


def _state_dicts_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_state_dicts_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(_state_dicts_equal(a, b) for a, b in zip(left, right, strict=True)))
    return left == right


def _require_mapper_state_equal(left: Any, right: Any, message: str) -> None:
    _require(_state_dicts_equal(left, right), message)


def _mapper_state_valid(state: Any) -> bool:
    expected = {
        "network.0.weight": (256, 200), "network.0.bias": (256,),
        "network.2.weight": (PIXEL_COUNT, 256), "network.2.bias": (PIXEL_COUNT,),
    }
    return (isinstance(state, Mapping) and set(state) == set(expected)
            and all(torch.is_tensor(state[key]) and state[key].dtype == torch.float32
                    and tuple(state[key].shape) == shape
                    and torch.isfinite(state[key]).all().item()
                    for key, shape in expected.items()))


def _optimizer_state_valid(state: Any, completed: int) -> bool:
    if not isinstance(state, Mapping) or not isinstance(state.get("state"), Mapping):
        return False
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        return False
    group = groups[0]
    if (group.get("lr") != 1.0e-4 or tuple(group.get("betas", ())) != (0.9, 0.999)
            or group.get("eps") != 1.0e-8 or group.get("weight_decay") != 0.0
            or group.get("amsgrad") is not False or group.get("maximize") is not False
            or len(group.get("params", [])) != 4):
        return False
    parameter_shapes = ((256, 200), (256,), (PIXEL_COUNT, 256), (PIXEL_COUNT,))
    parameters = group["params"]
    if completed == 0:
        return not state["state"]
    if set(state["state"]) != set(parameters):
        return False
    for parameter, shape in zip(parameters, parameter_shapes, strict=True):
        moment = state["state"].get(parameter)
        if (not isinstance(moment, Mapping) or set(moment) != {"step", "exp_avg", "exp_avg_sq"}
                or int(torch.as_tensor(moment["step"]).item()) != completed):
            return False
        for key in ("exp_avg", "exp_avg_sq"):
            value = moment[key]
            if (not torch.is_tensor(value) or value.dtype != torch.float32
                    or tuple(value.shape) != shape or not torch.isfinite(value).all().item()):
                return False
    return True


def _batch(population: Mapping[str, Any], schedule: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    epoch, within = divmod(step, 1000)
    _require(0 <= epoch < 5, "training schedule cursor is outside the fixed five epochs")
    indices = schedule[epoch, within]
    coefficients = population["train"]["coefficients"][indices]
    labels = population["train"]["labels"][indices]
    _require(torch.equal(labels, torch.arange(CLASS_COUNT))
             and tuple(coefficients.shape) == (CLASS_COUNT, PORT_COUNT),
             "training batch does not contain exactly one target per class")
    return indices.clone(), coefficients


def _update(mapper: torch.nn.Module, optimizer: torch.optim.Adam, model: Any,
            population: Mapping[str, Any], schedule: torch.Tensor,
            inputs: Mapping[str, Any], device: torch.device, step: int,
            *, audit: bool = False) -> dict[str, Any]:
    indices, coefficients = _batch(population, schedule, step)
    target = coefficients.to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(coefficients, inputs["scale"], device)
    matrix = inputs["matrix"].to(device=device, dtype=torch.float64)
    inverse = inputs["inverse"].to(device=device, dtype=torch.float64)
    scale = inputs["scale"].to(device=device, dtype=torch.float64)
    before = [parameter.detach().clone() for parameter in mapper.parameters()] if audit else None
    optimizer.zero_grad(set_to_none=True)
    amplitude, projection = projected._map_amplitude(mapper, features, target,
                                                      matrix, inverse, scale,
                                                      include_preclip=True)
    actual = projected._forward_amplitude(model, amplitude)
    coefficient_losses = pilot._phase_invariant_loss(actual["coefficients"], target)
    coefficient_loss = coefficient_losses.mean()
    box_loss = fit20._box_loss(projection["preclip_amplitude"].reshape(CLASS_COUNT, PIXEL_COUNT))
    total = coefficient_loss + BOX_LAMBDA * box_loss
    _require(torch.isfinite(total).item() and torch.isfinite(coefficient_losses).all().item(),
             "full-optics projected-box objective is nonfinite")
    total.backward()
    gradient_norms = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"mapper gradient is absent or nonfinite: {name}")
        gradient_norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()).item())
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, new.detach()) for old, new in
                     zip(before, mapper.parameters(), strict=True)),
                 "a declared mapper parameter tensor did not update")
    return {
        "completed_update": step + 1, "sample_indices": indices.tolist(),
        "labels": list(range(CLASS_COUNT)),
        "coefficient_loss": float(coefficient_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()), "total_loss": float(total.detach().cpu()),
        "coefficient_relative_error_mean": float(torch.sqrt(coefficient_losses.detach()).mean().cpu()),
        "coefficient_relative_error_rms": float(torch.sqrt(coefficient_losses.detach().mean()).cpu()),
        "clipped_fraction_mean": float(projection["clipped_fraction"].mean().detach().cpu()),
        "preclip_min": float(projection["preclip_min"].min().detach().cpu()),
        "preclip_max": float(projection["preclip_max"].max().detach().cpu()),
        "gradient_norms": gradient_norms, "loss_timing": "computed_before_this_adam_update",
        "full_production_coefficients": True,
    }


def _tree_delta(left: Any, right: Any) -> tuple[bool, float]:
    return projected._tree_max_delta(left, right)


def _checkpoint_payload(mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                        run_id: str, completed: int, history: Sequence[Mapping[str, Any]],
                        schedule: torch.Tensor, source_run_id: str) -> dict[str, Any]:
    return {
        "schema": "lightgen_g1_projected_box_fullpop_checkpoint.v1",
        "task_id": TASK_ID, "run_id": run_id, "source_run_id": source_run_id,
        "source_task_id": projected.TASK_ID, "source_mapper_updates": SOURCE_MAPPER_UPDATES,
        "new_update_budget": UPDATE_BUDGET, "completed_updates": completed,
        "checkpoint_step": completed, "next_schedule_cursor": completed,
        "next_batch_indices": schedule[completed // 1000, completed % 1000].clone()
        if completed < UPDATE_BUDGET else None,
        "mapper_state": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "torch_cpu_rng": torch.random.get_rng_state().clone(),
        "torch_cuda_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()],
        "history": copy.deepcopy(list(history)), "history_count": len(history),
        "optimizer_deepcopied": True,
        "schedule_identity": "g1_projected_box_fullpop_schedule_seed_2026102871:5x1000x10",
    }


def _validate_checkpoint(saved: Mapping[str, Any], run_id: str, source_run_id: str,
                         completed: int, history: Sequence[Mapping[str, Any]],
                         schedule: torch.Tensor) -> None:
    state_valid = _optimizer_state_valid(saved.get("optimizer_state"), completed)
    next_batch = saved.get("next_batch_indices")
    cursor_valid = ((completed == UPDATE_BUDGET and next_batch is None)
                    or (completed < UPDATE_BUDGET and torch.is_tensor(next_batch)
                        and torch.equal(next_batch, schedule[completed // 1000, completed % 1000])))
    _require(saved.get("schema") == "lightgen_g1_projected_box_fullpop_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("run_id") == run_id
             and saved.get("source_run_id") == source_run_id
             and saved.get("source_task_id") == projected.TASK_ID
             and saved.get("source_mapper_updates") == SOURCE_MAPPER_UPDATES
             and saved.get("new_update_budget") == UPDATE_BUDGET
             and saved.get("completed_updates") == completed
             and saved.get("checkpoint_step") == completed
             and saved.get("next_schedule_cursor") == completed
             and saved.get("history_count") == completed
             and len(saved.get("history", [])) == completed
             and list(saved.get("history", [])) == list(history)
             and _mapper_state_valid(saved.get("mapper_state"))
             and saved.get("optimizer_deepcopied") is True
             and saved.get("schedule_identity") ==
                 "g1_projected_box_fullpop_schedule_seed_2026102871:5x1000x10"
             and state_valid and cursor_valid,
             "full-population checkpoint identity, Adam, history, or schedule state is invalid")
    flat = schedule.reshape(UPDATE_BUDGET, CLASS_COUNT)
    for index, item in enumerate(history):
        _require(item.get("completed_update") == index + 1
                 and item.get("sample_indices") == flat[index].tolist()
                 and item.get("loss_timing") == "computed_before_this_adam_update"
                 and item.get("full_production_coefficients") is True,
                 f"training history row {index + 1} differs from the fixed production schedule")


def _save_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                     run_id: str, source_run_id: str, completed: int,
                     history: Sequence[Mapping[str, Any]], schedule: torch.Tensor) -> dict[str, Any]:
    rng_before = _rng_snapshot()
    _atomic_torch(path, _checkpoint_payload(mapper, optimizer, run_id=run_id,
                                             source_run_id=source_run_id, completed=completed,
                                             history=history, schedule=schedule))
    _require(_rng_equal(rng_before, _rng_snapshot()), "checkpoint save changed process RNG")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(saved, run_id, source_run_id, completed, history, schedule)
    return saved


def _restore_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Adam, *,
                        run_id: str, source_run_id: str,
                        schedule: torch.Tensor) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    _require(path.is_file(), f"explicit full-population checkpoint is missing: {path}", FileNotFoundError)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    completed = int(saved.get("completed_updates", -1))
    history = saved.get("history")
    _require(isinstance(history, list) and 0 <= completed < UPDATE_BUDGET,
             "explicit recovery checkpoint is terminal or has invalid progress")
    _validate_checkpoint(saved, run_id, source_run_id, completed, history, schedule)
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    _restore_rng({"cpu": saved["torch_cpu_rng"], "cuda": saved["torch_cuda_rng"]})
    return completed, copy.deepcopy(history), saved


def _atomic_history(path: Path, run_id: str, history: Sequence[Mapping[str, Any]], status: str) -> None:
    _atomic_json(path, {"schema": "lightgen_g1_projected_box_fullpop_history.v1",
                        "task_id": TASK_ID, "run_id": run_id,
                        "state": status, "source_mapper_updates": SOURCE_MAPPER_UPDATES,
                        "new_update_budget": UPDATE_BUDGET,
                        "completed_updates": len(history), "updates": list(history)})


def _box_contract() -> dict[str, float]:
    values = torch.tensor([-0.2, 0.0, 0.4, 1.0, 1.3], dtype=torch.float64, requires_grad=True)
    loss = fit20._box_loss(values)
    gradient, = torch.autograd.grad(loss, values)
    expected = ((torch.relu(-values.detach()).square()
                 + torch.relu(values.detach() - 1.0).square()).mean())
    expected_gradient = 2.0 * (values.detach() - values.detach().clamp(0.0, 1.0)) / values.numel()
    _require(torch.allclose(loss.detach(), expected, atol=1.0e-15, rtol=0.0)
             and torch.allclose(gradient, expected_gradient, atol=1.0e-15, rtol=0.0),
             "preclip box loss or analytic derivative disagrees with independent expression")
    epsilon = 1.0e-6
    finite = (float(fit20._box_loss(values.detach() + epsilon * torch.tensor([1, 0, 0, 0, -1]))
                                  .item())
              - float(fit20._box_loss(values.detach() - epsilon * torch.tensor([1, 0, 0, 0, -1]))
                       .item())) / (2.0 * epsilon)
    analytic = float((gradient * torch.tensor([1, 0, 0, 0, -1], dtype=torch.float64)).sum())
    _require(abs(finite - analytic) <= 1.0e-9, "preclip box finite-difference direction check failed")
    return {"value": float(loss.detach()), "gradient_max_abs_delta": 0.0,
            "directional_derivative": analytic, "finite_difference": finite}


def _training_batch(population: Mapping[str, Any], schedule: torch.Tensor,
                    inputs: Mapping[str, Any], mapper: torch.nn.Module,
                    optimizer: torch.optim.Adam, model: Any, device: torch.device,
                    step: int, *, audit: bool = False) -> dict[str, Any]:
    return _update(mapper, optimizer, model, population, schedule, inputs, device, step, audit=audit)


def _recovery_worker(config_path: Path, checkpoint_path: Path, output_path: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "independent recovery worker requires AutoDL CUDA")
    inputs = _load_inputs(config)
    population_payload = torch.load(PREFLIGHT_DIR / "fixed_population_and_schedule.pt",
                                    map_location="cpu", weights_only=False)
    population, schedule = _load_population_payload(population_payload)
    identity = _json(PREFLIGHT_DIR / "preflight_identity.json")
    run_id = str(identity["run_id"])
    source_run_id = str(identity["source_run_id"])
    device = torch.device("cuda:0")
    mapper = projected._new_mapper(device, inputs["source_mapper"]["mapper_state"])
    optimizer = _optimizer(mapper, config)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    completed, history, _ = _restore_checkpoint(checkpoint_path, mapper, optimizer,
                                                run_id=run_id, source_run_id=source_run_id,
                                                schedule=schedule)
    _require(completed == 3 and len(history) == 3,
             "independent recovery worker requires the exact third-step checkpoint")
    item = _training_batch(population, schedule, inputs, mapper, optimizer, model,
                           device, 3, audit=False)
    item["completed_update"] = 4
    history.append(item)
    payload = _checkpoint_payload(mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                                  completed=4, history=history, schedule=schedule)
    payload["engineering_adam_calls"] = 1
    _atomic_torch(output_path, payload)


def _load_population_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    population = {key: payload[key] for key in ("train", "validation", "development",
                                                 "training_probe", "known_reachable", "holdout",
                                                 "excluded_fit20_validation_indices", "role_contract")}
    schedule = payload["training_schedule"]
    return population, schedule


def _load_saved_population(path: Path, *, expected_run_id: str | None = None,
                           expected_source_run_id: str | None = None
                           ) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "lightgen_g1_projected_box_fullpop_population.v1"
             and payload.get("task_id") == TASK_ID
             and (expected_run_id is None or payload.get("run_id") == expected_run_id)
             and (expected_source_run_id is None
                  or payload.get("source_run_id") == expected_source_run_id),
             "saved full-population population identity is invalid")
    population, schedule = _load_population_payload(payload)
    _require(tuple(schedule.shape) == (5, 1000, CLASS_COUNT)
             and tuple(population["train"]["coefficients"].shape) == (TRAIN_ROWS, PORT_COUNT)
             and tuple(population["development"]["coefficients"].shape) == (DEV_ROWS, PORT_COUNT)
             and tuple(population["training_probe"]["coefficients"].shape) == (PROBE_ROWS, PORT_COUNT)
             and tuple(population["known_reachable"]["coefficients"].shape) == (TARGET_COUNT, PORT_COUNT)
             and tuple(population["holdout"]["coefficients"].shape) == (HOLDOUT_ROWS, PORT_COUNT),
             "saved role-labeled population or schedule has wrong shape")
    _require(_assert_population_independence(population) == TRAIN_ROWS + VALIDATION_ROWS + HOLDOUT_ROWS,
             "saved full-population coefficient roles are not row-wise independent")
    return population, schedule


def _recompute_population_source(config: Mapping[str, Any], inputs: Mapping[str, Any],
                                 evidence_dir: Path, identity: ResearchRunIdentity
                                 ) -> dict[str, Any]:
    expected, expected_schedule = _population(config, inputs, host="windows")
    saved, saved_schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    roles = ("train", "validation", "development", "training_probe", "known_reachable", "holdout")
    for role in roles:
        for key in ("coefficients", "labels", "proposal_ids", "source_validation_indices"):
            if key in expected[role]:
                _require(key in saved[role] and torch.equal(saved[role][key], expected[role][key]),
                         f"saved {role} population differs from its fixed source at {key}")
        _require(saved[role].get("source_role") == expected[role].get("source_role"),
                 f"saved {role} population source role changed")
    _require(torch.equal(saved_schedule, expected_schedule)
             and torch.equal(saved["excluded_fit20_validation_indices"],
                             expected["excluded_fit20_validation_indices"])
             and saved["role_contract"] == expected["role_contract"],
             "saved full-population schedule or role contract differs from deterministic source replay")
    raw = torch.load(evidence_dir / "fixed_population_and_schedule.pt",
                     map_location="cpu", weights_only=False)
    _require(raw.get("proposal_accounting") == expected["proposal_accounting"]
             and raw.get("rejection_rule") == expected["rejection_rule"]
             and raw.get("power_bound_w") == expected["power_bound_w"],
             "saved final holdout sampling accounting or rejection rule changed")
    return {"status": "FIXED_POPULATION_SOURCE_RECOMPUTED", "train": TRAIN_ROWS,
            "full_validation": VALIDATION_ROWS, "development": DEV_ROWS,
            "training_probe": PROBE_ROWS, "known_reachable": TARGET_COUNT,
            "new_final_holdout": HOLDOUT_ROWS,
            "unique_coefficient_rows": expected["unique_coefficient_rows"],
            "holdout_sampling_accounting_exact": True, "schedule_exact": True}


def _population_payload(population: Mapping[str, Any], schedule: torch.Tensor,
                        run_id: str, source_run_id: str) -> dict[str, Any]:
    return {"schema": "lightgen_g1_projected_box_fullpop_population.v1",
            "task_id": TASK_ID, "run_id": run_id, "source_run_id": source_run_id,
            "train": population["train"], "validation": population["validation"],
            "development": population["development"],
            "training_probe": population["training_probe"],
            "known_reachable": population["known_reachable"],
            "holdout": population["holdout"],
            "excluded_fit20_validation_indices": population["excluded_fit20_validation_indices"],
            "role_contract": population["role_contract"],
            "proposal_accounting": population["proposal_accounting"],
            "rejection_rule": population["rejection_rule"],
            "power_bound_w": population["power_bound_w"],
            "training_schedule": schedule.clone(),
            "populations": {"train_count": TRAIN_ROWS, "development_count": DEV_ROWS,
                            "training_probe_count": PROBE_ROWS, "known_reachable_count": TARGET_COUNT,
                            "new_final_holdout_count": HOLDOUT_ROWS}}


def _preflight_identity_path() -> Path:
    return PREFLIGHT_DIR / "preflight_identity.json"


def _run_recovery_preflight(config_path: Path, config: Mapping[str, Any],
                            inputs: Mapping[str, Any], population: Mapping[str, Any],
                            schedule: torch.Tensor, device: torch.device,
                            run_id: str, source_run_id: str) -> dict[str, Any]:
    recovery_dir = PREFLIGHT_DIR / "engineering_recovery"
    recovery_dir.mkdir(exist_ok=True)
    initial_rng = _rng_snapshot()
    torch.manual_seed(int(config["training"]["rng_seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
    mapper = projected._new_mapper(device, inputs["source_mapper"]["mapper_state"])
    optimizer = _optimizer(mapper, config)
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = projected._clone_state(model)
    mapper_before = [parameter.detach().clone() for parameter in mapper.parameters()]
    history = []
    for index in range(3):
        item = _training_batch(population, schedule, inputs, mapper, optimizer, model,
                               device, index, audit=index == 0)
        history.append(item)
    _require(all(not torch.equal(before, after.detach()) for before, after in
                 zip(mapper_before, mapper.parameters(), strict=True)),
             "real CUDA preflight did not update all mapper parameter tensors")
    cp_path = recovery_dir / "checkpoint_step_000003.pt"
    _save_checkpoint(cp_path, mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                     completed=3, history=history, schedule=schedule)
    live_item = _training_batch(population, schedule, inputs, mapper, optimizer, model,
                                device, 3, audit=False)
    live_history = [*history, live_item]
    live = _checkpoint_payload(mapper, optimizer, run_id=run_id, source_run_id=source_run_id,
                               completed=4, history=live_history, schedule=schedule)
    live["engineering_adam_calls"] = 1
    _atomic_torch(recovery_dir / "original_process_step4.pt", live)
    independent_path = recovery_dir / "independent_process_step4.pt"
    command = [sys.executable, "-m", MODULE_PATH, "--mode", "recovery-worker",
               "--config", str(config_path.resolve()), "--checkpoint", str(cp_path.resolve()),
               "--evidence-dir", str(recovery_dir.resolve()),
               "--worker-output", str(independent_path.resolve())]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _require(result.returncode == 0 and independent_path.is_file(),
             f"independent CUDA recovery process failed: {result.stderr[-3000:]}")
    restored = torch.load(independent_path, map_location="cpu", weights_only=False)
    mapper_equal, mapper_delta = _tree_delta(live["mapper_state"], restored["mapper_state"])
    adam_equal, adam_delta = _tree_delta(live["optimizer_state"], restored["optimizer_state"])
    live_loss = float(live_history[-1]["total_loss"])
    restored_loss = float(restored["history"][-1]["total_loss"])
    loss_delta = abs(live_loss - restored_loss)
    rng_equal = _rng_equal({"cpu": live["torch_cpu_rng"], "cuda": live["torch_cuda_rng"]},
                           {"cpu": restored["torch_cpu_rng"], "cuda": restored["torch_cuda_rng"]})
    live_indices = torch.as_tensor(live["history"][-1]["sample_indices"], dtype=torch.long)
    restored_indices = torch.as_tensor(restored["history"][-1]["sample_indices"], dtype=torch.long)
    schedule_equal = (torch.equal(live_indices, restored_indices)
                      and torch.equal(schedule[0, 3], live_indices))
    _require(mapper_equal and adam_equal and mapper_delta <= RECOVERY_TOLERANCE
             and adam_delta <= RECOVERY_TOLERANCE and loss_delta <= RECOVERY_TOLERANCE
             and rng_equal and schedule_equal
             and _optimizer_steps(live["optimizer_state"]) == [4] * 4
             and _optimizer_steps(restored["optimizer_state"]) == [4] * 4,
             "live and restored fourth full-population update do not agree")
    projected._assert_frozen(model, frozen_before, "full-population recovery frozen optics")
    _restore_rng(initial_rng)
    return {"pre_updates": 3, "live_step": 4, "restored_step": 4,
            "engineering_adam_calls": 5, "scientific_updates": 0,
            "mapper_max_abs_delta": mapper_delta, "adam_max_abs_delta": adam_delta,
            "total_loss_abs_delta": loss_delta, "rng_exact": rng_equal,
            "schedule_exact": schedule_equal, "full_four_step_history_saved": True,
            "frozen_optics_exact": True, "mapper_all_parameter_tensors_updated": True,
            "optimizer_serialization_deepcopied": True}


def _witness_preflight(inputs: Mapping[str, Any], population: Mapping[str, Any],
                       device: torch.device) -> dict[str, Any]:
    witness = population["known_reachable_witness"]
    known = population["known_reachable"]
    amplitudes = witness["amplitudes"].to(device="cpu", dtype=torch.float32)
    rows, measured = fit20._measure_commands(known, amplitudes, inputs, device, GRIDS)
    summary = projected._summarize_rows(rows)
    _require(summary["all_rows_passed"] and summary["passed_rows"] == TARGET_COUNT,
             "known-reachable positive-control witnesses do not pass the six-metric full-optics check")
    _atomic_torch(PREFLIGHT_DIR / "known_reachable_witness_replay.pt", {
        "task_id": TASK_ID, "population_role": "positive_control_only",
        "target_coefficients": known["coefficients"], "amplitudes": measured["amplitudes"],
        "outputs": measured["outputs"], "direct_raw_cmos": measured["direct_raw_cmos"],
        "rows": rows, "summary": summary,
    })
    return {"status": "PASS", "count": TARGET_COUNT, "passed": summary["passed_rows"],
            "six_metrics_recomputed": True, "used_as_training_labels": False,
            "used_for_inference_fallback": False, "summary": summary}


def _estimate_storage(config: Mapping[str, Any], inputs: Mapping[str, Any],
                      population: Mapping[str, Any], device: torch.device) -> dict[str, int]:
    known = population["known_reachable"]
    probe = {"holdout": {key: value[:10].clone() for key, value in known.items()
                          if torch.is_tensor(value)}}
    mapper = projected._new_mapper(device, inputs["source_mapper"]["mapper_state"]).eval()
    models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    frozen = {grid: projected._clone_state(model) for grid, model in models.items()}
    measured = _measure_shard(population=probe, baseline=mapper, candidate=mapper,
                              inputs=inputs, models=models, device=device,
                              start=0, stop=10, run_id="storage-size-probe")
    replay = {"schema": "lightgen_g1_projected_box_fullpop_cuda_command_replay.v1",
              "task_id": TASK_ID, "run_id": "storage-size-probe",
              "population_indices": torch.arange(10),
              "baseline_commands": measured["baseline_commands"],
              "candidate_commands": measured["candidate_commands"]}
    consumed = _consume_shard(measured, replay, probe, inputs["response_matrix"],
                              0, 10, "storage-size-probe")
    _require(all(len(consumed[role]) == 10 for role in consumed),
             "production measurement consumer did not reconsume its actual CUDA sample shard")
    wrong_target = copy.deepcopy(measured)
    wrong_target["target_coefficients"][0, 0] += 1.0 + 0.5j
    try:
        _consume_shard(wrong_target, replay, probe, inputs["response_matrix"],
                       0, 10, "storage-size-probe")
    except RuntimeError as error:
        wrong_target_rejected = "fixed source" in str(error)
    else:
        wrong_target_rejected = False
    _require(wrong_target_rejected,
             "production measurement consumer accepted a wrong target paired with actual CUDA outputs")
    wrong_encoding = copy.deepcopy(measured)
    wrong_encoding["baseline_commands"] = measured["baseline_commands"].square()
    wrong_encoding["candidate_commands"] = measured["candidate_commands"].square()
    wrong_replay = {**replay, "baseline_commands": wrong_encoding["baseline_commands"],
                    "candidate_commands": wrong_encoding["candidate_commands"]}
    try:
        _consume_shard(wrong_encoding, wrong_replay, probe, inputs["response_matrix"],
                       0, 10, "storage-size-probe")
    except RuntimeError as error:
        wrong_encoding_rejected = "command response differs from actual OLS output" in str(error)
    else:
        wrong_encoding_rejected = False
    _require(wrong_encoding_rejected,
             "production measurement consumer accepted a wrong amplitude/intensity encoding")
    probe_dir = PREFLIGHT_DIR / "storage_size_probe"
    probe_dir.mkdir(exist_ok=True)
    measurement_path = probe_dir / "measurement_shard.pt"
    replay_path = probe_dir / "command_replay_shard.pt"
    _atomic_torch(measurement_path, measured)
    _atomic_torch(replay_path, {
        "schema": "lightgen_g1_projected_box_fullpop_cuda_command_replay.v1",
        "task_id": TASK_ID, "run_id": "storage-size-probe",
        "population_indices": torch.arange(10),
        "baseline_commands": measured["baseline_commands"],
        "candidate_commands": measured["candidate_commands"],
        "batch_size": 1, "independent_process": True,
    })
    for grid in GRIDS:
        projected._assert_frozen(models[grid], frozen[grid],
                                 f"C optics grid {grid} during storage-size probe")
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in mapper.parameters())
    full_checkpoint_bytes = math.ceil(parameter_bytes * 3.1)
    checkpoint_peak_bytes = math.ceil(parameter_bytes * 1.1) + 7 * full_checkpoint_bytes
    measurement_bytes = (measurement_path.stat().st_size + replay_path.stat().st_size) * (
        HOLDOUT_ROWS // 10)
    copied_preflight_bytes = sum(path.stat().st_size for path in PREFLIGHT_DIR.rglob("*")
                                 if path.is_file())
    auxiliary_reserve_bytes = 128 * 1024 * 1024
    projected_peak_bytes = math.ceil((measurement_bytes + checkpoint_peak_bytes
                                      + copied_preflight_bytes + auxiliary_reserve_bytes) * 1.10)
    del models, mapper, measured
    torch.cuda.empty_cache()
    return {"measurement_and_independent_command_replay_bytes": measurement_bytes,
            "projected_checkpoint_peak_bytes": checkpoint_peak_bytes,
            "copied_preflight_evidence_bytes": copied_preflight_bytes,
            "auxiliary_reserve_bytes": auxiliary_reserve_bytes,
            "projected_peak_bytes_with_margin": projected_peak_bytes,
            "measurement_probe_file_bytes": measurement_path.stat().st_size,
            "command_replay_probe_file_bytes": replay_path.stat().st_size,
            "production_shard_reconsumed": True,
            "wrong_target_rejected": wrong_target_rejected,
            "wrong_amplitude_intensity_encoding_rejected": wrong_encoding_rejected}


def _load_control() -> tuple[ResearchRunControl, PreflightReceipt]:
    payload = _json(_preflight_identity_path())
    _require(payload.get("task_id") == TASK_ID, "full-population preflight identity is foreign")
    identity = ResearchRunIdentity.from_mapping(payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
    summary = _json(PREFLIGHT_DIR / "preflight_summary.json")
    _require_projector_preflight(summary, identity)
    _require(receipt.status == "READY" and receipt.authorization_id == AUTHORITY
             and receipt.source_semantics_id == identity.execution_source_semantics_id,
             "full-population execution requires the complete READY preflight receipt")
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    return control, receipt


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "full-population preflight requires AutoDL CUDA")
    _set_numerics()
    inputs = _load_inputs(config)
    repair_lineage = _validate_prior_b_repair(config, inputs, host="cloud")
    _require(not ATTEMPT_DIR.exists(),
             "full-population attempt path already exists; a new scientific attempt cannot reuse it",
             FileExistsError)
    population, schedule = _population(config, inputs, host="cloud")
    lineage = {"fit20_review_status": repair_lineage["predecessor_fit20_review_status"],
               "fit20_b_convergence_run_id": repair_lineage["predecessor_fit20_run_id"],
               "fit20_b_repair": repair_lineage,
               "v2_terminal_checkpoint": str(inputs["source_mapper_path"]),
               "v2_run_id": inputs["source_mapper"]["run_id"],
               "v2_terminal_step": inputs["source_mapper"]["completed_updates"]}
    prior_identity = None
    if PREFLIGHT_DIR.exists():
        summary_path = PREFLIGHT_DIR / "preflight_summary.json"
        _require(not (summary_path.is_file() and _json(summary_path).get("status") == "READY"),
                 "full-population preflight is already READY and terminal", FileExistsError)
        if _preflight_identity_path().is_file():
            prior_identity = _json(_preflight_identity_path())
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if prior_identity is None:
        run_id = f"g1boxfull_{uuid.uuid4().hex}"
        identity = _run_identity(config, run_id)
        _atomic_json(_preflight_identity_path(), {"task_id": TASK_ID, "run_id": run_id,
                                                  "source_run_id": inputs["source_mapper"]["run_id"],
                                                  "run_identity": identity.to_dict()})
        _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.json", config)
    else:
        run_id = str(prior_identity["run_id"])
        prior_config = copy.deepcopy(prior_identity["run_identity"].get("config_identity"))
        current_config = copy.deepcopy(config)
        projector_key = "v2_fixed_response_projector_evidence"
        prior_source = prior_config.get("source", {}) if isinstance(prior_config, dict) else {}
        current_source = current_config.get("source", {})
        projector_repair = (isinstance(prior_config, dict)
                            and projector_key not in prior_source
                            and current_source.get(projector_key) == config["source"][projector_key])
        if projector_repair:
            prior_source.pop(projector_key, None)
            current_source.pop(projector_key, None)
        identity = ResearchRunIdentity.from_mapping(prior_identity["run_identity"])
        expected_identity = _run_identity(config, run_id)
        _require(prior_identity.get("task_id") == TASK_ID
                 and prior_identity.get("source_run_id") == inputs["source_mapper"]["run_id"]
                 and (identity.execution_semantics() == expected_identity.execution_semantics()
                      or (projector_repair and prior_config == current_config)),
                 "existing full-population preflight identity or source differs")
        if identity.execution_semantics() != expected_identity.execution_semantics():
            identity_history = PREFLIGHT_DIR / "preflight_identity_before_fixed_projector_repair.json"
            config_history = PREFLIGHT_DIR / "run_config_before_fixed_projector_repair.json"
            if not identity_history.exists():
                _atomic_json(identity_history, prior_identity)
            if not config_history.exists():
                _atomic_json(config_history, _json(PREFLIGHT_DIR / "run_config_snapshot.json"))
            identity = expected_identity
            _atomic_json(_preflight_identity_path(), {
                "task_id": TASK_ID, "run_id": run_id,
                "source_run_id": inputs["source_mapper"]["run_id"],
                "run_identity": identity.to_dict(),
            })
            _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.json", config)
    source_run_id = str(inputs["source_mapper"]["run_id"])
    device = torch.device("cuda:0")
    projector_check = _verify_fixed_projector(
        inputs["matrix"].to(device=device, dtype=torch.float64),
        inputs["inverse"].to(device=device, dtype=torch.float64))
    projector_check.update({"source_path": str(inputs["projector_evidence_path"]),
                            "normalized_matrix_max_abs_error": inputs["projector_matrix_max_abs_error"]})
    _atomic_torch(PREFLIGHT_DIR / "fixed_population_and_schedule.pt",
                  _population_payload(population, schedule, run_id, source_run_id))
    _atomic_torch(PREFLIGHT_DIR / "response_projection.pt", {
        "task_id": TASK_ID, "response": inputs["response_matrix"].clone(),
        "matrix": inputs["matrix"].clone(), "inverse": inputs["inverse"].clone(),
        "scale": inputs["scale"].clone(), "camera_pixel_area_m2": 6.25e-12,
        "camera_gain_w_inverse": float(inputs["camera_gain"]),
    })
    box_check = _box_contract()
    witness = _witness_preflight(inputs, population, device)
    fit20_population = population["known_reachable"]
    probe_inputs = dict(inputs)
    probe_inputs["terminal_mapper"] = inputs["source_mapper"]
    response_direction = fit20._fit20_objective_gradient_check(
        probe_inputs, fit20_population, inputs["matrix"].to(device), inputs["inverse"].to(device),
        inputs["scale"].to(device), device)
    _require(response_direction.get("full_production_path") is True
             and response_direction.get("descent_total_loss", math.inf)
             < response_direction.get("center_total_loss", -math.inf),
             "full production Lc+5Lbox descent control failed")
    initial_rng = {"cpu": torch.random.get_rng_state().clone(),
                   "cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]}
    torch.manual_seed(int(config["training"]["rng_seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
    recovery = _run_recovery_preflight(config_path, config, inputs, population, schedule,
                                       device, run_id, source_run_id)
    _restore_rng(initial_rng)
    _atomic_json(PREFLIGHT_DIR / "engineering_recovery.json", recovery)
    storage = _estimate_storage(config, inputs, population, device)
    disk = shutil.disk_usage(RUNS_DIR.drive or ROOT)
    _require(disk.free >= storage["projected_peak_bytes_with_margin"],
             "actual free storage does not cover the measured shard, checkpoint, preflight-copy, and reserve estimate",
             OSError)
    correctness = {
        "passed": True, "status": "PASS", "explicit_v2_terminal_step_5000_verified": True,
        "new_adam_empty_at_start": True, "full_train_population_and_roles_verified": True,
        "holdout_seed_and_power_rejection_verified": True,
        "known_reachable_witnesses_passed_all_six_metrics": True,
        "fixed_response_matrix_and_moore_penrose_projector_verified": True,
        "box_loss_value_derivative_and_finite_difference_passed": True,
        "full_production_Lc_plus_5Lbox_descent_passed": True,
        "three_step_and_independent_fourth_recovery_passed": True,
        "all_mapper_groups_updated_and_optics_frozen": True,
        "scientific_updates": 0,
    }
    memory = torch.cuda.mem_get_info(device)
    resource = {"passed": True, "status": "PASS", "device": torch.cuda.get_device_name(device),
                "free_memory_bytes_observation": int(memory[0]),
                "total_memory_bytes_observation": int(memory[1]),
                "free_memory_is_gate": False,
                "data_disk_free_bytes_observation": int(disk.free),
                **storage}
    implementation = {"passed": True, "status": "PASS",
                      "selected_engine": "CUDA_full_C_encoder_OLS_decoder_production_path",
                      "new_scientific_updates": UPDATE_BUDGET,
                      "source_mapper_step": 5000, "source_adam_inherited": False,
                      "metric_threshold": GATE_LIMIT}
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(correctness=correctness, resource=resource,
                                         implementation=implementation,
                                         selected_engine="CUDA_full_C_encoder_OLS_decoder_production_path",
                                         resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    _atomic_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
    summary = {"task_id": TASK_ID, "run_id": run_id, "status": "READY",
               "lineage": lineage, "population_counts": {"train": TRAIN_ROWS,
               "development": DEV_ROWS, "training_probe": PROBE_ROWS,
               "known_reachable": TARGET_COUNT, "new_holdout": HOLDOUT_ROWS},
               "box_contract": box_check, "production_descent_check": response_direction,
               "fixed_projector_verification": projector_check,
               "positive_control": witness, "engineering_recovery": recovery,
               "resource_observations": resource, "storage_estimate": storage,
               "scientific_updates": 0}
    _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
    return {"status": "READY", "run_id": run_id, "engineering_adam_calls": 5,
            "scientific_updates": 0,
            "estimated_peak_bytes": storage["projected_peak_bytes_with_margin"]}


def _assert_attempt(attempt_dir: Path) -> ResearchRunIdentity:
    preflight = _json(attempt_dir / "preflight_evidence/preflight_identity.json")
    identity = ResearchRunIdentity.from_mapping(_json(attempt_dir / "run_identity.json"))
    expected = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    expected = expected.with_scientific_attempt(f"{expected.run_id}:attempt_01")
    _require(preflight.get("task_id") == TASK_ID
             and identity.execution_semantics() == expected.execution_semantics(),
             "full-population attempt differs from its READY task identity")
    receipt = PreflightReceipt.from_mapping(
        _json(attempt_dir / "preflight_evidence/preflight_receipt.json"))
    _require_projector_preflight(
        _json(attempt_dir / "preflight_evidence/preflight_summary.json"), expected)
    _require(receipt.status == "READY" and receipt.authorization_id == AUTHORITY
             and receipt.source_semantics_id == expected.execution_source_semantics_id,
             "full-population attempt did not consume its READY receipt")
    return identity


def _copy_preflight(attempt_dir: Path) -> None:
    destination = attempt_dir / "preflight_evidence"
    destination.mkdir(exist_ok=False)
    for source in PREFLIGHT_DIR.rglob("*"):
        if source.is_file():
            target = destination / source.relative_to(PREFLIGHT_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _task_state(attempt_dir: Path, state: str, completed: int) -> None:
    identity = _json(attempt_dir / "run_identity.json")
    _atomic_json(attempt_dir / "task_state.json", {
        "schema": "lightgen_g1_projected_box_fullpop_state.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"], "state": state,
        "completed_updates": completed, "authorized_new_updates": UPDATE_BUDGET,
        "source_mapper_updates": SOURCE_MAPPER_UPDATES,
        "cumulative_mapper_lineage_updates": LINEAGE_MAPPER_UPDATES,
        "frozen_optical_updates": 0, "updated_at_unix": time.time(),
    })


def _open_attempt(config: Mapping[str, Any], resume: bool) -> ResearchRunControl:
    control, receipt = _load_control()
    if not resume:
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01", receipt)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", control.identity.to_dict())
        _atomic_json(ATTEMPT_DIR / "run_config_snapshot.yaml", config)
        _copy_preflight(ATTEMPT_DIR)
        _atomic_torch(ATTEMPT_DIR / "fixed_population_and_schedule.pt",
                      torch.load(PREFLIGHT_DIR / "fixed_population_and_schedule.pt",
                                 map_location="cpu", weights_only=False))
        _task_state(ATTEMPT_DIR, RUN_SCIENTIFIC_ACTIVE, 0)
    else:
        _require(ATTEMPT_DIR.is_dir(), "explicit resume requires the existing full-population attempt")
        _assert_attempt(ATTEMPT_DIR)
        state = _json(ATTEMPT_DIR / "task_state.json")
        _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "INTERRUPTED"},
                 "closed full-population attempt cannot be resumed")
        control.open_scientific_attempt(ATTEMPT_DIR,
                                        f"{control.identity.run_id}:attempt_01", receipt, resume=True)
    return control


def _measure_observation(mapper: torch.nn.Module, population: Mapping[str, Any],
                         inputs: Mapping[str, Any], device: torch.device,
                         step: int) -> dict[str, Any]:
    models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    before = {grid: projected._clone_state(model) for grid, model in models.items()}
    results = {}
    for role, cohort in (("development", population["development"]),
                         ("training_probe", population["training_probe"])):
        rows = projected._measure_cohort(mapper, models, cohort, inputs["matrix"].to(device),
                                         inputs["inverse"].to(device), inputs["scale"].to(device),
                                         device, role=f"{role}_step_{step}", batch_size=10)
        values = np.asarray([float(row["coefficient_2048"]) for row in rows], dtype=np.float64)
        clips = np.asarray([float(row["clipped_fraction"]) for row in rows], dtype=np.float64)
        results[role] = {
            "count": len(rows), "coefficient_relative_error": {
                "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
                "median": float(np.median(values)), "maximum": float(values.max()),
            },
            "clipped_fraction": {"mean": float(clips.mean()),
                                 "median": float(np.median(clips)),
                                 "maximum": float(clips.max())},
            "per_class": {str(label): {
                "count": sum(int(row["label"]) == label for row in rows),
                "coefficient_error_mean": float(np.mean([row["coefficient_2048"]
                                                         for row in rows if int(row["label"]) == label])),
                "clipped_fraction_mean": float(np.mean([row["clipped_fraction"]
                                                        for row in rows if int(row["label"]) == label])),
            } for label in range(CLASS_COUNT)},
        }
        _write_observation_csv(ATTEMPT_DIR / "observations" / f"{role}_step_{step:04d}.csv", rows)
    for grid in GRIDS:
        projected._assert_frozen(models[grid], before[grid], f"C optics grid {grid} during observation")
    result = {"task_id": TASK_ID, "run_id": _json(ATTEMPT_DIR / "run_identity.json")["run_id"],
              "step": step, "source_mapper_updates": SOURCE_MAPPER_UPDATES,
              "observation_only": True, "no_model_selection": True,
              "cohort_results": results}
    _atomic_json(ATTEMPT_DIR / "observations" / f"step_{step:04d}.json", result)
    del models
    torch.cuda.empty_cache()
    return result


def _write_observation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["population_index", "label", "sample_index_within_class", "proposal_id",
              "coefficient_2048", "coefficient_4096", "native_cmos_2048", "native_cmos_4096",
              "cross_grid_coefficient", "cross_grid_native_cmos", "clipped_fraction"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _train(config_path: Path, config: Mapping[str, Any], inputs: Mapping[str, Any],
           attempt_dir: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "full-population training requires AutoDL CUDA")
    identity = _assert_attempt(attempt_dir)
    population, schedule = _load_saved_population(
        attempt_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    device = torch.device("cuda:0")
    mapper = projected._new_mapper(device, inputs["source_mapper"]["mapper_state"])
    optimizer = _optimizer(mapper, config)
    if optimizer.state:
        raise RuntimeError("full-population mapper must start with a new empty Adam")
    model = pilot._new_optical_model(inputs["checkpoint"], 2048, device)
    frozen_before = projected._clone_state(model)
    history: list[dict[str, Any]] = []
    if resume_checkpoint is None:
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        completed = 0
        _save_checkpoint(attempt_dir / "checkpoints" / "checkpoint_step_000000.pt",
                         mapper, optimizer, run_id=identity.run_id,
                         source_run_id=inputs["source_mapper"]["run_id"], completed=0,
                         history=history, schedule=schedule)
    else:
        completed, history, _ = _restore_checkpoint(
            resume_checkpoint, mapper, optimizer, run_id=identity.run_id,
            source_run_id=inputs["source_mapper"]["run_id"], schedule=schedule)
        _require(completed < UPDATE_BUDGET, "terminal full-population checkpoint cannot resume")
    (attempt_dir / "observations").mkdir(exist_ok=True)
    if completed == 0 and not (attempt_dir / "observations" / "step_0000.json").is_file():
        mapper.eval()
        _measure_observation(mapper, population, inputs, device, 0)
        mapper.train()
    _atomic_history(attempt_dir / "training_history.json", identity.run_id, history,
                    RUN_SCIENTIFIC_ACTIVE)
    _task_state(attempt_dir, RUN_SCIENTIFIC_ACTIVE, completed)
    for index in range(completed, UPDATE_BUDGET):
        mapper.train()
        record = _training_batch(population, schedule, inputs, mapper, optimizer, model,
                                 device, index, audit=index == completed)
        history.append(record)
        current = index + 1
        if current % 100 == 0:
            _atomic_history(attempt_dir / "training_history.json", identity.run_id, history,
                            RUN_SCIENTIFIC_ACTIVE)
            _atomic_json(attempt_dir / "progress.json", {
                "task_id": TASK_ID, "run_id": identity.run_id, "phase": "training",
                "completed_updates": current, "authorized_new_updates": UPDATE_BUDGET,
                "last_total_loss": record["total_loss"],
                "last_coefficient_loss": record["coefficient_loss"],
                "last_box_loss": record["box_loss"], "updated_at_unix": time.time(),
            })
            _task_state(attempt_dir, RUN_SCIENTIFIC_ACTIVE, current)
        if current % 500 == 0:
            _save_checkpoint(
                attempt_dir / "checkpoints" / "latest_checkpoint.pt", mapper, optimizer,
                run_id=identity.run_id, source_run_id=inputs["source_mapper"]["run_id"],
                completed=current, history=history, schedule=schedule)
            recent = attempt_dir / "checkpoints" / f"recent_{current:06d}.pt"
            shutil.copy2(attempt_dir / "checkpoints" / "latest_checkpoint.pt", recent)
            _require(torch.load(recent, map_location="cpu", weights_only=False)["completed_updates"] == current,
                     "recent checkpoint readback failed")
            recent_files = sorted((attempt_dir / "checkpoints").glob("recent_*.pt"))
            for stale in recent_files[:-2]:
                stale.unlink()
            if current in FIXED_CHECKPOINT_STEPS:
                shutil.copy2(recent, attempt_dir / "checkpoints" / f"checkpoint_step_{current:06d}.pt")
                _validate_checkpoint(torch.load(attempt_dir / "checkpoints" /
                                                f"checkpoint_step_{current:06d}.pt",
                                                map_location="cpu", weights_only=False),
                                     identity.run_id, inputs["source_mapper"]["run_id"],
                                     current, history, schedule)
        if current in OBSERVATION_STEPS[1:]:
            mapper.eval()
            _measure_observation(mapper, population, inputs, device, current)
            mapper.train()
        del record
    _atomic_history(attempt_dir / "training_history.json", identity.run_id, history,
                    "TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT")
    terminal = attempt_dir / "terminal_checkpoint.pt"
    _save_checkpoint(terminal, mapper, optimizer, run_id=identity.run_id,
                     source_run_id=inputs["source_mapper"]["run_id"],
                     completed=UPDATE_BUDGET, history=history, schedule=schedule)
    projected._assert_frozen(model, frozen_before, "C optical model throughout full-population training")
    _task_state(attempt_dir, "TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT", UPDATE_BUDGET)
    return {"status": "TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT",
            "completed_updates": len(history), "adam_steps": _optimizer_steps(optimizer.state_dict()),
            "terminal_checkpoint": str(terminal)}


def _execute(config_path: Path, config: Mapping[str, Any], inputs: Mapping[str, Any],
             *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    control = _open_attempt(config, resume_checkpoint is not None)
    result = _train(config_path, config, inputs, ATTEMPT_DIR,
                    resume_checkpoint=resume_checkpoint)
    return {"run_id": control.identity.run_id, **result}


def _map_one(mapper: torch.nn.Module, coefficients: torch.Tensor,
             inputs: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    feature = projected._normalized_features(coefficients.reshape(1, -1), inputs["scale"], device)
    target = coefficients.reshape(1, -1).to(device=device, dtype=torch.complex128)
    with torch.no_grad():
        amplitude, _ = projected._map_amplitude(
            mapper, feature, target, inputs["matrix"].to(device=device, dtype=torch.float64),
            inputs["inverse"].to(device=device, dtype=torch.float64),
            inputs["scale"].to(device=device, dtype=torch.float64),
        )
    return amplitude.detach().cpu().to(torch.float32)


def _measure_shard(population: Mapping[str, Any], baseline: torch.nn.Module,
                   candidate: torch.nn.Module, inputs: Mapping[str, Any],
                   models: Mapping[int, Any], device: torch.device,
                   start: int, stop: int, run_id: str) -> dict[str, Any]:
    target_cpu = population["holdout"]["coefficients"][start:stop]
    labels = population["holdout"]["labels"][start:stop]
    proposals = population["holdout"]["proposal_ids"][start:stop]
    baseline_commands, candidate_commands = [], []
    baseline_outputs = {grid: {"coefficients": [], "raw_cmos": []} for grid in GRIDS}
    candidate_outputs = {grid: {"coefficients": [], "raw_cmos": []} for grid in GRIDS}
    direct = {grid: [] for grid in GRIDS}
    rows = {"frozen_v2_baseline": [], "box_b_candidate": []}
    for offset in range(stop - start):
        idx = start + offset
        target = target_cpu[offset:offset + 1].to(device=device, dtype=torch.complex128)
        command_a = _map_one(baseline, target_cpu[offset], inputs, device).to(device)
        command_b = _map_one(candidate, target_cpu[offset], inputs, device).to(device)
        baseline_commands.append(command_a.cpu()[0])
        candidate_commands.append(command_b.cpu()[0])
        out_a, out_b, ref = {}, {}, {}
        for grid in GRIDS:
            with torch.no_grad():
                out_a[grid] = projected._forward_amplitude(models[grid], command_a)
                out_b[grid] = projected._forward_amplitude(models[grid], command_b)
                ref[grid] = projected._forward_target(models[grid], target)
            for key in ("coefficients", "raw_cmos"):
                baseline_outputs[grid][key].append(out_a[grid][key].detach().cpu()[0])
                candidate_outputs[grid][key].append(out_b[grid][key].detach().cpu()[0])
            direct[grid].append(ref[grid]["raw_cmos"].detach().cpu()[0])
        row_a = projected._error_row(out_a, target, {grid: ref[grid]["raw_cmos"] for grid in GRIDS}, 0)
        row_b = projected._error_row(out_b, target, {grid: ref[grid]["raw_cmos"] for grid in GRIDS}, 0)
        base_row = {"population_index": idx, "label": int(labels[offset]),
                    "sample_index_within_class": offset + start - int(labels[offset]) * 200,
                    "proposal_id": int(proposals[offset]), **row_a}
        box_row = {**base_row, **row_b}
        box_row["row_passed"] = all(float(box_row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
        base_row["row_passed"] = all(float(base_row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
        rows["frozen_v2_baseline"].append(base_row)
        rows["box_b_candidate"].append(box_row)
    return {
        "schema": "lightgen_g1_projected_box_fullpop_holdout_shard.v1",
        "task_id": TASK_ID, "run_id": run_id, "population_role": "new_final_holdout",
        "population_indices": torch.arange(start, stop), "labels": labels.clone(),
        "proposal_ids": proposals.clone(), "target_coefficients": target_cpu.clone(),
        "baseline_commands": torch.stack(baseline_commands).to(torch.float32),
        "candidate_commands": torch.stack(candidate_commands).to(torch.float32),
        "baseline_outputs": {grid: {key: torch.stack(items) for key, items in values.items()}
                             for grid, values in baseline_outputs.items()},
        "candidate_outputs": {grid: {key: torch.stack(items) for key, items in values.items()}
                              for grid, values in candidate_outputs.items()},
        "direct_raw_cmos": {grid: torch.stack(values) for grid, values in direct.items()},
        "rows": rows,
    }


def _independent_command_replay_worker(config_path: Path, evidence_dir: Path,
                                       output_dir: Path) -> None:
    config = _load_config(config_path)
    _set_numerics()
    _require(torch.cuda.is_available(), "independent terminal command replay requires AutoDL CUDA")
    inputs = _load_inputs(config)
    identity = _assert_attempt(evidence_dir)
    population, _ = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    baseline_state = inputs["source_mapper"]["mapper_state"]
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    _require(terminal.get("completed_updates") == UPDATE_BUDGET
             and terminal.get("run_id") == identity.run_id,
             "independent command replay requires this attempt's terminal checkpoint")
    _, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    fixed_terminal = torch.load(
        evidence_dir / "checkpoints" / f"checkpoint_step_{UPDATE_BUDGET:06d}.pt",
        map_location="cpu", weights_only=False)
    _require_terminal_checkpoint_bound(terminal, fixed_terminal, identity.run_id,
                                       inputs["source_mapper"]["run_id"], schedule)
    source_states = {
        "frozen_v2_baseline": {key: value.detach().cpu().clone()
                                for key, value in baseline_state.items()},
        "box_b_candidate": {key: value.detach().cpu().clone()
                             for key, value in terminal["mapper_state"].items()},
    }
    device = torch.device("cuda:0")
    baseline = projected._new_mapper(device, baseline_state).eval()
    candidate = projected._new_mapper(device, terminal["mapper_state"]).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "mapper_source_states.pt"
    source_payload = {
        "schema": "lightgen_g1_projected_box_fullpop_mapper_source_states.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "sources": {
            "frozen_v2_baseline": {
                "checkpoint_path": "source.v2_terminal_checkpoint",
                "checkpoint_step": 5000, "mapper_updates": SOURCE_MAPPER_UPDATES,
            },
            "box_b_candidate": {
                "checkpoint_path": "terminal_checkpoint.pt",
                "fixed_checkpoint_path": f"checkpoints/checkpoint_step_{UPDATE_BUDGET:06d}.pt",
                "checkpoint_step": UPDATE_BUDGET,
                "mapper_updates": LINEAGE_MAPPER_UPDATES,
            },
        },
        "mapper_states": source_states,
    }
    if source_path.is_file():
        existing_source = torch.load(source_path, map_location="cpu", weights_only=False)
        _require(_source_state_payload_equal(existing_source, source_payload),
                 "existing independent replay source states differ from loaded checkpoints")
    else:
        _atomic_torch(source_path, source_payload)
    for shard_no, start in enumerate(range(0, HOLDOUT_ROWS, 10)):
        stop = min(start + 10, HOLDOUT_ROWS)
        base, box = [], []
        for idx in range(start, stop):
            base.append(_map_one(baseline, population["holdout"]["coefficients"][idx], inputs, device)[0])
            box.append(_map_one(candidate, population["holdout"]["coefficients"][idx], inputs, device)[0])
        payload = {
            "schema": "lightgen_g1_projected_box_fullpop_cuda_command_replay.v1",
            "task_id": TASK_ID, "run_id": identity.run_id,
            "population_indices": torch.arange(start, stop),
            "baseline_commands": torch.stack(base), "candidate_commands": torch.stack(box),
            "batch_size": 1, "device": torch.cuda.get_device_name(device),
            "independent_process": True,
            "source_state_file": "mapper_source_states.pt",
            "source_state_roles": {"baseline": "frozen_v2_baseline",
                                   "candidate": "box_b_candidate"},
        }
        path = output_dir / f"commands_{shard_no:04d}.pt"
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            _require(saved.get("schema") == payload["schema"]
                     and saved.get("task_id") == TASK_ID and saved.get("run_id") == identity.run_id
                     and torch.equal(saved.get("population_indices"), payload["population_indices"])
                     and torch.equal(saved.get("baseline_commands"), payload["baseline_commands"])
                     and torch.equal(saved.get("candidate_commands"), payload["candidate_commands"]),
                     f"existing independent CUDA command replay differs: shard {shard_no}")
        else:
            _atomic_torch(path, payload)


def _source_state_payload_equal(left: Any, right: Mapping[str, Any]) -> bool:
    return (isinstance(left, Mapping)
            and left.get("schema") == right.get("schema")
            and left.get("task_id") == right.get("task_id")
            and left.get("run_id") == right.get("run_id")
            and left.get("sources") == right.get("sources")
            and isinstance(left.get("mapper_states"), Mapping)
            and all(_state_dicts_equal(left["mapper_states"].get(role), state)
                    for role, state in right["mapper_states"].items()))


def _require_terminal_checkpoint_bound(terminal: Mapping[str, Any], fixed: Mapping[str, Any],
                                       run_id: str, source_run_id: str,
                                       schedule: torch.Tensor) -> None:
    _validate_checkpoint(terminal, run_id, source_run_id, UPDATE_BUDGET,
                         terminal.get("history", []), schedule)
    _validate_checkpoint(fixed, run_id, source_run_id, UPDATE_BUDGET,
                         fixed.get("history", []), schedule)
    _require(_state_dicts_equal(terminal.get("mapper_state"), fixed.get("mapper_state"))
             and _state_dicts_equal(terminal.get("optimizer_state"), fixed.get("optimizer_state"))
             and terminal.get("history") == fixed.get("history")
             and torch.equal(terminal.get("torch_cpu_rng"), fixed.get("torch_cpu_rng"))
             and len(terminal.get("torch_cuda_rng", [])) == len(fixed.get("torch_cuda_rng", []))
             and all(torch.equal(left, right) for left, right in zip(
                 terminal.get("torch_cuda_rng", []), fixed.get("torch_cuda_rng", []), strict=True)),
             "terminal checkpoint mapper, Adam, history, or RNG differs from fixed step-5000 checkpoint")


def _measure_terminal(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "terminal full-population measurement requires AutoDL CUDA")
    identity = _assert_attempt(ATTEMPT_DIR)
    state = _json(ATTEMPT_DIR / "task_state.json")
    _require(state.get("state") in {"TRAINING_COMPLETE_AWAITING_HOLDOUT_MEASUREMENT",
                                     "MEASURING_NEW_FINAL_HOLDOUT"}
             and state.get("completed_updates") == UPDATE_BUDGET,
             "holdout is reserved until the fixed training terminal")
    terminal = torch.load(ATTEMPT_DIR / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    _validate_checkpoint(terminal, identity.run_id, inputs["source_mapper"]["run_id"],
                         UPDATE_BUDGET, terminal.get("history", []),
                         _load_saved_population(
                             ATTEMPT_DIR / "fixed_population_and_schedule.pt",
                             expected_run_id=identity.run_id,
                             expected_source_run_id=inputs["source_mapper"]["run_id"])[1])
    population, _ = _load_saved_population(
        ATTEMPT_DIR / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    device = torch.device("cuda:0")
    baseline = projected._new_mapper(device, inputs["source_mapper"]["mapper_state"]).eval()
    candidate = projected._new_mapper(device, terminal["mapper_state"]).eval()
    models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device) for grid in GRIDS}
    frozen = {grid: projected._clone_state(model) for grid, model in models.items()}
    shard_dir = ATTEMPT_DIR / "measurements" / "holdout"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _task_state(ATTEMPT_DIR, "MEASURING_NEW_FINAL_HOLDOUT", UPDATE_BUDGET)
    for shard_no, start in enumerate(range(0, HOLDOUT_ROWS, 10)):
        stop = min(start + 10, HOLDOUT_ROWS)
        shard_path = shard_dir / f"holdout_{shard_no:04d}.pt"
        if shard_path.is_file():
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == identity.run_id
                     and torch.equal(saved.get("population_indices"), torch.arange(start, stop)),
                     f"existing holdout shard {shard_no} is not a valid committed measurement")
        else:
            payload = _measure_shard(population, baseline, candidate, inputs, models,
                                     device, start, stop, identity.run_id)
            _atomic_torch(shard_path, payload)
        _atomic_json(ATTEMPT_DIR / "measurement_progress.json", {
            "task_id": TASK_ID, "run_id": identity.run_id,
            "completed_rows": stop, "total_rows": HOLDOUT_ROWS,
            "completed_shards": shard_no + 1, "updated_at_unix": time.time()})
    for grid in GRIDS:
        projected._assert_frozen(models[grid], frozen[grid], f"C optics grid {grid} during final holdout")
    del models, baseline, candidate
    torch.cuda.empty_cache()
    replay_dir = ATTEMPT_DIR / "measurements" / "cuda_command_replay"
    command = [sys.executable, "-m", MODULE_PATH, "--mode", "command-replay-worker",
               "--config", str(DEFAULT_CONFIG.resolve()), "--evidence-dir", str(ATTEMPT_DIR.resolve()),
               "--output-dir", str(replay_dir.resolve())]
    replay_result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _require(replay_result.returncode == 0 and replay_dir.is_dir(),
             f"independent terminal CUDA command replay failed: {replay_result.stderr[-3000:]}")
    decision = _finish_decision(ATTEMPT_DIR, inputs, write=True)
    _build_transfer_manifest(ATTEMPT_DIR)
    return {"status": decision["lifecycle_state"], "holdout_rows": HOLDOUT_ROWS,
            "shards": HOLDOUT_ROWS // 10,
            "candidate_summary": decision["candidate_summary"],
            "independent_command_replay": "PASSED"}


def _verify_transfer(evidence_dir: Path) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    receipt = _json(evidence_dir / "transfer_receipt.json")
    _require(manifest.get("schema") == "lightgen_g1_projected_box_fullpop_transfer.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list)
             and receipt.get("status") == "COMPLETE" and isinstance(receipt.get("files"), list),
             "full-population transfer manifest or receipt is invalid")
    expected = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in expected,
                 "full-population manifest path is unsafe or duplicated")
        expected[relative.as_posix()] = int(item["bytes"])
    actual_receipt = {}
    for item in receipt["files"]:
        relative = Path(str(item.get("path", "")))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and relative.as_posix() not in actual_receipt,
                 "full-population transfer receipt path is unsafe or duplicated")
        actual_receipt[relative.as_posix()] = item
    expected["transfer_manifest.json"] = (evidence_dir / "transfer_manifest.json").stat().st_size
    _require(set(expected) == set(actual_receipt),
             "full-population transfer receipt differs from manifest paths")
    files = []
    for relative, remote_size in sorted(expected.items()):
        path = evidence_dir / Path(relative)
        item = actual_receipt[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current == remote_size,
                 f"full-population transfer byte/status mismatch: {relative}")
        files.append({"path": relative, "remote_bytes": remote_size,
                      "receipt_local_bytes": int(item["local_size_bytes"]),
                      "current_bytes": current})
    _require(manifest.get("file_count") == len(manifest["files"])
             and manifest.get("bytes_total") == sum(item["bytes"] for item in manifest["files"]),
             "full-population manifest totals are inconsistent")
    _require(len(expected) == manifest["file_count"] + 1,
             "full-population transfer receipt must include the remote transfer manifest")
    return {"status": "COMPLETE", "file_count": len(files),
            "bytes_total": sum(item["current_bytes"] for item in files), "files": files}


def _consume_shard(shard: Mapping[str, Any], replay: Mapping[str, Any],
                   population: Mapping[str, Any], response: torch.Tensor,
                   start: int, stop: int, run_id: str,
                   source_states: Mapping[str, Any] | None = None,
                   ) -> dict[str, list[dict[str, Any]]]:
    _require(shard.get("schema") == "lightgen_g1_projected_box_fullpop_holdout_shard.v1"
             and shard.get("task_id") == TASK_ID and shard.get("run_id") == run_id
             and replay.get("schema") == "lightgen_g1_projected_box_fullpop_cuda_command_replay.v1"
             and replay.get("task_id") == TASK_ID and replay.get("run_id") == run_id
             and torch.equal(shard.get("population_indices"), torch.arange(start, stop))
             and torch.equal(replay.get("population_indices"), torch.arange(start, stop))
             and torch.equal(shard.get("target_coefficients"),
                             population["holdout"]["coefficients"][start:stop])
             and torch.equal(shard.get("labels"), population["holdout"]["labels"][start:stop])
             and torch.equal(shard.get("proposal_ids"), population["holdout"]["proposal_ids"][start:stop])
             and torch.equal(shard.get("baseline_commands"), replay.get("baseline_commands"))
             and torch.equal(shard.get("candidate_commands"), replay.get("candidate_commands")),
             "full-population shard target, identity, or command differs from its fixed source")
    if source_states is not None:
        _require(replay.get("source_state_file") == "mapper_source_states.pt"
                 and replay.get("source_state_roles") == {
                     "baseline": "frozen_v2_baseline", "candidate": "box_b_candidate"}
                 and _mapper_state_valid(source_states.get("frozen_v2_baseline"))
                 and _mapper_state_valid(source_states.get("box_b_candidate")),
                 "full-population command replay lacks its loaded mapper-state binding")
    rows = {"frozen_v2_baseline": [], "box_b_candidate": []}
    _require(all(len(shard.get("rows", {}).get(role, [])) == stop - start for role in rows),
             "full-population measurement rows are incomplete")
    for role, command_key, output_key in (
        ("frozen_v2_baseline", "baseline_commands", "baseline_outputs"),
        ("box_b_candidate", "candidate_commands", "candidate_outputs"),
    ):
        command = shard[command_key]
        _require(torch.is_tensor(command) and command.dtype == torch.float32
                 and tuple(command.shape) == (stop - start, 1, 200, 200)
                 and torch.isfinite(command).all().item()
                 and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                 f"malformed full-population amplitude command: {role}")
        predicted = command.reshape(stop - start, PIXEL_COUNT).to(torch.complex128) @ response.T
        outputs = shard[output_key]
        for grid in GRIDS:
            coefficients = outputs.get(grid, {}).get("coefficients")
            raw_cmos = outputs.get(grid, {}).get("raw_cmos")
            direct_cmos = shard.get("direct_raw_cmos", {}).get(grid)
            _require(torch.is_tensor(coefficients) and coefficients.dtype == torch.complex128
                     and tuple(coefficients.shape) == (stop - start, PORT_COUNT)
                     and torch.isfinite(coefficients.real).all().item()
                     and torch.isfinite(coefficients.imag).all().item()
                     and torch.is_tensor(raw_cmos) and raw_cmos.dtype == torch.float64
                     and tuple(raw_cmos.shape) == (stop - start, 1, 200, 200)
                     and torch.isfinite(raw_cmos).all().item()
                     and torch.is_tensor(direct_cmos) and direct_cmos.dtype == torch.float64
                     and tuple(direct_cmos.shape) == (stop - start, 1, 200, 200)
                     and torch.isfinite(direct_cmos).all().item(),
                     f"full-population raw output is malformed: {role}/{grid}")
        matrix_error = torch.sqrt(pilot._phase_invariant_loss(
            predicted, outputs[2048]["coefficients"].to(torch.complex128)))
        _require(float(matrix_error.max()) <= projected.MATRIX_LIMIT,
                 f"full-population command response differs from actual OLS output: {role}")
        for local in range(stop - start):
            index = start + local
            target = shard["target_coefficients"][local].to(torch.complex128)
            row_outputs = {grid: {key: value[local:local + 1] for key, value in outputs[grid].items()}
                           for grid in GRIDS}
            direct = {grid: shard["direct_raw_cmos"][grid][local:local + 1] for grid in GRIDS}
            metric = projected._error_row(row_outputs, target.reshape(1, -1), direct, 0)
            _require(_numeric_row_matches(shard["rows"][role][local], metric, 1.0e-14),
                     f"stored raw CMOS/coefficient metrics changed: {role}/{index}")
            row = {"population_index": index, "label": int(shard["labels"][local]),
                   "sample_index_within_class": index % 200,
                   "proposal_id": int(shard["proposal_ids"][local]), **metric}
            row["row_passed"] = all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS)
            _require(bool(shard["rows"][role][local].get("row_passed")) is row["row_passed"],
                     f"stored row decision changed: {role}/{index}")
            rows[role].append(row)
    return rows


def _rows_and_summary(evidence_dir: Path, population: Mapping[str, Any],
                      response: torch.Tensor, *,
                      replay_dir: Path | None = None,
                      source_states: Mapping[str, Any] | None = None
                      ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = {"frozen_v2_baseline": [], "box_b_candidate": []}
    run_id = _json(evidence_dir / "run_identity.json")["run_id"]
    for shard_no, start in enumerate(range(0, HOLDOUT_ROWS, 10)):
        path = evidence_dir / "measurements" / "holdout" / f"holdout_{shard_no:04d}.pt"
        replay_root = replay_dir or evidence_dir / "measurements" / "cuda_command_replay"
        replay_path = replay_root / f"commands_{shard_no:04d}.pt"
        _require(path.is_file() and replay_path.is_file(),
                 f"full-population holdout or command replay shard is missing: {shard_no}",
                 FileNotFoundError)
        shard = torch.load(path, map_location="cpu", weights_only=False)
        replay = torch.load(replay_path, map_location="cpu", weights_only=False)
        stop = min(start + 10, HOLDOUT_ROWS)
        _require(replay.get("batch_size") == 1 and replay.get("independent_process") is True,
                 f"terminal command replay is not an independent single-sample process: {shard_no}")
        consumed = _consume_shard(shard, replay, population, response, start, stop,
                                  run_id, source_states=source_states)
        for role in rows:
            rows[role].extend(consumed[role])
    summaries = {role: projected._summarize_rows(value) for role, value in rows.items()}
    return rows, summaries


def _numeric_row_matches(row: Mapping[str, Any], metric: Mapping[str, float], tolerance: float) -> bool:
    return all(key in row and math.isfinite(float(row[key]))
               and abs(float(row[key]) - float(metric[key])) <= tolerance for key in METRIC_KEYS)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["population_index", "label", "sample_index_within_class", "proposal_id",
              *METRIC_KEYS, "row_passed"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_report(path: Path, decision: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> None:
    terminal = decision["candidate_summary"]
    maxima = terminal["maxima"]
    box_start = float(history[0]["box_loss"])
    box_end = float(history[-1]["box_loss"])
    coefficient_start = float(history[0]["coefficient_loss"])
    coefficient_end = float(history[-1]["coefficient_loss"])
    report = f"""# G1全人口B候选：固定5,000步结果

## 结果与状态

新任务`{TASK_ID}`从原V2映射器终点新增恰好5,000步，全冻结光学更新为0。新固定留出为{terminal['denominator']}条，满足六项误差联合门槛的候选样本为{terminal['passed_rows']}/{terminal['denominator']}；科学状态为`{decision['lifecycle_state']}`。

## 科学认识

本轮唯一候选使用完整生产光路系数损失`Lc + 5Lbox`。首个与末个更新的更新前系数损失分别为{coefficient_start:.8g}/{coefficient_end:.8g}，越界惩罚分别为{box_start:.8g}/{box_end:.8g}。留出最大误差：2048/4096系数{maxima['coefficient_2048']:.8g}/{maxima['coefficient_4096']:.8g}，原生CMOS{maxima['native_cmos_2048']:.8g}/{maxima['native_cmos_4096']:.8g}，跨网格系数/CMOS{maxima['cross_grid_coefficient']:.8g}/{maxima['cross_grid_native_cmos']:.8g}。冻结V2基线只作同人口描述比较，本轮没有匹配A臂，不能单独归因越界惩罚的必要性。

## 目标缺口

最终判据仅检验固定新留出人口上的连续数值制备。它不证明人眼语义可读性、所有G1目标、实际器件或实物光路可用；真实几何仍为`PROJECT_GEOMETRY_UNMEASURED`。

## 工程完成与验证

V2起点、空Adam、人口角色、5,000条历史、固定日程、终点Adam步数、真实CUDA恢复及独立命令重放均由消费者复核。原始留出按10条分片保存目标、两指令、两候选两网格系数/原生CMOS和共用Decoder参考；Windows从分片独立重算指标与分母。

## 决策影响与下一步

{('全部2,000条满足六项门槛；本结果仅支持该固定新人口上的候选连续数值制备。' if decision['lifecycle_state'] == RUN_COMPLETED else '完整有效测量未满足全人口六项门槛；当前候选、固定损失和预算未达到目标。')}按固定计划停止，不自动续训、调权、扩网或转入量化、G4、评分或实物实验。历史G1`VALID_FAILED`、`PROJECT_GEOMETRY_UNMEASURED`及340张待评分状态保持。
"""
    path.write_text(report, encoding="utf-8")


def _finish_decision(evidence_dir: Path, inputs: Mapping[str, Any], *, write: bool,
                     replay_dir: Path | None = None,
                     source_states: Mapping[str, Any] | None = None) -> dict[str, Any]:
    identity = _assert_attempt(evidence_dir)
    population, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    history_payload = _json(evidence_dir / "training_history.json")
    history = history_payload["updates"]
    _require(len(history) == UPDATE_BUDGET and history_payload.get("completed_updates") == UPDATE_BUDGET
             and history_payload.get("run_id") == identity.run_id
             and _optimizer_steps(terminal["optimizer_state"]) == [UPDATE_BUDGET] * 4,
             "full-population terminal, history, or Adam count does not match the fixed budget")
    rows, summaries = _rows_and_summary(evidence_dir, population, inputs["response_matrix"],
                                        replay_dir=replay_dir, source_states=source_states)
    state = RUN_COMPLETED if summaries["box_b_candidate"]["all_rows_passed"] else RUN_VALID_FAILED
    decision = {
        "schema": "lightgen_g1_projected_box_fullpop_decision.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "lifecycle_state": state, "population_role": "new_fixed_final_holdout",
        "denominator": HOLDOUT_ROWS, "threshold": GATE_LIMIT,
        "metric_names": list(METRIC_KEYS), "candidate": "single_B_Lc_plus_5Lbox",
        "candidate_summary": summaries["box_b_candidate"],
        "frozen_v2_baseline_summary": summaries["frozen_v2_baseline"],
        "paired_comparison_is_descriptive_only": True,
        "new_scientific_updates": UPDATE_BUDGET,
        "source_mapper_updates": SOURCE_MAPPER_UPDATES,
        "cumulative_mapper_lineage_updates": LINEAGE_MAPPER_UPDATES,
        "adam_parameter_steps": _optimizer_steps(terminal["optimizer_state"]),
        "frozen_optical_updates": 0,
        "native_cmos_normalization": "none",
        "coefficient_alignment": "one_global_phase_per_sample",
        "rows": {key: value for key, value in rows.items()},
    }
    if write:
        _atomic_json(evidence_dir / "final_decision.json", decision)
        _write_rows_csv(evidence_dir / "candidate_holdout_rows.csv", rows["box_b_candidate"])
        _write_rows_csv(evidence_dir / "frozen_v2_baseline_holdout_rows.csv", rows["frozen_v2_baseline"])
        _write_report(evidence_dir / "final_report.md", decision, history)
        if evidence_dir.resolve() == ATTEMPT_DIR.resolve():
            control, receipt = _load_control()
            control.open_scientific_attempt(evidence_dir,
                                            f"{control.identity.run_id}:attempt_01",
                                            receipt, resume=True)
            if state == RUN_COMPLETED:
                control.complete()
            else:
                control.valid_failed()
        _task_state(evidence_dir, state, UPDATE_BUDGET)
        state_path = evidence_dir / "research_lifecycle_state.json"
        _atomic_json(state_path, {"task_id": TASK_ID, "run_id": identity.run_id,
                                  "state": state, "preflight_receipt_consumed": True,
                                  "new_scientific_updates": UPDATE_BUDGET,
                                  "frozen_optical_updates": 0,
                                  "control_state": state})
    return decision


def _build_transfer_manifest(evidence_dir: Path) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_reconsumption.json"}
    files = [{"path": path.relative_to(evidence_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(evidence_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    payload = {"schema": "lightgen_g1_projected_box_fullpop_transfer.v1",
               "task_id": TASK_ID, "run_id": _json(evidence_dir / "run_identity.json")["run_id"],
               "files": files, "file_count": len(files),
               "bytes_total": sum(item["bytes"] for item in files)}
    _atomic_json(evidence_dir / "transfer_manifest.json", payload)
    return payload


def _recompute_recovery_raw(evidence_dir: Path, identity: ResearchRunIdentity,
                           source_run_id: str, schedule: torch.Tensor) -> dict[str, Any]:
    recovery_dir = evidence_dir / "preflight_evidence" / "engineering_recovery"
    paths = (recovery_dir / "checkpoint_step_000003.pt",
             recovery_dir / "original_process_step4.pt",
             recovery_dir / "independent_process_step4.pt")
    _require(all(path.is_file() for path in paths),
             "full-population raw recovery states are missing", FileNotFoundError)
    checkpoint, live, restored = (torch.load(path, map_location="cpu", weights_only=False)
                                  for path in paths)
    prefix = checkpoint.get("history", [])
    live_history = live.get("history", [])
    restored_history = restored.get("history", [])
    _validate_checkpoint(checkpoint, identity.run_id, source_run_id, 3, prefix, schedule)
    _validate_checkpoint(live, identity.run_id, source_run_id, 4, live_history, schedule)
    _validate_checkpoint(restored, identity.run_id, source_run_id, 4, restored_history, schedule)
    mapper_equal, mapper_delta = _tree_delta(live["mapper_state"], restored["mapper_state"])
    adam_equal, adam_delta = _tree_delta(live["optimizer_state"], restored["optimizer_state"])
    history_equal = fit20._numeric_tree_matches(live_history, restored_history,
                                                 RECOVERY_TOLERANCE)
    rng_equal = _rng_equal(
        {"cpu": live["torch_cpu_rng"], "cuda": live["torch_cuda_rng"]},
        {"cpu": restored["torch_cpu_rng"], "cuda": restored["torch_cuda_rng"]},
    )
    losses = {key: abs(float(live_history[-1][key]) - float(restored_history[-1][key]))
              for key in ("total_loss", "coefficient_loss", "box_loss")}
    _require(mapper_equal and adam_equal and history_equal and rng_equal
             and mapper_delta <= RECOVERY_TOLERANCE
             and adam_delta <= RECOVERY_TOLERANCE
             and all(value <= RECOVERY_TOLERANCE for value in losses.values())
             and live.get("engineering_adam_calls") == restored.get("engineering_adam_calls") == 1
             and _optimizer_steps(live["optimizer_state"]) == [4] * 4
             and _optimizer_steps(restored["optimizer_state"]) == [4] * 4,
             "Windows raw recovery recomputation found divergent model, Adam, history, loss, RNG, or step")
    return {"status": "RAW_RECOVERY_RECOMPUTATION_PASSED", "step3_prefix_rows": len(prefix),
            "fourth_step_history_rows": len(live_history), "mapper_max_abs_delta": mapper_delta,
            "adam_max_abs_delta": adam_delta, "loss_abs_deltas": losses,
            "rng_exact": rng_equal, "history_exact": history_equal,
            "live_adam_steps": _optimizer_steps(live["optimizer_state"]),
            "restored_adam_steps": _optimizer_steps(restored["optimizer_state"]),
            "scientific_updates": 0, "engineering_adam_calls": 5}


def _verify_attempt_manifest(evidence_dir: Path) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    _require(manifest.get("schema") == "lightgen_g1_projected_box_fullpop_transfer.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list),
             "cloud full-population source manifest is invalid")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "cloud full-population source manifest has an unsafe or duplicate path")
        expected[key] = int(item["bytes"])
    checked = []
    for relative, remote_bytes in sorted(expected.items()):
        path = evidence_dir / Path(relative)
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and current == remote_bytes,
                 f"cloud full-population source file differs from its manifest: {relative}")
        checked.append({"path": relative, "remote_bytes": remote_bytes, "current_bytes": current})
    _require(manifest.get("file_count") == len(expected)
             and manifest.get("bytes_total") == sum(expected.values()),
             "cloud full-population source manifest totals are inconsistent")
    return {"status": "CLOUD_SOURCE_MANIFEST_MATCHED", "file_count": len(checked),
            "bytes_total": sum(item["current_bytes"] for item in checked), "files": checked}


def _build_review_manifest(review_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_review_receipt.json"}
    files = [{"path": path.relative_to(review_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(review_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    payload = {"schema": "lightgen_g1_projected_box_fullpop_review_transfer.v1",
               "task_id": TASK_ID, "run_id": run_id, "files": files,
               "file_count": len(files), "bytes_total": sum(item["bytes"] for item in files)}
    _atomic_json(review_dir / "transfer_manifest.json", payload)
    return payload


def _verify_review_transfer(review_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = _json(review_dir / "transfer_manifest.json")
    receipt = _json(review_dir / "transfer_receipt.json")
    expected_remote = RUNS_DIR.relative_to(ROOT).as_posix() + "/review_repair_01"
    remote_dir = str(receipt.get("remote_run_dir", "")).rstrip("/")
    _require(manifest.get("schema") == "lightgen_g1_projected_box_fullpop_review_transfer.v1"
             and manifest.get("task_id") == TASK_ID and manifest.get("run_id") == run_id
             and manifest.get("file_count") == len(manifest.get("files", []))
             and manifest.get("bytes_total") == sum(int(item["bytes"]) for item in manifest["files"])
             and receipt.get("status") == "COMPLETE"
             and receipt.get("remote_alias") == "autodl-monthly"
             and remote_dir.endswith("/" + expected_remote),
             "full-population repair transfer is not bound to this AutoDL repair directory")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "full-population repair manifest has an unsafe or duplicate path")
        expected[key] = int(item["bytes"])
    expected["transfer_manifest.json"] = (review_dir / "transfer_manifest.json").stat().st_size
    received: dict[str, Mapping[str, Any]] = {}
    for item in receipt.get("files", []):
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in received,
                 "full-population repair receipt has an unsafe or duplicate path")
        received[key] = item
    _require(set(expected) == set(received),
             "full-population repair receipt differs from its manifest paths")
    files = []
    for relative, remote_bytes in sorted(expected.items()):
        path = review_dir / Path(relative)
        item = received[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_bytes
                 and int(item.get("local_size_bytes", -1)) == remote_bytes
                 and current == remote_bytes,
                 f"full-population repair transfer byte/status mismatch: {relative}")
        files.append({"path": relative, "remote_bytes": remote_bytes,
                      "receipt_local_bytes": int(item["local_size_bytes"]),
                      "current_bytes": current})
    return {"status": "COMPLETE", "file_count": len(files),
            "bytes_total": sum(item["current_bytes"] for item in files), "files": files}


def _read_review_source_states(review_dir: Path, evidence_dir: Path,
                               inputs: Mapping[str, Any], identity: ResearchRunIdentity,
                               schedule: torch.Tensor) -> dict[str, Any]:
    path = review_dir / "command_replay" / "mapper_source_states.pt"
    _require(path.is_file(), f"review replay mapper source states are missing: {path}", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    fixed = torch.load(evidence_dir / "checkpoints" / f"checkpoint_step_{UPDATE_BUDGET:06d}.pt",
                       map_location="cpu", weights_only=False)
    _require_terminal_checkpoint_bound(terminal, fixed, identity.run_id,
                                       inputs["source_mapper"]["run_id"], schedule)
    expected = {
        "schema": "lightgen_g1_projected_box_fullpop_mapper_source_states.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "sources": {
            "frozen_v2_baseline": {
                "checkpoint_path": "source.v2_terminal_checkpoint",
                "checkpoint_step": 5000, "mapper_updates": SOURCE_MAPPER_UPDATES,
            },
            "box_b_candidate": {
                "checkpoint_path": "terminal_checkpoint.pt",
                "fixed_checkpoint_path": f"checkpoints/checkpoint_step_{UPDATE_BUDGET:06d}.pt",
                "checkpoint_step": UPDATE_BUDGET,
                "mapper_updates": LINEAGE_MAPPER_UPDATES,
            },
        },
        "mapper_states": {
            "frozen_v2_baseline": inputs["source_mapper"]["mapper_state"],
            "box_b_candidate": terminal["mapper_state"],
        },
    }
    _require(_source_state_payload_equal(payload, expected),
             "saved CUDA replay mapper state differs from the explicit V2 or terminal checkpoint")
    return payload["mapper_states"]


def _check_repair_replays(evidence_dir: Path, review_dir: Path,
                          source_states: Mapping[str, Any]) -> dict[str, Any]:
    compared = 0
    for shard_no, start in enumerate(range(0, HOLDOUT_ROWS, 10)):
        stop = min(start + 10, HOLDOUT_ROWS)
        new_path = review_dir / "command_replay" / f"commands_{shard_no:04d}.pt"
        old_path = evidence_dir / "measurements" / "cuda_command_replay" / f"commands_{shard_no:04d}.pt"
        measured_path = evidence_dir / "measurements" / "holdout" / f"holdout_{shard_no:04d}.pt"
        _require(new_path.is_file() and old_path.is_file() and measured_path.is_file(),
                 f"full-population source review is missing shard {shard_no}", FileNotFoundError)
        replay = torch.load(new_path, map_location="cpu", weights_only=False)
        old_replay = torch.load(old_path, map_location="cpu", weights_only=False)
        measured = torch.load(measured_path, map_location="cpu", weights_only=False)
        expected_indices = torch.arange(start, stop)
        _require(replay.get("batch_size") == 1 and replay.get("independent_process") is True
                 and replay.get("source_state_file") == "mapper_source_states.pt"
                 and replay.get("source_state_roles") == {
                     "baseline": "frozen_v2_baseline", "candidate": "box_b_candidate"}
                 and torch.equal(replay.get("population_indices"), expected_indices)
                 and torch.equal(old_replay.get("population_indices"), expected_indices)
                 and torch.equal(measured.get("population_indices"), expected_indices)
                 and torch.equal(replay.get("baseline_commands"), old_replay.get("baseline_commands"))
                 and torch.equal(replay.get("candidate_commands"), old_replay.get("candidate_commands"))
                 and torch.equal(replay.get("baseline_commands"), measured.get("baseline_commands"))
                 and torch.equal(replay.get("candidate_commands"), measured.get("candidate_commands")),
                 f"independent CUDA command replay does not match stored measurements: shard {shard_no}")
        compared += stop - start
    _require(compared == HOLDOUT_ROWS and _mapper_state_valid(source_states["frozen_v2_baseline"])
             and _mapper_state_valid(source_states["box_b_candidate"]),
             "full-population CUDA replay did not cover all targets and loaded sources")
    return {"status": "EXACT_SOURCE_BOUND_REPLAY_PASSED", "rows": compared,
            "shards": compared // 10, "baseline_and_candidate_commands_exact": True,
            "source_states_saved_once": True, "model_updates": 0}


def _review_saved_cloud(config_path: Path, config: Mapping[str, Any],
                        evidence_dir: Path, review_dir: Path) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "source review requires AutoDL CUDA")
    _require(review_dir.resolve() == (RUNS_DIR / "review_repair_01").resolve(),
             "source review output must use the fixed repair directory", ValueError)
    _require(review_dir != evidence_dir and evidence_dir.is_dir(),
             "source review requires a separate directory and the explicit source attempt", ValueError)
    inputs = _load_inputs(config)
    source_manifest = _verify_attempt_manifest(evidence_dir)
    identity = _assert_attempt(evidence_dir)
    population, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    fixed = torch.load(evidence_dir / "checkpoints" / f"checkpoint_step_{UPDATE_BUDGET:06d}.pt",
                       map_location="cpu", weights_only=False)
    _require_terminal_checkpoint_bound(terminal, fixed, identity.run_id,
                                       inputs["source_mapper"]["run_id"], schedule)
    session_path = review_dir / "review_session.json"
    if review_dir.exists():
        _require(session_path.is_file(),
                 f"existing full-population repair directory is not a resumable review: {review_dir}",
                 FileExistsError)
        session = _json(session_path)
        _require(session.get("task_id") == TASK_ID and session.get("run_id") == identity.run_id,
                 "existing source review directory belongs to another run")
        _require(not (review_dir / "transfer_receipt.json").exists()
                 and not (review_dir / "windows_review_receipt.json").exists(),
                 "completed source repair must be consumed with verify-review, not regenerated")
    else:
        review_dir.mkdir(parents=True)
        _atomic_json(session_path, {"task_id": TASK_ID, "run_id": identity.run_id,
                                    "scientific_updates": 0})
    replay_dir = review_dir / "command_replay"
    command = [sys.executable, "-m", MODULE_PATH, "--mode", "command-replay-worker",
               "--config", str(config_path.resolve()), "--evidence-dir", str(evidence_dir.resolve()),
               "--output-dir", str(replay_dir.resolve())]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _require(result.returncode == 0 and (replay_dir / "mapper_source_states.pt").is_file(),
             f"independent source-bound CUDA replay failed: {result.stderr[-3000:]}")
    source_states = _read_review_source_states(review_dir, evidence_dir, inputs, identity, schedule)
    replay_check = _check_repair_replays(evidence_dir, review_dir, source_states)
    recovery = _recompute_recovery_raw(evidence_dir, identity,
                                      inputs["source_mapper"]["run_id"], schedule)
    _require(recovery["status"] == "RAW_RECOVERY_RECOMPUTATION_PASSED",
             "full-population original recovery files did not pass raw state reconsumption")
    decision = _finish_decision(evidence_dir, inputs, write=False, replay_dir=replay_dir,
                                source_states=source_states)
    stored = _json(evidence_dir / "final_decision.json")
    _require(stored.get("lifecycle_state") == decision["lifecycle_state"]
             and fit20._numeric_tree_matches(stored.get("candidate_summary"),
                                             decision["candidate_summary"], 1.0e-14)
             and fit20._numeric_tree_matches(stored.get("frozen_v2_baseline_summary"),
                                             decision["frozen_v2_baseline_summary"], 1.0e-14),
             "source-bound replay changed the original full-population metric result")
    summary = {
        "schema": "lightgen_g1_projected_box_fullpop_review.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "status": "SOURCE_REPAIR_AND_COMMAND_REPLAY_PASSED",
        "scientific_updates": 0, "optical_updates": 0, "evaluator_updates": 0,
        "source_manifest": source_manifest,
        "source_mapper_states_saved_once": True,
        "v2_source_and_fixed_terminal_state_bound": True,
        "terminal_matches_fixed_step_5000": True,
        "command_replay": replay_check,
        "recovery_raw_recomputed": recovery,
        "all_2000_rows_recomputed": True,
        "original_decision_unchanged": True,
        "scientific_status": decision["lifecycle_state"],
        "candidate_summary": decision["candidate_summary"],
        "frozen_v2_baseline_summary": decision["frozen_v2_baseline_summary"],
    }
    _atomic_json(review_dir / "review_summary.json", summary)
    manifest = _build_review_manifest(review_dir, identity.run_id)
    return {"status": summary["status"], "run_id": identity.run_id,
            "review_dir": str(review_dir), "review_file_count": manifest["file_count"],
            "review_bytes": manifest["bytes_total"], "command_replay_rows": HOLDOUT_ROWS,
            "scientific_updates": 0, "recovery": recovery}


def _verify_saved(evidence_dir: Path, inputs: Mapping[str, Any],
                  review_dir: Path) -> dict[str, Any]:
    _require(evidence_dir.is_dir(), f"full-population evidence is missing: {evidence_dir}", FileNotFoundError)
    sizes = _verify_transfer(evidence_dir)
    identity = _assert_attempt(evidence_dir)
    manifest = _json(evidence_dir / "transfer_manifest.json")
    receipt = _json(evidence_dir / "transfer_receipt.json")
    expected_remote_run = ATTEMPT_DIR.relative_to(ROOT).as_posix()
    remote_run = str(receipt.get("remote_run_dir", "")).rstrip("/")
    _require(manifest.get("run_id") == identity.run_id
             and remote_run.endswith("/" + expected_remote_run)
             and receipt.get("remote_alias") == "autodl-monthly",
             "full-population transfer record is not bound to this AutoDL attempt")
    _validate_prior_b_repair(_load_config(DEFAULT_CONFIG), inputs, host="windows")
    population_source = _recompute_population_source(_load_config(DEFAULT_CONFIG), inputs,
                                                      evidence_dir, identity)
    population, schedule = _load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    terminal = torch.load(evidence_dir / "terminal_checkpoint.pt", map_location="cpu", weights_only=False)
    source_mapper = inputs["source_mapper"]
    _require(terminal.get("run_id") == identity.run_id
             and terminal.get("source_mapper_updates") == SOURCE_MAPPER_UPDATES
             and terminal.get("completed_updates") == UPDATE_BUDGET,
             "full-population terminal checkpoint is not the authorized endpoint")
    _validate_checkpoint(terminal, identity.run_id, source_mapper["run_id"],
                         UPDATE_BUDGET, terminal.get("history", []), schedule)
    history_payload = _json(evidence_dir / "training_history.json")
    history = history_payload.get("updates", [])
    _require(history_payload.get("task_id") == TASK_ID and history_payload.get("run_id") == identity.run_id
             and len(history) == UPDATE_BUDGET,
             "full-population training history is incomplete")
    fixed_terminal = None
    for checkpoint_step in FIXED_CHECKPOINT_STEPS:
        checkpoint_path = evidence_dir / "checkpoints" / f"checkpoint_step_{checkpoint_step:06d}.pt"
        _require(checkpoint_path.is_file(), f"fixed full-population checkpoint is missing: {checkpoint_step}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(checkpoint, identity.run_id, source_mapper["run_id"],
                             checkpoint_step, history[:checkpoint_step], schedule)
        if checkpoint_step == UPDATE_BUDGET:
            fixed_terminal = checkpoint
    _require(fixed_terminal is not None, "fixed terminal step-5000 checkpoint was not loaded")
    _require_terminal_checkpoint_bound(terminal, fixed_terminal, identity.run_id,
                                       source_mapper["run_id"], schedule)
    review_sizes = _verify_review_transfer(review_dir, identity.run_id)
    review_session = _json(review_dir / "review_session.json")
    _require(review_session.get("task_id") == TASK_ID
             and review_session.get("run_id") == identity.run_id
             and review_session.get("scientific_updates") == 0,
             "source repair session identity is invalid")
    review_summary = _json(review_dir / "review_summary.json")
    _require(review_summary.get("schema") == "lightgen_g1_projected_box_fullpop_review.v1"
             and review_summary.get("task_id") == TASK_ID
             and review_summary.get("run_id") == identity.run_id
             and review_summary.get("status") == "SOURCE_REPAIR_AND_COMMAND_REPLAY_PASSED"
             and review_summary.get("scientific_updates") == 0,
             "full-population source repair summary is invalid")
    source_states = _read_review_source_states(review_dir, evidence_dir, inputs, identity, schedule)
    replay_check = _check_repair_replays(evidence_dir, review_dir, source_states)
    decision = _finish_decision(evidence_dir, inputs, write=False,
                                replay_dir=review_dir / "command_replay",
                                source_states=source_states)
    stored = _json(evidence_dir / "final_decision.json")
    _require(stored.get("lifecycle_state") == decision["lifecycle_state"]
             and stored.get("new_scientific_updates") == UPDATE_BUDGET
             and fit20._numeric_tree_matches(stored.get("candidate_summary"),
                                             decision["candidate_summary"], 1.0e-14)
             and fit20._numeric_tree_matches(stored.get("frozen_v2_baseline_summary"),
                                             decision["frozen_v2_baseline_summary"], 1.0e-14),
             "saved final decision differs from independently recomputed raw holdout evidence")
    report = (evidence_dir / "final_report.md").read_text(encoding="utf-8")
    sections = ("## 结果与状态", "## 科学认识", "## 目标缺口",
                "## 工程完成与验证", "## 决策影响与下一步")
    _require(all(section in report for section in sections), "five-part full-population report is incomplete")
    recovery = _json(evidence_dir / "preflight_evidence" / "engineering_recovery.json")
    recovery_raw = _recompute_recovery_raw(evidence_dir, identity,
                                           inputs["source_mapper"]["run_id"], schedule)
    _require(recovery.get("scientific_updates") == 0 and recovery.get("engineering_adam_calls") == 5
             and recovery.get("mapper_max_abs_delta", math.inf) <= RECOVERY_TOLERANCE
             and recovery.get("adam_max_abs_delta", math.inf) <= RECOVERY_TOLERANCE
             and recovery.get("total_loss_abs_delta", math.inf) <= RECOVERY_TOLERANCE
             and recovery.get("rng_exact") is True and recovery.get("schedule_exact") is True,
             "full-population preflight raw recovery evidence is incomplete")
    _require(recovery_raw["status"] == "RAW_RECOVERY_RECOMPUTATION_PASSED"
             and recovery_raw["mapper_max_abs_delta"] <= RECOVERY_TOLERANCE
             and recovery_raw["adam_max_abs_delta"] <= RECOVERY_TOLERANCE
             and recovery_raw["rng_exact"] is True and recovery_raw["history_exact"] is True,
             "Windows raw recovery-state recomputation did not pass")
    _require(fit20._numeric_tree_matches(review_summary.get("recovery_raw_recomputed"),
                                         recovery_raw, 1.0e-14),
             "repair summary does not match the independently reconsumed recovery states")
    receipt = {"schema": "lightgen_g1_projected_box_fullpop_windows_review_receipt.v1",
               "task_id": TASK_ID, "run_id": identity.run_id,
               "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
               "scientific_status": decision["lifecycle_state"],
               "all_2000_rows_recomputed": True, "all_six_metrics_recomputed": True,
               "terminal_adam_steps": [UPDATE_BUDGET] * 4,
               "training_history_rows": len(history),
               "independent_cuda_command_replay_consumed": True,
               "source_bound_command_replay": replay_check,
               "source_population_roles_verified": True,
               "population_source_recomputed": population_source,
               "preflight_recovery_consumed": recovery,
               "preflight_recovery_raw_recomputed": recovery_raw,
               "file_size_reconciliation": sizes,
               "repair_file_size_reconciliation": review_sizes,
               "candidate_summary": decision["candidate_summary"],
               "frozen_v2_baseline_summary": decision["frozen_v2_baseline_summary"]}
    _atomic_json(review_dir / "windows_review_receipt.json", receipt)
    return receipt


def _append_transfer_receipt(evidence_dir: Path, receipt_path: Path) -> dict[str, Any]:
    _require(receipt_path.is_file(), f"explicit Windows transfer receipt is missing: {receipt_path}",
             FileNotFoundError)
    receipt = _json(receipt_path)
    _require(receipt.get("status") == "COMPLETE" and isinstance(receipt.get("files"), list),
             "Windows transfer receipt is invalid")
    if receipt_path.name != "transfer_receipt.json":
        shutil.copy2(receipt_path, evidence_dir / "transfer_receipt.json")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "verify-saved",
                                 "recovery-worker", "command-replay-worker",
                                 "review-saved", "verify-review"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="explicit checkpoint required for resume")
    parser.add_argument("--evidence-dir", type=Path, help="explicit attempt for measurement or verification")
    parser.add_argument("--review-dir", type=Path,
                        help="explicit source-binding repair directory for review-saved/verify-review")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--transfer-receipt", type=Path,
                        help="optional explicit local transfer receipt for Windows verification")
    parser.add_argument("--review-transfer-receipt", type=Path,
                        help="optional explicit local transfer receipt for source-binding repair")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode in {"recovery-worker", "command-replay-worker"}:
        _require(args.evidence_dir is not None, "worker mode requires explicit --evidence-dir", ValueError)
        if args.mode == "recovery-worker":
            _require(args.checkpoint is not None and args.worker_output is not None,
                     "recovery-worker requires explicit checkpoint and worker output", ValueError)
            _recovery_worker(config_path, args.checkpoint.resolve(), args.worker_output.resolve())
            result = {"status": "RECOVERY_STEP_COMPLETE"}
        else:
            _require(args.output_dir is not None and args.checkpoint is None,
                     "command-replay-worker requires explicit --output-dir", ValueError)
            _independent_command_replay_worker(config_path, args.evidence_dir.resolve(), args.output_dir.resolve())
            result = {"status": "INDEPENDENT_CUDA_COMMAND_REPLAY_COMPLETE"}
    elif args.mode == "preflight":
        _require(args.checkpoint is None and args.evidence_dir is None,
                 "preflight accepts no checkpoint or evidence directory", ValueError)
        result = _preflight(config_path, config)
    elif args.mode == "execute":
        _require(args.checkpoint is None and args.evidence_dir is None,
                 "execute starts only from the fixed V2 terminal mapper", ValueError)
        inputs = _load_inputs(config)
        result = _execute(config_path, config, inputs)
    elif args.mode == "resume":
        _require(args.checkpoint is not None and args.evidence_dir is None,
                 "resume requires an explicit --checkpoint", ValueError)
        inputs = _load_inputs(config)
        result = _execute(config_path, config, inputs, resume_checkpoint=args.checkpoint.resolve())
    elif args.mode == "measure":
        _require(args.evidence_dir is None and args.checkpoint is None,
                 "measure uses the fixed attempt terminal and accepts no substituted checkpoint", ValueError)
        inputs = _load_inputs(config)
        result = _measure_terminal(config, inputs)
    elif args.mode == "review-saved":
        _require(args.evidence_dir is not None and args.review_dir is not None
                 and args.checkpoint is None and args.transfer_receipt is None
                 and args.review_transfer_receipt is None,
                 "review-saved requires explicit --evidence-dir and a new --review-dir", ValueError)
        result = _review_saved_cloud(config_path, config, args.evidence_dir.resolve(),
                                     args.review_dir.resolve())
    else:
        _require(args.evidence_dir is not None and args.review_dir is not None
                 and args.checkpoint is None,
                 "verify-saved and verify-review require explicit --evidence-dir and --review-dir",
                 ValueError)
        if args.transfer_receipt is not None:
            _append_transfer_receipt(args.evidence_dir.resolve(), args.transfer_receipt.resolve())
        if args.review_transfer_receipt is not None:
            _append_transfer_receipt(args.review_dir.resolve(), args.review_transfer_receipt.resolve())
        inputs = _load_inputs(config)
        result = _verify_saved(args.evidence_dir.resolve(), inputs, args.review_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
