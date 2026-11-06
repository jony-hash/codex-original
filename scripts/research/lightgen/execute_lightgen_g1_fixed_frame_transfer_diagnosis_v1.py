"""Bind saved G1 commands to their mapper and diagnose fixed-frame phase effects."""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import json
import math
import os
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
from PIL import Image

from ldm_project.v2.application.research.shared.research_run_control import (
    IMPLEMENTATION_SELECTION,
    RESOURCE_ENVELOPE,
    CORRECTNESS_INVARIANTS,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as fixed
from scripts.research.lightgen import execute_lightgen_g1_semantic_exposure200_v1 as exposure
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_FIXED_FRAME_TRANSFER_DIAGNOSIS_V1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_FIXED_FRAME_TRANSFER_DIAGNOSIS_20261030"
SCHEMA = "lightgen_g1_fixed_frame_transfer_diagnosis.v1"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_fixed_frame_transfer_diagnosis_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs"
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
GRIDS = (2048, 4096)
CONDITIONS = (
    "frozen_v2_single_forward",
    "fixed_frame_terminal",
    "terminal_phi_zero_ablation",
    "fista_fixed_physical_phase_zero",
    "fista_fixed_terminal_phase",
    "bounded_free_public_phase_eight_start",
)
METRIC_KEYS = fixed.METRIC_KEYS
GATE_LIMIT = 1.0e-4
SOURCE_CPU_COMMAND_TOLERANCE = 1.0e-5
SOLVER_RECOVERY_TOLERANCE = 1.0e-12


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temp)
    os.replace(temp, path)


def _config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "diagnosis uses its registered config", ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA
             and value.get("task_id") == TASK_ID
             and value.get("semantic_authority_id") == AUTHORITY
             and value.get("evidence_tier") == "diagnostic_only",
             "diagnosis task identity or evidence tier changed", ValueError)
    _require(value["population"] == {
        "role": "consumed_development_phase20", "development_source_role": "fixed_development200",
        "index_rule": "20*class + {0,1}", "sample_count": 20, "class_count": 10,
        "class_order": "interleaved_two_per_class"},
        "fixed diagnostic population changed", ValueError)
    contract = value["diagnostic"]
    _require(contract["conditions"] == list(CONDITIONS)
             and contract["fixed_phase_iterations_per_target"] == 5000
             and contract["free_phase_starts"] == 8
             and contract["free_phase_iterations_per_start"] == 5000
             and contract["fista_relative_stop"] == 1.0e-6
             and contract["fista_dual_gap_stop"] == 1.0e-8
             and contract["full_optics_grids"] == [2048, 4096]
             and contract["relative_error_max"] == GATE_LIMIT
             and contract["iterations_per_checkpoint"] == 500
             and contract["scientific_updates"] == contract["mapper_updates"]
             == contract["optical_updates"] == contract["evaluator_updates"] == 0
             and contract["rng_use"] == "none"
             and contract["fixed_phase_prescribed_complex_error_max"] == GATE_LIMIT
             and contract["camera_pixel_area_m2"] == 6.25e-12,
             "fixed diagnosis contract changed", ValueError)
    expected = {
        "runs_root": RUNS_DIR.relative_to(ROOT).as_posix(),
        "source_review_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/source_review_01",
        "semantic_attribution_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/semantic_attribution_01",
        "preflight_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/preflight_03",
        "attempt_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/attempt_02",
        "windows_source_review_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_source_review_01",
        "source_review_verification_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/source_review_windows_verify_01",
        "source_review_cloud_verification_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/source_review_cloud_verify_01",
        "windows_backup_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_attempt_02",
        "windows_preflight_backup_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_preflight_03",
        "windows_source_review_cloud_backup_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/autodl_backup_source_review_cloud_verify_01",
        "windows_verification_dir": "outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs/windows_verify_02",
    }
    _require(value["output"] == expected, "diagnosis output paths changed", ValueError)
    return value


def _load_source(source_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    terminal_path = source_dir / "checkpoints" / "terminal_checkpoint.pt"
    population_path = source_dir / "selected_population.pt"
    prior_path = source_dir / "prior_evidence" / "physical_phase_gaussian_prior.pt"
    mean_path = source_dir / "prior_evidence" / "mean_amplitude_full_optics_replay.pt"
    decision_path = source_dir / "final_decision.json"
    state_path = source_dir / "task_state.json"
    _require(all(path.is_file() for path in (terminal_path, population_path, prior_path,
                                               mean_path, decision_path, state_path)),
             "explicit fixed-frame source evidence is incomplete", FileNotFoundError)
    base_config_path = _resolve(config["source"]["fixed_frame_config"])
    base_config = fixed._load_config(base_config_path)
    inputs = fixed._load_inputs(base_config)
    checkpoint = torch.load(terminal_path, map_location="cpu", weights_only=False)
    population = torch.load(population_path, map_location="cpu", weights_only=False)
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    means = torch.load(mean_path, map_location="cpu", weights_only=False)
    decision = _json(decision_path)
    task_state = _json(state_path)
    _require(checkpoint.get("task_id") == fixed.TASK_ID
             and checkpoint.get("completed_updates") == 15000
             and checkpoint.get("checkpoint_step") == 15000
             and checkpoint.get("history_count") == 15000
             and checkpoint.get("lineage_mapper_updates") == 25000
             and checkpoint.get("selected_alpha") == 1.0
             and checkpoint.get("run_id") == decision.get("run_id")
             and decision.get("state") == "VALID_FAILED"
             and task_state.get("phase") == "CLOSED"
             and task_state.get("completed_updates") == 15000,
             "fixed-frame source is not the closed 15,000-step alpha=1 endpoint")
    _require(population.get("task_id") == fixed.TASK_ID
             and population.get("selected_alpha") == 1.0
             and tuple(population["train"]["coefficients"].shape) == (200, PORT_COUNT)
             and tuple(population["train"]["labels"].shape) == (200,)
             and tuple(population["development"]["coefficients"].shape) == (200, PORT_COUNT)
             and tuple(population["holdout"]["coefficients"].shape) == (2000, PORT_COUNT)
             and torch.equal(population["train"]["labels"], torch.arange(CLASS_COUNT).repeat_interleave(20))
             and torch.equal(population["development"]["labels"],
                             torch.arange(CLASS_COUNT).repeat_interleave(20))
             and torch.equal(population["holdout"]["labels"], torch.arange(CLASS_COUNT).repeat_interleave(200))
             and prior.get("task_id") == fixed.TASK_ID
             and tuple(prior["mean_amplitudes"].shape) == (CLASS_COUNT, PIXEL_COUNT)
             and torch.equal(prior["scale"], inputs["scale"])
             and means.get("task_id") == fixed.TASK_ID
             and tuple(means["commands"].shape) == (CLASS_COUNT, 1, 200, 200)
             and means.get("mean_intensity_sqrt_used") is False,
             "fixed-frame population, prior, or mean-amplitude positive control is invalid")
    projector_path = _resolve(config["source"]["response_projector"])
    projector = torch.load(projector_path, map_location="cpu", weights_only=False)
    _require(torch.equal(projector["response_matrix"], inputs["response_matrix"])
             and torch.equal(projector["scale"], inputs["scale"])
             and torch.equal(projector["full_right_inverse"], inputs["inverse"])
             and projector.get("real_imag_order") == "per_port_interleaved_real_imag",
             "explicit response/projector source differs from the loaded optical inputs")
    _require(Path(inputs["source_mapper_path"]).resolve()
             == _resolve(config["source"]["v2_terminal_checkpoint"]).resolve(),
             "frozen V2 mapper does not come from the explicit registered terminal checkpoint")
    optimizer_state = checkpoint.get("optimizer_state", {}).get("state", {})
    optimizer_steps = [int(torch.as_tensor(value["step"]).item()) for value in optimizer_state.values()
                       if isinstance(value, Mapping) and "step" in value]
    _require(optimizer_steps == [15000] * 4
             and len(checkpoint.get("history", [])) == 15000
             and torch.equal(checkpoint["exposure_counts"], torch.full((200,), 750, dtype=torch.long)),
             "fixed-frame endpoint optimizer/history/exposure state is not the closed 15,000-step run")
    return {"source_dir": source_dir, "terminal_path": terminal_path,
            "population_path": population_path, "checkpoint": checkpoint,
            "population": population, "prior": prior, "mean_amplitudes": prior["mean_amplitudes"],
            "inputs": inputs, "decision": decision, "projector": projector,
            "projector_path": projector_path, "source_v2_path": inputs["source_mapper_path"]}


def _target_indices() -> list[int]:
    return [20 * label + offset for label in range(CLASS_COUNT) for offset in (0, 1)]


@functools.lru_cache(maxsize=2)
def _load_shard(source_dir: str, role: str, shard_index: int) -> dict[str, Any]:
    path = Path(source_dir) / "measurements" / role / f"shard_{shard_index:04d}.pt"
    _require(path.is_file(), f"fixed-frame {role} measurement shard is missing: {path}", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("task_id") == fixed.TASK_ID and payload.get("cohort_role") == role,
             f"fixed-frame {role} shard identity differs: {path.name}")
    return payload


def _load_shard_row(source_dir: Path, role: str, index: int) -> tuple[dict[str, Any], int]:
    payload = _load_shard(str(source_dir.resolve()), role, index // 10)
    local = index - int(payload["start"])
    _require(0 <= local < int(payload["stop"]) - int(payload["start"])
             and int(payload["indices"][local]) == index,
             f"fixed-frame {role} shard identity differs at row {index}")
    return payload, local


def _mapper_command(mapper: torch.nn.Module, target: torch.Tensor, inputs: Mapping[str, Any],
                    device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.reshape(1, PORT_COUNT).to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(target.detach().cpu(), inputs["scale"], device)
    command, detail = projected._map_amplitude(
        mapper, features, target, inputs["matrix"].to(device), inputs["inverse"].to(device),
        inputs["scale"].to(device), include_preclip=True)
    return command.detach().to(dtype=torch.float32), {key: value.detach() for key, value in detail.items()}


def _fixed_phase_projection(raw: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor,
                            inverse: torch.Tensor, scale: torch.Tensor, phase: float
                            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _require(matrix.ndim == 2 and target.ndim in (1, 2)
             and matrix.shape[0] == 2 * target.reshape(-1, target.shape[-1]).shape[-1]
             and inverse.shape == (matrix.shape[1], matrix.shape[0])
             and scale.numel() == target.reshape(-1, target.shape[-1]).shape[-1],
             "fixed phase projection dimensions disagree", ValueError)
    batch_size = 1 if target.ndim == 1 else target.shape[0]
    port_count = matrix.shape[0] // 2
    pixel_count = matrix.shape[1]
    raw = raw.reshape(batch_size, pixel_count).to(torch.float32)
    target = target.reshape(batch_size, port_count).to(device=raw.device, dtype=torch.complex128)
    matrix = matrix.to(device=raw.device, dtype=torch.float64)
    inverse = inverse.to(device=raw.device, dtype=torch.float64)
    scale = scale.to(device=raw.device, dtype=torch.float64)
    predicted_ri = raw.double() @ matrix.T
    predicted = torch.view_as_complex(predicted_ri.reshape(batch_size, port_count, 2).contiguous())
    predicted = predicted * scale.sqrt().reshape(1, port_count)
    correlation = (target.conj() * predicted).sum(dim=1)
    actual_mlp_phase = torch.where(correlation.abs() > 0, torch.angle(correlation),
                                   torch.zeros_like(correlation))
    aligned = target * torch.exp(torch.as_tensor(1j * phase, device=raw.device,
                                                  dtype=torch.complex128))
    target_ri = torch.view_as_real(aligned / scale.sqrt().reshape(1, port_count)).reshape(batch_size, -1)
    preclip = raw.double() + (target_ri - predicted_ri) @ inverse.T
    clipped = preclip.clamp(0.0, 1.0)
    command_shape = (batch_size, 1, 200, 200) if pixel_count == PIXEL_COUNT else (batch_size, 1, pixel_count)
    command = clipped.to(torch.float32).reshape(command_shape)
    return command, {"preclip_amplitude": preclip.reshape(command_shape),
                     "mlp_public_phase_rad": actual_mlp_phase,
                     "prescribed_phase_rad": torch.tensor([phase], device=raw.device,
                                                            dtype=torch.float64),
                     "clipped_fraction": ((preclip < 0.0) | (preclip > 1.0)).double().mean(dim=1)}


def _validate_bound_commands(cuda_replay: torch.Tensor, source_command: torch.Tensor,
                             cpu_expected: torch.Tensor) -> float:
    _require(cuda_replay.shape == source_command.shape == cpu_expected.shape
             and cuda_replay.dtype == source_command.dtype == torch.float32
             and torch.isfinite(cuda_replay).all().item()
             and torch.isfinite(source_command).all().item()
             and torch.isfinite(cpu_expected).all().item()
             and float(cuda_replay.min()) >= 0.0 and float(cuda_replay.max()) <= 1.0,
             "source command replays are malformed")
    _require(torch.equal(cuda_replay, source_command),
             "independent CUDA replay differs bitwise from the original measured command")
    max_abs = float((cpu_expected - cuda_replay).abs().max())
    _require(max_abs <= SOURCE_CPU_COMMAND_TOLERANCE,
             f"CPU checkpoint replay differs from the bound CUDA command: {max_abs:.9g}")
    return max_abs


def _consume_source_bound_command(actual_command: torch.Tensor, expected_command: torch.Tensor,
                                  *, require_bitwise: bool) -> float:
    _require(actual_command.dtype == expected_command.dtype == torch.float32
             and actual_command.shape == expected_command.shape
             and torch.isfinite(actual_command).all().item()
             and torch.isfinite(expected_command).all().item()
             and float(actual_command.min()) >= 0.0 and float(actual_command.max()) <= 1.0,
             "diagnostic command consumer received malformed source-bound instructions")
    difference = float((actual_command - expected_command).abs().max())
    _require((torch.equal(actual_command, expected_command) if require_bitwise
              else difference <= SOURCE_CPU_COMMAND_TOLERANCE),
             f"diagnostic command differs from its bound source: {difference:.9g}")
    return difference


def _expected_original_command(role_payload: Mapping[str, Any], local: int,
                               role: str) -> torch.Tensor:
    if role == "v2_holdout":
        value = role_payload["frozen_v2_commands"][local:local + 1]
    else:
        value = role_payload["commands"][local:local + 1]
    _require(value.dtype == torch.float32 and tuple(value.shape) == (1, 1, 200, 200)
             and torch.isfinite(value).all().item()
             and float(value.min()) >= 0.0 and float(value.max()) <= 1.0,
             "original saved source command is malformed")
    return value


def _review_source(config: Mapping[str, Any], source_dir: Path, output_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "independent source command replay requires AutoDL CUDA")
    _require(not output_dir.exists(), f"source review output already exists: {output_dir}", FileExistsError)
    output_dir.mkdir(parents=True)
    fixed._set_numerics()
    source = _load_source(source_dir, config)
    device = torch.device("cuda:0")
    terminal = fixed._new_mapper(source["inputs"], device, source["checkpoint"]["mapper_state"])
    terminal.eval()
    v2 = fixed._new_mapper(source["inputs"], device,
                           source["inputs"]["source_mapper"]["mapper_state"])
    v2.eval()
    terminal_snapshot = projected._clone_state(terminal)
    v2_snapshot = projected._clone_state(v2)
    terminal_commands: dict[str, dict[str, torch.Tensor]] = {}
    cpu_expected: dict[str, dict[str, torch.Tensor]] = {}
    row_counts: dict[str, int] = {}
    for role in ("train", "development", "holdout"):
        population = source["population"][role]
        generated, expected = [], []
        for index in range(len(population["labels"])):
            target = population["coefficients"][index:index + 1]
            command, _ = _mapper_command(terminal, target, source["inputs"], device)
            original_payload, local = _load_shard_row(source["source_dir"], role, index)
            original = _expected_original_command(original_payload, local, role)
            _require(torch.equal(command.cpu(), original),
                     f"CUDA terminal checkpoint does not reproduce saved {role} command {index}")
            generated.append(command.cpu().clone())
            if role == "holdout":
                v2_command, _ = _mapper_command(v2, target, source["inputs"], device)
                original_v2 = _expected_original_command(original_payload, local, "v2_holdout")
                _require(torch.equal(v2_command.cpu(), original_v2),
                         f"CUDA V2 checkpoint does not reproduce saved holdout command {index}")
                expected.append(v2_command.cpu().clone())
            if index % 100 == 99:
                _write_json(output_dir / "progress.json", {
                    "task_id": TASK_ID, "stage": "source_command_replay", "role": role,
                    "completed_rows": index + 1, "expected_rows": len(population["labels"])})
        terminal_commands[role] = {
            "indices": population["source_indices"].clone(), "labels": population["labels"].clone(),
            "proposal_ids": population["proposal_ids"].clone(),
            "target_coefficients": population["coefficients"].clone(),
            "commands": torch.cat(generated, dim=0)}
        row_counts[role] = len(generated)
        if role == "holdout":
            cpu_expected[role] = {"commands": torch.cat(expected, dim=0)}
    v2_source = {
        "indices": source["population"]["holdout"]["source_indices"].clone(),
        "labels": source["population"]["holdout"]["labels"].clone(),
        "proposal_ids": source["population"]["holdout"]["proposal_ids"].clone(),
        "target_coefficients": source["population"]["holdout"]["coefficients"].clone(),
        "commands": cpu_expected["holdout"]["commands"],
    }
    _save(output_dir / "model_binding.pt", {
        "schema": f"{TASK_ID}.model_binding.v1", "task_id": TASK_ID,
        "source_run_id": source["checkpoint"]["run_id"],
        "terminal_checkpoint": str(source["terminal_path"]),
        "terminal_step": 15000,
        "terminal_loaded_state": terminal_snapshot,
        "v2_checkpoint": str(source["source_v2_path"]),
        "v2_loaded_state": v2_snapshot,
        "response_projector": str(source["projector_path"]),
        "response_matrix": source["inputs"]["response_matrix"].clone(),
        "right_inverse": source["inputs"]["inverse"].clone(),
        "scale": source["inputs"]["scale"].clone(),
    })
    _save(output_dir / "terminal_command_replay.pt", {
        "schema": f"{TASK_ID}.terminal_command_replay.v1", "task_id": TASK_ID,
        "source_run_id": source["checkpoint"]["run_id"], "roles": terminal_commands})
    _save(output_dir / "v2_holdout_command_replay.pt", {
        "schema": f"{TASK_ID}.v2_command_replay.v1", "task_id": TASK_ID,
        "source_checkpoint": str(source["source_v2_path"]), "role": v2_source})
    correction = (
        "# 历史报告事实修订\n\n"
        "原始报告 `windows_verify_01/closeout_report.md` 保留不改。其科学认识段中“开发人口已满足的参照资格”表述不准确："
        "完整双网格见证属于固定训练人口200条；历史开发人口没有逐目标参照资格证据。\n\n"
        "本修订只更正证据角色，不更改15,000步终点、原 `VALID_FAILED` 结论或语义代理数值。\n"
    )
    (output_dir / "report_correction.md").write_text(correction, encoding="utf-8")
    summary = {
        "schema": f"{TASK_ID}.source_review.v1", "task_id": TASK_ID,
        "status": "SOURCE_REPLAY_PRODUCED", "source_run_id": source["checkpoint"]["run_id"],
        "terminal_checkpoint_step": 15000, "terminal_rows_replayed": sum(row_counts.values()),
        "terminal_rows_by_role": row_counts, "v2_holdout_rows_replayed": len(v2_source["labels"]),
        "cuda_command_equality": "bitwise", "scientific_updates": 0,
        "mapper_updates": 0, "optical_updates": 0, "evaluator_updates": 0,
        "original_saved_measurements_regenerated": False,
        "history_correction": "training witnesses do not qualify development rows",
    }
    _write_json(output_dir / "source_review_summary.json", summary)
    _write_json(output_dir / "source_review_run_config.json", config)
    _write_manifest(output_dir)
    return summary


def _write_manifest(directory: Path) -> dict[str, Any]:
    files = [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(directory.rglob("*")) if path.is_file()
             and path.name not in {"artifact_manifest.json", "transfer_receipt.json"}]
    manifest = {"schema": f"{TASK_ID}.artifact_manifest.v1", "task_id": TASK_ID,
                "files": files, "total_bytes": sum(row["bytes"] for row in files)}
    _write_json(directory / "artifact_manifest.json", manifest)
    return manifest


def _verify_manifest_files(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    _require(manifest_path.is_file(), f"artifact manifest is missing: {manifest_path}", FileNotFoundError)
    manifest = _json(manifest_path)
    entries = manifest.get("files")
    _require(manifest.get("task_id") == TASK_ID and isinstance(entries, list),
             "artifact manifest task identity or file table is invalid")
    paths = [row.get("path") for row in entries]
    _require(all(isinstance(value, str) for value in paths) and len(paths) == len(set(paths)),
             "artifact manifest contains duplicate or invalid paths")
    for row in entries:
        relative = Path(str(row["path"]))
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 "artifact manifest contains a path escape")
        path = directory / relative
        _require(path.is_file() and path.stat().st_size == int(row["bytes"]),
                 f"artifact file is missing or has a different byte count: {relative}")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name not in {"artifact_manifest.json", "transfer_receipt.json"}}
    telemetry_path = directory / "resource_telemetry.csv"
    unregistered = actual - set(paths)
    if "resource_telemetry.csv" in unregistered:
        _require(telemetry_path.is_file() and not telemetry_path.is_symlink(),
                 "AutoDL resource telemetry sidecar must be a regular file")
        # autodl_run.ps1 copies this operational sidecar after the task writes its manifest.
        unregistered.remove("resource_telemetry.csv")
    _require(not unregistered, "artifact manifest does not enumerate the current evidence files exactly")
    _require(int(manifest.get("total_bytes", -1)) == sum(int(row["bytes"]) for row in entries),
             "artifact manifest total byte count is inconsistent")
    return manifest


def _verify_transfer_receipt(backup_root: Path, expected_paths: set[str] | None = None) -> dict[str, Any]:
    backup_root = backup_root.resolve()
    receipt_path = backup_root / "transfer_receipt.json"
    _require(receipt_path.is_file(), f"completed transfer receipt is missing: {receipt_path}", FileNotFoundError)
    receipt = _json(receipt_path)
    entries = receipt.get("files")
    allowed_remote_root = (
        "/root/autodl-tmp/codex-original/outputs/20261030_lightgen_g1_fixed_frame_transfer_diagnosis_v1/runs")
    remote_run_dir = str(receipt.get("remote_run_dir", "")).rstrip("/")
    _require(receipt.get("status") == "COMPLETE" and isinstance(entries, list)
             and Path(str(receipt.get("destination", ""))).resolve() == backup_root
             and (remote_run_dir == allowed_remote_root
                  or remote_run_dir.startswith(allowed_remote_root + "/")),
             "transfer receipt is incomplete or points at a different run")
    paths = [row.get("path") for row in entries]
    _require(all(isinstance(value, str) for value in paths) and len(paths) == len(set(paths)),
             "transfer receipt contains duplicate or invalid file paths")
    for row in entries:
        relative = Path(str(row["path"]))
        _require(not relative.is_absolute() and ".." not in relative.parts
                 and row.get("status") == "COPIED" and int(row.get("scp_exit_code", -1)) == 0,
                 "transfer receipt records an unsuccessful or unsafe file")
        path = backup_root / relative
        actual_bytes = path.stat().st_size if path.is_file() else -1
        _require(actual_bytes == int(row.get("remote_size_bytes", -2))
                 and actual_bytes == int(row.get("local_size_bytes", -3)),
                 f"transferred file is missing or its current size differs from the receipt: {relative}")
    if expected_paths is not None:
        _require(set(paths) == expected_paths,
                 "transfer receipt file list differs from the registered evidence manifests")
    return receipt


def _manifest_transfer_paths(directory: Path, prefix: str = "") -> set[str]:
    manifest = _verify_manifest_files(directory)
    own_manifest = f"{prefix}artifact_manifest.json"
    files = {f"{prefix}{row['path']}" for row in manifest["files"]}
    files.add(own_manifest)
    return files


def _cpu_command_rows(mapper_state: Mapping[str, torch.Tensor], targets: torch.Tensor,
                      inputs: Mapping[str, Any], batch_size: int = 10) -> torch.Tensor:
    device = torch.device("cpu")
    mapper = fixed._new_mapper(inputs, device, mapper_state)
    mapper.eval()
    values = []
    with torch.no_grad():
        for start in range(0, targets.shape[0], batch_size):
            target = targets[start:start + batch_size].to(torch.complex128)
            features = projected._normalized_features(target, inputs["scale"], device)
            command, _ = projected._map_amplitude(
                mapper, features, target, inputs["matrix"].cpu(), inputs["inverse"].cpu(),
                inputs["scale"].cpu())
            values.append(command.float().cpu())
    return torch.cat(values)


def _verify_source(config: Mapping[str, Any], source_dir: Path, review_dir: Path,
                   report_dir: Path | None = None) -> dict[str, Any]:
    source = _load_source(source_dir, config)
    review_dir = review_dir.resolve()
    source_manifest = _verify_manifest_files(review_dir)
    expected_transfer_paths = {row["path"] for row in source_manifest["files"]}
    expected_transfer_paths.add("artifact_manifest.json")
    _verify_transfer_receipt(review_dir, expected_transfer_paths)
    summary = _json(review_dir / "source_review_summary.json")
    binding = torch.load(review_dir / "model_binding.pt", map_location="cpu", weights_only=False)
    terminal = torch.load(review_dir / "terminal_command_replay.pt", map_location="cpu", weights_only=False)
    v2 = torch.load(review_dir / "v2_holdout_command_replay.pt", map_location="cpu", weights_only=False)
    _require(summary.get("schema") == f"{TASK_ID}.source_review.v1"
             and summary.get("status") == "SOURCE_REPLAY_PRODUCED"
             and binding.get("schema") == f"{TASK_ID}.model_binding.v1"
             and binding.get("source_run_id") == source["checkpoint"]["run_id"]
             and binding.get("terminal_step") == 15000
             and fixed._compare_states(binding["terminal_loaded_state"],
                                       source["checkpoint"]["mapper_state"]) == 0.0
             and fixed._compare_states(binding["v2_loaded_state"],
                                       source["inputs"]["source_mapper"]["mapper_state"]) == 0.0
             and torch.equal(binding["response_matrix"], source["inputs"]["response_matrix"])
             and torch.equal(binding["right_inverse"], source["inputs"]["inverse"])
             and torch.equal(binding["scale"], source["inputs"]["scale"]),
             "CUDA source binding is not tied to the explicit checkpoints and frozen projection")
    _require(terminal.get("task_id") == TASK_ID and v2.get("task_id") == TASK_ID
             and terminal.get("source_run_id") == source["checkpoint"]["run_id"],
             "saved source command replay identity differs")
    cpu_max_abs = 0.0
    terminal_rows = 0
    for role in ("train", "development", "holdout"):
        replay = terminal["roles"][role]
        population = source["population"][role]
        expected_shape = tuple(population["coefficients"].shape)
        _require(replay.get("commands").dtype == torch.float32
                 and tuple(replay["commands"].shape) == (expected_shape[0], 1, 200, 200)
                 and torch.equal(replay["indices"], population["source_indices"])
                 and torch.equal(replay["labels"], population["labels"])
                 and torch.equal(replay["proposal_ids"], population["proposal_ids"])
                 and torch.equal(replay["target_coefficients"], population["coefficients"]),
                 f"terminal replay population identity differs for {role}")
        original_commands = []
        for index in range(expected_shape[0]):
            payload, local = _load_shard_row(source["source_dir"], role, index)
            original_commands.append(_expected_original_command(payload, local, role))
        original = torch.cat(original_commands)
        cpu_expected = _cpu_command_rows(binding["terminal_loaded_state"], population["coefficients"],
                                         source["inputs"])
        cpu_max_abs = max(cpu_max_abs, _validate_bound_commands(replay["commands"], original, cpu_expected))
        terminal_rows += expected_shape[0]
    population = source["population"]["holdout"]
    replay = v2["role"]
    original_commands = []
    for index in range(len(population["labels"])):
        payload, local = _load_shard_row(source["source_dir"], "holdout", index)
        original_commands.append(_expected_original_command(payload, local, "v2_holdout"))
    original = torch.cat(original_commands)
    cpu_expected = _cpu_command_rows(binding["v2_loaded_state"], population["coefficients"], source["inputs"])
    cpu_max_abs = max(cpu_max_abs, _validate_bound_commands(replay["commands"], original, cpu_expected))
    _require(torch.equal(replay["indices"], population["source_indices"])
             and torch.equal(replay["labels"], population["labels"])
             and torch.equal(replay["proposal_ids"], population["proposal_ids"])
             and torch.equal(replay["target_coefficients"], population["coefficients"]),
             "V2 holdout CUDA replay is not bound to the fixed holdout targets")
    result = {"schema": f"{TASK_ID}.source_review_verification.v1", "task_id": TASK_ID,
              "status": "SOURCE_REPLAY_VERIFIED", "source_run_id": source["checkpoint"]["run_id"],
              "terminal_rows_verified": terminal_rows,
              "v2_holdout_rows_verified": int(population["labels"].numel()),
              "cuda_command_bitwise_equal_to_original": True,
              "cpu_checkpoint_replay_max_abs_difference": cpu_max_abs,
              "cpu_replay_tolerance": SOURCE_CPU_COMMAND_TOLERANCE,
              "source_review_manifest_files": len(source_manifest["files"]),
              "source_review_manifest_bytes": int(source_manifest["total_bytes"]),
              "model_binding_verified": True, "response_projector_binding_verified": True,
              "training_witness_scope_correction": True, "scientific_updates": 0}
    if report_dir is not None:
        _require(not report_dir.exists(), f"source verification report already exists: {report_dir}", FileExistsError)
        report_dir.mkdir(parents=True)
        _write_json(report_dir / "source_review_verification.json", result)
        _write_manifest(report_dir)
    return result


def _transition(reference: np.ndarray, candidate: np.ndarray,
                labels: np.ndarray, mask: np.ndarray) -> dict[str, int | float]:
    ref_ok, cand_ok = reference == labels, candidate == labels
    count = int(mask.sum())
    return {
        "count": count,
        "reference_correct_candidate_correct": int(np.sum(mask & ref_ok & cand_ok)),
        "reference_correct_candidate_wrong": int(np.sum(mask & ref_ok & ~cand_ok)),
        "reference_wrong_candidate_correct": int(np.sum(mask & ~ref_ok & cand_ok)),
        "reference_wrong_candidate_wrong": int(np.sum(mask & ~ref_ok & ~cand_ok)),
        "wrong_prediction_preserved": int(np.sum(mask & ~ref_ok & ~cand_ok
                                                   & (reference == candidate))),
        "prediction_changed": int(np.sum(mask & (reference != candidate))),
        "net_accuracy_change": (float(np.mean(cand_ok[mask]) - np.mean(ref_ok[mask]))
                                if count else 0.0),
    }


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _semantic_audit(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"semantic audit output already exists: {output_dir}", FileExistsError)
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True)
    pop = torch.load(source_dir / "selected_population.pt", map_location="cpu", weights_only=False)
    prior = torch.load(source_dir / "prior_evidence" / "physical_phase_gaussian_prior.pt",
                       map_location="cpu", weights_only=False)
    semantic = _json(source_dir / "holdout_semantic_summary.json")
    saved_readouts = torch.load(source_dir / "holdout_semantic_readouts.pt",
                                map_location="cpu", weights_only=False)
    holdout = pop["holdout"]
    labels = holdout["labels"].long().numpy()
    proposals = holdout["proposal_ids"].long().numpy()
    _require(pop.get("task_id") == fixed.TASK_ID and semantic.get("evidence_role") ==
             "new_fixed_holdout_2000_semantic_proxy"
             and tuple(labels.shape) == (2000,) and tuple(proposals.shape) == (2000,)
             and torch.equal(saved_readouts["labels"].long(), holdout["labels"].long())
             and torch.equal(saved_readouts["proposal_ids"].long(), holdout["proposal_ids"].long()),
             "saved semantic predictions/readouts do not bind to the fixed holdout population")
    conditions = ("old_g1_direct_decoder", "new_prior_direct_decoder",
                  "frozen_v2_mapped", "fixed_frame_mapped")
    readout_conditions = saved_readouts["readouts"]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for condition in conditions:
        _require(condition in readout_conditions
                 and tuple(readout_conditions[condition].shape) == (2000, 1, 28, 28),
                 f"saved readout is missing for {condition}")
        predictions[condition] = {}
        for family in ("ED", "EC"):
            values = np.asarray(semantic["conditions"][condition]["evaluators"][family][
                "predicted_labels"], dtype=np.int64)
            _require(values.shape == (2000,) and np.all((values >= 0) & (values < CLASS_COUNT)),
                     f"saved {family} predictions are malformed for {condition}")
            predictions[condition][family] = values

    pairwise: dict[str, Any] = {}
    for left, right in (("new_prior_direct_decoder", "frozen_v2_mapped"),
                        ("frozen_v2_mapped", "fixed_frame_mapped"),
                        ("new_prior_direct_decoder", "fixed_frame_mapped")):
        pairwise[f"{left}_to_{right}"] = {
            family: {"overall": _transition(predictions[left][family], predictions[right][family],
                                             labels, np.ones(2000, dtype=bool)),
                     "by_class": {str(label): _transition(
                         predictions[left][family], predictions[right][family], labels,
                         labels == label) for label in range(CLASS_COUNT)}}
            for family in ("ED", "EC")}

    class_accuracy = {}
    for condition in conditions:
        class_accuracy[condition] = {
            family: {str(label): float(np.mean(predictions[condition][family][labels == label] == label))
                     for label in range(CLASS_COUNT)}
            for family in ("ED", "EC")}

    area = float(semantic["camera_pixel_area_m2"])
    gain = float(semantic["camera_gain_w_inverse"])
    _require(abs(area - 6.25e-12) <= 1.0e-22 and gain > 0.0,
             "saved semantic readout calibration is invalid")
    brightness = {condition: {str(label): {"pixels": 0, "above_one_after_gain": 0,
                                           "at_zero_after_clip": 0, "at_one_after_clip": 0}
                              for label in range(CLASS_COUNT)} for condition in conditions}
    numeric_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ("train", "development", "holdout")}
    for role in numeric_rows:
        role_pop = pop[role]
        paths = sorted((source_dir / "measurements" / role).glob("shard_*.pt"))
        expected = len(role_pop["labels"])
        _require(len(paths) == expected // 10, f"saved {role} shard count is incomplete")
        for path in paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            _require(shard.get("task_id") == fixed.TASK_ID and shard.get("cohort_role") == role,
                     f"saved semantic/numeric shard identity differs: {path.name}")
            for local, saved in enumerate(shard["rows"]):
                index = int(shard["start"]) + local
                _require(index == int(shard["indices"][local])
                         and int(saved["label"]) == int(role_pop["labels"][index])
                         and int(saved["proposal_id"]) == int(role_pop["proposal_ids"][index])
                         and torch.equal(shard["target_coefficients"][local],
                                         role_pop["coefficients"][index]),
                         f"saved {role} target identity differs at row {index}")
                command = shard["commands"][local]
                numeric_rows[role].append({
                    "index": index, "label": int(saved["label"]),
                    "proposal_id": int(saved["proposal_id"]),
                    "max_coefficient_error": max(float(saved["coefficient_2048"]),
                                                  float(saved["coefficient_4096"])),
                    "max_native_cmos_error": max(float(saved["native_cmos_2048"]),
                                                  float(saved["native_cmos_4096"])),
                    "amplitude_boundary_saturation_fraction": float(
                        ((command <= 0.0) | (command >= 1.0)).double().mean()),
                })
            if role == "holdout":
                for local in range(int(shard["stop"]) - int(shard["start"])):
                    new_label = int(shard["labels"][local])
                    old_label = int(shard["old_g1_labels"][local])
                    raws = {
                        "old_g1_direct_decoder": shard["old_g1_direct_raw_cmos_2048"][local],
                        "new_prior_direct_decoder": shard["direct_raw_cmos"][2048][local],
                        "frozen_v2_mapped": shard["frozen_v2_outputs"][2048]["raw_cmos"][local],
                        "fixed_frame_mapped": shard["candidate_outputs"][2048]["raw_cmos"][local],
                    }
                    for condition, raw in raws.items():
                        label = old_label if condition == "old_g1_direct_decoder" else new_label
                        calibrated = raw.double() * area * gain
                        clipped = calibrated.clamp(0.0, 1.0)
                        item = brightness[condition][str(label)]
                        item["pixels"] += int(raw.numel())
                        item["above_one_after_gain"] += int((calibrated > 1.0).sum())
                        item["at_zero_after_clip"] += int((clipped == 0.0).sum())
                        item["at_one_after_clip"] += int((clipped == 1.0).sum())
    train = pop["train"]["coefficients"].to(torch.complex128)
    scale = prior["scale"].to(torch.float64)
    train_features = fixed._raw_features(train, scale)
    nearest_by_role: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    for role, rows in numeric_rows.items():
        role_pop = pop[role]
        features = fixed._raw_features(role_pop["coefficients"].to(torch.complex128), scale)
        nearest_parts = []
        for start in range(0, len(features), 50):
            distances = torch.cdist(features[start:start + 50], train_features)
            if role == "train":
                for local in range(distances.shape[0]):
                    distances[local, start + local] = float("inf")
            nearest_parts.append(distances.min(dim=1).values)
        nearest = torch.cat(nearest_parts).numpy()
        errors = np.asarray([row["max_coefficient_error"] for row in rows], dtype=np.float64)
        cmos = np.asarray([row["max_native_cmos_error"] for row in rows], dtype=np.float64)
        saturation = np.asarray([row["amplitude_boundary_saturation_fraction"] for row in rows],
                                dtype=np.float64)
        order = np.argsort(nearest, kind="stable")
        groups = np.empty(len(rows), dtype=np.int64)
        groups[order] = np.minimum(np.arange(len(rows)) * 4 // len(rows), 3)
        by_class = {}
        for label in range(CLASS_COUNT):
            mask = role_pop["labels"].numpy() == label
            by_class[str(label)] = {
                "count": int(mask.sum()), "nearest_train_l2_mean": float(nearest[mask].mean()),
                "nearest_train_l2_median": float(np.median(nearest[mask])),
                "coefficient_error_mean": float(errors[mask].mean()),
                "native_cmos_error_mean": float(cmos[mask].mean()),
                "amplitude_boundary_saturation_fraction_mean": float(saturation[mask].mean()),
            }
        quartile_summary = []
        for group in range(4):
            mask = groups == group
            row = {"role": role, "quartile": group + 1, "count": int(mask.sum()),
                   "nearest_train_l2_min": float(nearest[mask].min()),
                   "nearest_train_l2_max": float(nearest[mask].max()),
                   "coefficient_error_mean": float(errors[mask].mean()),
                   "native_cmos_error_mean": float(cmos[mask].mean()),
                   "amplitude_boundary_saturation_fraction_mean": float(saturation[mask].mean()),
                   "pearson_distance_coefficient_error": _corr(nearest[mask], errors[mask]),
                   "pearson_distance_cmos_error": _corr(nearest[mask], cmos[mask]),
                   "pearson_saturation_coefficient_error": _corr(saturation[mask], errors[mask])}
            quartile_summary.append(row)
            quartile_rows.append(row)
        nearest_by_role[role] = {
            "role": ("fixed_train200_leave_one_out" if role == "train" else role),
            "count": len(rows), "distance_definition": "Euclidean distance in frozen-scale normalized interleaved real-imag coefficient coordinates",
            "nearest_train_l2_mean": float(nearest.mean()),
            "nearest_train_l2_median": float(np.median(nearest)),
            "correlation_is_descriptive_only": True,
            "by_class": by_class, "distance_quartiles": quartile_summary,
        }
        for index, item in enumerate(rows):
            sample_rows.append({"role": role, "index": index, "label": item["label"],
                                "proposal_id": item["proposal_id"],
                                "nearest_train_l2": float(nearest[index]),
                                "max_coefficient_error": item["max_coefficient_error"],
                                "max_native_cmos_error": item["max_native_cmos_error"],
                                "amplitude_boundary_saturation_fraction":
                                    item["amplitude_boundary_saturation_fraction"],
                                "distance_quartile": int(groups[index] + 1)})
    brightness_summary = {
        condition: {label: {**values,
                            "fraction_above_one_after_gain": values["above_one_after_gain"] / values["pixels"],
                            "fraction_at_zero_after_clip": values["at_zero_after_clip"] / values["pixels"],
                            "fraction_at_one_after_clip": values["at_one_after_clip"] / values["pixels"]}
                    for label, values in classes.items()}
        for condition, classes in brightness.items()}
    result = {
        "schema": f"{TASK_ID}.semantic_attribution.v1", "task_id": TASK_ID,
        "status": "SEMANTIC_ATTRIBUTION_COMPLETE", "count": 2000,
        "prediction_source": "existing holdout_semantic_summary.json; no evaluator rerun",
        "readout_source": "existing holdout_semantic_readouts.pt; no optical output regenerated",
        "pairwise_same_target_transitions": pairwise,
        "class_accuracy": class_accuracy,
        "same_class_diversity_from_saved_summary": semantic["same_class_diversity"],
        "brightness_clipping": brightness_summary,
        "prior_comparison": {
            "old_g1_vs_new_physical_phase_prior": "class-level distribution comparison; coefficient samples are not paired",
            "new_prior_direct_to_frozen_v2_to_terminal": "same new fixed holdout target population",
            "human_readability": "not_measured", "novelty": "not_measured",
        },
        "nearest_training_relationship": nearest_by_role,
        "amplitude_boundary_saturation_note": "post-clip command mass at 0 or 1; not the saved pre-clip exceedance fraction",
        "correlation_claim": False, "LPIPS": "not_computed",
        "camera_pixel_area_m2": area, "camera_gain_w_inverse": gain,
        "causal_attribution": "descriptive only; associations do not isolate a cause",
    }
    _write_json(output_dir / "semantic_attribution_summary.json", result)
    for name, rows in (("semantic_transitions.csv", [
            {"comparison": comparison, "evaluator": family, "population_index": index,
             "label": int(labels[index]), "proposal_id": int(proposals[index]),
             "reference_prediction": int(predictions[left][family][index]),
             "candidate_prediction": int(predictions[right][family][index])}
            for comparison, left, right in (
                ("new_prior_to_v2", "new_prior_direct_decoder", "frozen_v2_mapped"),
                ("v2_to_terminal", "frozen_v2_mapped", "fixed_frame_mapped"),
                ("new_prior_to_terminal", "new_prior_direct_decoder", "fixed_frame_mapped"))
            for family in ("ED", "EC") for index in range(2000)]),
            ("numeric_distance_by_sample.csv", sample_rows),
            ("numeric_distance_quartiles.csv", quartile_rows)):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            _require(bool(rows), f"cannot write empty semantic report {name}")
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _write_manifest(output_dir)
    return result


class _StopAtSolverIteration(Exception):
    pass


def _capture_free_state(response: torch.Tensor, target: torch.Tensor, lipschitz: float,
                        stop_at: int, initial_state: Mapping[str, Any] | None = None
                        ) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def checkpoint(state: dict[str, Any]) -> None:
        if int(state["iteration"]) == stop_at:
            captured.update(copy.deepcopy(state))
            raise _StopAtSolverIteration

    try:
        reach._bounded_phase_solve(response, target, starts=8, max_steps=5000,
                                   stop_error=1.0e-6, lipschitz=lipschitz,
                                   initial_state=None if initial_state is None else dict(initial_state),
                                   checkpoint=checkpoint, checkpoint_interval=1)
    except _StopAtSolverIteration:
        pass
    _require(captured and int(captured["iteration"]) == stop_at,
             f"free phase solver did not expose recovery cursor {stop_at}")
    return captured


def _recovery_step4(checkpoint_path: Path, output_path: Path, source_dir: Path,
                    config_path: Path) -> dict[str, Any]:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == f"{TASK_ID}.solver_recovery_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("step") == 3,
             "solver recovery worker checkpoint identity or cursor is invalid")
    source = _load_source(source_dir.resolve(), _config(config_path.resolve()))
    _require(saved.get("source_run_id") == source["checkpoint"]["run_id"],
             "solver recovery checkpoint is bound to another frozen source")
    fixed._set_numerics()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    response = source["inputs"]["response_matrix"].to(device=device, dtype=torch.complex128)
    target = saved["target"].to(device=device, dtype=torch.complex128)
    state3 = saved["solver_state"]
    kind = saved["kind"]
    captured: dict[str, Any] = {}
    if kind == "fixed":
        def checkpoint(state: dict[str, Any]) -> None:
            if int(state["iteration"]) == 4:
                captured.update(copy.deepcopy(state))

        exposure._fista_fixed_phase_solve(
            response, target, float(saved["phase_rad"]), max_steps=4,
            lipschitz=float(saved["lipschitz"]), initial_state=state3,
            checkpoint=checkpoint, checkpoint_interval=1)
    else:
        def checkpoint(state: dict[str, Any]) -> None:
            if int(state["iteration"]) == 4:
                captured.update(copy.deepcopy(state))
                raise _StopAtSolverIteration

        try:
            reach._bounded_phase_solve(
                response, target, starts=8, max_steps=5000, stop_error=1.0e-6,
                lipschitz=float(saved["lipschitz"]), initial_state=state3,
                checkpoint=checkpoint, checkpoint_interval=1)
        except _StopAtSolverIteration:
            pass
    _require(captured and int(captured["iteration"]) == 4,
             "independent solver recovery did not complete the fourth iteration")
    result = {"schema": f"{TASK_ID}.solver_recovery_step4.v1", "task_id": TASK_ID,
              "source_run_id": source["checkpoint"]["run_id"],
              "kind": kind, "step": 4, "solver_state": captured,
              "loss": _solver_state_loss(kind, response, target, captured),
              "rng_state_cpu": torch.get_rng_state().clone(),
              "rng_state_cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]
              if torch.cuda.is_available() else []}
    _save(output_path, result)
    return result


def _analytic_solver_controls(device: torch.device) -> dict[str, Any]:
    response = torch.eye(2, dtype=torch.complex128, device=device)
    inside = torch.tensor([0.23 + 0j, 0.78 + 0j], dtype=torch.complex128, device=device)
    lip = reach._projected_spectral_norm(response)
    fixed_inside = exposure._fista_fixed_phase_solve(response, inside, 0.0, max_steps=10,
                                                     lipschitz=lip)
    inside_error = float(torch.linalg.vector_norm(
        response @ fixed_inside["amplitude"].to(device).to(torch.complex128) - inside)
        / torch.linalg.vector_norm(inside))
    rotated = inside * torch.exp(torch.tensor(0.71j, dtype=torch.complex128, device=device))
    fixed_zero = exposure._fista_fixed_phase_solve(response, rotated, 0.0, max_steps=20,
                                                    lipschitz=lip)
    zero_error = float(torch.linalg.vector_norm(
        response @ fixed_zero["amplitude"].to(device).to(torch.complex128) - rotated)
        / torch.linalg.vector_norm(rotated))
    fixed_rotated = exposure._fista_fixed_phase_solve(response, rotated, -0.71, max_steps=10,
                                                       lipschitz=lip)
    rotated_error = float(torch.linalg.vector_norm(
        response @ fixed_rotated["amplitude"].to(device).to(torch.complex128) - rotated
        * torch.exp(torch.tensor(-0.71j, dtype=torch.complex128, device=device)))
        / torch.linalg.vector_norm(rotated))
    excessive = torch.tensor([1.4 + 0j, 0j], dtype=torch.complex128, device=device)
    bounded = exposure._fista_fixed_phase_solve(response, excessive, 0.0, max_steps=20,
                                                 lipschitz=lip)
    excessive_error = float(torch.linalg.vector_norm(
        response @ bounded["amplitude"].to(device).to(torch.complex128) - excessive)
        / torch.linalg.vector_norm(excessive))
    _require(inside_error <= 1.0e-6 and zero_error > GATE_LIMIT
             and rotated_error <= 1.0e-6 and excessive_error > GATE_LIMIT,
             "analytic box, fixed-phase, or public-phase solver control failed")
    amp = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64, device=device)
    real_response = torch.eye(4, dtype=torch.float64, device=device)
    complex_target = torch.tensor([[0.2 + 0.3j, 0.6 + 0.1j]], dtype=torch.complex128, device=device)
    scale = torch.ones(2, dtype=torch.float64, device=device)
    command, details = _fixed_phase_projection(amp.float().reshape(1, 4), complex_target,
                                               real_response, real_response, scale, 0.0)
    expected_ri = torch.view_as_real(complex_target).reshape(1, 4)
    projection_error = float((command.reshape(1, 4).double() - expected_ri).abs().max())
    _require(projection_error <= 1.0e-6 and float(details["prescribed_phase_rad"][0]) == 0.0,
             "zero-phase fixed projection failed its analytic real-imaginary reference")
    x = torch.tensor([0.31, 0.67], dtype=torch.float64, device=device)
    target = torch.tensor([0.2 + 0.1j, 0.6 - 0.3j], dtype=torch.complex128, device=device)
    matrix = torch.tensor([[1.0 + 0.2j, -0.3 + 0.4j],
                           [0.1 - 0.5j, 0.8 + 0.1j]], dtype=torch.complex128, device=device)
    desired = target * torch.exp(torch.tensor(0.43j, dtype=torch.complex128, device=device))
    residual = x.to(torch.complex128) @ matrix.T - desired
    gradient = (residual.reshape(1, -1) @ matrix.conj()).real.reshape(-1)
    def objective(value: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((value.to(torch.complex128) @ matrix.T - desired).abs().square())
    eps = 1.0e-6
    numeric = torch.stack([(objective(x + torch.eye(2, device=device, dtype=torch.float64)[i] * eps)
                            - objective(x - torch.eye(2, device=device, dtype=torch.float64)[i] * eps))
                           / (2.0 * eps) for i in range(2)])
    gradient_error = float((gradient - numeric).abs().max())
    direction_change = float(objective(x - 1.0e-3 * gradient) - objective(x))
    _require(gradient_error <= 1.0e-5 and direction_change < 0.0,
             "fixed-phase complex gradient or descent direction failed")
    return {"inside_box_relative_error": inside_error,
            "fixed_zero_phase_rotated_target_relative_error": zero_error,
            "rotated_public_phase_relative_error": rotated_error,
            "out_of_range_target_relative_error": excessive_error,
            "phi_zero_projection_max_abs": projection_error,
            "complex_gradient_max_abs_error": gradient_error,
            "complex_gradient_descent_objective_change": direction_change,
            "support_bound_positive_and_feasible": True}


def _nested_max_delta(left: Any, right: Any) -> float:
    if torch.is_tensor(left):
        _require(torch.is_tensor(right) and left.shape == right.shape,
                 "solver recovery tensor structures differ")
        return float((left.to(torch.float64) - right.to(torch.float64)).abs().max()) if left.numel() else 0.0
    if isinstance(left, Mapping):
        _require(isinstance(right, Mapping) and set(left) == set(right),
                 "solver recovery mapping structures differ")
        return max((_nested_max_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (tuple, list)):
        _require(isinstance(right, (tuple, list)) and len(left) == len(right),
                 "solver recovery sequence structures differ")
        return max((_nested_max_delta(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left is right else float("inf")
    if isinstance(left, int) or isinstance(right, int):
        return 0.0 if type(left) is type(right) and left == right else float("inf")
    if isinstance(left, float) and isinstance(right, float):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _solver_recovery_case(kind: str, response: torch.Tensor, target: torch.Tensor,
                          phase: float, lipschitz: float, directory: Path,
                          run_id: str, source_dir: Path,
                          source_run_id: str) -> dict[str, Any]:
    if kind == "fixed":
        initial = _initial_phase_specific_solver_state(
            response, target, torch.full((response.shape[1],), 0.5,
                                         dtype=torch.float64, device=response.device), phase)
        third_state: dict[str, Any] = {}
        exposure._fista_fixed_phase_solve(
            response, target, phase, max_steps=3, lipschitz=lipschitz,
            initial_state=initial, checkpoint=lambda state: third_state.update(copy.deepcopy(state)),
            checkpoint_interval=1)
    else:
        third_state = _capture_free_state(response, target, lipschitz, 3)
    checkpoint_path = directory / f"{kind}_solver_step_00003.pt"
    rng_before_save = {"cpu": torch.get_rng_state().clone(),
                       "cuda": [item.clone() for item in torch.cuda.get_rng_state_all()]}
    _save(checkpoint_path, {"schema": f"{TASK_ID}.solver_recovery_checkpoint.v1",
                            "task_id": TASK_ID, "run_id": run_id, "kind": kind, "step": 3,
                            "source_run_id": source_run_id,
                            "target": target.detach().cpu().clone(), "phase_rad": phase,
                            "lipschitz": lipschitz,
                            "solver_state": copy.deepcopy(third_state)})
    _require(torch.equal(rng_before_save["cpu"], torch.get_rng_state())
             and all(torch.equal(a, b) for a, b in zip(
                 rng_before_save["cuda"], torch.cuda.get_rng_state_all(), strict=True)),
             "solver checkpoint serialization changed RNG state")
    if kind == "fixed":
        direct_state: dict[str, Any] = {}
        exposure._fista_fixed_phase_solve(
            response, target, phase, max_steps=4, lipschitz=lipschitz,
            initial_state=third_state,
            checkpoint=lambda state: direct_state.update(copy.deepcopy(state)),
            checkpoint_interval=1)
    else:
        direct_state = _capture_free_state(response, target, lipschitz, 4, third_state)
    direct_loss = _solver_state_loss(kind, response, target, direct_state)
    direct_path = directory / f"{kind}_solver_direct_step_00004.pt"
    _save(direct_path, {"solver_state": direct_state, "loss": direct_loss})
    replay_path = directory / f"{kind}_solver_recovered_step_00004.pt"
    worker = [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_g1_fixed_frame_transfer_diagnosis_v1",
              "--mode", "solver-recovery-worker", "--config", str(DEFAULT_CONFIG),
              "--source-dir", str(source_dir.resolve()),
              "--checkpoint", str(checkpoint_path), "--output-dir", str(replay_path)]
    completed = subprocess.run(worker, cwd=ROOT, capture_output=True, text=True, check=False)
    _require(completed.returncode == 0 and replay_path.is_file(),
             f"independent {kind} solver recovery worker failed: {completed.stderr[-2000:]}")
    recovered = torch.load(replay_path, map_location="cpu", weights_only=False)
    model_delta = _nested_max_delta(direct_state, recovered["solver_state"])
    loss_delta = abs(direct_loss - float(recovered["loss"]))
    _require(model_delta <= SOLVER_RECOVERY_TOLERANCE
             and loss_delta <= SOLVER_RECOVERY_TOLERANCE
             and recovered.get("source_run_id") == source_run_id
             and recovered.get("kind") == kind and recovered.get("step") == 4,
             f"direct and restored fourth solver iteration differ for {kind}")
    return {"kind": kind, "step3_checkpoint": str(checkpoint_path),
            "direct_step4": str(direct_path), "recovered_step4": str(replay_path),
            "state_max_abs_delta": model_delta, "loss_abs_delta": loss_delta,
            "rng_preserved_by_save": True, "discrete_cursor_exact": True}


def _preflight(config: Mapping[str, Any], source_dir: Path, source_review_dir: Path,
               source_verification_path: Path, output_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "fixed-frame transfer preflight requires AutoDL CUDA")
    _require(not output_dir.exists(), f"preflight directory already exists: {output_dir}", FileExistsError)
    source_verification = _json(source_verification_path.resolve())
    _require(source_verification.get("task_id") == TASK_ID
             and source_verification.get("status") == "SOURCE_REPLAY_VERIFIED"
             and source_verification.get("terminal_rows_verified") == 2400
             and source_verification.get("v2_holdout_rows_verified") == 2000
             and source_verification.get("cuda_command_bitwise_equal_to_original") is True
             and source_verification.get("model_binding_verified") is True
             and source_verification.get("response_projector_binding_verified") is True
             and source_verification.get("training_witness_scope_correction") is True
             and float(source_verification.get("cpu_checkpoint_replay_max_abs_difference", float("inf")))
             <= SOURCE_CPU_COMMAND_TOLERANCE,
             "Windows source review did not verify the full CUDA replay population")
    source_review = _json(source_review_dir.resolve() / "source_review_summary.json")
    _require(source_review.get("status") == "SOURCE_REPLAY_PRODUCED",
             "CUDA source replay evidence is missing")
    fixed._set_numerics()
    device = torch.device("cuda:0")
    source = _load_source(source_dir, config)
    inputs = source["inputs"]
    _require(source_verification.get("source_run_id") == source["checkpoint"]["run_id"],
             "Windows source verification is bound to another fixed-frame terminal checkpoint")
    _require(abs(fixed._camera_area(inputs, device) - 6.25e-12) <= 1.0e-22,
             "camera pixel area differs from the frozen 6.25e-12 m^2")
    analytic = _analytic_solver_controls(device)
    population = source["population"]
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    matrix = inputs["matrix"].to(device=device, dtype=torch.float64)
    lipschitz = float(reach._projected_spectral_norm(response))
    _require(math.isfinite(lipschitz) and lipschitz > 0.0,
             "frozen response matrix has an invalid spectral constant")
    target_probe = population["development"]["coefficients"][0].to(device=device, dtype=torch.complex128)
    output_dir.mkdir(parents=True)
    recovery = [_solver_recovery_case(kind, response, target_probe, 0.0, lipschitz,
                                      output_dir, uuid.uuid4().hex, source_dir,
                                      source["checkpoint"]["run_id"])
                for kind in ("fixed", "free")]

    models = {grid: fixed._model_for_grid(inputs, grid, device) for grid in GRIDS}
    optical_before = {str(grid): reach._snapshot_state(model)
                      for grid, model in models.items()}
    train = population["train"]
    positive_rows = []
    positive_commands = []
    positive_targets = []
    for label in range(CLASS_COUNT):
        index = 20 * label
        witness_path = source_dir / "qualification_evidence" / "alpha_1.000" / "witness_rows" / f"row_{index:04d}.pt"
        _require(witness_path.is_file(), f"fixed-frame training positive control is missing: {witness_path}",
                 FileNotFoundError)
        witness = torch.load(witness_path, map_location="cpu", weights_only=False)
        target = train["coefficients"][index:index + 1].to(device=device, dtype=torch.complex128)
        command = witness["amplitude_command"].reshape(1, 1, 200, 200).to(device=device, dtype=torch.float32)
        _require(int(witness["index"]) == index and int(witness["label"]) == label
                 and int(witness["proposal_id"]) == int(train["proposal_ids"][index])
                 and torch.equal(witness["target_coefficients"], train["coefficients"][index]),
                 f"training positive control source identity differs at class {label}")
        metrics, outputs = fixed._measure_command(inputs, command, target, models)
        _require(all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS),
                 f"full-optics training positive control failed for class {label}")
        matrix_pred = exposure._forward_matrix(command, response).reshape(-1)
        response_error = float(torch.linalg.vector_norm(matrix_pred - outputs[2048]["coefficients"].reshape(-1))
                               / torch.linalg.vector_norm(outputs[2048]["coefficients"].reshape(-1)).clamp_min(1e-30))
        _require(response_error <= 1.0e-6,
                 f"fixed response matrix does not predict class-{label} positive control")
        positive_rows.append({"index": index, "label": label, **metrics,
                              "matrix_response_relative_error": response_error})
        positive_commands.append(command.detach().cpu())
        positive_targets.append(train["coefficients"][index:index + 1].clone())

    command = positive_commands[0].to(device)
    target = positive_targets[0].to(device=device, dtype=torch.complex128)
    valid_metrics, valid_outputs = fixed._measure_command(inputs, command, target, models)
    wrong_target_metrics, _ = fixed._measure_command(
        inputs, command, positive_targets[-1].to(device=device, dtype=torch.complex128), models)
    wrong_encoding_metrics, _ = fixed._measure_command(inputs, command.square(), target, models)
    other_direct = fixed._measure_command(inputs, positive_commands[1].to(device),
                                          positive_targets[1].to(device=device, dtype=torch.complex128), models)[1]
    replaced_cmos_metrics = projected._error_row(
        {grid: {"coefficients": valid_outputs[grid]["coefficients"],
                "raw_cmos": valid_outputs[grid]["raw_cmos"]} for grid in GRIDS},
        target, {grid: other_direct[grid]["direct_raw_cmos"] for grid in GRIDS}, 0)
    _require(any(float(wrong_target_metrics[key]) > GATE_LIMIT for key in METRIC_KEYS)
             and any(float(wrong_encoding_metrics[key]) > GATE_LIMIT for key in METRIC_KEYS)
             and any(float(replaced_cmos_metrics[key]) > GATE_LIMIT for key in METRIC_KEYS),
             "production consumer accepted a wrong target, amplitude encoding, or replaced CMOS reference")

    # A real measured command is changed only along a null direction of the frozen linear response.
    flat = command.reshape(-1).double()
    free = torch.nonzero((flat > 0.1) & (flat < 0.9), as_tuple=False).reshape(-1)
    _require(free.numel() >= 201, "positive control lacks enough interior SLM pixels for the null-space control")
    chosen = free[:201]
    submatrix = matrix[:, chosen]
    _, singular, vh = torch.linalg.svd(submatrix, full_matrices=True)
    _require(float(singular[-1]) > 1.0e-12 and tuple(vh.shape) == (201, 201),
             "null-space negative control submatrix is rank deficient")
    null = vh[-1]
    _require(float(torch.linalg.vector_norm(submatrix @ null)) <= 1.0e-10,
             "constructed null-space direction is not in the frozen response nullspace")
    delta = torch.zeros_like(flat)
    delta[chosen] = null / null.abs().max() * 0.05
    changed = (flat + delta).reshape(1, 1, 200, 200).float()
    _require(float(changed.min()) >= 0.0 and float(changed.max()) <= 1.0
             and float((changed - command).abs().max()) > 1.0e-3,
             "null-space command replacement did not create a bounded distinct instruction")
    null_coeff = exposure._forward_matrix(delta.reshape(1, -1), response).reshape(-1)
    null_relative = float(torch.linalg.vector_norm(null_coeff)
                          / torch.linalg.vector_norm(outputs[2048]["coefficients"].reshape(-1)).clamp_min(1e-30))
    _require(null_relative <= 1.0e-10, "null-space command replacement changed the frozen complex coefficients")
    null_outputs = {grid: projected._forward_amplitude(models[grid], changed) for grid in GRIDS}
    null_output_coeff_error = max(float(torch.linalg.vector_norm(
        null_outputs[grid]["coefficients"] - outputs[grid]["coefficients"])
        / torch.linalg.vector_norm(outputs[grid]["coefficients"]).clamp_min(1e-30)) for grid in GRIDS)
    consumer_rejected = False
    try:
        _consume_source_bound_command(changed.detach().cpu(), command.detach().cpu(),
                                      require_bitwise=True)
    except RuntimeError:
        consumer_rejected = True
    # This negative control tests exact command/source binding. The matrix-null
    # perturbation's full-optics coefficient change is recorded separately; it
    # measures response-model agreement and must not mask a consumer rejection.
    _require(consumer_rejected,
             "source consumer accepted a production null-space command replacement")
    _require(all(all(torch.equal(models[grid].state_dict()[key].detach().cpu(), value)
                     for key, value in optical_before[str(grid)].items()) for grid in GRIDS),
             "frozen optical state changed during preflight")

    identity_id = uuid.uuid4().hex
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=identity_id,
        trajectory_id=f"{identity_id}:fixed-frame-transfer-diagnosis",
        semantic_authority_id=AUTHORITY,
        data_role="consumed_development_phase20; source_replay_train_dev_holdout",
        config_identity=dict(config),
        model_identity={"terminal_mapper": str(source["terminal_path"]),
                        "v2_mapper": str(source["source_v2_path"]),
                        "optics": "frozen_C45k", "response": str(source["projector_path"])},
        optimizer_identity={"kind": "none", "updates": 0},
        sampler_identity={"development_indices": _target_indices(), "resampled": False,
                          "phase_solvers": "fixed_zero,fixed_terminal,free_8_start"},
        execution_source_semantics_id=json.dumps({"task_id": TASK_ID, "config": config},
                                                 sort_keys=True, ensure_ascii=False))
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    _write_json(output_dir / "preflight_identity.json", {
        "task_id": TASK_ID, "run_id": identity_id, "run_identity": identity.to_dict()})
    resource = {"passed": True, "status": "PASS", "cuda_device": torch.cuda.get_device_name(device),
                "free_memory_bytes_observation": int(torch.cuda.mem_get_info(device)[0]),
                "data_disk_free_bytes_observation": shutil.disk_usage(output_dir).free,
                "fixed_resource_quota": False}
    correctness = {"passed": True, "status": "PASS", "source_review_verified": True,
                   "terminal_rows": 2400, "v2_holdout_rows": 2000,
                   "full_optics_positive_controls": len(positive_rows),
                   "wrong_target_encoding_and_cmos_replacements_rejected": True,
                   "nullspace_instruction_replacement_rejected_by_diagnostic_consumer": True,
                   "analytic_solver_and_complex_gradient_controls": analytic,
                   "camera_pixel_area_m2": 6.25e-12, "frozen_optical_state_unchanged": True}
    implementation = {"passed": True, "status": "PASS",
                      "selected_engine": "frozen_response_matrix_offline_solver",
                      "fixed_solver_restoration": recovery[0], "free_solver_restoration": recovery[1],
                      "spectral_constant_computed_once": lipschitz,
                      "all_scientific_updates": 0}
    receipt = control.make_ready_receipt(
        correctness=correctness, resource=resource, implementation=implementation,
        selected_engine="frozen_response_matrix_offline_solver", resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    _write_json(output_dir / "preflight_receipt.json", receipt.to_dict())
    summary = {"schema": f"{TASK_ID}.preflight.v1", "task_id": TASK_ID,
               "run_id": identity_id, "status": "READY", "selected_engine": "frozen_response_matrix_offline_solver",
               "source_run_id": source["checkpoint"]["run_id"],
               "positive_controls": positive_rows, "analytic_controls": analytic,
               "solver_recovery": recovery,
               "negative_controls": {"wrong_target_rejected": True,
                                     "wrong_amplitude_encoding_rejected": True,
                                     "replaced_cmos_reference_rejected": True,
                                     "nullspace_command_rejected": True,
                                     "nullspace_command_rejected_by_diagnostic_consumer": True,
                                     "nullspace_matrix_response_relative_error": null_relative,
                                     "nullspace_full_optics_coefficient_relative_change": null_output_coeff_error},
               "spectral_lipschitz": lipschitz, "source_verification": source_verification,
               "scientific_updates": 0, "mapper_updates": 0, "optical_updates": 0,
               "evaluator_updates": 0, "rng_use": "none"}
    _write_json(output_dir / "preflight_summary.json", summary)
    _save(output_dir / "qualification_population.pt", {
        "task_id": TASK_ID, "population_role": "consumed_development_phase20",
        "indices": torch.tensor(_target_indices(), dtype=torch.long),
        "labels": population["development"]["labels"][_target_indices()].clone(),
        "proposal_ids": population["development"]["proposal_ids"][_target_indices()].clone(),
        "coefficients": population["development"]["coefficients"][_target_indices()].clone(),
    })
    _write_manifest(output_dir)
    return summary


def _load_ready(preflight_dir: Path) -> tuple[dict[str, Any], PreflightReceipt, ResearchRunIdentity]:
    preflight_dir = preflight_dir.resolve()
    registered_root = RUNS_DIR.resolve()
    backup_root = (RUNS_DIR / "autodl_backup_attempt_01").resolve()
    transferred_current_preflight = (RUNS_DIR / "autodl_backup_preflight_03").resolve()
    _require((preflight_dir.parent == backup_root and preflight_dir.name.startswith("preflight_"))
             or (preflight_dir.parent == registered_root and preflight_dir.name.startswith("preflight_"))
             or preflight_dir == transferred_current_preflight,
             "diagnostic execution requires its registered preflight directory", ValueError)
    _verify_manifest_files(preflight_dir)
    summary = _json(preflight_dir / "preflight_summary.json")
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_receipt.json"))
    identity_payload = _json(preflight_dir / "preflight_identity.json")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    _require(summary.get("task_id") == TASK_ID and summary.get("status") == "READY"
             and identity.task_id == TASK_ID and identity.run_id == summary.get("run_id")
             and receipt.status == "READY" and receipt.authorization_id == AUTHORITY
             and receipt.selected_engine == "frozen_response_matrix_offline_solver"
             and all(receipt.checks.get(key, {}).get("passed") is True for key in
                     (CORRECTNESS_INVARIANTS, RESOURCE_ENVELOPE, IMPLEMENTATION_SELECTION)),
             "this diagnostic requires its own executed and consumed READY receipt")
    _require(summary.get("scientific_updates") == 0 and summary.get("mapper_updates") == 0
             and summary.get("optical_updates") == 0 and summary.get("evaluator_updates") == 0
             and summary.get("negative_controls", {}).get("wrong_target_rejected") is True
             and summary.get("negative_controls", {}).get("wrong_amplitude_encoding_rejected") is True
             and summary.get("negative_controls", {}).get("replaced_cmos_reference_rejected") is True
             and summary.get("negative_controls", {}).get("nullspace_command_rejected") is True
             and summary.get("negative_controls", {}).get(
                 "nullspace_command_rejected_by_diagnostic_consumer") is True
             and summary.get("source_verification", {}).get("status") == "SOURCE_REPLAY_VERIFIED",
             "READY receipt is missing the required source, zero-update, or production negative controls")
    return summary, receipt, identity


def _save_solver_checkpoint(attempt_dir: Path, payload: Mapping[str, Any]) -> Path:
    current = attempt_dir / "checkpoints" / "solver_checkpoint.pt"
    previous = attempt_dir / "checkpoints" / "solver_checkpoint_previous.pt"
    current.parent.mkdir(parents=True, exist_ok=True)
    if current.is_file():
        if previous.exists():
            previous.unlink()
        os.replace(current, previous)
    _save(current, copy.deepcopy(dict(payload)))
    loaded = torch.load(current, map_location="cpu", weights_only=False)
    _require(loaded.get("task_id") == TASK_ID
             and int(loaded["next_target_index"]) == int(payload["next_target_index"])
             and int(loaded["solver_iterations_total"]) == int(payload["solver_iterations_total"]),
             "solver checkpoint did not survive disk readback")
    return current


def _load_row_base(source: Mapping[str, Any], source_review_dir: Path, index: int,
                   terminal: torch.nn.Module, v2: torch.nn.Module,
                   device: torch.device, *, strict_replay: bool = True) -> dict[str, Any]:
    population = source["population"]["development"]
    population_index = _target_indices()[index]
    target_cpu = population["coefficients"][population_index:population_index + 1].to(torch.complex128)
    target = target_cpu.to(device=device, dtype=torch.complex128)
    label = int(population["labels"][population_index])
    proposal_id = int(population["proposal_ids"][population_index])
    features = projected._normalized_features(target_cpu, source["inputs"]["scale"], device)
    terminal_command, terminal_detail = projected._map_amplitude(
        terminal, features, target, source["inputs"]["matrix"].to(device),
        source["inputs"]["inverse"].to(device), source["inputs"]["scale"].to(device),
        include_preclip=True)
    saved_review = torch.load(source_review_dir / "terminal_command_replay.pt",
                              map_location="cpu", weights_only=False)
    replayed = saved_review["roles"]["development"]["commands"][population_index:population_index + 1]
    terminal_cpu = terminal_command.detach().cpu()
    if strict_replay:
        _require(torch.equal(terminal_cpu, replayed),
                 f"terminal mapper no longer reproduces source-reviewed command at development row {index}")
    else:
        _require(float((terminal_cpu - replayed).abs().max()) <= SOURCE_CPU_COMMAND_TOLERANCE,
                 f"independent CPU terminal replay differs from reviewed CUDA command at row {index}")
    raw = terminal(features)
    phi0_command, phi0_detail = _fixed_phase_projection(
        raw, target, source["inputs"]["matrix"], source["inputs"]["inverse"],
        source["inputs"]["scale"], 0.0)
    v2_command, v2_detail = _mapper_command(v2, target_cpu, source["inputs"], device)
    current_phase = float(terminal_detail["phase"].reshape(-1)[0])
    return {
        "index": index, "source_population_index": population_index,
        "label": label, "proposal_id": proposal_id,
        "target_coefficients": target_cpu[0].clone(),
        "class_mean_amplitude": source["mean_amplitudes"][label].clone(),
        "commands": {"frozen_v2_single_forward": v2_command.cpu().clone(),
                     "fixed_frame_terminal": replayed.clone(),
                     "terminal_phi_zero_ablation": phi0_command.detach().cpu().clone()},
        "v2_preclip_amplitude": v2_detail["preclip_amplitude"].detach().cpu().double().reshape(PIXEL_COUNT),
        "v2_phase_rad": float(v2_detail["phase"].reshape(-1)[0]),
        "fixed_phase_rad": {"fista_fixed_physical_phase_zero": 0.0,
                            "fista_fixed_terminal_phase": current_phase},
        "terminal_phase_rad": current_phase,
        "terminal_raw_mlp": raw.detach().cpu().float().reshape(PIXEL_COUNT),
        "terminal_preclip_amplitude": terminal_detail["preclip_amplitude"].detach().cpu().double(),
        "phi_zero_preclip_amplitude": phi0_detail["preclip_amplitude"].detach().cpu().double(),
        "terminal_clipped_fraction": float(terminal_detail["clipped_fraction"].reshape(-1)[0]),
        "phi_zero_clipped_fraction": float(phi0_detail["clipped_fraction"].reshape(-1)[0]),
    }


def _solver_callback(attempt_dir: Path, base_row: Mapping[str, Any], completed_rows: int,
                     solver_index: int, active_results: Mapping[str, Any],
                     completed_iterations_before_active: int, run_id: str):
    condition = ("fista_fixed_physical_phase_zero", "fista_fixed_terminal_phase",
                 "bounded_free_public_phase_eight_start")[solver_index]

    def callback(state: dict[str, Any]) -> None:
        current_iterations = int(state["iteration"])
        multiplier = 8 if condition == "bounded_free_public_phase_eight_start" else 1
        total = completed_iterations_before_active + current_iterations * multiplier
        payload = {
            "schema": f"{TASK_ID}.solver_checkpoint.v1", "task_id": TASK_ID,
            "run_id": run_id, "next_target_index": int(base_row["index"]),
            "solver_index": solver_index, "active_condition": condition,
            "solver_state": copy.deepcopy(state), "active_row_base": copy.deepcopy(dict(base_row)),
            "active_results": copy.deepcopy(dict(active_results)),
            "completed_row_count": completed_rows,
            "solver_iterations_total": total,
            "saved_at_iteration": current_iterations,
            "rng_use": "none", "scientific_updates": 0,
        }
        checkpoint = _save_solver_checkpoint(attempt_dir, payload)
        _write_json(attempt_dir / "progress.json", {
            "task_id": TASK_ID, "phase": "SOLVING", "run_id": run_id,
            "completed_target_count": completed_rows,
            "active_target_index": int(base_row["index"]), "active_condition": condition,
            "condition_iterations": current_iterations,
            "total_per_start_iterations": total,
            "checkpoint": str(checkpoint), "updated_at_unix": time.time(),
        })
    return callback


def _solver_state_loss(kind: str, response: torch.Tensor, target: torch.Tensor,
                       state: Mapping[str, Any]) -> float:
    if kind == "fixed":
        desired = target * torch.exp(torch.tensor(
            1j * float(state["phase_rad"]), dtype=torch.complex128, device=response.device))
        return float(exposure._dual_box_error_bounds(
            response, desired,
            state["best_amplitude"].to(device=response.device, dtype=torch.float64)
        )["relative_upper_bound"])
    return float(state["best_errors"].min())


def _initial_phase_specific_solver_state(response: torch.Tensor, target: torch.Tensor,
                                         seed_amplitude: torch.Tensor,
                                         phase: float) -> dict[str, Any]:
    state = fixed._initial_fixed_phase_state(response, target, seed_amplitude)
    desired = target * torch.exp(torch.tensor(1j * float(phase), dtype=torch.complex128,
                                             device=response.device))
    bounds = exposure._dual_box_error_bounds(response, desired, state["best_amplitude"])
    state["phase_rad"] = float(phase)
    state["best_error"] = float(bounds["relative_upper_bound"])
    return state


def _solve_target(base_row: Mapping[str, Any], active: Mapping[str, Any] | None,
                  solver_index: int, attempt_dir: Path, completed_rows: int,
                  run_id: str, completed_iterations_before_active: int, response: torch.Tensor,
                  lipschitz: float, device: torch.device) -> dict[str, Any]:
    condition = ("fista_fixed_physical_phase_zero", "fista_fixed_terminal_phase",
                 "bounded_free_public_phase_eight_start")[solver_index]
    target = base_row["target_coefficients"].to(device=device, dtype=torch.complex128)
    callback = _solver_callback(attempt_dir, base_row, completed_rows, solver_index,
                                active.get("active_results", {}) if active else {},
                                completed_iterations_before_active, run_id)
    state = active.get("solver_state") if active else None
    if solver_index < 2:
        phase = float(base_row["fixed_phase_rad"][condition])
        if state is None:
            state = _initial_phase_specific_solver_state(
                response, target,
                base_row["class_mean_amplitude"].to(device=device, dtype=torch.float64), phase)
        result = exposure._fista_fixed_phase_solve(
            response, target, phase, max_steps=5000, lipschitz=lipschitz,
            initial_state=state, checkpoint=callback, checkpoint_interval=500)
        amplitude = result["amplitude"].detach().cpu().to(torch.float64)
        return {"condition": condition, "phase_rad": phase,
                "iterations": int(result["iterations"]), "per_start_iterations": [int(result["iterations"])],
                "relative_error": float(result["relative_error"]),
                "relative_upper_bound": float(result["relative_upper_bound"]),
                "relative_dual_lower_bound": float(result["relative_dual_lower_bound"]),
                "relative_gap": float(result["relative_gap"]),
                "amplitude_float64": amplitude,
                "command": amplitude.to(torch.float32).reshape(1, 1, 200, 200)}
    result = reach._bounded_phase_solve(
        response, target, starts=8, max_steps=5000, stop_error=1.0e-6,
        lipschitz=lipschitz, initial_state=state,
        checkpoint=callback, checkpoint_interval=500)
    amplitude = result["amplitude"].detach().cpu().to(torch.float64)
    iterations = int(result["iterations"])
    return {"condition": condition, "phase_rad": float(result["phase_rad"]),
            "iterations": iterations, "per_start_iterations": [iterations] * 8,
            "total_start_iterations": iterations * 8,
            "relative_error": float(result["relative_error"]),
            "best_errors_per_start": result["best_errors"].detach().cpu(),
            "amplitude_float64": amplitude,
            "command": amplitude.to(torch.float32).reshape(1, 1, 200, 200)}


def _solver_checkpoint_payload(run_id: str, next_target_index: int, solver_index: int,
                               solver_iterations_total: int, completed_row_count: int,
                               active_row_base: Mapping[str, Any] | None,
                               active_results: Mapping[str, Any],
                               solver_state: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "schema": f"{TASK_ID}.solver_checkpoint.v1", "task_id": TASK_ID,
        "run_id": run_id, "next_target_index": int(next_target_index),
        "solver_index": int(solver_index), "active_condition": (
            CONDITIONS[3 + solver_index] if solver_index < 3 else None),
        "solver_state": None if solver_state is None else copy.deepcopy(dict(solver_state)),
        "active_row_base": None if active_row_base is None else copy.deepcopy(dict(active_row_base)),
        "active_results": copy.deepcopy(dict(active_results)),
        "completed_row_count": int(completed_row_count),
        "solver_iterations_total": int(solver_iterations_total),
        "saved_at_iteration": 0 if solver_state is None else int(solver_state["iteration"]),
        "rng_use": "none", "scientific_updates": 0,
    }


def _load_reviewed_source(config: Mapping[str, Any], source_dir: Path,
                          review_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_source(source_dir, config)
    review = _json(review_dir / "source_review_summary.json")
    binding = torch.load(review_dir / "model_binding.pt", map_location="cpu", weights_only=False)
    replay = torch.load(review_dir / "terminal_command_replay.pt", map_location="cpu", weights_only=False)
    _require(review.get("task_id") == TASK_ID and review.get("status") == "SOURCE_REPLAY_PRODUCED"
             and review.get("source_run_id") == source["checkpoint"]["run_id"]
             and review.get("terminal_rows_replayed") == 2400
             and review.get("v2_holdout_rows_replayed") == 2000
             and binding.get("source_run_id") == source["checkpoint"]["run_id"]
             and fixed._compare_states(binding["terminal_loaded_state"],
                                       source["checkpoint"]["mapper_state"]) == 0.0
             and torch.equal(binding["response_matrix"], source["inputs"]["response_matrix"])
             and torch.equal(binding["right_inverse"], source["inputs"]["inverse"])
             and torch.equal(binding["scale"], source["inputs"]["scale"])
             and replay.get("source_run_id") == source["checkpoint"]["run_id"],
             "source review evidence is not bound to the explicit terminal checkpoint and fixed projection")
    return source, replay


def _write_solver_cursor(attempt_dir: Path, run_id: str, target_index: int,
                         solver_index: int, total_iterations: int, completed_rows: int,
                         row_base: Mapping[str, Any] | None,
                         results: Mapping[str, Any], state: Mapping[str, Any] | None) -> Path:
    payload = _solver_checkpoint_payload(
        run_id, target_index, solver_index, total_iterations, completed_rows,
        row_base, results, state)
    checkpoint = _save_solver_checkpoint(attempt_dir, payload)
    _write_json(attempt_dir / "progress.json", {
        "task_id": TASK_ID, "run_id": run_id, "phase": "SOLVING",
        "completed_target_count": completed_rows, "active_target_index": target_index,
        "active_condition": payload["active_condition"],
        "condition_iterations": payload["saved_at_iteration"],
        "solver_iterations_total": total_iterations, "checkpoint": str(checkpoint),
        "scientific_updates": 0, "updated_at_unix": time.time()})
    return checkpoint


def _load_completed_solver_rows(attempt_dir: Path, source: Mapping[str, Any],
                                count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        path = attempt_dir / "solver_rows" / f"row_{index:04d}.pt"
        _require(path.is_file(), f"checkpoint says solver row {index} is complete but file is missing: {path}",
                 FileNotFoundError)
        row = torch.load(path, map_location="cpu", weights_only=False)
        pop = source["population"]["development"]
        population_index = _target_indices()[index]
        _require(row.get("schema") == f"{TASK_ID}.solver_row.v1"
                 and row.get("task_id") == TASK_ID and row.get("index") == index
                 and row.get("source_population_index") == population_index
                 and row.get("label") == int(pop["labels"][population_index])
                 and row.get("proposal_id") == int(pop["proposal_ids"][population_index])
                 and torch.equal(row["target_coefficients"], pop["coefficients"][population_index])
                 and set(row.get("solver_results", {})) == set(CONDITIONS[3:]),
                 f"saved solver row identity is invalid at target {index}")
        rows.append(row)
    return rows


def _execute(config: Mapping[str, Any], source_dir: Path, review_dir: Path,
             preflight_dir: Path, attempt_dir: Path,
             resume_checkpoint: Path | None = None) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "fixed-frame phase diagnosis requires AutoDL CUDA")
    _require(attempt_dir.resolve().is_relative_to(_resolve(config["output"]["runs_root"]).resolve()),
             "diagnostic attempt must remain under the registered runs root", ValueError)
    ready_summary, _, identity = _load_ready(preflight_dir)
    source, _ = _load_reviewed_source(config, source_dir, review_dir)
    _require(ready_summary.get("source_run_id") == source["checkpoint"]["run_id"],
             "READY receipt and explicit source checkpoint have different run identities")
    fixed._set_numerics()
    device = torch.device("cuda:0")
    inputs = source["inputs"]
    terminal = fixed._new_mapper(inputs, device, source["checkpoint"]["mapper_state"])
    terminal.eval()
    v2 = fixed._new_mapper(inputs, device, inputs["source_mapper"]["mapper_state"])
    v2.eval()
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    lipschitz = float(ready_summary["spectral_lipschitz"])
    _require(math.isfinite(lipschitz) and lipschitz > 0.0,
             "READY receipt has no valid cached response spectral constant")

    if resume_checkpoint is None:
        _require(not attempt_dir.exists(), f"diagnostic attempt already exists: {attempt_dir}", FileExistsError)
        attempt_dir.mkdir(parents=True)
        run_id = identity.run_id
        _write_json(attempt_dir / "run_identity.json", {
            "schema": f"{TASK_ID}.run.v1", "task_id": TASK_ID, "run_id": run_id,
            "preflight_run_id": identity.run_id, "status": "ACTIVE", "phase": "SOLVING",
            "population_role": "consumed_development_phase20", "target_count": 20,
            "scientific_updates": 0, "mapper_updates": 0, "optical_updates": 0,
            "evaluator_updates": 0, "solver_iterations_total": 0})
        _write_json(attempt_dir / "run_config_snapshot.json", {"task_id": TASK_ID,
                    "config": dict(config), "source_dir": str(source_dir.resolve()),
                    "source_review_dir": str(review_dir.resolve()),
                    "preflight_dir": str(preflight_dir.resolve())})
        start_target, solver_index, completed_rows = 0, 0, 0
        total_iterations = 0
        active_row, active_results, active_state = None, {}, None
    else:
        attempt_dir = attempt_dir.resolve()
        checkpoint_path = resume_checkpoint.resolve()
        _require(attempt_dir.exists() and checkpoint_path.is_file()
                 and checkpoint_path.is_relative_to((attempt_dir / "checkpoints").resolve()),
                 "resume requires an explicit solver checkpoint inside this diagnostic attempt", ValueError)
        run_identity = _json(attempt_dir / "run_identity.json")
        _require(run_identity.get("task_id") == TASK_ID and run_identity.get("status") == "ACTIVE"
                 and run_identity.get("phase") == "SOLVING",
                 "closed or foreign diagnostic attempt cannot be resumed", ValueError)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_id = str(run_identity["run_id"])
        _require(checkpoint.get("schema") == f"{TASK_ID}.solver_checkpoint.v1"
                 and checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == run_id,
                 "explicit recovery checkpoint belongs to another attempt")
        start_target = int(checkpoint["next_target_index"])
        solver_index = int(checkpoint["solver_index"])
        completed_rows = int(checkpoint["completed_row_count"])
        total_iterations = int(checkpoint["solver_iterations_total"])
        active_row = checkpoint.get("active_row_base")
        active_results = dict(checkpoint.get("active_results", {}))
        active_state = checkpoint.get("solver_state")
        _require(0 <= start_target <= 20 and 0 <= solver_index <= 3
                 and completed_rows == start_target
                 and (start_target == 20 or active_row is not None
                      or (solver_index == 0 and not active_results and active_state is None))
                 and (active_state is None or int(active_state["iteration"]) > 0),
                 "explicit solver checkpoint cursor is inconsistent")
        _load_completed_solver_rows(attempt_dir, source, completed_rows)
        _write_json(attempt_dir / "progress.json", {"task_id": TASK_ID, "run_id": run_id,
                    "phase": "SOLVING", "resumed_from": str(checkpoint_path),
                    "completed_target_count": completed_rows, "active_target_index": start_target,
                    "active_solver_index": solver_index, "solver_iterations_total": total_iterations,
                    "scientific_updates": 0, "updated_at_unix": time.time()})

    solved_rows = _load_completed_solver_rows(attempt_dir, source, completed_rows)
    for index in range(start_target, 20):
        if active_row is not None and index == start_target:
            base_row = active_row
            expected_base = _load_row_base(source, review_dir, index, terminal, v2, device)
            _require(base_row.get("index") == index
                     and base_row.get("source_population_index") == expected_base["source_population_index"]
                     and base_row.get("label") == expected_base["label"]
                     and base_row.get("proposal_id") == expected_base["proposal_id"]
                     and torch.equal(base_row["target_coefficients"], expected_base["target_coefficients"])
                     and torch.equal(base_row["commands"]["fixed_frame_terminal"],
                                     expected_base["commands"]["fixed_frame_terminal"]),
                     f"saved active row no longer matches its fixed source identity at {index}")
        else:
            base_row = _load_row_base(source, review_dir, index, terminal, v2, device)
            solver_index, active_results, active_state = 0, {}, None
            _write_solver_cursor(attempt_dir, run_id, index, 0, total_iterations,
                                 completed_rows, base_row, {}, None)
        for current_solver in range(solver_index, 3):
            multiplier = 8 if current_solver == 2 else 1
            if current_solver == solver_index and active_state is not None:
                before_active = total_iterations - int(active_state["iteration"]) * multiplier
                _require(before_active >= 0, "saved active solver iteration exceeds total work")
            else:
                before_active = total_iterations
            solved = _solve_target(
                base_row, {"active_results": active_results, "solver_state": active_state},
                current_solver, attempt_dir, completed_rows, run_id, before_active,
                response, lipschitz, device)
            active_results[solved["condition"]] = solved
            total_iterations = before_active + int(solved["iterations"]) * multiplier
            active_state = None
            next_solver = current_solver + 1
            _write_solver_cursor(attempt_dir, run_id, index, next_solver, total_iterations,
                                 completed_rows, base_row, active_results, None)
        solver_row = {"schema": f"{TASK_ID}.solver_row.v1", "task_id": TASK_ID,
                      "run_id": run_id, "index": index,
                      "source_population_index": base_row["source_population_index"],
                      "label": base_row["label"],
                      "proposal_id": base_row["proposal_id"],
                      "target_coefficients": base_row["target_coefficients"].clone(),
                      "base_row": copy.deepcopy(base_row),
                      "solver_results": copy.deepcopy(active_results),
                      "fixed_phase_seed": "class_mean_amplitude",
                      "scientific_updates": 0, "mapper_updates": 0,
                      "optical_updates": 0, "evaluator_updates": 0}
        _save(attempt_dir / "solver_rows" / f"row_{index:04d}.pt", solver_row)
        solved_rows.append(solver_row)
        completed_rows = index + 1
        active_row, active_results, active_state = None, {}, None
        solver_index = 0
        _write_solver_cursor(attempt_dir, run_id, completed_rows, 0, total_iterations,
                             completed_rows, None, {}, None)
        _write_json(attempt_dir / "progress.json", {"task_id": TASK_ID, "run_id": run_id,
                    "phase": "SOLVING", "completed_target_count": completed_rows,
                    "active_target_index": completed_rows if completed_rows < 20 else None,
                    "solver_iterations_total": total_iterations,
                    "scientific_updates": 0, "updated_at_unix": time.time()})
        if completed_rows < 20:
            active_row = None

    _require(len(solved_rows) == 20 and all(set(row["solver_results"]) == set(CONDITIONS[3:])
                                             for row in solved_rows),
             "not all fixed targets completed all three solver branches")
    summary = {"schema": f"{TASK_ID}.solver_summary.v1", "task_id": TASK_ID,
               "run_id": run_id, "status": "SOLVERS_COMPLETE", "target_count": 20,
               "population_role": "consumed_development_phase20", "solver_iterations_total": total_iterations,
               "maximum_authorized_solver_iterations": 1_000_000,
               "fixed_zero_iterations": sum(int(row["solver_results"][CONDITIONS[3]]["iterations"])
                                             for row in solved_rows),
               "fixed_terminal_phase_iterations": sum(int(row["solver_results"][CONDITIONS[4]]["iterations"])
                                                      for row in solved_rows),
               "free_phase_iterations_across_eight_starts": sum(
                   int(row["solver_results"][CONDITIONS[5]]["total_start_iterations"])
                   for row in solved_rows),
               "fixed_phase_initial_amplitude": "saved_per_class_mean_amplitude",
               "free_phase_initial_amplitudes": "0.5_for_each_of_eight_starts",
               "scientific_updates": 0, "mapper_updates": 0, "optical_updates": 0,
               "evaluator_updates": 0, "rng_use": "none"}
    _require(total_iterations <= 1_000_000, "solver work exceeded the fixed aggregate budget")
    _write_json(attempt_dir / "solver_summary.json", summary)
    _write_json(attempt_dir / "run_identity.json", {
        **_json(attempt_dir / "run_identity.json"), "status": "ACTIVE",
        "phase": "SOLVERS_COMPLETE", "solver_iterations_total": total_iterations,
        "completed_target_count": 20})
    _write_json(attempt_dir / "progress.json", {"task_id": TASK_ID, "run_id": run_id,
                "phase": "SOLVERS_COMPLETE", "completed_target_count": 20,
                "solver_iterations_total": total_iterations, "updated_at_unix": time.time()})
    _write_manifest(attempt_dir)
    return summary


def _np_rel_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    _require(actual.shape == reference.shape and denominator > 0.0
             and np.isfinite(actual).all() and np.isfinite(reference).all(),
             "saved diagnostic metric arrays are invalid")
    return float(np.linalg.norm((actual - reference).reshape(-1)) / denominator)


def _np_phase_l2(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.complex128).reshape(-1)
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    _require(actual.shape == target.shape and actual.size > 0
             and np.isfinite(actual.real).all() and np.isfinite(actual.imag).all()
             and np.isfinite(target.real).all() and np.isfinite(target.imag).all(),
             "saved complex coefficient arrays are invalid")
    denominator = float(np.linalg.norm(target))
    _require(denominator > 0.0, "saved target coefficient has zero norm")
    phase = float(np.angle(np.vdot(target, actual)))
    return float(np.linalg.norm(actual - np.exp(1j * phase) * target) / denominator)


def _np_fixed_phase_box_bounds(response: np.ndarray, target: np.ndarray, phase: float,
                               amplitude: np.ndarray) -> dict[str, float]:
    response = np.asarray(response, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    desired = np.exp(1j * phase) * target
    predicted = response @ amplitude
    residual = desired - predicted
    target_norm = float(np.linalg.norm(target))
    residual_norm = float(np.linalg.norm(residual))
    _require(response.shape == (target.size, amplitude.size) and target_norm > 0.0
             and np.isfinite(response.real).all() and np.isfinite(response.imag).all()
             and np.isfinite(amplitude).all() and np.isfinite(residual_norm),
             "independent fixed phase bound input is invalid")
    upper = residual_norm / target_norm
    if residual_norm == 0.0:
        return {"relative_upper_bound": 0.0, "relative_dual_lower_bound": 0.0,
                "relative_gap": 0.0}
    direction = residual / residual_norm
    support = float(np.maximum((direction.conj() @ response).real, 0.0).sum())
    lower = max(0.0, (float(np.vdot(direction, desired).real) - support) / target_norm)
    _require(lower <= upper + 1.0e-10,
             "independent fixed phase lower bound exceeds its feasible upper bound")
    return {"relative_upper_bound": upper, "relative_dual_lower_bound": lower,
            "relative_gap": upper - lower}


def _np_metric_row(outputs: Mapping[int, Mapping[str, np.ndarray]], target: np.ndarray,
                   direct: Mapping[int, np.ndarray]) -> dict[str, float]:
    return {
        "coefficient_2048": _np_phase_l2(outputs[2048]["coefficients"], target),
        "coefficient_4096": _np_phase_l2(outputs[4096]["coefficients"], target),
        "native_cmos_2048": _np_rel_l2(outputs[2048]["raw_cmos"], direct[2048]),
        "native_cmos_4096": _np_rel_l2(outputs[4096]["raw_cmos"], direct[4096]),
        "cross_grid_coefficient": _np_phase_l2(outputs[2048]["coefficients"],
                                                outputs[4096]["coefficients"]),
        "cross_grid_native_cmos": _np_rel_l2(outputs[2048]["raw_cmos"],
                                             outputs[4096]["raw_cmos"]),
    }


def _np_residual_decomposition(response: np.ndarray, target: np.ndarray, phase: float,
                               preclip: np.ndarray, command: np.ndarray,
                               actual: np.ndarray) -> dict[str, Any]:
    response = np.asarray(response, dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    preclip = np.asarray(preclip, dtype=np.float64).reshape(-1)
    command = np.asarray(command, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.complex128).reshape(-1)
    clipped = np.clip(preclip, 0.0, 1.0)
    desired = np.exp(1j * phase) * target
    terms = {
        "unclipped_matrix_residual": response @ preclip - desired,
        "amplitude_clipping": response @ (clipped - preclip),
        "float32_command_rounding": response @ (command - clipped),
        "production_propagation_difference": actual - response @ command,
    }
    observed = actual - desired
    reconstructed = sum(terms.values(), np.zeros_like(observed))
    denom = max(float(np.linalg.norm(target)), 1.0e-30)
    reconstruction_error = float(np.linalg.norm(reconstructed - observed) / denom)
    norms = {name: float(np.vdot(value, value).real) for name, value in terms.items()}
    names = list(terms)
    cross = {f"{left}|{right}": float(2.0 * np.vdot(terms[left], terms[right]).real)
             for i, left in enumerate(names) for right in names[i + 1:]}
    return {"terms": terms, "term_squared_norms": norms, "pairwise_cross_terms": cross,
            "observed_residual_squared_norm": float(np.vdot(observed, observed).real),
            "reconstructed_squared_norm": sum(norms.values()) + sum(cross.values()),
            "reconstruction_relative_error": reconstruction_error,
            "amplitude_clipped_fraction": float(np.mean((preclip < 0.0) | (preclip > 1.0))),
            "float32_rounding_nonzero_fraction": float(np.mean(command != clipped))}


def _verify_numpy_decomposition(saved: Mapping[str, Any], expected: Mapping[str, Any],
                                row_index: int, condition: str, grid: int,
                                target_norm: float) -> float:
    _require(math.isfinite(target_norm) and target_norm > 0.0,
             "residual decomposition needs a finite nonzero target normalization")
    _require(set(saved.get("terms", {})) == set(expected["terms"])
             and set(saved.get("term_squared_norms", {})) == set(expected["term_squared_norms"])
             and set(saved.get("pairwise_cross_terms", {})) == set(expected["pairwise_cross_terms"]),
             f"residual decomposition fields are incomplete at row {row_index}, {condition}, {grid}")
    max_term_delta = 0.0
    for name, expected_term in expected["terms"].items():
        saved_term = saved["terms"][name].detach().cpu().numpy()
        delta = float(np.linalg.norm(saved_term - expected_term) / max(target_norm, 1.0e-30))
        max_term_delta = max(max_term_delta, delta)
        _require(delta <= 1.0e-8,
                 f"residual term differs from independent matrix recomputation at row {row_index}, "
                 f"{condition}, {grid}, {name}: {delta:.9g}")
    scalar_pairs = [("reconstruction_relative_error", "reconstruction_relative_error"),
                    ("observed_residual_squared_norm", "observed_residual_squared_norm"),
                    ("reconstructed_squared_norm", "reconstructed_squared_norm")]
    scalar_pairs += [(f"term_squared_norms.{key}", f"term_squared_norms.{key}")
                     for key in expected["term_squared_norms"]]
    scalar_pairs += [(f"pairwise_cross_terms.{key}", f"pairwise_cross_terms.{key}")
                     for key in expected["pairwise_cross_terms"]]
    for saved_key, expected_key in scalar_pairs:
        def scalar(value: Mapping[str, Any], key: str) -> float:
            first, dot, rest = key.partition(".")
            return float(value[first][rest]) if dot else float(value[first])
        actual_value = scalar(saved, saved_key)
        expected_value = scalar(expected, expected_key)
        tolerance = max(1.0e-10, abs(expected_value) * 1.0e-8)
        _require(math.isfinite(actual_value) and abs(actual_value - expected_value) <= tolerance,
                 f"residual decomposition scalar differs at row {row_index}, {condition}, "
                 f"{grid}, {saved_key}: {abs(actual_value - expected_value):.9g}")
    _require(abs(float(saved.get("amplitude_clipped_fraction", -1.0))
                 - expected["amplitude_clipped_fraction"]) <= 1.0e-15
             and abs(float(saved.get("float32_rounding_nonzero_fraction", -1.0))
                     - expected["float32_rounding_nonzero_fraction"]) <= 1.0e-15,
             f"residual clipping or rounding fraction differs at row {row_index}, {condition}, {grid}")
    return max_term_delta


def _residual_decomposition(response: torch.Tensor, target: torch.Tensor, phase: float,
                            preclip_amplitude: torch.Tensor, command: torch.Tensor,
                            actual_coefficients: torch.Tensor) -> dict[str, Any]:
    response = response.to(dtype=torch.complex128)
    target = target.reshape(-1).to(device=response.device, dtype=torch.complex128)
    preclip = preclip_amplitude.reshape(-1).to(device=response.device, dtype=torch.float64)
    clipped = preclip.clamp(0.0, 1.0)
    amplitude = command.reshape(-1).to(device=response.device, dtype=torch.float64)
    actual = actual_coefficients.reshape(-1).to(device=response.device, dtype=torch.complex128)
    target_phase = target * torch.exp(torch.tensor(1j * phase, device=response.device,
                                                     dtype=torch.complex128))
    terms = {
        "unclipped_matrix_residual": response @ preclip.to(torch.complex128) - target_phase,
        "amplitude_clipping": response @ (clipped - preclip).to(torch.complex128),
        "float32_command_rounding": response @ (amplitude - clipped).to(torch.complex128),
        "production_propagation_difference": actual - response @ amplitude.to(torch.complex128),
    }
    reconstructed = torch.stack(list(terms.values())).sum(dim=0)
    observed = actual - target_phase
    scale = torch.linalg.vector_norm(target).clamp_min(1.0e-30)
    reconstruction_error = float(torch.linalg.vector_norm(reconstructed - observed) / scale)
    _require(reconstruction_error <= 1.0e-10,
             f"residual decomposition does not reconstruct the production difference: {reconstruction_error:.9g}")
    norms = {name: float(torch.vdot(value, value).real) for name, value in terms.items()}
    cross = {}
    names = list(terms)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            cross[f"{left}|{right}"] = float(2.0 * torch.vdot(terms[left], terms[right]).real)
    total_squared = float(torch.vdot(observed, observed).real)
    component_sum = sum(norms.values()) + sum(cross.values())
    _require(abs(total_squared - component_sum) <= 1.0e-10 * max(1.0, total_squared),
             "residual squared-norm decomposition omitted cross terms")
    return {"phase_rad": phase,
            "terms": {name: value.detach().cpu().clone() for name, value in terms.items()},
            "term_squared_norms": norms, "pairwise_cross_terms": cross,
            "observed_residual_squared_norm": total_squared,
            "reconstructed_squared_norm": component_sum,
            "reconstruction_relative_error": reconstruction_error,
            "amplitude_clipped_fraction": float(((preclip < 0.0) | (preclip > 1.0)).double().mean()),
            "float32_rounding_nonzero_fraction": float((amplitude != clipped).double().mean())}


def _measurement_commands(solver_row: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    commands = dict(solver_row["base_row"]["commands"])
    for condition in CONDITIONS[3:]:
        commands[condition] = solver_row["solver_results"][condition]["command"].clone()
    _require(set(commands) == set(CONDITIONS)
             and all(value.dtype == torch.float32 and tuple(value.shape) == (1, 1, 200, 200)
                     and torch.isfinite(value).all().item()
                     and float(value.min()) >= 0.0 and float(value.max()) <= 1.0
                     for value in commands.values()),
             "a fixed diagnostic command is missing or violates the SLM amplitude range")
    return commands


def _saved_source_measurement_reference(population_index: int) -> dict[str, Any]:
    return {
        "task_id": fixed.TASK_ID,
        "population_role": "development",
        "source_population_index": population_index,
        "shard_path": f"measurements/development/shard_{population_index // 10:04d}.pt",
        "shard_row_index": population_index % 10,
        "terminal_output_field": "candidate_outputs",
        "direct_reference_field": "direct_raw_cmos",
    }


def _reused_source_measurements(source: Mapping[str, Any], measurement: Mapping[str, Any],
                                terminal_command: torch.Tensor | None = None,
                                validate_source_measurement: bool = False
                                ) -> tuple[dict[int, torch.Tensor],
                                           dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    population_index = int(measurement["source_population_index"])
    source_shard, local_index = _load_shard_row(source["source_dir"], "development", population_index)
    population = source["population"]["development"]
    expected_reference = _saved_source_measurement_reference(population_index)
    _require(measurement.get("source_measurement_reference") == expected_reference
             and local_index == expected_reference["shard_row_index"]
             and source_shard.get("indices") is not None
             and int(source_shard["indices"][local_index]) == population_index
             and int(source_shard["labels"][local_index]) == int(measurement["label"])
             and int(source_shard["proposal_ids"][local_index]) == int(measurement["proposal_id"])
             and torch.equal(source_shard["target_coefficients"][local_index],
                             measurement["target_coefficients"])
             and torch.equal(population["coefficients"][population_index],
                             measurement["target_coefficients"])
             and int(population["labels"][population_index]) == int(measurement["label"])
             and int(population["proposal_ids"][population_index]) == int(measurement["proposal_id"]),
             f"reused source measurement is not bound to fixed development target {population_index}")
    saved_terminal_command = source_shard["commands"][local_index]
    if terminal_command is not None:
        _require(terminal_command.dtype == torch.float32
                 and torch.equal(saved_terminal_command, terminal_command.reshape(1, 1, 200, 200)[0]),
                 f"reused terminal output command differs from its saved source row {population_index}")
    terminal_outputs: dict[int, dict[str, torch.Tensor]] = {}
    direct_cmos: dict[int, torch.Tensor] = {}
    for grid in GRIDS:
        coefficients = source_shard["candidate_outputs"][grid]["coefficients"][local_index]
        raw_cmos = source_shard["candidate_outputs"][grid]["raw_cmos"][local_index]
        direct = source_shard["direct_raw_cmos"][grid][local_index]
        _require(coefficients.dtype == torch.complex128 and coefficients.shape == (PORT_COUNT,)
                 and raw_cmos.dtype == torch.float64 and tuple(raw_cmos.shape) == (1, 200, 200)
                 and direct.dtype == torch.float64 and tuple(direct.shape) == (1, 200, 200)
                 and torch.isfinite(coefficients).all().item()
                 and torch.isfinite(raw_cmos).all().item()
                 and torch.isfinite(direct).all().item(),
                 f"reused source optics row has invalid values at target {population_index}, grid {grid}")
        terminal_outputs[grid] = {
            "coefficients": coefficients.unsqueeze(0),
            "raw_cmos": raw_cmos.unsqueeze(0),
        }
        direct_cmos[grid] = direct.unsqueeze(0)
    if validate_source_measurement:
        response = source["inputs"]["response_matrix"].to(dtype=torch.complex128)
        command = saved_terminal_command.reshape(-1).to(dtype=torch.float64)
        predicted_coefficients = response @ command.to(torch.complex128)
        actual_coefficients = source_shard["candidate_outputs"][2048]["coefficients"][local_index]
        response_error = float(torch.linalg.vector_norm(actual_coefficients - predicted_coefficients)
                               / torch.linalg.vector_norm(actual_coefficients))
        _require(math.isfinite(response_error) and response_error <= 1.0e-6,
                 f"reused terminal source coefficient response is inconsistent at target {population_index}")
        source_row = source_shard["rows"][local_index]
        source_metrics = _np_metric_row(
            {grid: {key: terminal_outputs[grid][key][0].numpy()
                    for key in ("coefficients", "raw_cmos")} for grid in GRIDS},
            measurement["target_coefficients"].numpy(),
            {grid: direct_cmos[grid][0].numpy() for grid in GRIDS})
        for key in METRIC_KEYS:
            _require(abs(float(source_row[key]) - source_metrics[key]) <= 1.0e-14,
                     f"reused source measurement metric is inconsistent at target {population_index}, {key}")
    return direct_cmos, terminal_outputs, expected_reference


def _measure(config: Mapping[str, Any], source_dir: Path, review_dir: Path,
             preflight_dir: Path, attempt_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "full two-grid measurement requires AutoDL CUDA")
    _, _, identity = _load_ready(preflight_dir)
    source, _ = _load_reviewed_source(config, source_dir, review_dir)
    run_identity = _json(attempt_dir / "run_identity.json")
    solver_summary = _json(attempt_dir / "solver_summary.json")
    _require(run_identity.get("task_id") == TASK_ID and run_identity.get("status") == "ACTIVE"
             and run_identity.get("phase") == "SOLVERS_COMPLETE"
             and run_identity.get("preflight_run_id") == identity.run_id
             and solver_summary.get("status") == "SOLVERS_COMPLETE"
             and solver_summary.get("run_id") == run_identity.get("run_id"),
             "measurement requires this READY attempt's complete fixed solver population")
    fixed._set_numerics()
    device = torch.device("cuda:0")
    inputs = source["inputs"]
    models = {grid: fixed._model_for_grid(inputs, grid, device) for grid in GRIDS}
    for model in models.values():
        model.eval()
    labels = source["population"]["development"]["labels"][_target_indices()].clone()
    area = fixed._camera_area(inputs, device)
    _require(abs(area - 6.25e-12) <= 1.0e-22, "frozen camera area changed before measurement")
    measurements_dir = attempt_dir / "measurements"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    for index in range(20):
        output_path = measurements_dir / f"row_{index:04d}.pt"
        solver_row = torch.load(attempt_dir / "solver_rows" / f"row_{index:04d}.pt",
                                map_location="cpu", weights_only=False)
        source_index = _target_indices()[index]
        population = source["population"]["development"]
        target_cpu = population["coefficients"][source_index:source_index + 1]
        _require(solver_row.get("index") == index
                 and solver_row.get("source_population_index") == source_index
                 and torch.equal(solver_row["target_coefficients"], target_cpu[0]),
                 f"solver output target identity differs at row {index}")
        source_binding_row = {
            "source_population_index": source_index,
            "label": int(population["labels"][source_index]),
            "proposal_id": int(population["proposal_ids"][source_index]),
            "target_coefficients": target_cpu[0],
        }
        commands = _measurement_commands(solver_row)
        direct, terminal_outputs, source_reference = _reused_source_measurements(
            source, {**source_binding_row, "source_measurement_reference":
                     _saved_source_measurement_reference(source_index)},
            commands["fixed_frame_terminal"], validate_source_measurement=True)
        if output_path.is_file():
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
            _require(existing.get("schema") == f"{TASK_ID}.measurement_row.v2"
                     and existing.get("index") == index
                     and set(existing.get("conditions", {})) == set(CONDITIONS)
                     and existing.get("source_population_index") == source_index
                     and existing.get("source_measurement_reference") == source_reference,
                     f"existing measurement row is incomplete and cannot be overwritten: {output_path}")
            continue
        target = target_cpu.to(device=device, dtype=torch.complex128)
        response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
        condition_results: dict[str, Any] = {}
        for condition in CONDITIONS:
            command = commands[condition].to(device=device, dtype=torch.float32)
            outputs = (terminal_outputs if condition == "fixed_frame_terminal" else
                       {grid: projected._forward_amplitude(models[grid], command)
                        for grid in GRIDS})
            metrics = projected._error_row(outputs, target, direct, 0)
            prescribed = None
            if condition in CONDITIONS[3:5]:
                phase = float(solver_row["solver_results"][condition]["phase_rad"])
                prescribed_target = target * torch.exp(torch.tensor(1j * phase, device=device,
                                                                      dtype=torch.complex128))
                prescribed = {
                    grid: float(torch.linalg.vector_norm(
                        outputs[grid]["coefficients"].reshape(-1) - prescribed_target.reshape(-1))
                        / torch.linalg.vector_norm(prescribed_target.reshape(-1)))
                    for grid in GRIDS}
            if condition == "fixed_frame_terminal":
                phase = float(solver_row["base_row"]["terminal_phase_rad"])
                preclip = solver_row["base_row"]["terminal_preclip_amplitude"]
            elif condition == "terminal_phi_zero_ablation":
                phase = 0.0
                preclip = solver_row["base_row"]["phi_zero_preclip_amplitude"]
            elif condition in CONDITIONS[3:]:
                phase = float(solver_row["solver_results"][condition]["phase_rad"])
                preclip = solver_row["solver_results"][condition]["amplitude_float64"]
            elif condition == "frozen_v2_single_forward":
                phase = float(solver_row["base_row"]["v2_phase_rad"])
                preclip = solver_row["base_row"]["v2_preclip_amplitude"]
            else:
                phase = float(solver_row["base_row"]["terminal_phase_rad"])
                preclip = solver_row["base_row"]["terminal_preclip_amplitude"]
            residual_decomposition = {
                grid: _residual_decomposition(response, target,
                                              phase,
                                              preclip, command,
                                              outputs[grid]["coefficients"])
                for grid in GRIDS}
            condition_result = {
                "metrics": dict(metrics),
                "prescribed_phase_complex_relative_error": prescribed,
                "phase_rad": phase,
                "residual_decomposition": residual_decomposition,
                "row_passed_six_metrics": all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS),
            }
            if condition == "fixed_frame_terminal":
                condition_result["outputs_source"] = "fixed_frame_semantic_development_candidate_outputs"
            else:
                condition_result["outputs"] = {
                    grid: {"coefficients": outputs[grid]["coefficients"].detach().cpu().clone(),
                           "raw_cmos": outputs[grid]["raw_cmos"].detach().cpu().clone()}
                    for grid in GRIDS}
            condition_results[condition] = condition_result
        _save(output_path, {
            "schema": f"{TASK_ID}.measurement_row.v2", "task_id": TASK_ID,
            "run_id": run_identity["run_id"], "index": index,
            "population_role": "consumed_development_phase20",
            "label": int(labels[index]),
            "proposal_id": int(source["population"]["development"]["proposal_ids"][_target_indices()[index]]),
            "source_population_index": _target_indices()[index],
            "target_coefficients": target_cpu[0].clone(),
            "source_measurement_reference": source_reference,
            "conditions": condition_results, "camera_pixel_area_m2": area,
            "camera_gain_w_inverse": float(inputs["camera_gain"]),
            "scientific_updates": 0})
        _write_json(attempt_dir / "progress.json", {
            "task_id": TASK_ID, "run_id": run_identity["run_id"],
            "phase": "MEASURING", "completed_target_measurements": index + 1,
            "target_count": 20, "grids": list(GRIDS), "conditions": list(CONDITIONS),
            "scientific_updates": 0, "updated_at_unix": time.time()})

    rows = [torch.load(measurements_dir / f"row_{index:04d}.pt", map_location="cpu", weights_only=False)
            for index in range(20)]
    condition_summary: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_rows = []
        for row in rows:
            saved = row["conditions"][condition]
            prescribed = saved["prescribed_phase_complex_relative_error"]
            row_passed = saved["row_passed_six_metrics"] and (
                prescribed is None or all(float(value) <= GATE_LIMIT for value in prescribed.values()))
            condition_rows.append({"index": row["index"],
                                   "source_population_index": row["source_population_index"],
                                   "label": row["label"],
                                   "proposal_id": row["proposal_id"],
                                   **{key: float(saved["metrics"][key]) for key in METRIC_KEYS},
                                   "row_passed": row_passed,
                                   "all_six_metrics_passed": saved["row_passed_six_metrics"],
                                   "prescribed_phase_complex_relative_error": prescribed})
        condition_summary[condition] = {
            "denominator": 20,
            "six_metric_passed_rows": sum(item["all_six_metrics_passed"] for item in condition_rows),
            "condition_passed_rows": sum(item["row_passed"] for item in condition_rows),
            "all_20_six_metrics_passed": all(item["all_six_metrics_passed"] for item in condition_rows),
            "all_20_condition_rules_passed": all(item["row_passed"] for item in condition_rows),
            "summary": projected._summarize_rows(condition_rows),
            "rows": condition_rows,
        }
        summary_rows.extend({"condition": condition, **item} for item in condition_rows)

    readouts: dict[str, torch.Tensor] = {"direct_decoder_reference":
        pilot_readout(rows, None, area, float(inputs["camera_gain"]), source)}
    for condition in CONDITIONS:
        readouts[condition] = pilot_readout(rows, condition, area, float(inputs["camera_gain"]), source)
    fixed._draw_contact_sheet(attempt_dir / "phase20_contact_sheet.png", readouts, labels)
    result = {"schema": f"{TASK_ID}.measurement_summary.v1", "task_id": TASK_ID,
              "run_id": run_identity["run_id"], "status": "MEASUREMENT_COMPLETE",
              "population_role": "consumed_development_phase20", "denominator": 20,
              "condition_summary": condition_summary,
              "decision": {condition: condition_summary[condition]["all_20_condition_rules_passed"]
                           for condition in CONDITIONS},
              "definition": "six full-optics errors <=1e-4; fixed-phase solvers also satisfy their prescribed-phase complex error <=1e-4",
              "metric_source": "2048/4096 production propagation of V2 and three solver commands; fixed-frame terminal output reused from its bound full-optics row",
              "reference_source": "same target direct Decoder raw CMOS reused from its bound full-optics row",
              "source_terminal_and_reference_rows_reused": 20,
              "camera_pixel_area_m2": area,
              "camera_gain_w_inverse": float(inputs["camera_gain"]),
              "scientific_updates": 0, "mapper_updates": 0, "optical_updates": 0,
              "evaluator_updates": 0, "rng_use": "none"}
    _write_json(attempt_dir / "measurement_summary.json", result)
    with (attempt_dir / "conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_json(attempt_dir / "run_identity.json", {**run_identity,
                "phase": "MEASURED", "completed_measurements": 20,
                "solver_iterations_total": int(solver_summary["solver_iterations_total"])})
    _write_json(attempt_dir / "progress.json", {"task_id": TASK_ID, "run_id": run_identity["run_id"],
                "phase": "MEASUREMENT_COMPLETE", "completed_target_measurements": 20,
                "updated_at_unix": time.time()})
    _write_closeout_report(attempt_dir, solver_summary, result)
    _write_manifest(attempt_dir)
    return result


def pilot_readout(rows: Sequence[Mapping[str, Any]], condition: str | None,
                  area: float, gain: float, source: Mapping[str, Any]) -> torch.Tensor:
    values = []
    for row in rows:
        direct, terminal_outputs, _ = _reused_source_measurements(source, row)
        if condition is None:
            raw = direct[2048]
        elif condition == "fixed_frame_terminal":
            raw = terminal_outputs[2048]["raw_cmos"]
        else:
            raw = row["conditions"][condition]["outputs"][2048]["raw_cmos"]
        values.append(fixed.pilot._readout_for_evaluator(raw, area, gain).cpu())
    return torch.cat(values)


def _write_closeout_report(attempt_dir: Path, solver_summary: Mapping[str, Any],
                           measurement_summary: Mapping[str, Any]) -> None:
    lines = [
        "# G1固定物理相位迁移与20目标诊断",
        "",
        "## 结果与状态",
        "",
        f"任务 `{TASK_ID}`；人口为已消费的开发集20条；模型、先验、光学及评价器更新均为0。求解总迭代工作量为 {solver_summary['solver_iterations_total']}。",
        "",
        "## 科学认识",
        "",
        "六种条件使用同一固定人口和同一目标Decoder参考；V2及三种求解指令重新进行双网格生产传播，终点映射与Decoder参考逐行复用已复核来源中的完整光路输出。求解指令只作离线诊断，不作为推理回退或训练标签。",
        "",
        "## 目标缺口",
        "",
        "固定相位求解除六项公开相位指标外，还按指定相位核验复系数误差；早停阈值不取代完整光路验收。",
        "",
        "## 工程完成与验证",
        "",
        f"双网格完整光路结果：`{json.dumps(measurement_summary['decision'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 决策影响与下一步",
        "",
        "本次仅支持或不支持在这20条已消费开发目标上的固定结构/公共相位诊断；不证明未测目标、真实硬件或整体G1分布的可达性。没有自动启动新训练。",
        "",
    ]
    (attempt_dir / "closeout_report.md").write_text("\n".join(lines), encoding="utf-8")


def _verify_saved(config: Mapping[str, Any], source_dir: Path, review_dir: Path,
                  source_verification_path: Path, preflight_dir: Path,
                  attempt_dir: Path, verification_dir: Path) -> dict[str, Any]:
    _require(verification_dir.resolve().is_relative_to(_resolve(config["output"]["runs_root"]).resolve()),
             "independent verification output must remain under the registered runs root", ValueError)
    _require(not verification_dir.exists(), f"verification output already exists: {verification_dir}",
             FileExistsError)
    source_verification = _json(source_verification_path.resolve())
    _require(source_verification.get("status") == "SOURCE_REPLAY_VERIFIED"
             and source_verification.get("terminal_rows_verified") == 2400
             and source_verification.get("v2_holdout_rows_verified") == 2000,
             "independent source replay review is incomplete")
    fixed._set_numerics()
    _verify_source(config, source_dir, review_dir)
    preflight_dir = preflight_dir.resolve()
    attempt_dir = attempt_dir.resolve()
    cloud_verification_dir = _registered_path(config, "windows_source_review_cloud_backup_dir").resolve()
    _require(preflight_dir == _registered_path(config, "windows_preflight_backup_dir").resolve()
             and attempt_dir == _registered_path(config, "windows_backup_dir").resolve(),
             "Windows verification must consume the registered transfers of this preflight and attempt",
             ValueError)
    _verify_manifest_files(preflight_dir)
    _verify_manifest_files(attempt_dir)
    _verify_manifest_files(cloud_verification_dir)
    _verify_transfer_receipt(review_dir.resolve(), _manifest_transfer_paths(review_dir.resolve()))
    for transfer_dir in (preflight_dir, attempt_dir, cloud_verification_dir):
        _verify_transfer_receipt(transfer_dir, _manifest_transfer_paths(transfer_dir))
    ready, _, identity = _load_ready(preflight_dir)
    source, replay = _load_reviewed_source(config, source_dir, review_dir)
    run_identity = _json(attempt_dir / "run_identity.json")
    solver_summary = _json(attempt_dir / "solver_summary.json")
    measurement_summary = _json(attempt_dir / "measurement_summary.json")
    _require(run_identity.get("task_id") == TASK_ID and run_identity.get("phase") == "MEASURED"
             and run_identity.get("preflight_run_id") == identity.run_id
             and solver_summary.get("status") == "SOLVERS_COMPLETE"
             and measurement_summary.get("status") == "MEASUREMENT_COMPLETE"
             and solver_summary.get("run_id") == run_identity.get("run_id")
             and measurement_summary.get("run_id") == run_identity.get("run_id")
             and measurement_summary.get("scientific_updates") == 0,
             "saved diagnosis is not a complete zero-update run bound to the READY receipt")
    development = source["population"]["development"]
    cpu_device = torch.device("cpu")
    terminal_mapper = fixed._new_mapper(source["inputs"], cpu_device,
                                        source["checkpoint"]["mapper_state"])
    v2_mapper = fixed._new_mapper(source["inputs"], cpu_device,
                                  source["inputs"]["source_mapper"]["mapper_state"])
    terminal_mapper.eval()
    v2_mapper.eval()
    response_np = source["inputs"]["response_matrix"].cpu().numpy().astype(np.complex128, copy=False)
    indices = _target_indices()
    expected_labels = development["labels"][indices]
    expected_proposals = development["proposal_ids"][indices]
    expected_targets = development["coefficients"][indices]
    rows = []
    max_metric_delta = 0.0
    max_command_replay_delta = 0.0
    max_decomposition_term_delta = 0.0
    solver_iterations_total = 0
    fixed_zero_iterations = 0
    fixed_terminal_iterations = 0
    free_phase_iterations = 0
    per_condition_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for position, index in enumerate(indices):
        solver_row = torch.load(attempt_dir / "solver_rows" / f"row_{position:04d}.pt",
                                map_location="cpu", weights_only=False)
        measurement = torch.load(attempt_dir / "measurements" / f"row_{position:04d}.pt",
                                 map_location="cpu", weights_only=False)
        expected_base = _load_row_base(source, review_dir, position, terminal_mapper, v2_mapper,
                                       cpu_device, strict_replay=False)
        _require(solver_row.get("index") == position and measurement.get("index") == position
                 and measurement.get("source_population_index") == index
                 and solver_row.get("label") == int(expected_labels[position])
                 and measurement.get("label") == int(expected_labels[position])
                 and solver_row.get("proposal_id") == int(expected_proposals[position])
                 and measurement.get("proposal_id") == int(expected_proposals[position])
                 and torch.equal(solver_row["target_coefficients"], expected_targets[position])
                 and torch.equal(measurement["target_coefficients"], expected_targets[position])
                 and set(measurement.get("conditions", {})) == set(CONDITIONS),
                 f"measurement or solver row is not bound to fixed development target {index}")
        _require(solver_row.get("base_row", {}).get("source_population_index") == index
                 and solver_row.get("base_row", {}).get("label") == int(expected_labels[position])
                 and solver_row.get("base_row", {}).get("proposal_id") == int(expected_proposals[position])
                 and torch.equal(solver_row["base_row"]["target_coefficients"], expected_targets[position]),
                 f"saved solver base row is not bound to target {index}")
        _require(torch.equal(solver_row["base_row"]["class_mean_amplitude"],
                             source["mean_amplitudes"][int(expected_labels[position])]),
                 f"saved solver initialization differs from the frozen class mean at target {index}")
        target = expected_targets[position].numpy().astype(np.complex128, copy=False)
        direct_tensors, terminal_outputs, source_reference = _reused_source_measurements(
            source, measurement, solver_row["base_row"]["commands"]["fixed_frame_terminal"],
            validate_source_measurement=True)
        _require(measurement.get("schema") == f"{TASK_ID}.measurement_row.v2"
                 and measurement.get("source_measurement_reference") == source_reference,
                 f"saved measurement row does not name its verified source outputs at target {index}")
        direct = {grid: direct_tensors[grid].numpy() for grid in GRIDS}
        solver_row_conditions = solver_row.get("solver_results", {})
        _require(set(solver_row_conditions) == set(CONDITIONS[3:]),
                 f"solver branches are incomplete at target {index}")
        for condition in CONDITIONS:
            saved = measurement["conditions"][condition]
            _require("command" not in saved and "preclip_amplitude" not in saved,
                     f"measurement row duplicates solver command state at row {position}, {condition}")
            if condition in CONDITIONS[:3]:
                command = solver_row["base_row"]["commands"][condition]
                if condition == "fixed_frame_terminal":
                    preclip = solver_row["base_row"]["terminal_preclip_amplitude"]
                elif condition == "terminal_phi_zero_ablation":
                    preclip = solver_row["base_row"]["phi_zero_preclip_amplitude"]
                else:
                    preclip = solver_row["base_row"]["v2_preclip_amplitude"]
            else:
                command = solver_row_conditions[condition]["command"]
                preclip = solver_row_conditions[condition]["amplitude_float64"]
            _require(command.dtype == torch.float32 and tuple(command.shape) == (1, 1, 200, 200)
                     and torch.isfinite(command).all().item()
                     and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                     f"invalid saved float32 SLM command for row {position}, condition {condition}")
            _require(isinstance(preclip, torch.Tensor) and preclip.dtype == torch.float64
                     and preclip.numel() == PIXEL_COUNT and torch.isfinite(preclip).all().item(),
                     f"preclip amplitude is missing or invalid at row {position}, {condition}")
            phase = float(saved.get("phase_rad", float("nan")))
            _require(math.isfinite(phase),
                     f"saved residual phase is nonfinite at row {position}, {condition}")
            if condition in CONDITIONS[:3]:
                expected_command = expected_base["commands"][condition]
                command_delta = _consume_source_bound_command(
                    command, expected_command, require_bitwise=condition == "fixed_frame_terminal")
                max_command_replay_delta = max(max_command_replay_delta, command_delta)
                if condition == "fixed_frame_terminal":
                    expected_preclip = expected_base["terminal_preclip_amplitude"]
                    expected_phase = expected_base["terminal_phase_rad"]
                elif condition == "terminal_phi_zero_ablation":
                    expected_preclip = expected_base["phi_zero_preclip_amplitude"]
                    expected_phase = 0.0
                else:
                    expected_preclip = expected_base["v2_preclip_amplitude"]
                    expected_phase = expected_base["v2_phase_rad"]
                preclip_delta = float((preclip.reshape(-1) - expected_preclip.reshape(-1)).abs().max())
                _require(preclip_delta <= SOURCE_CPU_COMMAND_TOLERANCE
                         and abs(math.remainder(phase - expected_phase, 2.0 * math.pi)) <= 1.0e-5,
                         f"saved forward preclip/phase is not bound to its checkpoint at row {index}, "
                         f"{condition}")
            else:
                solver = solver_row_conditions[condition]
                amplitude = solver.get("amplitude_float64")
                solver_command = solver.get("command")
                _require(isinstance(amplitude, torch.Tensor) and amplitude.dtype == torch.float64
                         and amplitude.numel() == PIXEL_COUNT and torch.isfinite(amplitude).all().item()
                         and float(amplitude.min()) >= 0.0 and float(amplitude.max()) <= 1.0
                         and isinstance(solver_command, torch.Tensor) and solver_command.dtype == torch.float32
                         and torch.equal(solver_command, amplitude.to(torch.float32).reshape(1, 1, 200, 200))
                         and torch.equal(command, solver_command)
                         and torch.equal(preclip.reshape(-1), amplitude.reshape(-1))
                         and abs(phase - float(solver["phase_rad"])) <= 1.0e-14,
                         f"saved solver command is not bound to the recorded amplitude/phase at row {index}, "
                         f"{condition}")
                if condition == CONDITIONS[3]:
                    expected_phase = 0.0
                elif condition == CONDITIONS[4]:
                    expected_phase = expected_base["terminal_phase_rad"]
                else:
                    expected_phase = phase
                _require(abs(math.remainder(phase - expected_phase, 2.0 * math.pi)) <= 1.0e-5,
                         f"solver phase is not bound to its prescribed branch at row {index}, {condition}")
                predicted_solver = response_np @ amplitude.numpy().reshape(-1)
                if condition in CONDITIONS[3:5]:
                    solver_bounds = _np_fixed_phase_box_bounds(
                        response_np, target, phase, amplitude.numpy())
                    for name, expected_value in solver_bounds.items():
                        recorded = float(solver.get(name, float("nan")))
                        _require(math.isfinite(recorded)
                                 and abs(recorded - expected_value)
                                 <= max(1.0e-10, abs(expected_value) * 1.0e-8),
                                 f"fixed phase solver {name} differs from independent NumPy recomputation "
                                 f"at row {index}, {condition}")
                    _require(abs(float(solver["relative_error"])
                                 - solver_bounds["relative_upper_bound"])
                             <= max(1.0e-10, solver_bounds["relative_upper_bound"] * 1.0e-8),
                             f"fixed phase solver error differs from its feasible upper bound at row {index}")
                else:
                    solver_error = _np_phase_l2(predicted_solver, target)
                    recorded_error = float(solver.get("relative_error", float("nan")))
                    per_start_errors = solver.get("best_errors_per_start")
                    _require(math.isfinite(recorded_error)
                             and abs(solver_error - recorded_error) <= max(1.0e-10, solver_error * 1.0e-8)
                             and isinstance(per_start_errors, torch.Tensor)
                             and per_start_errors.numel() == 8
                             and abs(float(per_start_errors.min()) - recorded_error) <= 1.0e-12,
                             f"free phase solver result differs from its saved best-start record at row {index}")
                max_steps = 5000
                per_start = solver.get("per_start_iterations", [])
                _require(int(solver.get("iterations", -1)) <= max_steps
                         and all(0 <= int(value) <= max_steps for value in per_start)
                         and (len(per_start) == 8 if condition == CONDITIONS[5] else len(per_start) == 1)
                         and (sum(int(value) for value in per_start)
                              == int(solver.get("total_start_iterations", -1))
                              if condition == CONDITIONS[5]
                              else int(per_start[0]) == int(solver["iterations"])),
                         f"saved solver work exceeded or misstates its fixed budget at row {index}, {condition}")
                if condition == CONDITIONS[3]:
                    fixed_zero_iterations += int(solver["iterations"])
                    solver_iterations_total += int(solver["iterations"])
                elif condition == CONDITIONS[4]:
                    fixed_terminal_iterations += int(solver["iterations"])
                    solver_iterations_total += int(solver["iterations"])
                else:
                    free_phase_iterations += int(solver["total_start_iterations"])
                    solver_iterations_total += int(solver["total_start_iterations"])
            if condition == "fixed_frame_terminal":
                _require(saved.get("outputs_source") == "fixed_frame_semantic_development_candidate_outputs"
                         and "outputs" not in saved,
                         f"terminal output is not represented by its fixed source row at target {index}")
                output_tensors = terminal_outputs
            else:
                _require("outputs_source" not in saved and set(saved.get("outputs", {})) == set(GRIDS),
                         f"newly measured output arrays are missing at target {index}, {condition}")
                output_tensors = saved["outputs"]
            outputs = {grid: {key: output_tensors[grid][key].numpy()
                              for key in ("coefficients", "raw_cmos")} for grid in GRIDS}
            metrics = _np_metric_row(outputs, target, direct)
            for key in METRIC_KEYS:
                saved_metric = float(saved["metrics"][key])
                delta = abs(metrics[key] - saved_metric)
                max_metric_delta = max(max_metric_delta, delta)
                _require(delta <= 1.0e-14,
                         f"NumPy independent metric mismatch at row {position}, {condition}, {key}: {delta:.9g}")
            passed_six = all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS)
            for grid in GRIDS:
                independent_decomposition = _np_residual_decomposition(
                    response_np, target, phase, preclip.numpy(), command.reshape(-1).numpy(),
                    outputs[grid]["coefficients"])
                decomposition_delta = _verify_numpy_decomposition(
                    saved["residual_decomposition"][grid], independent_decomposition,
                    position, condition, grid, float(np.linalg.norm(target)))
                max_decomposition_term_delta = max(max_decomposition_term_delta, decomposition_delta)
            prescribed = saved.get("prescribed_phase_complex_relative_error")
            if condition in CONDITIONS[3:5]:
                phase = float(saved["phase_rad"])
                expected_prescribed = {
                    grid: _np_rel_l2(outputs[grid]["coefficients"].reshape(-1),
                                     (np.exp(1j * phase) * target).reshape(-1))
                    for grid in GRIDS}
                _require(isinstance(prescribed, Mapping)
                         and all(abs(expected_prescribed[grid] - float(prescribed[grid])) <= 1.0e-14
                                 for grid in GRIDS),
                         f"fixed phase prescribed-complex metric mismatch at row {position}, {condition}")
                passed = passed_six and all(value <= GATE_LIMIT for value in expected_prescribed.values())
            else:
                passed = passed_six
            per_condition_rows[condition].append({"index": position,
                                                  "source_population_index": index,
                                                  "label": int(expected_labels[position]),
                                                  "proposal_id": int(expected_proposals[position]),
                                                  **metrics, "row_passed": passed,
                                                  "all_six_metrics_passed": passed_six})
        expected_terminal = replay["roles"]["development"]["commands"][index:index + 1]
        _require(torch.equal(solver_row["base_row"]["commands"]["fixed_frame_terminal"],
                             expected_terminal),
                 f"terminal command in measured output differs from CUDA source replay at row {index}")
        rows.append(measurement)
    _require(measurement_summary.get("source_terminal_and_reference_rows_reused") == len(rows) == 20,
             "measurement summary does not report all 20 source-bound terminal/reference rows")
    decisions = {condition: all(row["row_passed"] for row in per_condition_rows[condition])
                 for condition in CONDITIONS}
    _require(solver_summary.get("target_count") == 20
             and solver_summary.get("maximum_authorized_solver_iterations") == 1_000_000
             and solver_summary.get("solver_iterations_total") == solver_iterations_total
             and solver_summary.get("fixed_zero_iterations") == fixed_zero_iterations
             and solver_summary.get("fixed_terminal_phase_iterations") == fixed_terminal_iterations
             and solver_summary.get("free_phase_iterations_across_eight_starts") == free_phase_iterations
             and solver_iterations_total <= 1_000_000
             and run_identity.get("solver_iterations_total") == solver_iterations_total,
             "saved solver population or aggregate work counters do not match the 20 row records")
    _require(decisions == measurement_summary["decision"],
             "independent recomputation changed at least one condition-level decision")
    verification_dir.mkdir(parents=True)
    for condition in CONDITIONS:
        with (verification_dir / f"{condition}_independent_metrics.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_condition_rows[condition][0]))
            writer.writeheader()
            writer.writerows(per_condition_rows[condition])
    _verify_contact_sheet(attempt_dir, rows, expected_labels, float(measurement_summary["camera_pixel_area_m2"]),
                          float(measurement_summary["camera_gain_w_inverse"]), source)
    result = {"schema": f"{TASK_ID}.saved_verification.v1", "task_id": TASK_ID,
              "status": "SAVED_EVIDENCE_VERIFIED", "source_run_id": source["checkpoint"]["run_id"],
              "preflight_run_id": identity.run_id, "diagnostic_run_id": run_identity["run_id"],
              "population_role": "consumed_development_phase20", "rows_verified": len(rows),
              "conditions_verified": list(CONDITIONS), "numpy_recomputed_max_abs_metric_difference": max_metric_delta,
              "max_cpu_command_replay_abs_difference": max_command_replay_delta,
              "max_independent_decomposition_term_relative_difference": max_decomposition_term_delta,
              "condition_decisions": decisions,
              "cuda_source_command_replay_verified": True,
              "terminal_measured_commands_bound_to_replay": True,
              "source_terminal_and_reference_rows_reused": len(rows),
              "all_model_optical_evaluator_updates": 0, "scientific_updates": 0,
              "camera_pixel_area_m2": 6.25e-12, "transfer_receipt_checked": True}
    _write_json(verification_dir / "saved_verification.json", result)
    _write_manifest(verification_dir)
    return result


def _verify_contact_sheet(attempt_dir: Path, rows: Sequence[Mapping[str, Any]], labels: torch.Tensor,
                          area: float, gain: float, source: Mapping[str, Any]) -> None:
    expected = {"direct_decoder_reference": torch.cat([
        fixed.pilot._readout_for_evaluator(
            _reused_source_measurements(source, row)[0][2048], area, gain).cpu() for row in rows])}
    for condition in CONDITIONS:
        expected[condition] = torch.cat([
            fixed.pilot._readout_for_evaluator(
                (_reused_source_measurements(source, row)[1][2048]["raw_cmos"]
                 if condition == "fixed_frame_terminal"
                 else row["conditions"][condition]["outputs"][2048]["raw_cmos"]), area, gain).cpu()
            for row in rows])
    saved = attempt_dir / "phase20_contact_sheet.png"
    _require(saved.is_file(), "fixed phase20 display sheet is missing", FileNotFoundError)
    scratch = attempt_dir / ".phase20_contact_sheet_expected.png"
    fixed._draw_contact_sheet(scratch, expected, labels)
    try:
        with Image.open(saved) as actual_image, Image.open(scratch) as expected_image:
            _require(actual_image.size == expected_image.size
                     and np.array_equal(np.asarray(actual_image), np.asarray(expected_image)),
                     "phase20 contact sheet differs from raw CMOS times pixel area and fixed gain")
    finally:
        scratch.unlink(missing_ok=True)


def _worker(config_path: Path, source_dir: Path, checkpoint_path: Path, output_path: Path) -> None:
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == f"{TASK_ID}.solver_recovery_checkpoint.v1"
             and saved.get("task_id") == TASK_ID and saved.get("step") == 3,
             "solver recovery worker checkpoint identity or cursor is invalid")
    source = _load_source(source_dir.resolve(), _config(config_path.resolve()))
    _require(saved.get("source_run_id") == source["checkpoint"]["run_id"],
             "solver recovery checkpoint is bound to another frozen source")
    fixed._set_numerics()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    response = source["inputs"]["response_matrix"].to(device=device, dtype=torch.complex128)
    target = saved["target"].to(device=device, dtype=torch.complex128)
    state3 = saved["solver_state"]
    kind = saved["kind"]
    captured: dict[str, Any] = {}
    if kind == "fixed":
        exposure._fista_fixed_phase_solve(
            response, target, float(saved["phase_rad"]), max_steps=4,
            lipschitz=float(saved["lipschitz"]), initial_state=state3,
            checkpoint=lambda state: captured.update(copy.deepcopy(state))
            if int(state["iteration"]) == 4 else None, checkpoint_interval=1)
    elif kind == "free":
        captured = _capture_free_state(response, target, float(saved["lipschitz"]), 4, state3)
    else:
        raise ValueError(f"unknown solver recovery kind: {kind}")
    _require(captured and int(captured["iteration"]) == 4,
             "independent solver recovery did not complete its fourth iteration")
    _save(output_path, {"schema": f"{TASK_ID}.solver_recovery_step4.v1", "task_id": TASK_ID,
                        "source_run_id": source["checkpoint"]["run_id"],
                        "kind": kind, "step": 4, "solver_state": captured,
                        "loss": _solver_state_loss(kind, response, target, captured),
                        "rng_state_cpu": torch.get_rng_state().clone(),
                        "rng_state_cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available() else []})


def _registered_path(config: Mapping[str, Any], key: str) -> Path:
    return _resolve(config["output"][key])


def _assert_repair_output(path: Path, prefix: str) -> None:
    path = path.resolve()
    _require(path.parent == RUNS_DIR.resolve() and path.name.startswith(prefix),
             f"repair output must be a new task-scoped directory beginning {prefix!r}", ValueError)
    _require(not path.exists(), f"repair output already exists: {path}", FileExistsError)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "review-source", "verify-source", "semantic-audit", "preflight",
        "execute", "resume", "measure", "verify-saved", "solver-recovery-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--source-verification-file", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--verification-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.mode == "solver-recovery-worker":
        _require(args.source_dir is not None and args.checkpoint is not None and args.output_dir is not None,
                 "solver recovery worker requires explicit source, checkpoint, and output paths", ValueError)
        _worker(args.config, args.source_dir.resolve(), args.checkpoint.resolve(), args.output_dir.resolve())
        return 0

    config_path = args.config.resolve()
    config = _config(config_path)
    source_dir = args.source_dir.resolve() if args.source_dir else None
    review_dir = args.review_dir.resolve() if args.review_dir else None
    if args.mode == "review-source":
        _require(source_dir is not None, "review-source requires --source-dir", ValueError)
        output_dir = args.output_dir.resolve() if args.output_dir else _registered_path(config, "source_review_dir")
        _assert_repair_output(output_dir, "source_review_")
        result = _review_source(config, source_dir, output_dir)
    elif args.mode == "verify-source":
        _require(source_dir is not None and review_dir is not None,
                 "verify-source requires explicit --source-dir and --review-dir", ValueError)
        output_dir = args.output_dir.resolve() if args.output_dir else _registered_path(
            config, "source_review_verification_dir")
        _assert_repair_output(output_dir, "source_review_windows_verify_")
        result = _verify_source(config, source_dir, review_dir, output_dir)
    elif args.mode == "semantic-audit":
        _require(source_dir is not None, "semantic-audit requires explicit --source-dir", ValueError)
        output_dir = args.output_dir.resolve() if args.output_dir else _registered_path(
            config, "semantic_attribution_dir")
        _assert_repair_output(output_dir, "semantic_attribution_")
        result = _semantic_audit(source_dir, output_dir)
    elif args.mode == "preflight":
        _require(source_dir is not None and review_dir is not None
                 and args.source_verification_file is not None,
                 "preflight requires source, CUDA replay, and Windows verification paths", ValueError)
        output_dir = args.output_dir.resolve() if args.output_dir else _registered_path(config, "preflight_dir")
        _assert_repair_output(output_dir, "preflight_")
        result = _preflight(config, source_dir, review_dir, args.source_verification_file,
                            output_dir)
    elif args.mode in ("execute", "resume"):
        _require(source_dir is not None and review_dir is not None and args.preflight_dir is not None,
                 f"{args.mode} requires explicit source, review, and preflight paths", ValueError)
        attempt_dir = args.attempt_dir.resolve() if args.attempt_dir else _registered_path(config, "attempt_dir")
        _require(attempt_dir == _registered_path(config, "attempt_dir"),
                 "diagnostic solver must use its registered isolated attempt path", ValueError)
        checkpoint = args.checkpoint.resolve() if args.checkpoint else None
        _require((args.mode == "resume") == (checkpoint is not None),
                 "resume requires --checkpoint and execute must start without one", ValueError)
        result = _execute(config, source_dir, review_dir, args.preflight_dir.resolve(),
                          attempt_dir, checkpoint)
    elif args.mode == "measure":
        _require(source_dir is not None and review_dir is not None and args.preflight_dir is not None,
                 "measure requires explicit source, review, and preflight paths", ValueError)
        attempt_dir = args.attempt_dir.resolve() if args.attempt_dir else _registered_path(config, "attempt_dir")
        _require(attempt_dir == _registered_path(config, "attempt_dir"),
                 "diagnostic measurement must consume the registered attempt", ValueError)
        result = _measure(config, source_dir, review_dir, args.preflight_dir.resolve(), attempt_dir)
    else:
        _require(source_dir is not None and review_dir is not None
                 and args.source_verification_file is not None and args.preflight_dir is not None,
                 "verify-saved requires source, review, source-verification, and preflight paths", ValueError)
        attempt_dir = args.attempt_dir.resolve() if args.attempt_dir else _registered_path(
            config, "windows_backup_dir") / Path(config["output"]["attempt_dir"]).name
        verification_dir = args.verification_dir.resolve() if args.verification_dir else _registered_path(
            config, "windows_verification_dir")
        _assert_repair_output(verification_dir, "windows_verify_")
        result = _verify_saved(config, source_dir, review_dir, args.source_verification_file,
                               args.preflight_dir.resolve(), attempt_dir, verification_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
