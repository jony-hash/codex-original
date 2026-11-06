"""Replay fixed Gaussian endpoints and verify isolated checkpoint recovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

import scripts.research.lightgen.diagnose_lightgen_gaussian_j_frozen_attribution_phase_density_v1 as jdiag
import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref
import scripts.research.lightgen.execute_lightgen_small_joint_update_readability_pilot_v1 as joint
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json
from ldm_project.v2.evidence.io import cpu_copy


ROOT = Path(__file__).resolve().parents[3]
JOINT_CONFIG = Path("configs/research/lightgen/small_joint_update_readability_pilot_v1.yaml")
JDIAG_CONFIG = Path("configs/research/lightgen/gaussian_j_frozen_attribution_phase_density_v1.yaml")
OUTPUT_REL = Path("outputs/20261024_lightgen_gaussian_production_replay_v1/runs/attempt_02")
C_G4_REL = Path(
    "outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/runs/"
    "autodl_backup_attempt_01/generation/C_seed_2026101501/G4_stream_0.pt"
)
RTOL = 2e-6
ATOL = 5e-8
RECOVERY_ATOL = 5e-8
BATCH_SIZE = 10
ENDPOINTS = ("C", "D0", "J")
PHASE_KEYS = {"runtime.encoder.phase", "runtime.generator.phase"}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def _write_update_log(stage: Path, calls: list[dict[str, Any]]) -> None:
    payload = {"schema": "lightgen_gaussian_recovery_updates.v1", "calls": calls}
    path = stage / "engineering_update_log.json"
    _write_json(path, payload)
    _require(_read_json(path) == payload, "engineering update log failed atomic readback")


def _start_logged_update(stage: Path, calls: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    _require(entry.get("status") == "CALL_STARTED" and entry.get("call_number") == len(calls) + 1,
             "engineering update log call order changed")
    calls.append(entry)
    _write_update_log(stage, calls)


def _complete_logged_update(stage: Path, calls: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    _require(calls and calls[-1] is entry and entry.get("status") == "CALL_STARTED",
             "engineering update log has no matching in-flight call")
    entry["status"] = "COMPLETED"
    _write_update_log(stage, calls)


def _optimizer_steps_by_phase(optimizer: torch.optim.Adam,
                              model: torch.nn.Module) -> dict[str, int]:
    encoder, decoder = base._phase_groups(model)
    encoder_step, decoder_step = joint._optimizer_steps(optimizer, encoder, decoder)
    return {"encoder_adam_step": encoder_step, "decoder_adam_step": decoder_step}


def _write_task_progress(output: Path, phase: str, status: str,
                         completed: int | None = None, total: int | None = None,
                         optimizer_step: int | None = None) -> None:
    progress: dict[str, Any] = {
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "status": status,
        "current_stage": phase,
    }
    if completed is not None:
        progress["current_task_local_update"] = completed
    if total is not None:
        progress["total_task_local_updates"] = total
    if optimizer_step is not None:
        progress["current_optimizer_step"] = optimizer_step
    _write_json(output / "progress.json", progress)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV evidence is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite task evidence: {path}")
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)
    temporary.replace(path)


def _relative(path: Path) -> str:
    absolute = (path if path.is_absolute() else ROOT / path).absolute()
    root = ROOT.absolute()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        # Some production resolvers canonicalize a task-local historical alias
        # before returning it. Map that target back through its registered alias.
        for registered_run in sorted((root / "outputs").glob(
                "*/runs/autodl_backup_attempt_01")):
            try:
                resolved_run = registered_run.resolve(strict=True)
                suffix = absolute.relative_to(resolved_run)
            except (OSError, ValueError):
                continue
            return (registered_run / suffix).absolute().relative_to(root).as_posix()
        raise ValueError(f"path is outside the task root and registered historical inputs: {path}")


def _torch_load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is missing: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(value, dict), f"expected checkpoint/evidence mapping: {path}")
    return value


def _read_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    joint_config = joint._read_config(ROOT / JOINT_CONFIG)
    j_config = jdiag._read_config(ROOT / JDIAG_CONFIG)
    return joint_config, j_config


def _load_inputs(device: torch.device) -> dict[str, Any]:
    joint_config, j_config = _read_configs()
    fixed = jdiag._load_fixed(j_config, device)
    fixed["source"]["training_labels"] = fixed["labels"]
    fixed["source"]["training_indices"] = fixed["train_rows"]
    fixed["source"]["training_pixels"] = fixed["pixels"]
    source_path = ROOT / joint_config["source"]["checkpoint"]
    source_dev_path = source_path.parent / "dev_examples_step_045000.pt"
    source_metrics_path = source_path.parent / "reconstruction_metrics_dev.csv"
    source_dev = _torch_load(source_dev_path)
    c_g4 = _torch_load(ROOT / C_G4_REL)
    d0_dev_path = fixed["control_dir"] / "D0" / "dev_display_examples.pt"
    d0_g4_path = fixed["control_dir"] / "D0" / "generated_cohort.pt"
    j_dir = ROOT / j_config["inputs"]["joint_optical_dir"] / "J"
    j_dev_path = j_dir / "dev_display_examples.pt"
    j_g4_path = j_dir / "generated_cohort.pt"
    endpoints = {
        "C": {
            "checkpoint": fixed["source"]["checkpoint"],
            "dev": source_dev,
            "g4": c_g4,
            "dev_path": source_dev_path,
            "g4_path": ROOT / C_G4_REL,
            "dev_metrics": _read_csv(source_metrics_path),
            "generation_metrics": None,
        },
        "D0": {
            "checkpoint": fixed["checkpoints"]["D050000"],
            "dev": _torch_load(d0_dev_path),
            "g4": _torch_load(d0_g4_path),
            "dev_path": d0_dev_path,
            "g4_path": d0_g4_path,
            "dev_metrics": _read_csv(fixed["control_dir"] / "D0" / "reconstruction_metrics_dev.csv"),
            "generation_metrics": _read_csv(fixed["control_dir"] / "D0" / "generation_metrics_per_image.csv"),
        },
        "J": {
            "checkpoint": fixed["checkpoints"]["J50000"],
            "dev": _torch_load(j_dev_path),
            "g4": _torch_load(j_g4_path),
            "dev_path": j_dev_path,
            "g4_path": j_g4_path,
            "dev_metrics": _read_csv(j_dir / "reconstruction_metrics_dev.csv"),
            "generation_metrics": _read_csv(j_dir / "generation_metrics_per_image.csv"),
        },
    }
    source = fixed["source"]["checkpoint"]
    j_start = fixed["checkpoints"]["J45000"]
    _require(source["model"].keys() == j_start["model"].keys()
             and all(torch.equal(source["model"][name], j_start["model"][name])
                     for name in source["model"]),
             "C source checkpoint and J 45,000-step lineage snapshot differ")
    _require(source_dev.get("task_id") == source.get("task_id")
             and source_dev.get("branch") == "C" and source_dev.get("global_step") == 45_000
             and source_dev.get("run_id") == source.get("run_id"),
             "C saved DEV evidence does not match the loaded C 45,000-step checkpoint")
    _require(tuple(source_dev.get("rows", torch.empty(0)).shape) == (20,)
             and tuple(source_dev.get("ed_logits", torch.empty(0)).shape) == (20, 10)
             and tuple(source_dev.get("ec_logits", torch.empty(0)).shape) == (20, 10),
             "C saved DEV views or historical logits are incomplete")
    _require(c_g4.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
             and c_g4.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
             and c_g4.get("model") == "C_seed_2026101501" and c_g4.get("method") == "G4"
             and c_g4.get("model_seed") == 2026101501 and c_g4.get("stream") == 0
             and c_g4.get("global_step") == 45_000,
             "C saved G4 replay cohort is not the registered C endpoint")
    for endpoint, record in endpoints.items():
        dev, g4, checkpoint = record["dev"], record["g4"], record["checkpoint"]
        expected_step = {"C": 45_000, "D0": 50_000, "J": 50_000}[endpoint]
        _require(checkpoint.get("completed_global_step") == expected_step
                 and tuple(checkpoint.get("model", {}).get("runtime.encoder.phase", torch.empty(0)).shape)
                 == (2, 1, 100, 100), f"{endpoint} checkpoint model or endpoint identity changed")
        if endpoint != "C":
            _require(dev.get("task_id") == checkpoint.get("task_id")
                     and dev.get("branch") == endpoint and dev.get("global_step") == expected_step
                     and dev.get("run_id") == checkpoint.get("run_id"),
                     f"{endpoint} saved DEV endpoint identity changed")
            _require(g4.get("task_id") == checkpoint.get("task_id")
                     and g4.get("branch") == endpoint and g4.get("global_step") == expected_step
                     and g4.get("run_id") == checkpoint.get("run_id"),
                     f"{endpoint} saved G4 endpoint identity changed")
        rows = dev.get("rows")
        labels = dev.get("labels")
        _require(isinstance(rows, torch.Tensor) and tuple(rows.shape) == (20,)
                 and isinstance(labels, torch.Tensor) and tuple(labels.shape) == (20,),
                 f"{endpoint} DEV saved rows are incomplete")
        _require(tuple(g4.get("ols_coefficients", torch.empty(0)).shape) == (10_000, 100)
                 and g4["ols_coefficients"].is_complex()
                 and tuple(g4.get("requested_labels", torch.empty(0)).shape) == (10_000,)
                 and tuple(g4.get("component_ids", torch.empty(0)).shape) == (10_000,)
                 and tuple(g4.get("proposal_ids", torch.empty(0)).shape) == (10_000,),
                 f"{endpoint} G4 coefficient or identity population is incomplete")
    d0_g4 = endpoints["D0"]["g4"]
    for key in ("ols_coefficients", "requested_labels", "component_ids", "proposal_ids"):
        _require(torch.equal(c_g4[key], d0_g4[key]), f"C and fixed D0 G4 identity differs: {key}")
        _require(torch.equal(endpoints["J"]["g4"][key], d0_g4[key]),
                 f"J and fixed D0 G4 identity differs: {key}")
    dev_identity = {}
    for endpoint, record in endpoints.items():
        dev = record["dev"]
        selected = _selected_positions(dev["labels"].tolist())
        dev_identity[endpoint] = [(int(dev["rows"][i]), int(dev["labels"][i])) for i in selected]
    _require(dev_identity["C"] == dev_identity["D0"] == dev_identity["J"],
             "DEV records must be joined by row and label; endpoint row order differs")
    return {
        "joint_config": joint_config, "j_config": j_config, "fixed": fixed,
        "pixels": fixed["pixels"], "labels": fixed["labels"],
        "train_rows": fixed["train_rows"], "dev_rows": fixed["dev_rows"],
        "endpoints": endpoints,
    }


def _selected_positions(labels: list[int]) -> list[int]:
    selected: list[int] = []
    for label in range(10):
        positions = [index for index, value in enumerate(labels) if int(value) == label]
        _require(len(positions) >= 2, f"saved population does not contain two rows for class {label}")
        selected.extend(positions[:2])
    return selected


def _interface_spec(joint_config: dict[str, Any]) -> str:
    optics = joint_config["optics"]
    rows = [
        ("类别与随机输入→复系数", "训练证据使用每条100维复数OLS系数；G4为四分量先验，C生成端点使用固定保存系数。",
         "当前为软件采样接口，没有实测设备把类别/随机数制备为100路复振幅。",
         "测量100通道幅相可编程范围、通道串扰和重复性；明确公共相位规范与端口相对相位。"),
        ("直接制备Decoder输入", f"当前模型为{optics['ols_ports']}路Gaussian S12，端口节距{optics['port_pitch_m'] * 1e6:g} μm，1/e场半径{optics['field_1e_radius_m'] * 1e6:g} μm；传输设为理想单位矩阵。",
         "配置标记PROJECT_GEOMETRY_UNMEASURED；没有实测复传输矩阵、端口坐标、相位漂移或输入面标定。",
         "在实际Decoder入口面测量复场与端口坐标，取得含幅相、串扰和损耗的传输矩阵。"),
        ("保留完整Encoder链路", f"模拟链路为2层Encoder相位面、每面100×100参数，传播网格{optics['propagation_grid']}。",
         "从类别图像到输入面的SLM/照明配置、打印器件误差和实测层间传播均未知。",
         "先测输入调制器与Encoder出口复场，再核对实际传播与Decoder入口映射。"),
        ("双光子打印衍射层", f"数值模型使用{optics['wavelength_m'] * 1e9:g} nm波长、2层Encoder和2层Decoder相位面。",
         "打印波长适配、材料折射率/透过率、相位-高度曲线、有效口径、收缩、表面误差、基底和层间装调均无实测或厂商依据。",
         "在下单前取得材料与工艺标定片数据，并用实测相位-高度曲线和公差复核器件设计。"),
        ("工厂光纤组件", f"数值OLS表示{optics['ols_ports']}个理想通道；配置几何是10×10、{optics['port_pitch_m'] * 1e6:g} μm节距，T=I。",
         "真实芯径/包层、MFD、NA、端面、损耗、串扰、通道坐标和复传输矩阵未知；模型节距不是厂商规格。",
         "取得组件图纸，并在工作波长下测量端口复场、损耗、串扰及相位随时间的稳定性。"),
        ("探测与读出", "软件视图为固定camera_gain、截断到[0,1]后池化到28×28；200×200视图是模型生成的预览。",
         "相机型号、像素尺寸、成像倍率、曝光/增益、暗场/平场、QE、满阱和饱和行为未知。",
         "用暗场、平场和已知功率标定相机响应、动态范围、空间采样与固定增益。"),
    ]
    lines = [
        "# 输入接口与器件规格边界",
        "",
        "本表区分当前模拟配置与实际硬件证据。T=I、Gaussian几何及200×200视图只描述软件路径，不作为器件实测值。",
        "",
        "| 环节 | 当前模型值与来源 | 未知或未测量 | 下一验证动作 |",
        "|---|---|---|---|",
    ]
    for stage, current, unknown, next_action in rows:
        values = [stage, current, unknown, next_action]
        lines.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in values) + " |")
    lines.append("")
    return "\n".join(lines)


def _checkpoint_optimizer_steps(checkpoint: dict[str, Any]) -> dict[str, int]:
    recorded = checkpoint.get("optimizer_steps")
    if isinstance(recorded, dict):
        _require(set(recorded) == {"encoder_adam_step", "decoder_adam_step"}
                 and all(type(value) is int for value in recorded.values()),
                 "checkpoint optimizer step declaration is malformed")
        return {key: int(value) for key, value in recorded.items()}
    optimizer = checkpoint.get("optimizer")
    _require(isinstance(optimizer, dict), "checkpoint optimizer state is missing")
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    _require(len(groups) == 1 and len(groups[0].get("params", [])) == 2 and len(states) == 2,
             "C source checkpoint must contain both phase Adam states")
    phase_steps = []
    for parameter_id in groups[0]["params"]:
        state = states.get(parameter_id, states.get(str(parameter_id)))
        _require(isinstance(state, dict) and "step" in state,
                 "C source checkpoint is missing an Encoder or Decoder Adam state")
        phase_steps.append(int(torch.as_tensor(state["step"]).item()))
    return {"encoder_adam_step": phase_steps[0], "decoder_adam_step": phase_steps[1]}


def _build_run_config(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    output_rel = output.resolve().relative_to(ROOT.resolve()).as_posix()
    records = {}
    for endpoint, item in inputs["endpoints"].items():
        dev, g4, checkpoint = item["dev"], item["g4"], item["checkpoint"]
        records[endpoint] = {
            "global_step": int(checkpoint["completed_global_step"]),
            "checkpoint_run_id": checkpoint.get("run_id"),
            "optimizer_steps": _checkpoint_optimizer_steps(checkpoint),
            "dev_path": _relative(item["dev_path"]),
            "dev_task_id": dev.get("task_id"), "dev_run_id": dev.get("run_id"),
            "dev_rows": [int(x) for x in dev["rows"].tolist()],
            "dev_labels": [int(x) for x in dev["labels"].tolist()],
            "g4_path": _relative(item["g4_path"]),
            "g4_task_id": g4.get("task_id"), "g4_run_id": g4.get("run_id"),
            "g4_count": int(g4["requested_labels"].numel()),
            "g4_coefficients_shape": list(g4["ols_coefficients"].shape),
        }
    native_coverage = {endpoint: _native_coverage(inputs, endpoint) for endpoint in ENDPOINTS}
    native_matched = {endpoint: len(value[0]) for endpoint, value in native_coverage.items()}
    native_missing = {endpoint: len(value[1]) for endpoint, value in native_coverage.items()}
    _require(sum(native_matched.values()) == 80 and sum(native_missing.values()) == 40,
             "fixed historical native-view coverage differs from the registered 80/40 population")
    return {
        "schema": "lightgen_gaussian_production_replay_run.v1",
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "evidence_tier": "engineering_only",
        "output_dir": output_rel,
        "inputs": {
            "joint_config": JOINT_CONFIG.as_posix(), "j_diagnostic_config": JDIAG_CONFIG.as_posix(),
            "C_source_checkpoint": inputs["joint_config"]["source"]["checkpoint"],
            "C_G4_saved_cohort": C_G4_REL.as_posix(),
            "D0_control_dir": inputs["j_config"]["inputs"]["control_evidence_dir"],
            "J_optical_dir": inputs["j_config"]["inputs"]["joint_optical_dir"],
            "MNIST_training_file": inputs["j_config"]["inputs"]["data_file"],
            "E_D_checkpoint": inputs["joint_config"]["evaluation"]["evaluator_checkpoint"],
            "E_C_checkpoint": inputs["joint_config"]["evaluation"]["ec_checkpoint"],
            "J_recovery_checkpoint": (
                inputs["j_config"]["inputs"]["joint_optical_dir"]
                + "/checkpoint_step_002500.pt"
            ),
        },
        "roles": {"train_rows": int(inputs["train_rows"].numel()),
                  "dev_rows": int(inputs["dev_rows"].numel()), "official_test_reads": 0},
        "replay": {"endpoints": records, "dev_rows_per_endpoint": 20,
                   "g4_rows_per_endpoint": 20, "total_rows": 120,
                   "native_reference_rows": 80,
                   "native_reference_rows_by_endpoint": native_matched,
                   "native_reference_missing_rows_by_endpoint": native_missing,
                   "grid": 2048, "rtol": RTOL, "atol": ATOL,
                   "join_rule": "DEV by dataset row and label; G4 by saved cohort row, proposal and component"},
        "C_preflight": {"production_entry": "joint._run_preflight", "engineering_adam_calls": 5},
        "J_recovery": {"source_global_step": 47_500, "schedule_cursor": 2_500,
                       "post_three_global_step": 47_503, "post_four_global_step": 47_504,
                       "engineering_adam_calls": 5, "model_optimizer_atol": RECOVERY_ATOL},
        "scientific_updates": 0, "prior_fits": 0, "new_sampling": 0,
        "official_test_reads": 0, "lpips": 0,
        "historical_system_gate": "VALID_FAILED",
        "hardware_geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }


def _build_input_manifest(inputs: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lightgen_gaussian_production_replay_inputs.v1",
        "task_id": run_config["task_id"],
        "source_status": "EXPLICIT_INPUTS_VALIDATED",
        "endpoints": run_config["replay"]["endpoints"],
        "dev_join": "row_and_label",
        "g4_identity_equal_across_C_D0_J": True,
        "optimizer_and_step_identity": {
            key: {"global_step": int(inputs["endpoints"][key]["checkpoint"]["completed_global_step"]),
                  "optimizer_steps": _checkpoint_optimizer_steps(
                      inputs["endpoints"][key]["checkpoint"]
                  )}
            for key in ENDPOINTS
        },
    }


def _prepare(output: Path) -> dict[str, Any]:
    preexisting = list(output.iterdir()) if output.exists() else []
    unarchived = [path for path in preexisting if path.name != "engineering"]
    if unarchived and not (output / "run_config.json").is_file():
        raise RuntimeError(f"output path is nonempty but has no task config; refusing overwrite: {output}")
    inputs = _load_inputs(torch.device("cpu"))
    run_config = _build_run_config(inputs, output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.is_file():
        existing = _read_json(config_path)
        _require(existing == run_config, "prepared run configuration differs from loaded input identities")
    else:
        _write_json(config_path, run_config)
    input_manifest = _build_input_manifest(inputs, run_config)
    manifest_path = output / "input_manifest.json"
    if manifest_path.is_file():
        _require(_read_json(manifest_path) == input_manifest, "input manifest changed on resume")
    else:
        _write_json(manifest_path, input_manifest)
    interface_path = output / "interface_spec.md"
    interface = _interface_spec(inputs["joint_config"])
    if interface_path.is_file():
        _require(interface_path.read_text(encoding="utf-8") == interface,
                 "interface specification changed on resume")
    else:
        interface_path.write_text(interface, encoding="utf-8")
    result = {"status": "PREPARED", "task_id": run_config["task_id"],
              "input_manifest": str(manifest_path.relative_to(ROOT)),
              "validated_endpoints": list(ENDPOINTS), "replay_items": 120,
              "native_reference_rows": 80, "native_reference_missing_rows": 40,
              "data_roles": input_manifest["optimizer_and_step_identity"]}
    marker = output / "prepare_result.json"
    if marker.is_file():
        _require(_read_json(marker) == result, "prepare phase evidence changed on resume")
    else:
        _write_json(marker, result)
    return result


def _load_run_config(output: Path) -> dict[str, Any]:
    run = _read_json(output / "run_config.json")
    output_rel = output.resolve().relative_to(ROOT.resolve()).as_posix()
    _require(run.get("schema") == "lightgen_gaussian_production_replay_run.v1"
             and run.get("output_dir") == output_rel
             and run.get("replay", {}).get("total_rows") == 120
             and run.get("replay", {}).get("native_reference_rows") == 80
             and run.get("replay", {}).get("native_reference_rows_by_endpoint")
                 == {"C": 40, "D0": 20, "J": 20}
             and run.get("replay", {}).get("native_reference_missing_rows_by_endpoint")
                 == {"C": 0, "D0": 20, "J": 20},
             "run configuration is missing, changed, or belongs to a different attempt")
    return run


def _latest_complete_dir(stage_root: Path, complete_name: str) -> Path | None:
    if not stage_root.is_dir():
        return None
    for candidate in sorted(stage_root.glob("attempt_*"), reverse=True):
        if candidate.is_dir() and (candidate / complete_name).is_file():
            return candidate
    return None


def _new_stage_dir(stage_root: Path) -> Path:
    stage_root.mkdir(parents=True, exist_ok=True)
    existing = [candidate for candidate in stage_root.glob("attempt_*") if candidate.is_dir()]
    _require(not existing,
             f"incomplete stage evidence blocks automatic retry; inspect and resume explicitly: {existing}")
    result = stage_root / "attempt_01"
    result.mkdir(parents=False, exist_ok=False)
    return result


def _execution_stage(stage_root: Path, complete_name: str) -> Path | None:
    if not stage_root.is_dir():
        return None
    candidates = sorted(path for path in stage_root.glob("attempt_*") if path.is_dir())
    incomplete = [path for path in candidates if not (path / complete_name).is_file()]
    _require(not incomplete,
             f"incomplete stage evidence blocks automatic retry; inspect and resume explicitly: {incomplete}")
    return candidates[-1] if candidates else None


def _close_cuda_events(events: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]],
                       device: torch.device) -> list[dict[str, Any]]:
    torch.cuda.synchronize(device)
    return [{"name": name, "rows": rows, "device": torch.cuda.get_device_name(device),
             "elapsed_ms": float(start.elapsed_time(end))}
            for name, rows, start, end in events]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().double() - right.detach().double()).abs().max()) if left.numel() else 0.0


def _np_ssim(image: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Independent float64 NumPy implementation of the registered 7x7 SSIM formula."""
    x = np.asarray(image, dtype=np.float64)[:, 0]
    y = np.asarray(target, dtype=np.float64)[:, 0]
    coordinates = np.arange(-3, 4, dtype=np.float64)
    kernel_1d = np.exp(-(coordinates ** 2) / (2.0 * 1.5 ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)

    def conv(values: np.ndarray) -> np.ndarray:
        padded = np.pad(values, ((0, 0), (3, 3), (3, 3)), mode="constant")
        output = np.zeros_like(values, dtype=np.float64)
        for row in range(7):
            for col in range(7):
                output += kernel[row, col] * padded[:, row:row + values.shape[1],
                                                     col:col + values.shape[2]]
        return output

    mx, my = conv(x), conv(y)
    vx = conv(x * x) - mx * mx
    vy = conv(y * y) - my * my
    vxy = conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    values = ((2.0 * mx * my + c1) * (2.0 * vxy + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2)
    )
    return values.mean(axis=(1, 2))


def _csv_lookup(rows: list[dict[str, str]], key_name: str) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        key = int(row[key_name])
        _require(key not in result, f"duplicate source metrics key {key_name}={key}")
        result[key] = row
    return result


def _selected_g4_positions(cohort: dict[str, Any]) -> list[int]:
    return _selected_positions(cohort["requested_labels"].tolist())


def _native_reference(pack: dict[str, Any], kind: str, position: int) -> torch.Tensor | None:
    if kind == "DEV":
        key = "native_images" if "native_images" in pack else "view_native"
        values = pack.get(key)
        if not isinstance(values, torch.Tensor):
            return None
        _require(values.ndim == 4 and tuple(values.shape[-2:]) == (200, 200)
                 and 0 <= position < values.shape[0],
                 f"saved {key} native view has the wrong shape or row index")
        return values[position]
    indices, values = pack.get("native_capture_indices"), pack.get("native_capture_images")
    if not isinstance(indices, torch.Tensor) or not isinstance(values, torch.Tensor):
        return None
    _require(indices.ndim == 1 and values.ndim == 4 and values.shape[0] == indices.numel()
             and tuple(values.shape[-2:]) == (200, 200),
             "saved G4 native captures have the wrong identity or shape")
    matches = torch.nonzero(indices.cpu().long() == int(position), as_tuple=False).flatten()
    _require(matches.numel() <= 1, f"duplicate saved native capture for G4 row {position}")
    return values[int(matches[0])] if matches.numel() == 1 else None


def _native_coverage(inputs: dict[str, Any], endpoint: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    matched, missing = [], []
    record = inputs["endpoints"][endpoint]
    populations = (
        ("DEV", record["dev"], _selected_positions(record["dev"]["labels"].tolist())),
        ("G4", record["g4"], _selected_g4_positions(record["g4"])),
    )
    for kind, pack, positions in populations:
        for position in positions:
            target = matched if _native_reference(pack, kind, position) is not None else missing
            target.append((kind, int(position)))
    return matched, missing


def _load_evaluators(config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    ed = base._load_evaluator(ROOT / config["evaluation"]["evaluator_checkpoint"], device)
    ec, _qualification = decoder_ref._load_ec_evaluator(
        ROOT / config["evaluation"]["ec_checkpoint"], device,
    )
    ed.eval()
    ec.eval()
    _require(all(not p.requires_grad for p in ed.parameters())
             and all(not p.requires_grad for p in ec.parameters()),
             "frozen confirmation evaluator has trainable parameters")
    return ed, ec


def _compare_record(
    endpoint: str, kind: str, position: int, inputs: dict[str, Any],
    saved_view: torch.Tensor, replay_view: torch.Tensor, ed_logits: torch.Tensor,
    ec_logits: torch.Tensor, target: torch.Tensor | None,
    saved_ed_reference: torch.Tensor | None = None,
    saved_ec_reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    record = inputs["endpoints"][endpoint]
    pack = record["dev"] if kind == "DEV" else record["g4"]
    labels = pack["labels"] if kind == "DEV" else pack["requested_labels"]
    label = int(labels[position])
    ed_pred, ed_requested_margin, ed_top_margin, ed_cpu_logits = decoder_ref._logit_summary(
        ed_logits, torch.tensor([label], device=ed_logits.device),
    )
    ec_pred, ec_requested_margin, ec_top_margin, ec_cpu_logits = decoder_ref._logit_summary(
        ec_logits, torch.tensor([label], device=ec_logits.device),
    )
    row: dict[str, Any] = {
        "endpoint": endpoint, "kind": kind, "label": label,
        "position": position,
        "view_max_abs_error": _max_abs(replay_view, saved_view),
        "view_reference_max_abs": float(saved_view.detach().abs().max()),
        "view_allclose": bool(torch.allclose(replay_view, saved_view, rtol=RTOL, atol=ATOL)),
        "ed_logit_max_abs_error": None,
        "ec_logit_max_abs_error": None,
        "ed_logits_allclose": None,
        "ec_logits_allclose": None,
        "ed_prediction": int(ed_pred[0]), "ec_prediction": int(ec_pred[0]),
        "ed_requested_margin": float(ed_requested_margin[0]),
        "ed_top_margin": float(ed_top_margin[0]),
        "ec_requested_margin": float(ec_requested_margin[0]),
        "ec_top_margin": float(ec_top_margin[0]),
        "predictions_match_saved": None,
        "mse": None, "psnr_db": None, "ssim_float32": None,
        "ssim_float64": None, "ssim_numpy": None,
        "saved_metrics_match": None,
        "saved_logit_reference": None,
        "native_reference_status": None,
        "native_max_abs_error": None,
        "native_allclose": None,
    }
    saved_ed = pack.get("ed_logits")
    saved_ec = pack.get("ec_logits")
    if saved_ed is None and kind == "DEV":
        row["saved_logit_reference"] = "frozen evaluators recomputed on saved endpoint view"
        _require(saved_ed_reference is not None and saved_ec_reference is not None,
                 "DEV replay needs frozen evaluator logits from the saved view")
        saved_ed = saved_ed_reference
        saved_ec = saved_ec_reference
    else:
        row["saved_logit_reference"] = "historically saved full evaluator logits"
        saved_ed = saved_ed[position:position + 1].to(ed_logits.device)
        saved_ec = saved_ec[position:position + 1].to(ec_logits.device)
    row["ed_logit_max_abs_error"] = _max_abs(ed_logits, saved_ed)
    row["ec_logit_max_abs_error"] = _max_abs(ec_logits, saved_ec)
    row["ed_logit_reference_max_abs"] = float(saved_ed.detach().abs().max())
    row["ec_logit_reference_max_abs"] = float(saved_ec.detach().abs().max())
    row["ed_logits_allclose"] = bool(torch.allclose(ed_logits, saved_ed, rtol=RTOL, atol=ATOL))
    row["ec_logits_allclose"] = bool(torch.allclose(ec_logits, saved_ec, rtol=RTOL, atol=ATOL))
    row["predictions_match_saved"] = bool(
        int(ed_cpu_logits.argmax(1)[0]) == int(saved_ed.argmax(1)[0])
        and int(ec_cpu_logits.argmax(1)[0]) == int(saved_ec.argmax(1)[0])
    )
    if kind == "DEV":
        row["dataset_row"] = int(pack["rows"][position])
    else:
        row.update({"cohort_row": position,
                    "proposal_id": int(pack["proposal_ids"][position]),
                    "component_id": int(pack["component_ids"][position])})
    if target is not None:
        mse = (replay_view - target).square().mean().clamp_min(1.0e-12)
        psnr = -10.0 * torch.log10(mse)
        ssim32 = ssim_batch(replay_view, target)
        ssim64 = ssim_batch(replay_view.double(), target.double())
        ssim_np = _np_ssim(replay_view.detach().cpu().numpy(), target.detach().cpu().numpy())
        _require(bool(np.allclose(ssim64.detach().cpu().numpy(), ssim_np, rtol=1e-10, atol=1e-12)),
                 "production float64 SSIM differs from the independent NumPy reference")
        row.update({"mse": float(mse), "psnr_db": float(psnr),
                    "ssim_float32": float(ssim32[0]), "ssim_float64": float(ssim64[0]),
                    "ssim_numpy": float(ssim_np[0])})
        source_metric = _csv_lookup(record["dev_metrics"], "row")[int(row["dataset_row"])]
        _require(int(source_metric["label"]) == label,
                 f"saved DEV metric row has a different label: {row['dataset_row']}")
        metric_comparisons = {
            "mse": math.isclose(row["mse"], float(source_metric["mse"]), rel_tol=RTOL, abs_tol=ATOL),
            "psnr_db": math.isclose(row["psnr_db"], float(source_metric["psnr_db"]),
                                    rel_tol=RTOL, abs_tol=ATOL),
            "ssim": math.isclose(row["ssim_float32"], float(source_metric["ssim"]),
                                 rel_tol=RTOL, abs_tol=ATOL),
        }
        saved_pred_ed = int(source_metric["ed_prediction"])
        saved_pred_ec = int(source_metric["ec_prediction"])
        ed_margin_name = "ed_target_margin" if endpoint == "C" else "ed_requested_margin"
        ec_margin_name = "ec_target_margin" if endpoint == "C" else "ec_requested_margin"
        ed_top_name = "ed_margin" if endpoint == "C" else "ed_top_margin"
        ec_top_name = "ec_margin" if endpoint == "C" else "ec_top_margin"
        metric_comparisons["ed_prediction"] = int(ed_pred[0]) == saved_pred_ed
        metric_comparisons["ec_prediction"] = int(ec_pred[0]) == saved_pred_ec
        metric_comparisons["ed_requested_margin"] = math.isclose(
            row["ed_requested_margin"], float(source_metric[ed_margin_name]), rel_tol=RTOL, abs_tol=ATOL,
        )
        metric_comparisons["ec_requested_margin"] = math.isclose(
            row["ec_requested_margin"], float(source_metric[ec_margin_name]), rel_tol=RTOL, abs_tol=ATOL,
        )
        metric_comparisons["ed_top_margin"] = math.isclose(
            row["ed_top_margin"], float(source_metric[ed_top_name]), rel_tol=RTOL, abs_tol=ATOL,
        )
        metric_comparisons["ec_top_margin"] = math.isclose(
            row["ec_top_margin"], float(source_metric[ec_top_name]), rel_tol=RTOL, abs_tol=ATOL,
        )
        row["saved_metrics_match"] = metric_comparisons
    else:
        saved = record["generation_metrics"]
        if saved is None:
            saved_row = {
                "requested_label": label,
                "component_id": int(pack["component_ids"][position]),
                "proposal_id": int(pack["proposal_ids"][position]),
                "ed_prediction": int(pack["ed_predictions"][position]),
                "ec_prediction": int(pack["ec_predictions"][position]),
                "ed_requested_margin": float(pack["ed_requested_margin"][position]),
                "ec_requested_margin": float(pack["ec_requested_margin"][position]),
                "ed_top_margin": float(pack["ed_top_margin"][position]),
                "ec_top_margin": float(pack["ec_top_margin"][position]),
            }
        else:
            saved_row = _csv_lookup(saved, "cohort_row")[position]
        row["g4_saved_metric_match"] = {
            "identity": int(saved_row["requested_label"]) == label
                and int(saved_row["component_id"]) == int(pack["component_ids"][position])
                and int(saved_row["proposal_id"]) == int(pack["proposal_ids"][position]),
            "ed_prediction": int(ed_pred[0]) == int(saved_row["ed_prediction"]),
            "ec_prediction": int(ec_pred[0]) == int(saved_row["ec_prediction"]),
            "ed_requested_margin": math.isclose(row["ed_requested_margin"],
                float(saved_row["ed_requested_margin"]), rel_tol=RTOL, abs_tol=ATOL),
            "ec_requested_margin": math.isclose(row["ec_requested_margin"],
                float(saved_row["ec_requested_margin"]), rel_tol=RTOL, abs_tol=ATOL),
            "ed_top_margin": math.isclose(row["ed_top_margin"],
                float(saved_row["ed_top_margin"]), rel_tol=RTOL, abs_tol=ATOL),
            "ec_top_margin": math.isclose(row["ec_top_margin"],
                float(saved_row["ec_top_margin"]), rel_tol=RTOL, abs_tol=ATOL),
        }
    return row


def _expected_replay_rows(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    expected = []
    for endpoint in ENDPOINTS:
        record = inputs["endpoints"][endpoint]
        dev, g4 = record["dev"], record["g4"]
        for position in _selected_positions(dev["labels"].tolist()):
            row = int(dev["rows"][position])
            label = int(dev["labels"][position])
            _require(int(inputs["labels"][row]) == label, f"{endpoint} DEV row and label disagree: {row}")
            _require(bool(torch.isin(torch.tensor([row]), inputs["dev_rows"]).all()),
                     f"{endpoint} DEV row is outside the registered DEV split: {row}")
            expected.append({"endpoint": endpoint, "kind": "DEV", "position": position,
                             "dataset_row": row, "label": label})
        for position in _selected_g4_positions(g4):
            expected.append({"endpoint": endpoint, "kind": "G4", "position": position,
                             "cohort_row": position, "label": int(g4["requested_labels"][position]),
                             "proposal_id": int(g4["proposal_ids"][position]),
                             "component_id": int(g4["component_ids"][position])})
    return expected


def _validate_replay_rows(rows: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
    expected_by_key = {(item["endpoint"], item["kind"], item["position"]): item for item in expected}
    _require(len(expected_by_key) == 120, "expected replay population does not contain 120 unique rows")
    actual_by_key = {}
    for row in rows:
        endpoint, kind, position = row.get("endpoint"), row.get("kind"), row.get("position")
        _require(endpoint in ENDPOINTS and kind in {"DEV", "G4"}, "replay row has an unknown endpoint or kind")
        key = (endpoint, kind, position)
        _require(key not in actual_by_key, f"duplicate replay row: {key}")
        actual_by_key[key] = row
    _require(actual_by_key.keys() == expected_by_key.keys(), "replay output has missing or unexpected rows")
    for key, required in expected_by_key.items():
        actual = actual_by_key[key]
        for name in ("dataset_row", "cohort_row", "label", "proposal_id", "component_id"):
            if name in required:
                _require(int(actual.get(name, -1)) == int(required[name]),
                         f"replay row identity mismatch for {key}: {name}")


def _replay_tensor_identity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("endpoint", "kind", "position", "dataset_row", "cohort_row", "label",
            "proposal_id", "component_id")
    return [{key: row[key] for key in keys
             if key in row and row[key] not in (None, "")} for row in rows]


def _validate_replay_tensor_evidence(
    payload: dict[str, Any], rows: list[dict[str, Any]], endpoint: str,
) -> None:
    _require(payload.get("schema") == "lightgen_gaussian_replay_tensors.v1"
             and payload.get("endpoint") == endpoint
             and payload.get("rows") == _replay_tensor_identity(rows),
             f"{endpoint} saved replay tensors have the wrong row identity")
    expected_shapes = {
        "saved_views": (40, 1, 28, 28), "replay_views": (40, 1, 28, 28),
        "saved_ed_logits": (40, 10), "replay_ed_logits": (40, 10),
        "saved_ec_logits": (40, 10), "replay_ec_logits": (40, 10),
    }
    for name, shape in expected_shapes.items():
        value = payload.get(name)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                 f"{endpoint} replay tensor {name} is missing or malformed")
        _require(torch.isfinite(value).all().item(), f"{endpoint} replay tensor {name} is nonfinite")
    native = payload.get("native")
    _require(isinstance(native, dict), f"{endpoint} replay native tensor record is missing")
    native_rows = [(row["kind"], int(row["position"])) for row in rows
                   if row.get("native_reference_status") == "MATCHED"]
    missing_rows = [(row["kind"], int(row["position"])) for row in rows
                    if row.get("native_reference_status") == "MISSING_HISTORICAL_REFERENCE"]
    _require(native.get("row_keys") == [[kind, position] for kind, position in native_rows]
             and native.get("missing_row_keys") == [[kind, position] for kind, position in missing_rows],
             f"{endpoint} native-reference coverage differs from the fixed selected rows")
    native_saved, native_replay = native.get("saved"), native.get("replay")
    _require(isinstance(native_saved, torch.Tensor) and isinstance(native_replay, torch.Tensor)
             and tuple(native_saved.shape) == (len(native_rows), 1, 200, 200)
             and tuple(native_replay.shape) == (len(native_rows), 1, 200, 200),
             f"{endpoint} native replay tensors do not match the available historical population")
    _require(torch.isfinite(native_saved).all().item() and torch.isfinite(native_replay).all().item(),
             f"{endpoint} native replay tensors are nonfinite")
    row_by_key = {(row["kind"], int(row["position"])): row for row in rows}
    for index, row in enumerate(rows):
        pairs = (
            ("view", "saved_views", "replay_views", "view_max_abs_error", "view_allclose"),
            ("ed", "saved_ed_logits", "replay_ed_logits", "ed_logit_max_abs_error", "ed_logits_allclose"),
            ("ec", "saved_ec_logits", "replay_ec_logits", "ec_logit_max_abs_error", "ec_logits_allclose"),
        )
        for _name, saved_key, replay_key, error_key, allclose_key in pairs:
            saved, replayed = payload[saved_key][index], payload[replay_key][index]
            error = _max_abs(saved, replayed)
            allclose = bool(torch.allclose(saved, replayed, rtol=RTOL, atol=ATOL))
            _require(math.isclose(error, float(row[error_key]), rel_tol=0.0, abs_tol=1e-12)
                     and allclose == bool(row[allclose_key]),
                     f"{endpoint} replay tensor consumer recomputed a different {_name} result")
        saved_ed, replay_ed = payload["saved_ed_logits"][index], payload["replay_ed_logits"][index]
        saved_ec, replay_ec = payload["saved_ec_logits"][index], payload["replay_ec_logits"][index]
        prediction_match = (int(saved_ed.argmax()) == int(replay_ed.argmax())
                            and int(saved_ec.argmax()) == int(replay_ec.argmax()))
        _require(prediction_match == bool(row["predictions_match_saved"]),
                 f"{endpoint} evaluator prediction identity differs from replay tensors")
    native_index = {key: index for index, key in enumerate(native_rows)}
    for key, index in native_index.items():
        row = row_by_key[key]
        error = _max_abs(native_saved[index], native_replay[index])
        allclose = bool(torch.allclose(native_saved[index], native_replay[index], rtol=RTOL, atol=ATOL))
        _require(math.isclose(error, float(row["native_max_abs_error"]), rel_tol=0.0, abs_tol=1e-12)
                 and allclose == bool(row["native_allclose"]),
                 f"{endpoint} native replay tensor consumer recomputed a different result")
    for key in missing_rows:
        row = row_by_key[key]
        _require(row.get("native_max_abs_error") is None and row.get("native_allclose") is None,
                 f"{endpoint} missing historical native reference was assigned a fabricated comparison")


def _validate_replay_tensor_source_arrays(
    payload: dict[str, Any], rows: list[dict[str, Any]], inputs: dict[str, Any], endpoint: str,
) -> None:
    """Bind saved tensors in the replay bundle to the explicitly loaded historical inputs."""
    record = inputs["endpoints"][endpoint]
    native_index = 0
    for index, row in enumerate(rows):
        kind, position = row["kind"], int(row["position"])
        pack = record["dev"] if kind == "DEV" else record["g4"]
        view_key = "images" if kind == "G4" or endpoint == "C" else "view28"
        source_view = pack[view_key][position].detach().cpu()
        saved_view = payload["saved_views"][index].detach().cpu()
        _require(torch.equal(saved_view, source_view),
                 f"{endpoint} {kind} saved replay tensor is not the registered historical view at {position}")

        historical_ed, historical_ec = pack.get("ed_logits"), pack.get("ec_logits")
        if isinstance(historical_ed, torch.Tensor) and isinstance(historical_ec, torch.Tensor):
            _require(torch.equal(payload["saved_ed_logits"][index].detach().cpu(),
                                 historical_ed[position].detach().cpu())
                     and torch.equal(payload["saved_ec_logits"][index].detach().cpu(),
                                     historical_ec[position].detach().cpu()),
                     f"{endpoint} {kind} saved logits are not the registered historical logits at {position}")
        else:
            _require(kind == "DEV" and endpoint in {"D0", "J"}
                     and row.get("saved_logit_reference")
                         == "frozen evaluators recomputed on saved endpoint view",
                     f"{endpoint} {kind} has neither historical logits nor an explicit saved-view reference")

        source_native = _native_reference(pack, kind, position)
        native_status = row.get("native_reference_status")
        if source_native is None:
            _require(native_status == "MISSING_HISTORICAL_REFERENCE",
                     f"{endpoint} {kind} row {position} claims an unavailable native reference")
        else:
            _require(native_status == "MATCHED"
                     and torch.equal(payload["native"]["saved"][native_index].detach().cpu(),
                                     source_native.detach().cpu()),
                     f"{endpoint} {kind} native tensor is not the registered historical reference at {position}")
            native_index += 1
    _require(native_index == payload["native"]["saved"].shape[0],
             f"{endpoint} native tensor consumer did not consume every historical reference")


def _validate_replay_tensor_metrics(
    payload: dict[str, Any], rows: list[dict[str, Any]], inputs: dict[str, Any], endpoint: str,
) -> None:
    """Recompute logged views and classifier summaries from the serialized tensors."""
    record = inputs["endpoints"][endpoint]
    for index, row in enumerate(rows):
        kind, position, label = row["kind"], int(row["position"]), int(row["label"])
        pack = record["dev"] if kind == "DEV" else record["g4"]
        target = torch.tensor([label], dtype=torch.long)
        metric_summaries = {}
        for name, pred_key, requested_key, top_key in (
            ("ed", "ed_prediction", "ed_requested_margin", "ed_top_margin"),
            ("ec", "ec_prediction", "ec_requested_margin", "ec_top_margin"),
        ):
            summaries = []
            for prefix in ("saved", "replay"):
                logits = payload[f"{prefix}_{name}_logits"][index:index + 1]
                prediction, requested_margin, top_margin, _ = decoder_ref._logit_summary(logits, target)
                summaries.append((int(prediction[0]), float(requested_margin[0]), float(top_margin[0])))
                _require(int(prediction[0]) == int(row[pred_key])
                         and math.isclose(float(requested_margin[0]), float(row[requested_key]),
                                          rel_tol=RTOL, abs_tol=ATOL)
                         and math.isclose(float(top_margin[0]), float(row[top_key]),
                                          rel_tol=RTOL, abs_tol=ATOL),
                         f"{endpoint} {kind} {name.upper()} tensor summary differs from its row at {position}")
            metric_summaries[name] = summaries[-1]

        if kind == "DEV":
            target_key = "targets" if endpoint == "C" else "target28"
            target_view = pack[target_key][position:position + 1].detach().cpu()
            replay_view = payload["replay_views"][index:index + 1].detach().cpu()
            mse = (replay_view - target_view).square().mean().clamp_min(1.0e-12)
            psnr = -10.0 * torch.log10(mse)
            ssim32 = ssim_batch(replay_view, target_view)
            ssim64 = ssim_batch(replay_view.double(), target_view.double())
            ssim_numpy = _np_ssim(replay_view.numpy(), target_view.numpy())
            for name, computed in (("mse", float(mse)), ("psnr_db", float(psnr)),
                                   ("ssim_float32", float(ssim32[0])),
                                   ("ssim_float64", float(ssim64[0])),
                                   ("ssim_numpy", float(ssim_numpy[0]))):
                _require(math.isclose(float(row[name]), computed, rel_tol=RTOL, abs_tol=ATOL),
                         f"{endpoint} DEV {name} differs when recomputed from replay tensors at {position}")
            _require(math.isclose(float(ssim64[0]), float(ssim_numpy[0]),
                                  rel_tol=1e-10, abs_tol=1e-12),
                     f"{endpoint} DEV independent SSIM check failed at {position}")
            dataset_row = int(row["dataset_row"])
            source_metric = _csv_lookup(record["dev_metrics"], "row")[dataset_row]
            ed_margin_key = "ed_target_margin" if endpoint == "C" else "ed_requested_margin"
            ec_margin_key = "ec_target_margin" if endpoint == "C" else "ec_requested_margin"
            ed_top_key = "ed_margin" if endpoint == "C" else "ed_top_margin"
            ec_top_key = "ec_margin" if endpoint == "C" else "ec_top_margin"
            expected_metrics = {
                "mse": math.isclose(float(row["mse"]), float(source_metric["mse"]),
                                     rel_tol=RTOL, abs_tol=ATOL),
                "psnr_db": math.isclose(float(row["psnr_db"]), float(source_metric["psnr_db"]),
                                         rel_tol=RTOL, abs_tol=ATOL),
                "ssim": math.isclose(float(row["ssim_float32"]), float(source_metric["ssim"]),
                                     rel_tol=RTOL, abs_tol=ATOL),
                "ed_prediction": metric_summaries["ed"][0] == int(source_metric["ed_prediction"]),
                "ec_prediction": metric_summaries["ec"][0] == int(source_metric["ec_prediction"]),
                "ed_requested_margin": math.isclose(metric_summaries["ed"][1],
                    float(source_metric[ed_margin_key]), rel_tol=RTOL, abs_tol=ATOL),
                "ec_requested_margin": math.isclose(metric_summaries["ec"][1],
                    float(source_metric[ec_margin_key]), rel_tol=RTOL, abs_tol=ATOL),
                "ed_top_margin": math.isclose(metric_summaries["ed"][2],
                    float(source_metric[ed_top_key]), rel_tol=RTOL, abs_tol=ATOL),
                "ec_top_margin": math.isclose(metric_summaries["ec"][2],
                    float(source_metric[ec_top_key]), rel_tol=RTOL, abs_tol=ATOL),
            }
            _require(row.get("saved_metrics_match") == expected_metrics and all(expected_metrics.values()),
                     f"{endpoint} DEV final consumer could not reproduce source metric agreement at {position}")
        else:
            if record["generation_metrics"] is None:
                source_metric = {
                    "requested_label": int(pack["requested_labels"][position]),
                    "component_id": int(pack["component_ids"][position]),
                    "proposal_id": int(pack["proposal_ids"][position]),
                    "ed_prediction": int(pack["ed_predictions"][position]),
                    "ec_prediction": int(pack["ec_predictions"][position]),
                    "ed_requested_margin": float(pack["ed_requested_margin"][position]),
                    "ec_requested_margin": float(pack["ec_requested_margin"][position]),
                    "ed_top_margin": float(pack["ed_top_margin"][position]),
                    "ec_top_margin": float(pack["ec_top_margin"][position]),
                }
            else:
                source_metric = _csv_lookup(record["generation_metrics"], "cohort_row")[position]
            expected_metrics = {
                "identity": int(source_metric["requested_label"]) == label
                    and int(source_metric["component_id"]) == int(pack["component_ids"][position])
                    and int(source_metric["proposal_id"]) == int(pack["proposal_ids"][position]),
                "ed_prediction": metric_summaries["ed"][0] == int(source_metric["ed_prediction"]),
                "ec_prediction": metric_summaries["ec"][0] == int(source_metric["ec_prediction"]),
                "ed_requested_margin": math.isclose(metric_summaries["ed"][1],
                    float(source_metric["ed_requested_margin"]), rel_tol=RTOL, abs_tol=ATOL),
                "ec_requested_margin": math.isclose(metric_summaries["ec"][1],
                    float(source_metric["ec_requested_margin"]), rel_tol=RTOL, abs_tol=ATOL),
                "ed_top_margin": math.isclose(metric_summaries["ed"][2],
                    float(source_metric["ed_top_margin"]), rel_tol=RTOL, abs_tol=ATOL),
                "ec_top_margin": math.isclose(metric_summaries["ec"][2],
                    float(source_metric["ec_top_margin"]), rel_tol=RTOL, abs_tol=ATOL),
            }
            _require(row.get("g4_saved_metric_match") == expected_metrics and all(expected_metrics.values()),
                     f"{endpoint} G4 final consumer could not reproduce source metric agreement at {position}")


def _run_replay(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for production replay")
    evaluator, ec_evaluator = _load_evaluators(inputs["joint_config"], device)
    evaluator_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    rows_out: list[dict[str, Any]] = []
    stage_summaries = {}
    replay_completed = 0
    for endpoint in ENDPOINTS:
        _write_task_progress(output, "REPLAY", "RUNNING", replay_completed, 120)
        stage_root = output / "replay" / endpoint
        complete = _execution_stage(stage_root, "replay_summary.json")
        if complete is not None:
            summary = _read_json(complete / "replay_summary.json")
            _require(summary.get("status") == "PASSED" and summary.get("endpoint") == endpoint,
                     f"completed replay evidence is invalid: {complete}")
            rows = _read_csv(complete / "replay_rows.csv")
            for row in rows:
                for key in ("position", "label"):
                    row[key] = int(row[key])
                for key in ("dataset_row", "cohort_row", "proposal_id", "component_id"):
                    if row.get(key) not in (None, ""):
                        row[key] = int(row[key])
                if row.get("native_max_abs_error") not in (None, ""):
                    row["native_max_abs_error"] = float(row["native_max_abs_error"])
                for key in ("view_allclose", "ed_logits_allclose", "ec_logits_allclose",
                            "predictions_match_saved", "native_allclose"):
                    if row.get(key) is not None:
                        row[key] = row[key].lower() == "true"
                    elif key == "native_allclose":
                        row[key] = None
            _validate_replay_tensor_evidence(
                _torch_load(complete / "replay_tensors.pt"), rows, endpoint,
            )
            _validate_replay_tensor_source_arrays(
                _torch_load(complete / "replay_tensors.pt"), rows, inputs, endpoint,
            )
            rows_out.extend(rows)
            replay_completed += len(rows)
            _write_task_progress(output, "REPLAY", "RUNNING", replay_completed, 120)
            stage_summaries[endpoint] = summary
            continue
        stage = _new_stage_dir(stage_root)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        record = inputs["endpoints"][endpoint]
        if endpoint == "C":
            model = base._model("small", 2048, device, inputs["fixed"]["baseline_cfg"])
            model.load_state_dict(record["checkpoint"]["model"], strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        else:
            model_key = {"D0": "D050000", "J": "J50000"}[endpoint]
            model = jdiag._model(model_key, 2048, inputs["fixed"])
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()),
                 f"{endpoint} frozen replay model has trainable parameters")
        events: list[tuple[str, int, torch.cuda.Event, torch.cuda.Event]] = []
        endpoint_rows: list[dict[str, Any]] = []
        saved_views_all, replay_views_all = [], []
        saved_ed_all, replay_ed_all, saved_ec_all, replay_ec_all = [], [], [], []
        native_row_keys, native_saved_all, native_replay_all, native_missing_keys = [], [], [], []
        for kind in ("DEV", "G4"):
            pack = record["dev"] if kind == "DEV" else record["g4"]
            positions = (_selected_positions(pack["labels"].tolist()) if kind == "DEV"
                         else _selected_g4_positions(pack))
            for chunk_index, start in enumerate(range(0, len(positions), BATCH_SIZE)):
                selected = positions[start:start + BATCH_SIZE]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                if kind == "DEV":
                    dataset_rows = pack["rows"].index_select(0, torch.tensor(selected)).long()
                    image = image_batch(inputs["pixels"], dataset_rows, "small", device)
                    target = (pack["targets"] if endpoint == "C" else pack["target28"])
                    saved_batch = pack["images"] if endpoint == "C" else pack["view28"]
                    target_batch = target.index_select(0, torch.tensor(selected)).to(device)
                    saved_views = saved_batch.index_select(0, torch.tensor(selected)).to(device)
                    row_ids = dataset_rows.tolist()
                else:
                    coefficient = pack["ols_coefficients"].index_select(0, torch.tensor(selected)).to(device)
                    saved_views = pack["images"].index_select(0, torch.tensor(selected)).to(device)
                    target_batch = None
                    row_ids = selected
                start_event.record()
                if kind == "DEV":
                    result = base._forward(model, image)
                else:
                    result = model.forward_from_ols(coefficient)
                raw = result["physical_w_per_pixel"]
                gain = inputs["fixed"]["camera_gain"].to(device=device, dtype=torch.float32)
                replay_views = resize_view(raw * gain)
                replay_native = (raw * gain).clamp(0.0, 1.0)
                with torch.inference_mode():
                    replay_ed = evaluator(replay_views)
                    replay_ec = decoder_ref._ec_logits(ec_evaluator, replay_views)
                if kind == "DEV" and endpoint != "C":
                    with torch.inference_mode():
                        saved_ed = evaluator(saved_views)
                        saved_ec = decoder_ref._ec_logits(ec_evaluator, saved_views)
                else:
                    saved_ed = pack["ed_logits"].index_select(0, torch.tensor(selected)).to(device)
                    saved_ec = pack["ec_logits"].index_select(0, torch.tensor(selected)).to(device)
                end_event.record()
                _require(torch.allclose(replay_views, saved_views, rtol=RTOL, atol=ATOL),
                         f"{endpoint} {kind} view replay failed for selected rows {row_ids}")
                _require(torch.allclose(replay_ed, saved_ed, rtol=RTOL, atol=ATOL)
                         and torch.allclose(replay_ec, saved_ec, rtol=RTOL, atol=ATOL)
                         and torch.equal(replay_ed.argmax(1), saved_ed.argmax(1))
                         and torch.equal(replay_ec.argmax(1), saved_ec.argmax(1)),
                         f"{endpoint} {kind} evaluator replay failed for selected rows {row_ids}")
                for offset, position in enumerate(selected):
                    row = _compare_record(
                        endpoint, kind, position, inputs, saved_views[offset:offset + 1], replay_views[offset:offset + 1],
                        replay_ed[offset:offset + 1], replay_ec[offset:offset + 1],
                        None if target_batch is None else target_batch[offset:offset + 1],
                        None if kind != "DEV" or endpoint == "C" else saved_ed[offset:offset + 1],
                        None if kind != "DEV" or endpoint == "C" else saved_ec[offset:offset + 1],
                    )
                    native_reference = _native_reference(pack, kind, position)
                    if native_reference is None:
                        row["native_reference_status"] = "MISSING_HISTORICAL_REFERENCE"
                        native_missing_keys.append([kind, int(position)])
                    else:
                        native_reference = native_reference.to(device=device, dtype=replay_native.dtype)
                        native_current = replay_native[offset]
                        native_error = _max_abs(native_current, native_reference)
                        native_allclose = bool(torch.allclose(
                            native_current, native_reference, rtol=RTOL, atol=ATOL,
                        ))
                        row.update({"native_reference_status": "MATCHED",
                                    "native_max_abs_error": native_error,
                                    "native_allclose": native_allclose})
                        _require(native_allclose,
                                 f"{endpoint} {kind} historical 200x200 view replay failed at {position}")
                        native_row_keys.append([kind, int(position)])
                        native_saved_all.append(native_reference.detach().cpu().clone())
                        native_replay_all.append(native_current.detach().cpu().clone())
                    _require(row["view_allclose"] and row["ed_logits_allclose"] and row["ec_logits_allclose"]
                             and row["predictions_match_saved"],
                             f"{endpoint} {kind} row comparison failed at position {position}")
                    if kind == "DEV":
                        _require(all(row["saved_metrics_match"].values()),
                                 f"{endpoint} DEV saved per-row metric differs at dataset row {row['dataset_row']}")
                        _require(np.allclose([row["ssim_float64"]], [row["ssim_numpy"]],
                                             rtol=1e-10, atol=1e-12),
                                 f"{endpoint} DEV float64 SSIM independent reference differs")
                    elif "g4_saved_metric_match" in row:
                        _require(all(row["g4_saved_metric_match"].values()),
                                 f"{endpoint} G4 saved per-row evaluator metric differs at cohort row {position}")
                    endpoint_rows.append(row)
                saved_views_all.append(saved_views.detach().cpu().clone())
                replay_views_all.append(replay_views.detach().cpu().clone())
                saved_ed_all.append(saved_ed.detach().cpu().clone())
                replay_ed_all.append(replay_ed.detach().cpu().clone())
                saved_ec_all.append(saved_ec.detach().cpu().clone())
                replay_ec_all.append(replay_ec.detach().cpu().clone())
                replay_completed += len(selected)
                _write_task_progress(output, "REPLAY", "RUNNING", replay_completed, 120)
                events.append((f"{kind}_chunk_{chunk_index + 1}", len(selected), start_event, end_event))
        cuda_events = _close_cuda_events(events, device)
        tensor_evidence = {
            "schema": "lightgen_gaussian_replay_tensors.v1", "endpoint": endpoint,
            "rows": _replay_tensor_identity(endpoint_rows),
            "saved_views": torch.cat(saved_views_all), "replay_views": torch.cat(replay_views_all),
            "saved_ed_logits": torch.cat(saved_ed_all), "replay_ed_logits": torch.cat(replay_ed_all),
            "saved_ec_logits": torch.cat(saved_ec_all), "replay_ec_logits": torch.cat(replay_ec_all),
            "native": {
                "row_keys": native_row_keys, "missing_row_keys": native_missing_keys,
                "saved": (torch.stack(native_saved_all) if native_saved_all else
                          torch.empty((0, 1, 200, 200), dtype=torch.float32)),
                "replay": (torch.stack(native_replay_all) if native_replay_all else
                           torch.empty((0, 1, 200, 200), dtype=torch.float32)),
            },
        }
        _validate_replay_tensor_evidence(tensor_evidence, endpoint_rows, endpoint)
        _validate_replay_tensor_source_arrays(tensor_evidence, endpoint_rows, inputs, endpoint)
        atomic_torch_save_new(stage / "replay_tensors.pt", tensor_evidence)
        summary = {
            "schema": "lightgen_gaussian_production_replay_endpoint.v1",
            "status": "PASSED", "endpoint": endpoint,
            "checkpoint_global_step": int(record["checkpoint"]["completed_global_step"]),
            "production_grid": 2048, "device": torch.cuda.get_device_name(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
            "rows_by_kind": {"DEV": 20, "G4": 20}, "cuda_events": cuda_events,
            "cuda_event_scope": "CUDA model propagation, readout, and frozen evaluator forward; excludes CPU row checks",
            "ed_replay_rows": 40, "ec_replay_rows": 40,
            "saved_view_evaluator_reference_rows": 20 if endpoint in {"D0", "J"} else 0,
            "ed_saved_view_reference_rows": 20 if endpoint in {"D0", "J"} else 0,
            "ec_saved_view_reference_rows": 20 if endpoint in {"D0", "J"} else 0,
            "native_reference_rows": len(native_row_keys),
            "native_reference_missing_rows": len(native_missing_keys),
            "elapsed_seconds": time.perf_counter() - started,
            "rtol": RTOL, "atol": ATOL,
        }
        _write_csv_new(stage / "replay_rows.csv", endpoint_rows)
        _write_json(stage / "replay_summary.json", summary)
        stage_summaries[endpoint] = summary
        rows_out.extend(endpoint_rows)
        del model
        torch.cuda.empty_cache()
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "E_D_replay") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_replay") == 0.0,
             "frozen evaluator state changed during production replay")
    _validate_replay_rows(rows_out, _expected_replay_rows(inputs))
    result = {
        "schema": "lightgen_gaussian_production_replay.v1", "status": "PASSED",
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1", "rows": len(rows_out),
        "device": torch.cuda.get_device_name(device), "endpoints": stage_summaries,
        "evaluator_frozen_after_replay": True,
        "ed_rows": sum(item["ed_replay_rows"] for item in stage_summaries.values()),
        "ec_rows": sum(item["ec_replay_rows"] for item in stage_summaries.values()),
        "ed_saved_view_reference_rows": sum(
            item["ed_saved_view_reference_rows"] for item in stage_summaries.values()),
        "ec_saved_view_reference_rows": sum(
            item["ec_saved_view_reference_rows"] for item in stage_summaries.values()),
        "ed_total_forward_rows": sum(item["ed_replay_rows"]
                                      + item["ed_saved_view_reference_rows"]
                                      for item in stage_summaries.values()),
        "ec_total_forward_rows": sum(item["ec_replay_rows"]
                                      + item["ec_saved_view_reference_rows"]
                                      for item in stage_summaries.values()),
    }
    result_path = output / "replay_result.json"
    if result_path.is_file():
        _require(_read_json(result_path) == result, "replay summary changed on resume")
    else:
        _write_json(result_path, result)
    _write_task_progress(output, "REPLAY", "PASSED", 120, 120)
    return result


def _run_c_preflight(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    stage_root = output / "engineering" / "c_preflight"
    complete = _execution_stage(stage_root, "c_preflight_result.json")
    if complete is not None:
        result = _read_json(complete / "c_preflight_result.json")
        _require(result.get("status") == "PASSED" and result.get("engineering_adam_calls") == 5,
                 "saved C preflight result is incomplete")
        return result
    stage = _new_stage_dir(stage_root)
    _write_task_progress(output, "C_PREFLIGHT", "RUNNING", 0, 5, 45_000)
    config = inputs["joint_config"]
    source_path = ROOT / config["source"]["checkpoint"]
    control_dir = ROOT / config["control"]["evidence_dir"]
    cohort_path = ROOT / config["generation"]["cohort"]
    reference_dir = ROOT / config["readability_reference"]["directory"]
    fixed = joint._load_fixed_inputs(config, source_path, control_dir, cohort_path, reference_dir)
    pixels, labels, train_indices, dev_indices = base._load_data(ROOT / config["data"]["training_file"])
    fixed["source"]["training_labels"] = labels
    fixed["source"]["training_indices"] = train_indices
    fixed["source"]["training_pixels"] = pixels
    schedule = decoder_ref._schedule(labels, train_indices)
    _require(tuple(schedule.shape) == (50_000, 10), "C preflight reconstructed schedule has the wrong shape")
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for the C production preflight")
    ed, ec = _load_evaluators(config, device)
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    started = time.perf_counter()
    args = argparse.Namespace()
    preflight, control, receipt, identity = joint._run_preflight(
        args, config, ROOT / JOINT_CONFIG, fixed, pixels, labels, train_indices, dev_indices,
        schedule, device, ed, ec, stage,
    )
    end_event.record()
    torch.cuda.synchronize(device)
    _require(preflight.get("status") == "READY" and preflight.get("engineering_adam_calls") == 5
             and preflight.get("preflight_rolled_back") is True
             and preflight.get("frozen_evaluators_unchanged") is True,
             "C production preflight did not satisfy its engineering contract")
    _require(len(preflight["D0_saved_replay"]["dev"]) == 10
             and len(preflight["D0_saved_replay"]["generation"]) == 10,
             "C production preflight evaluator reference batches changed")
    receipt_dict = receipt.to_dict()
    result = {
        "schema": "lightgen_gaussian_c_preflight.v1", "status": "PASSED",
        "task_id": "LIGHTGEN_SMALL_JOINT_UPDATE_READABILITY_PILOT_V1",
        "run_identity": identity.to_dict(), "preflight": preflight,
        "ready_receipt": receipt_dict,
        "engineering_adam_calls": 5, "scientific_updates": 0,
        "official_test_reads": 0,
        "evaluator_forward_rows_by_role": {
            "E_D": {"current_DEV": 10, "saved_view_DEV_reference": 10,
                    "current_G4": 10, "total": 30},
            "E_C": {"current_DEV": 10, "saved_view_DEV_reference": 10,
                    "current_G4": 10, "total": 30},
        },
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "elapsed_event_ms": float(start_event.elapsed_time(end_event)),
                 "event_scope": "CUDA timeline around production preflight; includes independent-process recovery wait",
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())},
        "wall_elapsed_seconds": time.perf_counter() - started,
        "staging_dir": str(stage.relative_to(ROOT)),
    }
    _write_json(stage / "c_preflight_result.json", result)
    _write_task_progress(output, "C_PREFLIGHT", "PASSED", 5, 5, 45_004)
    return result


def _optimizer_phase_steps(payload: dict[str, Any]) -> tuple[int, int]:
    optimizer = payload.get("optimizer")
    _require(isinstance(optimizer, dict), "checkpoint optimizer state is missing")
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    _require(len(groups) == 1 and len(groups[0].get("params", [])) == 2
             and len(states) == 2, "checkpoint must contain both Encoder and Decoder Adam states")
    steps = []
    for parameter_id in groups[0]["params"]:
        state = states.get(parameter_id, states.get(str(parameter_id)))
        _require(isinstance(state, dict) and "step" in state, "one Adam phase state is missing")
        steps.append(int(torch.as_tensor(state["step"]).item()))
    declared = payload.get("optimizer_steps")
    _require(declared == {"encoder_adam_step": steps[0], "decoder_adam_step": steps[1]},
             "declared optimizer steps do not match stored phase states")
    return steps[0], steps[1]


def _validate_j_cursor(schedule: torch.Tensor, start: int, labels: torch.Tensor,
                       train_rows: torch.Tensor, count: int = 4) -> torch.Tensor:
    _require((start, count) in {(2_500, 4), (2_503, 1)}
             and tuple(schedule.shape) == (5_000, 10),
             "J recovery cursor or fixed schedule shape changed")
    rows = schedule[start:start + count]
    _require(tuple(rows.shape) == (count, 10)
             and torch.equal(labels[rows], torch.arange(10).expand(count, 10))
             and bool(torch.isin(rows, train_rows).all()),
             "J next four recovery batches are not the registered TRAIN rows")
    return rows


def _assert_only_phase_groups_changed(before: dict[str, torch.Tensor],
                                      after: dict[str, torch.Tensor]) -> None:
    _require(before.keys() == after.keys(), "model state keys changed across engineering updates")
    for name in before:
        if name in PHASE_KEYS:
            _require(not torch.equal(before[name], after[name]),
                     f"declared trainable phase did not update: {name}")
        else:
            _require(torch.equal(before[name], after[name]),
                     f"frozen model group changed during recovery: {name}")


def _validate_j_snapshot(payload: dict[str, Any], source: dict[str, Any],
                          config: dict[str, Any], original: dict[str, Any]) -> int:
    completed = joint._validate_checkpoint(payload, payload.get("run_id", ""), source,
                                           config, original["schedule"])
    _require(completed == 2_503 and payload.get("completed_global_step") == 47_503
             and payload.get("origin_checkpoint_run_id") == original.get("run_id")
             and payload.get("origin_checkpoint_global_step") == 47_500
             and payload.get("engineering_updates_from_origin") == 3,
             "J saved engineering snapshot lost its true global checkpoint position")
    _require(_optimizer_phase_steps(payload) == (47_503, 47_503),
             "J snapshot Adam counters do not match global step 47,503")
    for key in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(payload[key].cpu(), source["checkpoint"][key].cpu()),
                 f"J recovery changed frozen source reference {key}")
    return completed


def _worker_update(staging_dir: Path, config_path: Path, previous_config_path: Path) -> None:
    config = joint._read_config(config_path)
    previous_config = decoder_ref._read_config(previous_config_path)
    snapshot = _torch_load(staging_dir / "checkpoint_step_002503.pt")
    original = _torch_load(staging_dir / "j_original_checkpoint_metadata.pt")
    source_checkpoint = _torch_load(ROOT / config["source"]["checkpoint"])
    source = {"checkpoint": source_checkpoint}
    _validate_j_snapshot(snapshot, source, config, original)
    _require(torch.cuda.is_available(), "independent J restoration requires CUDA")
    device = torch.device("cuda")
    model, optimizer = joint._restore_joint_state(snapshot, previous_config, device)
    torch.cuda.set_rng_state_all([item.cpu() for item in snapshot["cuda_rng"]])
    torch.random.set_rng_state(snapshot["cpu_rng"].cpu())
    pixels, labels, train_rows, _dev = base._load_data(ROOT / config["data"]["training_file"])
    source["training_labels"] = labels
    source["training_indices"] = train_rows
    source["training_pixels"] = pixels
    rows = _validate_j_cursor(snapshot["schedule"], 2_503, labels, train_rows, count=1)
    update_log = _read_json(staging_dir / "engineering_update_log.json")
    calls = update_log.get("calls")
    _require(isinstance(calls, list) and len(calls) == 5
             and all(call.get("status") == "COMPLETED" for call in calls[:4])
             and calls[4].get("status") == "CALL_STARTED"
             and calls[4].get("branch") == "independent_process_fourth"
             and calls[4].get("sample_rows") == [int(value) for value in rows[0].tolist()],
             "independent J worker requires four completed calls and one matching durable start record")
    call = calls[4]
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    loss = joint._joint_update(model, optimizer, pixels, rows[0], device, source)
    end_event.record()
    torch.cuda.synchronize(device)
    encoder, decoder = base._phase_groups(model)
    result_path = staging_dir / "recovery_fourth_update.pt"
    result = {
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "origin_checkpoint_run_id": original["run_id"],
        "completed_global_step": 47_504,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": _optimizer_steps_by_phase(optimizer, model),
        "loss": loss,
        "schedule_cursor": 2_503, "rows": rows[0].detach().cpu().clone(),
        "cpu_rng": torch.random.get_rng_state().cpu().clone(),
        "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "elapsed_event_ms": float(start_event.elapsed_time(end_event)),
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())},
    }
    atomic_torch_save_new(result_path, result)
    call.update({"global_step_after": 47_504, "optimizer_steps": result["optimizer_steps"],
                 "loss": loss, "elapsed_event_ms": result["cuda"]["elapsed_event_ms"],
                 "peak_reserved_vram_bytes": result["cuda"]["peak_reserved_vram_bytes"]})
    _complete_logged_update(staging_dir, calls, call)


def _finalize_j_recovery_existing_stage(
    inputs: dict[str, Any], output: Path, stage: Path,
) -> dict[str, Any]:
    finalization_started = time.perf_counter()
    config = inputs["joint_config"]
    fixed = inputs["fixed"]
    source = fixed["source"]
    original_path = ROOT / inputs["j_config"]["inputs"]["joint_optical_dir"] / "checkpoint_step_002500.pt"
    original = _torch_load(original_path)
    metadata = _torch_load(stage / "j_original_checkpoint_metadata.pt")
    _require(original.get("run_id") == metadata.get("run_id")
             and original.get("completed_global_step") == metadata.get("completed_global_step") == 47_500
             and original.get("completed_updates") == metadata.get("completed_updates") == 2_500
             and original.get("optimizer_steps") == metadata.get("optimizer_steps")
             and torch.equal(original["schedule"], metadata["schedule"])
             and original.get("history") == metadata.get("history"),
             "J source checkpoint differs from its persisted pre-update metadata")
    rows = _validate_j_cursor(original["schedule"], 2_500,
                              inputs["labels"], inputs["train_rows"])
    _require(_optimizer_phase_steps(original) == (47_500, 47_500),
             "J original checkpoint is missing a real 47,500-step Adam phase state")
    joint._validate_checkpoint(original, original["run_id"], source, config, original["schedule"])
    for name, value in source["checkpoint"]["model"].items():
        if name not in PHASE_KEYS:
            _require(torch.equal(value, original["model"][name]),
                     f"original J checkpoint changed frozen model state {name}")
    for key in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(original[key], source["checkpoint"][key]),
                 f"original J checkpoint changed frozen reference {key}")
    _require(original.get("weights") == source["checkpoint"].get("weights")
             and original.get("structural_weight") == source["checkpoint"].get("structural_weight"),
             "original J checkpoint changed frozen loss references")

    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for J recovery evidence finalization")
    previous_config = fixed["baseline_cfg"]
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    try:
        restored_model, restored_optimizer = joint._restore_joint_state(
            original, previous_config, device,
        )
        zero_model_error = base._assert_close(
            original["model"], base._state_snapshot(restored_model), "J_resume_zero_update_model",
        )
        zero_optimizer_error = base._assert_close(
            original["optimizer"], cpu_copy(restored_optimizer.state_dict()),
            "J_resume_zero_update_optimizer",
        )
    finally:
        torch.random.set_rng_state(cpu_rng_before)
        torch.cuda.set_rng_state_all(cuda_rng_before)
    zero_steps = list(_optimizer_phase_steps(original))
    _require(zero_model_error == 0.0 and zero_optimizer_error == 0.0
             and zero_steps == [47_500, 47_500],
             "J finalization-only restore did not exactly reload the original model and Adam state")

    snapshot = _torch_load(stage / "checkpoint_step_002503.pt")
    _validate_j_snapshot(snapshot, source, config, original)
    _require(snapshot["history"] == original["history"]
             and _optimizer_phase_steps(snapshot) == (47_503, 47_503),
             "persisted J 47,503 snapshot changed history or Adam phase state")
    direct = _torch_load(stage / "direct_fourth_update.pt")
    recovered = _torch_load(stage / "recovery_fourth_update.pt")
    expected_rows = [[int(value) for value in row.tolist()] for row in rows]
    log_path = stage / "engineering_update_log.json"
    log = _read_json(log_path)
    calls = _validate_recovery_update_log(
        log, "calls",
        ["original_process", "original_process", "original_process",
         "original_process_fourth", "independent_process_fourth"],
        47_501, expected_rows + [expected_rows[3]],
        [2_500, 2_501, 2_502, 2_503, 2_503],
    )
    schedule_row = _validate_j_cursor(
        snapshot["schedule"], 2_503, inputs["labels"], inputs["train_rows"], count=1,
    )
    _require([int(value) for value in schedule_row[0].tolist()] == expected_rows[3]
             and direct["rows"].tolist() == expected_rows[3]
             and recovered["rows"].tolist() == expected_rows[3],
             "persisted J fourth-step rows differ from the original schedule")
    pair = _validate_recovery_pair(
        direct, recovered,
        {"encoder_adam_step": 47_504, "decoder_adam_step": 47_504},
        rows[3], {"task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
                  "origin_checkpoint_run_id": original["run_id"],
                  "completed_global_step": 47_504, "schedule_cursor": 2_503},
    )
    _require(calls[3]["optimizer_steps"] == direct["optimizer_steps"]
             and calls[4]["optimizer_steps"] == recovered["optimizer_steps"]
             and base._assert_close(calls[3]["loss"], direct["loss"],
                                    "J_resume_direct_log_loss") <= RECOVERY_ATOL
             and base._assert_close(calls[4]["loss"], recovered["loss"],
                                    "J_resume_worker_log_loss") <= RECOVERY_ATOL
             and calls[3]["elapsed_event_ms"] == direct["cuda"]["elapsed_event_ms"]
             and calls[4]["elapsed_event_ms"] == recovered["cuda"]["elapsed_event_ms"],
             "persisted J fourth-step artifacts differ from the durable call log")
    disk_after = _torch_load(original_path)
    _require(disk_after.get("run_id") == original.get("run_id")
             and base._assert_close(disk_after["model"], original["model"],
                                    "J_resume_source_disk_model") == 0.0
             and base._assert_close(disk_after["optimizer"], original["optimizer"],
                                    "J_resume_source_disk_optimizer") == 0.0
             and disk_after.get("optimizer_steps") == original.get("optimizer_steps"),
             "original J checkpoint changed on disk during finalization-only restore")
    cuda_events = [float(call["elapsed_event_ms"]) for call in calls]
    artifact_span_seconds = max(0.0, log_path.stat().st_mtime -
                                (stage / "j_original_checkpoint_metadata.pt").stat().st_mtime)
    result = {
        "schema": "lightgen_gaussian_j_recovery.v1", "status": "PASSED",
        "source_checkpoint_global_step": 47_500,
        "source_checkpoint_run_id": original["run_id"],
        "next_four_schedule_cursors": [2_500, 2_501, 2_502, 2_503],
        "next_four_sample_rows": expected_rows,
        "snapshot_global_step": 47_503,
        "snapshot_optimizer_steps": list(_optimizer_phase_steps(snapshot)),
        "zero_update_restore": {"model_max_abs_error": zero_model_error,
                                 "optimizer_max_abs_error": zero_optimizer_error,
                                 "optimizer_steps": zero_steps},
        "snapshot_historical_history_rows": len(snapshot["history"]),
        "engineering_update_log": calls,
        "direct_fourth_global_step": 47_504,
        "independent_fourth_global_step": recovered["completed_global_step"],
        "independent_recovery": {"passed": True,
                                 "model_max_abs_error": pair["model_max_abs_error"],
                                 "optimizer_max_abs_error": pair["optimizer_max_abs_error"],
                                 "loss_max_abs_error": pair["loss_max_abs_error"],
                                 "tolerance": RECOVERY_ATOL},
        "cuda": {"device": calls[0]["device"],
                 "three_update_event_ms": cuda_events[:3],
                 "direct_fourth_event_ms": cuda_events[3],
                 "independent_fourth": recovered["cuda"],
                 "peak_reserved_vram_bytes": max(
                     int(call["peak_reserved_vram_bytes"]) for call in calls)},
        "engineering_adam_calls": 5, "scientific_updates": 0,
        "official_test_reads": 0, "recovery_finalized_from_existing_updates": True,
        "elapsed_seconds": artifact_span_seconds,
        "elapsed_seconds_source": "persisted_stage_file_timestamp_span",
        "finalization_elapsed_seconds": time.perf_counter() - finalization_started,
    }
    _write_json(stage / "j_recovery_result.json", result)
    _write_json(output / "engineering" / "j_recovery_finalization_receipt.json", {
        "schema": "lightgen_gaussian_j_recovery_finalization.v1",
        "status": "COMPLETED_WITHOUT_NEW_UPDATES", "stage": str(stage.relative_to(output)),
        "existing_completed_calls_consumed": 5, "additional_adam_calls": 0,
        "source_checkpoint_reloaded_from_disk": True,
        "source_model_max_abs_error": 0.0, "source_optimizer_max_abs_error": 0.0,
        "recovery_model_max_abs_error": pair["model_max_abs_error"],
        "recovery_optimizer_max_abs_error": pair["optimizer_max_abs_error"],
    })
    _write_task_progress(output, "J_RECOVERY", "PASSED", 5, 5, 47_504)
    return result


def _run_j_recovery(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    stage_root = output / "engineering" / "j_recovery"
    existing_stages = sorted(path for path in stage_root.glob("attempt_*") if path.is_dir())
    if existing_stages:
        _require(len(existing_stages) == 1,
                 "multiple incomplete J recovery stages exist; refusing to select one automatically")
        stage = existing_stages[0]
        result_path = stage / "j_recovery_result.json"
        if result_path.is_file():
            result = _read_json(result_path)
            _require(result.get("status") == "PASSED" and result.get("engineering_adam_calls") == 5,
                     "saved J recovery result is incomplete")
            return result
        log_path = stage / "engineering_update_log.json"
        _require(log_path.is_file(),
                 "incomplete J recovery stage has no durable update log; refusing to repeat updates")
        log = _read_json(log_path)
        calls = log.get("calls")
        _require(isinstance(calls, list) and len(calls) == 5
                 and all(call.get("status") == "COMPLETED" for call in calls),
                 "incomplete J recovery stage cannot be resumed without repeating Adam calls")
        _write_task_progress(output, "J_RECOVERY", "RUNNING", 5, 5, 47_503)
        return _finalize_j_recovery_existing_stage(inputs, output, stage)
    stage = _new_stage_dir(stage_root)
    _write_task_progress(output, "J_RECOVERY", "RUNNING", 0, 5, 47_500)
    config = inputs["joint_config"]
    fixed = inputs["fixed"]
    source = fixed["source"]
    original_path = (ROOT / inputs["j_config"]["inputs"]["joint_optical_dir"]
                     / "checkpoint_step_002500.pt")
    original = _torch_load(original_path)
    _require(original.get("completed_global_step") == 47_500
             and original.get("completed_updates") == 2_500
             and original.get("schedule_start_global_step") == 45_000,
             "explicit J checkpoint is not the registered global 47,500 state")
    schedule = original["schedule"]
    rows = _validate_j_cursor(schedule, 2_500, inputs["labels"], inputs["train_rows"])
    _require(_optimizer_phase_steps(original) == (47_500, 47_500),
             "original J checkpoint is missing a real 47,500-step Adam phase state")
    joint._validate_checkpoint(original, original["run_id"], source, config, schedule)
    frozen_before = source["checkpoint"]["model"]
    for name in frozen_before:
        if name not in PHASE_KEYS:
            _require(torch.equal(frozen_before[name], original["model"][name]),
                     f"original J checkpoint changed frozen model state {name}")
    for key in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(original[key], source["checkpoint"][key]),
                 f"original J checkpoint changed frozen reference {key}")
    _require(original.get("weights") == source["checkpoint"].get("weights")
             and original.get("structural_weight") == source["checkpoint"].get("structural_weight"),
             "original J checkpoint changed frozen loss references")
    metadata_path = stage / "j_original_checkpoint_metadata.pt"
    atomic_torch_save_new(metadata_path, {
        "run_id": original["run_id"], "completed_global_step": original["completed_global_step"],
        "completed_updates": original["completed_updates"], "schedule": original["schedule"],
        "history": original["history"], "optimizer_steps": original["optimizer_steps"],
    })
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for J checkpoint restoration")
    previous_config = fixed["baseline_cfg"]
    model, optimizer = joint._restore_joint_state(original, previous_config, device)
    model_before = base._state_snapshot(model)
    optimizer_before = cpu_copy(optimizer.state_dict())
    zero_update_model_error = base._assert_close(
        original["model"], model_before, "J_zero_update_loaded_model",
    )
    zero_update_optimizer_error = base._assert_close(
        original["optimizer"], optimizer_before, "J_zero_update_loaded_optimizer",
    )
    zero_update_steps = list(_optimizer_phase_steps(original))
    _require(zero_update_model_error == 0.0 and zero_update_optimizer_error == 0.0
             and zero_update_steps == [47_500, 47_500],
             "J zero-update load did not restore the exact model and both Adam states")
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    torch.cuda.reset_peak_memory_stats()
    torch.random.set_rng_state(original["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in original["cuda_rng"]])
    update_events = []
    update_log: list[dict[str, Any]] = []
    _write_update_log(stage, update_log)
    started = time.perf_counter()
    for offset in range(3):
        call = {"call_number": offset + 1, "branch": "original_process",
                "status": "CALL_STARTED", "schedule_cursor": 2_500 + offset,
                "global_step_before": 47_500 + offset,
                "sample_rows": [int(value) for value in rows[offset].tolist()],
                "device": torch.cuda.get_device_name(device)}
        _start_logged_update(stage, update_log, call)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        loss = joint._joint_update(model, optimizer, inputs["pixels"], rows[offset], device, source)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start_event.elapsed_time(end_event))
        call.update({"global_step_after": 47_501 + offset,
                     "optimizer_steps": _optimizer_steps_by_phase(optimizer, model),
                     "loss": loss, "elapsed_event_ms": elapsed_ms,
                     "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))})
        _complete_logged_update(stage, update_log, call)
        update_events.append(elapsed_ms)
        _write_task_progress(output, "J_RECOVERY", "RUNNING", offset + 1, 5,
                             47_501 + offset)
    encoder, decoder = base._phase_groups(model)
    _require(joint._optimizer_steps(optimizer, encoder, decoder) == (47_503, 47_503),
             "three J engineering updates did not advance both Adam states to 47,503")
    snapshot = joint._checkpoint_payload(
        model, optimizer, 2_503, uuid.uuid4().hex, source, config, schedule,
        list(original["history"]),
        float(original.get("elapsed_seconds_continuation", 0.0)) + time.perf_counter() - started,
    )
    snapshot.update({"origin_checkpoint_run_id": original["run_id"],
                     "origin_checkpoint_global_step": 47_500,
                     "engineering_updates_from_origin": 3})
    _assert_only_phase_groups_changed(model_before, snapshot["model"])
    snapshot_path = stage / "checkpoint_step_002503.pt"
    atomic_torch_save_new(snapshot_path, snapshot)
    reread_snapshot = _torch_load(snapshot_path)
    _validate_j_snapshot(reread_snapshot, source, config, original)
    _require(reread_snapshot["history"] == original["history"],
             "J engineering recovery rewrote historical training history")
    _require(base._assert_close(snapshot["model"], reread_snapshot["model"],
                                "J_engineering_snapshot_model") == 0.0
             and base._assert_close(snapshot["optimizer"], reread_snapshot["optimizer"],
                                    "J_engineering_snapshot_optimizer") == 0.0,
             "J complete engineering snapshot failed serialized readback")
    call = {"call_number": 4, "branch": "original_process_fourth", "status": "CALL_STARTED",
            "schedule_cursor": 2_503, "global_step_before": 47_503,
            "sample_rows": [int(value) for value in rows[3].tolist()],
            "device": torch.cuda.get_device_name(device)}
    _start_logged_update(stage, update_log, call)
    direct_start, direct_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    direct_start.record()
    direct_loss = joint._joint_update(model, optimizer, inputs["pixels"], rows[3], device, source)
    direct_end.record()
    torch.cuda.synchronize(device)
    direct_model = base._state_snapshot(model)
    direct_optimizer = cpu_copy(optimizer.state_dict())
    direct_steps = _optimizer_steps_by_phase(optimizer, model)
    direct_event_ms = float(direct_start.elapsed_time(direct_end))
    _write_task_progress(output, "J_RECOVERY", "RUNNING", 4, 5, 47_504)
    direct_path = stage / "direct_fourth_update.pt"
    atomic_torch_save_new(direct_path, {
        "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "origin_checkpoint_run_id": original["run_id"],
        "completed_global_step": 47_504,
        "model": direct_model,
        "optimizer": direct_optimizer,
        "optimizer_steps": direct_steps,
        "loss": direct_loss,
        "schedule_cursor": 2_503, "rows": rows[3].detach().cpu().clone(),
        "cpu_rng": torch.random.get_rng_state().cpu().clone(),
        "cuda_rng": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "elapsed_event_ms": direct_event_ms,
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())},
    })
    call.update({"global_step_after": 47_504, "optimizer_steps": direct_steps,
                 "loss": direct_loss, "elapsed_event_ms": direct_event_ms,
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device))})
    _complete_logged_update(stage, update_log, call)
    worker_call = {"call_number": 5, "branch": "independent_process_fourth",
                   "status": "CALL_STARTED", "schedule_cursor": 2_503,
                   "global_step_before": 47_503,
                   "sample_rows": [int(value) for value in rows[3].tolist()],
                   "device": torch.cuda.get_device_name(device)}
    _start_logged_update(stage, update_log, worker_call)
    _run_j_worker(stage, [sys.executable, "-m",
                          "scripts.research.lightgen.verify_lightgen_gaussian_production_replay_v1",
                          "--j-recovery-worker-dir", str(stage), "--config", str(ROOT / JOINT_CONFIG),
                          "--previous-config", str(ROOT / config["control"]["previous_config"])])
    recovered = _torch_load(stage / "recovery_fourth_update.pt")
    direct = _torch_load(direct_path)
    pair_errors = _validate_recovery_pair(
        direct, recovered, {"encoder_adam_step": 47_504, "decoder_adam_step": 47_504},
        rows[3], {"task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
                  "origin_checkpoint_run_id": original["run_id"],
                  "completed_global_step": 47_504, "schedule_cursor": 2_503},
    )
    model_error = pair_errors["model_max_abs_error"]
    optimizer_error = pair_errors["optimizer_max_abs_error"]
    loss_error = pair_errors["loss_max_abs_error"]
    update_log = _read_json(stage / "engineering_update_log.json")["calls"]
    _require(len(update_log) == 5 and all(call.get("status") == "COMPLETED" for call in update_log),
             "J per-update durable log is incomplete after independent recovery")
    _require(_optimizer_phase_steps(reread_snapshot) == (47_503, 47_503),
             "saved J snapshot does not retain both Adam phase states")
    torch.random.set_rng_state(cpu_rng_before)
    torch.cuda.set_rng_state_all(cuda_rng_before)
    original_after = _torch_load(original_path)
    _require(original_after.get("run_id") == original.get("run_id")
             and original_after.get("optimizer_steps") == original.get("optimizer_steps")
             and base._assert_close(original_after["model"], original["model"],
                                    "J_source_disk_model_after_updates") == 0.0
             and base._assert_close(original_after["optimizer"], original["optimizer"],
                                    "J_source_disk_optimizer_after_updates") == 0.0,
             "J source checkpoint changed on disk or in memory during recovery")
    restored_model, restored_optimizer = joint._restore_joint_state(
        original_after, previous_config, device,
    )
    _require(base._assert_close(model_before, base._state_snapshot(restored_model),
                                "J_original_restore_model") == 0.0
             and base._assert_close(optimizer_before, cpu_copy(restored_optimizer.state_dict()),
                                    "J_original_restore_optimizer") == 0.0,
             "J source model or optimizer state changed during recovery")
    result = {
        "schema": "lightgen_gaussian_j_recovery.v1", "status": "PASSED",
        "source_checkpoint_global_step": 47_500,
        "source_checkpoint_run_id": original["run_id"],
        "next_four_schedule_cursors": [2_500, 2_501, 2_502, 2_503],
        "next_four_sample_rows": [[int(x) for x in row.tolist()] for row in rows],
        "snapshot_global_step": 47_503,
        "snapshot_optimizer_steps": list(_optimizer_phase_steps(reread_snapshot)),
        "zero_update_restore": {"model_max_abs_error": zero_update_model_error,
                                 "optimizer_max_abs_error": zero_update_optimizer_error,
                                 "optimizer_steps": zero_update_steps},
        "snapshot_historical_history_rows": len(reread_snapshot["history"]),
        "engineering_update_log": update_log,
        "direct_fourth_global_step": 47_504,
        "independent_fourth_global_step": recovered["completed_global_step"],
        "independent_recovery": {"passed": True, "model_max_abs_error": model_error,
                                 "optimizer_max_abs_error": optimizer_error,
                                 "loss_max_abs_error": loss_error,
                                 "tolerance": RECOVERY_ATOL},
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "three_update_event_ms": update_events,
                 "direct_fourth_event_ms": direct_event_ms,
                 "independent_fourth": recovered["cuda"],
                 "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved())},
        "engineering_adam_calls": 5, "scientific_updates": 0,
        "official_test_reads": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(stage / "j_recovery_result.json", result)
    _write_task_progress(output, "J_RECOVERY", "PASSED", 5, 5, 47_504)
    return result


def _validate_cuda_evidence(summary: dict[str, Any]) -> None:
    events = summary.get("cuda_events")
    _require(summary.get("status") == "PASSED"
             and isinstance(summary.get("device_name"), str) and summary["device_name"]
             and isinstance(events, list) and len(events) == 4,
             "replay CUDA evidence is incomplete")
    _require(summary.get("cuda_event_scope")
             == "CUDA model propagation, readout, and frozen evaluator forward; excludes CPU row checks",
             "replay CUDA event scope is missing or ambiguous")
    names = {event.get("name") for event in events}
    _require(names == {"DEV_chunk_1", "DEV_chunk_2", "G4_chunk_1", "G4_chunk_2"},
             "replay CUDA event names do not cover all production batches")
    for event in events:
        _require(event.get("device") == summary["device_name"]
                 and isinstance(event.get("elapsed_ms"), (int, float))
                 and math.isfinite(float(event["elapsed_ms"])) and event["elapsed_ms"] > 0
                 and event.get("rows") == 10,
                 "replay CUDA event has missing or invalid device/timing/row evidence")


def _validate_recovery_update_log(
    payload: dict[str, Any], entry_key: str, expected_branches: list[str],
    first_global_step: int, expected_rows: list[list[int]],
    expected_schedule_cursors: list[int],
) -> list[dict[str, Any]]:
    entries = payload.get(entry_key)
    _require(isinstance(entries, list) and len(entries) == 5
             and len(expected_rows) == 5 and len(expected_schedule_cursors) == 5,
             "recovery update log must contain exactly five calls")
    for index, (entry, branch, sample_rows) in enumerate(
            zip(entries, expected_branches, expected_rows, strict=True)):
        global_after = first_global_step + index
        if index == 4:
            global_after = first_global_step + 3
        global_before = global_after - 1
        expected_steps = {"encoder_adam_step": global_after,
                          "decoder_adam_step": global_after}
        loss = entry.get("loss")
        cuda_ms = entry.get("elapsed_event_ms")
        _require(entry.get("call_number") == index + 1
                 and entry.get("branch") == branch
                 and entry.get("status") == "COMPLETED"
                 and entry.get("global_step_before") == global_before
                 and entry.get("global_step_after") == global_after
                 and entry.get("schedule_cursor") == expected_schedule_cursors[index]
                 and entry.get("optimizer_steps") == expected_steps
                 and entry.get("sample_rows") == sample_rows
                 and isinstance(entry.get("device"), str) and entry["device"]
                 and isinstance(cuda_ms, (int, float)) and math.isfinite(float(cuda_ms))
                 and float(cuda_ms) > 0
                 and isinstance(entry.get("peak_reserved_vram_bytes"), int)
                 and entry["peak_reserved_vram_bytes"] > 0
                 and isinstance(loss, dict) and loss
                 and all(isinstance(value, (int, float)) and math.isfinite(float(value))
                         for value in loss.values()),
                 f"recovery update log entry {index + 1} is incomplete or invalid")
    return entries


def _validate_recovery_pair(
    direct: dict[str, Any], recovered: dict[str, Any], expected_steps: dict[str, int],
    expected_rows: torch.Tensor, identity: dict[str, Any],
) -> dict[str, float]:
    direct_rows, recovered_rows = direct.get("rows"), recovered.get("rows")
    direct_cpu_rng, recovered_cpu_rng = direct.get("cpu_rng"), recovered.get("cpu_rng")
    direct_cuda_rng, recovered_cuda_rng = direct.get("cuda_rng"), recovered.get("cuda_rng")
    expected_step_tuple = (expected_steps["encoder_adam_step"], expected_steps["decoder_adam_step"])
    _require(all(direct.get(key) == value and recovered.get(key) == value
                 for key, value in identity.items())
             and direct.get("optimizer_steps") == expected_steps
             and recovered.get("optimizer_steps") == expected_steps
             and _optimizer_phase_steps(direct) == expected_step_tuple
             and _optimizer_phase_steps(recovered) == expected_step_tuple
             and isinstance(direct_rows, torch.Tensor) and isinstance(recovered_rows, torch.Tensor)
             and torch.equal(direct_rows, recovered_rows)
             and torch.equal(direct_rows, expected_rows.cpu())
             and isinstance(direct_cpu_rng, torch.Tensor) and isinstance(recovered_cpu_rng, torch.Tensor)
             and torch.equal(direct_cpu_rng, recovered_cpu_rng)
             and isinstance(direct_cuda_rng, list) and isinstance(recovered_cuda_rng, list)
             and len(direct_cuda_rng) == len(recovered_cuda_rng)
             and all(torch.equal(left, right) for left, right in zip(
                 direct_cuda_rng, recovered_cuda_rng, strict=True)),
             "direct and independent recovery identities, rows, optimizer mapping, or RNG differ")
    model_error = base._assert_close(direct["model"], recovered["model"], "recovery_pair_model")
    optimizer_error = base._assert_close(direct["optimizer"], recovered["optimizer"],
                                         "recovery_pair_optimizer")
    loss_error = base._assert_close(direct["loss"], recovered["loss"], "recovery_pair_loss")
    _require(model_error <= RECOVERY_ATOL and optimizer_error <= RECOVERY_ATOL
             and loss_error <= RECOVERY_ATOL,
             "direct and independent recovery states exceed the fixed tolerance")
    return {"model_max_abs_error": model_error, "optimizer_max_abs_error": optimizer_error,
            "loss_max_abs_error": loss_error}


def _run_j_worker(stage: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    log = _read_json(stage / "engineering_update_log.json")
    calls = log.get("calls")
    _require(isinstance(calls, list) and len(calls) == 5
             and all(call.get("status") == "COMPLETED" for call in calls[:4])
             and calls[4].get("status") == "CALL_STARTED",
             "J worker must start only after four completed calls and a durable fifth-call record")
    _require(_read_json(stage / "engineering_update_log.json") == log,
             "J worker pre-start log readback changed")
    worker = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    _require(worker.returncode == 0,
             "independent J recovery process failed: " + (worker.stderr[-2500:] or worker.stdout[-2500:]))
    return worker


def _combined_csv_value_matches(expected: Any, actual: str) -> bool:
    if expected is None:
        return actual == ""
    if isinstance(expected, dict):
        try:
            return json.loads(actual) == expected
        except json.JSONDecodeError:
            return False
    if isinstance(expected, bool):
        return actual.lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except ValueError:
            return False
    return actual == str(expected)


def _verify(inputs: dict[str, Any], output: Path) -> dict[str, Any]:
    run_config = _load_run_config(output)
    expected_config = _build_run_config(inputs, output)
    _require(run_config == expected_config,
             "prepared run configuration differs from the currently loaded explicit inputs")
    _require((output / "input_manifest.json").is_file()
             and (output / "interface_spec.md").is_file(),
             "prepare evidence or interface table is missing")
    _require(_read_json(output / "input_manifest.json") == _build_input_manifest(inputs, run_config)
             and (output / "interface_spec.md").read_text(encoding="utf-8")
                 == _interface_spec(inputs["joint_config"]),
             "prepared input manifest or interface table differs from current inputs")
    prepare_result = _read_json(output / "prepare_result.json")
    _require(prepare_result.get("status") == "PREPARED"
             and prepare_result.get("validated_endpoints") == list(ENDPOINTS)
             and prepare_result.get("replay_items") == 120
             and prepare_result.get("native_reference_rows") == 80
             and prepare_result.get("native_reference_missing_rows") == 40,
             "prepare result does not declare the fixed input and native-reference population")
    expected = _expected_replay_rows(inputs)
    all_rows = []
    replay_summaries = {}
    for endpoint in ENDPOINTS:
        stage = _latest_complete_dir(output / "replay" / endpoint, "replay_summary.json")
        _require(stage is not None, f"complete {endpoint} replay evidence is missing")
        summary = _read_json(stage / "replay_summary.json")
        _validate_cuda_evidence(summary)
        rows = _read_csv(stage / "replay_rows.csv")
        _require(len(rows) == 40 and summary.get("rows_by_kind") == {"DEV": 20, "G4": 20},
                 f"{endpoint} replay output has a missing row")
        for row in rows:
            for key in ("position", "label"):
                row[key] = int(row[key])
            for key in ("dataset_row", "cohort_row", "proposal_id", "component_id"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            if row.get("native_max_abs_error") not in (None, ""):
                row["native_max_abs_error"] = float(row["native_max_abs_error"])
            else:
                row["native_max_abs_error"] = None
            for key in ("view_allclose", "ed_logits_allclose", "ec_logits_allclose",
                        "predictions_match_saved", "native_allclose"):
                if key == "native_allclose" and row.get(key) in (None, ""):
                    row[key] = None
                else:
                    row[key] = row[key].lower() == "true"
            if row["kind"] == "DEV":
                metric = json.loads(row["saved_metrics_match"])
                _require(all(metric.values()), f"{endpoint} DEV source metrics did not match")
                row["saved_metrics_match"] = metric
            elif row.get("g4_saved_metric_match"):
                metric = json.loads(row["g4_saved_metric_match"])
                _require(all(metric.values()), f"{endpoint} G4 source metrics did not match")
                row["g4_saved_metric_match"] = metric
            for key in ("view_allclose", "ed_logits_allclose", "ec_logits_allclose",
                        "predictions_match_saved"):
                _require(row[key], f"{endpoint} saved view/logit comparison failed")
            for error_name, reference_name in (
                ("view_max_abs_error", "view_reference_max_abs"),
                ("ed_logit_max_abs_error", "ed_logit_reference_max_abs"),
                ("ec_logit_max_abs_error", "ec_logit_reference_max_abs"),
            ):
                _require(float(row[error_name]) <= ATOL + RTOL * float(row[reference_name]),
                         f"{endpoint} stored maximum error exceeds the registered allclose bound")
            all_rows.append(row)
        tensors_path = stage / "replay_tensors.pt"
        _require(tensors_path.is_file(), f"{endpoint} replay tensor evidence is missing")
        tensor_payload = _torch_load(tensors_path)
        _validate_replay_tensor_evidence(tensor_payload, rows, endpoint)
        _validate_replay_tensor_source_arrays(tensor_payload, rows, inputs, endpoint)
        _validate_replay_tensor_metrics(tensor_payload, rows, inputs, endpoint)
        native_matched, native_missing = _native_coverage(inputs, endpoint)
        expected_native = {key: "MATCHED" for key in native_matched}
        expected_native.update({key: "MISSING_HISTORICAL_REFERENCE" for key in native_missing})
        _require({(row["kind"], int(row["position"])): row.get("native_reference_status")
                  for row in rows} == expected_native,
                 f"{endpoint} native comparison rows do not match the fixed historical coverage")
        _require(summary.get("native_reference_rows") == len(native_matched)
                 and summary.get("native_reference_missing_rows") == len(native_missing),
                 f"{endpoint} 200x200 historical reference coverage differs from the registered rows")
        replay_summaries[endpoint] = summary
    _validate_replay_rows(all_rows, expected)
    _require(len(all_rows) == 120 and all(row["view_allclose"] for row in all_rows),
             "120-row replay consumer did not accept the exact fixed population")
    _require(sum(int(value["native_reference_rows"]) for value in replay_summaries.values()) == 80
             and sum(int(value["native_reference_missing_rows"])
                     for value in replay_summaries.values()) == 40,
             "historical native replay population is not 80 measured and 40 explicitly missing")
    _require(sum(int(value["ed_replay_rows"]) for value in replay_summaries.values()) == 120
             and sum(int(value["ec_replay_rows"]) for value in replay_summaries.values()) == 120
             and sum(int(value["ed_saved_view_reference_rows"])
                     for value in replay_summaries.values()) == 40
             and sum(int(value["ec_saved_view_reference_rows"])
                     for value in replay_summaries.values()) == 40,
             "frozen evaluator forward rows are not separated into replay and saved-view roles")
    replay_result = _read_json(output / "replay_result.json")
    _require(replay_result.get("status") == "PASSED" and replay_result.get("rows") == 120
             and replay_result.get("endpoints") == replay_summaries
             and replay_result.get("ed_rows") == 120 and replay_result.get("ec_rows") == 120
             and replay_result.get("ed_saved_view_reference_rows") == 40
             and replay_result.get("ec_saved_view_reference_rows") == 40
             and replay_result.get("ed_total_forward_rows") == 160
             and replay_result.get("ec_total_forward_rows") == 160,
             "combined replay receipt differs from actual endpoint tensors and evaluator row roles")
    replay_path = output / "replay_rows.csv"
    if not replay_path.exists():
        _write_csv_new(replay_path, all_rows)
    else:
        saved_rows = _read_csv(replay_path)
        _require(len(saved_rows) == 120, "combined replay table has an incomplete population")
        normalized_rows = []
        for row in saved_rows:
            normalized = dict(row)
            for key in ("position", "label", "dataset_row", "cohort_row", "proposal_id", "component_id"):
                if normalized.get(key) not in (None, ""):
                    normalized[key] = int(normalized[key])
            normalized_rows.append(normalized)
        _validate_replay_rows(normalized_rows, expected)
        saved_by_key = {(row["endpoint"], row["kind"], int(row["position"])): row for row in saved_rows}
        _require(len(saved_by_key) == 120, "combined replay table has duplicate rows")
        for expected_row in all_rows:
            key = (expected_row["endpoint"], expected_row["kind"], int(expected_row["position"]))
            saved_row = saved_by_key[key]
            _require(all(_combined_csv_value_matches(value, saved_row.get(name, ""))
                         for name, value in expected_row.items()),
                     f"combined replay table differs from endpoint evidence at {key}")
    c_stage = _latest_complete_dir(output / "engineering" / "c_preflight", "c_preflight_result.json")
    j_stage = _latest_complete_dir(output / "engineering" / "j_recovery", "j_recovery_result.json")
    _require(c_stage is not None and j_stage is not None, "C preflight or J recovery evidence is missing")
    c_result = _read_json(c_stage / "c_preflight_result.json")
    j_result = _read_json(j_stage / "j_recovery_result.json")
    _require(c_result.get("status") == "PASSED" and c_result.get("engineering_adam_calls") == 5
             and c_result.get("preflight", {}).get("status") == "READY"
             and c_result.get("preflight", {}).get("preflight_rolled_back") is True,
             "C production preflight evidence is incomplete")
    c_zero_restore = c_result.get("preflight", {}).get("zero_update_restore", {})
    _require(c_zero_restore.get("model_max_abs_error") == 0.0
             and c_zero_restore.get("optimizer_max_abs_error") == 0.0
             and c_zero_restore.get("optimizer_steps")
                 == {"encoder_adam_step": 45_000, "decoder_adam_step": 45_000},
             "C zero-update model/Adam restore evidence is missing or invalid")
    _require(j_result.get("status") == "PASSED" and j_result.get("engineering_adam_calls") == 5
             and j_result.get("snapshot_global_step") == 47_503
             and j_result.get("direct_fourth_global_step") == 47_504
             and j_result.get("independent_fourth_global_step") == 47_504
             and j_result.get("independent_recovery", {}).get("passed") is True
             and j_result.get("independent_recovery", {}).get("model_max_abs_error", math.inf) <= RECOVERY_ATOL
             and j_result.get("independent_recovery", {}).get("optimizer_max_abs_error", math.inf) <= RECOVERY_ATOL,
             "J checkpoint restoration evidence is incomplete")
    j_zero_restore = j_result.get("zero_update_restore", {})
    _require(j_zero_restore.get("model_max_abs_error") == 0.0
             and j_zero_restore.get("optimizer_max_abs_error") == 0.0
             and j_zero_restore.get("optimizer_steps") == [47_500, 47_500],
             "J zero-update model/Adam restore evidence is missing or invalid")
    _require((j_stage / "checkpoint_step_002503.pt").is_file()
             and (j_stage / "recovery_fourth_update.pt").is_file()
             and (j_stage / "direct_fourth_update.pt").is_file()
             and (j_stage / "engineering_update_log.json").is_file(),
             "J serialized recovery snapshot, independent step, or separate engineering log is missing")
    source = inputs["fixed"]["source"]
    joint_config = inputs["joint_config"]
    original = _torch_load(
        ROOT / inputs["j_config"]["inputs"]["joint_optical_dir"] / "checkpoint_step_002500.pt",
    )
    snapshot = _torch_load(j_stage / "checkpoint_step_002503.pt")
    _validate_j_snapshot(snapshot, source, joint_config, original)
    j_rows = [list(map(int, row)) for row in j_result.get("next_four_sample_rows", [])]
    _require(len(j_rows) == 4, "J recovery result does not list the four registered TRAIN batches")
    j_expected_log_rows = j_rows + [j_rows[3]]
    j_log = _read_json(j_stage / "engineering_update_log.json")
    j_calls = _validate_recovery_update_log(
        j_log, "calls",
        ["original_process", "original_process", "original_process",
         "original_process_fourth", "independent_process_fourth"],
        47_501, j_expected_log_rows, [2_500, 2_501, 2_502, 2_503, 2_503],
    )
    _require(j_result.get("engineering_update_log") == j_calls
             and j_result.get("cuda", {}).get("three_update_event_ms")
                 == [call["elapsed_event_ms"] for call in j_calls[:3]],
             "J summary differs from the durable per-call update log")
    schedule_rows = _validate_j_cursor(snapshot["schedule"], 2_503,
                                       inputs["labels"], inputs["train_rows"], count=1)
    _require([int(value) for value in schedule_rows[0].tolist()] == j_rows[3]
             and j_rows == [[int(value) for value in row.tolist()]
                            for row in _validate_j_cursor(
                                original["schedule"], 2_500, inputs["labels"], inputs["train_rows"])]
             and j_result.get("snapshot_optimizer_steps") == [47_503, 47_503],
             "J snapshot cursor or saved batch identities differ from the registered schedule")
    original_j = inputs["fixed"]["checkpoints"]["J47500"]
    _require(torch.equal(original_j["model"]["runtime.encoder.phase"],
                         original["model"]["runtime.encoder.phase"])
             and torch.equal(original_j["model"]["runtime.generator.phase"],
                             original["model"]["runtime.generator.phase"])
             and original_j["optimizer_steps"] == original["optimizer_steps"],
             "J original checkpoint changed after recovery evidence was produced")
    direct = _torch_load(j_stage / "direct_fourth_update.pt")
    recovered = _torch_load(j_stage / "recovery_fourth_update.pt")
    j_step_declaration = {"encoder_adam_step": 47_504, "decoder_adam_step": 47_504}
    j_pair_errors = _validate_recovery_pair(
        direct, recovered, j_step_declaration, torch.tensor(j_rows[3], dtype=torch.long),
        {"task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
         "origin_checkpoint_run_id": original["run_id"],
         "completed_global_step": 47_504, "schedule_cursor": 2_503},
    )
    direct_model_error = j_pair_errors["model_max_abs_error"]
    direct_optimizer_error = j_pair_errors["optimizer_max_abs_error"]
    direct_loss_error = j_pair_errors["loss_max_abs_error"]
    _require(j_calls[3]["optimizer_steps"] == direct["optimizer_steps"]
             and j_calls[4]["optimizer_steps"] == recovered["optimizer_steps"]
             and base._assert_close(j_calls[3]["loss"], direct["loss"], "J_direct_log_loss") <= RECOVERY_ATOL
             and base._assert_close(j_calls[4]["loss"], recovered["loss"], "J_worker_log_loss") <= RECOVERY_ATOL
             and j_calls[3]["elapsed_event_ms"] == direct["cuda"]["elapsed_event_ms"]
             and j_calls[4]["elapsed_event_ms"] == recovered["cuda"]["elapsed_event_ms"],
             "J fourth-step artifacts differ from the per-call durable log")
    _require(direct_model_error <= RECOVERY_ATOL and direct_optimizer_error <= RECOVERY_ATOL
             and direct_loss_error <= RECOVERY_ATOL
             and math.isclose(direct_model_error, j_result["independent_recovery"]["model_max_abs_error"],
                              rel_tol=0.0, abs_tol=1e-15)
             and math.isclose(direct_optimizer_error,
                              j_result["independent_recovery"]["optimizer_max_abs_error"],
                              rel_tol=0.0, abs_tol=1e-15)
             and math.isclose(direct_loss_error,
                              j_result["independent_recovery"]["loss_max_abs_error"],
                              rel_tol=0.0, abs_tol=1e-15),
             "J independent fourth-step final consumer rejected model/Adam/loss agreement")
    c_recovery_input = _torch_load(c_stage / "recovery_input.pt")
    c_direct = _torch_load(c_stage / "direct_fourth_update.pt")
    c_recovered = _torch_load(c_stage / "recovery_fourth_update.pt")
    c_schedule = decoder_ref._schedule(inputs["labels"], inputs["train_rows"])
    c_rows = [[int(value) for value in row.tolist()] for row in c_schedule[45_000:45_004]]
    c_log = _read_json(c_stage / "engineering_update_log.json")
    c_calls = _validate_recovery_update_log(
        c_log, "updates",
        ["original_process", "original_process", "original_process",
         "original_process_fourth", "independent_process_fourth"],
        45_001, c_rows + [c_rows[3]], [45_000, 45_001, 45_002, 45_003, 45_003],
    )
    c_trials = c_result.get("preflight", {}).get("engineering_updates")
    _require(isinstance(c_trials, list) and len(c_trials) == 3
             and all(base._assert_close(c_calls[index]["loss"], c_trials[index],
                                        f"C_logged_update_{index + 1}_loss") <= RECOVERY_ATOL
                     for index in range(3)),
             "C first three durable call losses differ from the production preflight result")
    c_input_checkpoint = c_recovery_input.get("source_checkpoint", {})
    _require(c_recovery_input.get("task_id") == joint.TASK_ID
             and c_recovery_input.get("completed_updates") == 3
             and c_recovery_input.get("config_identity") == joint_config
             and c_input_checkpoint.get("run_id") == source["checkpoint"].get("run_id")
             and c_input_checkpoint.get("completed_global_step") == 45_000
             and base._assert_close(c_input_checkpoint.get("model"), source["checkpoint"]["model"],
                                    "C_recovery_input_origin_model") == 0.0
             and base._assert_close(c_input_checkpoint.get("optimizer"),
                                    source["checkpoint"]["optimizer"],
                                    "C_recovery_input_origin_optimizer") == 0.0
             and torch.equal(c_recovery_input.get("rows"), c_schedule[45_000:45_004])
             and _optimizer_phase_steps(c_recovery_input) == (45_003, 45_003)
             and c_direct.get("task_id") == joint.TASK_ID
             and c_recovered.get("task_id") == joint.TASK_ID
             and c_direct.get("completed_updates") == c_recovered.get("completed_updates") == 4,
             "C recovery input or direct/independent update identity is incomplete")
    c_pair_errors = _validate_recovery_pair(
        c_direct, c_recovered,
        {"encoder_adam_step": 45_004, "decoder_adam_step": 45_004},
        c_schedule[45_003],
        {"task_id": joint.TASK_ID,
         "origin_checkpoint_run_id": source["checkpoint"].get("run_id"),
         "completed_global_step": 45_004, "schedule_cursor": 45_003},
    )
    c_model_error = c_pair_errors["model_max_abs_error"]
    c_optimizer_error = c_pair_errors["optimizer_max_abs_error"]
    c_loss_error = c_pair_errors["loss_max_abs_error"]
    _require(c_calls[3]["optimizer_steps"] == c_direct["optimizer_steps"]
             and c_calls[4]["optimizer_steps"] == c_recovered["optimizer_steps"]
             and base._assert_close(c_calls[3]["loss"], c_direct["loss"], "C_direct_log_loss")
                 <= RECOVERY_ATOL
             and base._assert_close(c_calls[4]["loss"], c_recovered["loss"], "C_worker_log_loss")
                 <= RECOVERY_ATOL
             and c_calls[3]["elapsed_event_ms"] == c_direct["cuda"]["elapsed_event_ms"]
             and c_calls[4]["elapsed_event_ms"] == c_recovered["cuda"]["elapsed_event_ms"],
             "C fourth-step artifacts differ from the per-call durable log")
    c_recovery_summary = c_result["preflight"]["independent_process_recovery"]
    _require(c_result["preflight"]["status"] == "READY"
             and c_result["preflight"]["preflight_rolled_back"] is True
             and c_result["preflight"]["frozen_evaluators_unchanged"] is True
             and c_recovery_summary.get("passed") is True
             and c_model_error <= RECOVERY_ATOL and c_optimizer_error <= RECOVERY_ATOL
             and c_loss_error <= RECOVERY_ATOL
             and math.isclose(c_model_error, c_recovery_summary["direct_vs_restored_model_max_abs"],
                              rel_tol=0.0, abs_tol=1e-15)
             and math.isclose(c_optimizer_error,
                              c_recovery_summary["direct_vs_restored_optimizer_max_abs"],
                              rel_tol=0.0, abs_tol=1e-15)
             and c_result.get("evaluator_forward_rows_by_role") == {
                 "E_D": {"current_DEV": 10, "saved_view_DEV_reference": 10,
                         "current_G4": 10, "total": 30},
                 "E_C": {"current_DEV": 10, "saved_view_DEV_reference": 10,
                         "current_G4": 10, "total": 30},
             },
             "C production preflight recovery comparison, rollback, or evaluator counts are incomplete")
    for endpoint in ENDPOINTS:
        _validate_cuda_evidence(replay_summaries[endpoint])
    _require(c_result.get("cuda", {}).get("device")
             and math.isfinite(float(c_result["cuda"].get("elapsed_event_ms", math.nan)))
             and c_result["cuda"]["elapsed_event_ms"] > 0
             and int(c_result["cuda"].get("peak_reserved_vram_bytes", 0)) > 0
             and j_result.get("cuda", {}).get("device")
             and [call["elapsed_event_ms"] for call in j_calls[:3]]
                 == j_result.get("cuda", {}).get("three_update_event_ms")
             and math.isfinite(float(j_result["cuda"].get("direct_fourth_event_ms", math.nan)))
             and j_result["cuda"]["direct_fourth_event_ms"] > 0
             and math.isfinite(float(j_result["cuda"].get("independent_fourth", {}).get(
                 "elapsed_event_ms", math.nan)))
             and j_result["cuda"]["independent_fourth"]["elapsed_event_ms"] > 0,
             "C/J recovery CUDA execution evidence is incomplete")
    report = _closeout_report(all_rows, c_result, j_result)
    report_path = output / "closeout_report.md"
    if report_path.exists():
        _require(report_path.read_text(encoding="utf-8") == report,
                 "closeout report differs from reconsumed production evidence")
    else:
        report_path.write_text(report, encoding="utf-8")
    receipt = {
        "schema": "lightgen_gaussian_production_replay_validation.v1",
        "status": "ENGINEERING_ACCEPTED", "task_id": "LIGHTGEN_GAUSSIAN_PRODUCTION_REPLAY_V1",
        "replay_rows": len(all_rows), "endpoint_rows": {key: 40 for key in ENDPOINTS},
        "native_reference_rows": sum(row["native_reference_status"] == "MATCHED"
                                      for row in all_rows),
        "native_reference_missing_rows": sum(
            row["native_reference_status"] == "MISSING_HISTORICAL_REFERENCE" for row in all_rows),
        "E_D_replay_rows": 120, "E_D_saved_view_reference_rows": 40,
        "E_C_replay_rows": 120, "E_C_saved_view_reference_rows": 40,
        "C_E_D_forward_rows": 30, "C_E_C_forward_rows": 30,
        "C_engineering_adam_calls": c_result["engineering_adam_calls"],
        "J_engineering_adam_calls": j_result["engineering_adam_calls"],
        "scientific_updates": 0, "official_test_reads": 0,
        "frozen_replay_cuda_devices": {key: replay_summaries[key]["device_name"] for key in ENDPOINTS},
        "J_recovery_model_max_abs_error": j_result["independent_recovery"]["model_max_abs_error"],
        "J_recovery_optimizer_max_abs_error": j_result["independent_recovery"]["optimizer_max_abs_error"],
        "verified_at_unix": time.time(),
    }
    receipt_path = output / "engineering" / "final_validation_receipt.json"
    if receipt_path.is_file():
        existing = _read_json(receipt_path)
        _require({key: value for key, value in existing.items() if key != "verified_at_unix"}
                 == {key: value for key, value in receipt.items() if key != "verified_at_unix"},
                 "final validation receipt conflicts with reconsumed evidence")
        receipt = existing
    else:
        _write_json(receipt_path, receipt)
    return receipt


def _closeout_report(rows: list[dict[str, Any]], c_result: dict[str, Any],
                     j_result: dict[str, Any]) -> str:
    replay_count = len(rows)
    max_view = max(float(row["view_max_abs_error"]) for row in rows)
    max_ed = max(float(row["ed_logit_max_abs_error"]) for row in rows)
    max_ec = max(float(row["ec_logit_max_abs_error"]) for row in rows)
    native_rows = sum(row["native_reference_status"] == "MATCHED" for row in rows)
    native_missing = sum(row["native_reference_status"] == "MISSING_HISTORICAL_REFERENCE"
                         for row in rows)
    lines = [
        "# Gaussian生产复现与生成接口定界报告",
        "",
        "## 结果与状态",
        "本轮完成冻结C/D0/J端点的120条2048网格生产重放、C生产预检及真实J 47,500步状态的独立进程恢复。工程验收通过；历史系统`VALID_FAILED`保持。",
        "",
        "## 科学认识",
        "本轮只核验当前生产代码与既有Gaussian证据的重放及续接能力，科学模型判断未改变。复现通过仅覆盖已登记的C 45,000步、D0/J 50,000步及所抽DEV/G4记录，不构成生成质量、硬件可达或模型采用证据。",
        "",
        "## 目标缺口",
        "语义生成仍缺少类别正确率、跨轨迹稳健性、独立人眼确认及多样性/新颖性证据；真实光纤复传输矩阵、打印材料与相位响应、相机标定也未测量。`PROJECT_GEOMETRY_UNMEASURED`保持。",
        "",
        "## 工程完成与验证",
        f"冻结重放{replay_count}/120条，视图最大差{max_view:.3e}，E_D/E_C最大logit差分别为{max_ed:.3e}/{max_ec:.3e}。复核已有200×200原生图{native_rows}条；另有{native_missing}条D0/J G4样本没有历史原生参考，未补造参考。冻结评价器前向按角色计数：E_D与E_C各120条重放、40条保存视图复算；C预检中E_D与E_C各30条。C预检执行{c_result['engineering_adam_calls']}次工程Adam调用并完成回滚；J恢复从47,500步接续，保存47,503步快照，原进程与独立进程的第四步均到47,504步，模型/Adam最大绝对差分别为{j_result['independent_recovery']['model_max_abs_error']:.3e}/{j_result['independent_recovery']['optimizer_max_abs_error']:.3e}。官方TEST、科学更新、先验拟合、采样与LPIPS均为0。接口缺项见`interface_spec.md`。",
        "",
        "## 决策影响与下一步",
        "工程链通过后，可据此另行设计生成基线定界；本轮未授权且未启动后继科学任务。建议路线仍为生成基线→单因素候选→独立语义确认→实物输入/传输/探测验证。直接Decoder输入所需复场制备与完整Encoder链路条件分别列于接口表，未知硬件值继续待测。",
        "",
    ]
    return "\n".join(lines)


def _main_worker(args: argparse.Namespace) -> None:
    _worker_update(args.j_recovery_worker_dir.resolve(), args.config.resolve(), args.previous_config.resolve())
    print(json.dumps({"status": "PASSED", "worker": "independent_J_process"}), flush=True)


def _record_phase_failure(output: Path, phase: str, error: BaseException) -> None:
    stage_name = {"replay": "replay", "c-preflight": "c_preflight", "j-recovery": "j_recovery"}.get(phase)
    calls: list[dict[str, Any]] = []
    if stage_name is not None:
        stage_root = output / "replay" if stage_name == "replay" else output / "engineering" / stage_name
        candidates = sorted(path for path in stage_root.glob("attempt_*") if path.is_dir()) if stage_root.exists() else []
        if candidates:
            log_path = candidates[-1] / "engineering_update_log.json"
            if log_path.is_file():
                value = _read_json(log_path)
                entries = value.get("calls", value.get("updates", []))
                if isinstance(entries, list):
                    calls = entries
    completed = sum(call.get("status") == "COMPLETED" for call in calls)
    in_flight = [call.get("call_number") for call in calls if call.get("status") == "CALL_STARTED"]
    receipt = {
        "schema": "lightgen_gaussian_production_replay_phase_failure.v1",
        "status": "STOPPED_INCOMPLETE", "phase": phase.upper().replace("-", "_"),
        "failure_type": type(error).__name__, "failure_reason": str(error),
        "engineering_adam_calls_completed": completed,
        "engineering_calls_in_flight_at_stop": in_flight,
        "scientific_updates": 0, "official_test_reads": 0,
    }
    receipt_path = output / "engineering" / f"{phase.replace('-', '_')}_failure_receipt.json"
    if receipt_path.exists():
        sequence = 2
        while True:
            candidate = receipt_path.with_name(
                f"{phase.replace('-', '_')}_failure_receipt_{sequence:02d}.json",
            )
            if not candidate.exists():
                receipt_path = candidate
                break
            sequence += 1
    _write_json(receipt_path, receipt)
    total = 120 if phase == "replay" else (5 if phase in {"c-preflight", "j-recovery"} else None)
    _write_task_progress(output, phase.upper().replace("-", "_"), "STOPPED_INCOMPLETE",
                         completed if total is not None else None, total)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "replay", "c-preflight", "j-recovery", "verify"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / JOINT_CONFIG, help=argparse.SUPPRESS)
    parser.add_argument("--previous-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--j-recovery-worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.j_recovery_worker_dir is not None:
        _require(args.previous_config is not None, "J recovery worker needs the explicit prior config")
        _main_worker(args)
        return
    _require(args.phase is not None, "select one production replay phase")
    _require(args.output_dir is not None, "normal production phases require an explicit --output-dir")
    output = args.output_dir.resolve()
    expected_output = (ROOT / OUTPUT_REL).resolve()
    _require(output == expected_output, "output directory differs from the authorized attempt_02")
    try:
        if args.phase == "prepare":
            result = _prepare(output)
        elif args.phase == "verify":
            _load_run_config(output)
            _write_task_progress(output, "VERIFY", "RUNNING", 0, 1)
            inputs = _load_inputs(torch.device("cpu"))
            result = _verify(inputs, output)
        elif args.phase in {"replay", "c-preflight", "j-recovery"}:
            _load_run_config(output)
            phase = args.phase.upper().replace("-", "_")
            total = 120 if args.phase == "replay" else 5
            optimizer_step = 45_000 if args.phase == "c-preflight" else (
                47_500 if args.phase == "j-recovery" else None
            )
            _write_task_progress(output, phase, "RUNNING", 0, total, optimizer_step)
            _require(torch.cuda.is_available(), "CUDA unavailable; formal production stages require CUDA")
            inputs = _load_inputs(torch.device("cuda"))
            if args.phase == "replay":
                result = _run_replay(inputs, output)
            elif args.phase == "c-preflight":
                result = _run_c_preflight(inputs, output)
            else:
                result = _run_j_recovery(inputs, output)
        else:
            raise RuntimeError(f"unknown phase: {args.phase}")
    except BaseException as error:
        _record_phase_failure(output, args.phase, error)
        raise
    final_progress = {
        "prepare": (1, 1, None), "verify": (1, 1, None),
        "replay": (120, 120, None), "c-preflight": (5, 5, 45_004),
        "j-recovery": (5, 5, 47_504),
    }[args.phase]
    _write_task_progress(output, args.phase.upper().replace("-", "_"), result["status"],
                         *final_progress)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    _main()
