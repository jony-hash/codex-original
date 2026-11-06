"""Locate training-to-generation OLS input support differences without updates."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as frame
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from scripts.research.lightgen import execute_lightgen_recent_optical_fullpath_semantic_v1 as recent


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_TRAIN_GENERATION_OLS_SUPPORT_V1"
CONFIG_PATH = ROOT / "configs/research/lightgen/train_generation_ols_support_v1.yaml"
SOURCE_ATTEMPT_DIR = ROOT / "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/attempt_03"
SOURCE_ATTEMPT_BACKUP_DIR = ROOT / "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/autodl_backup_attempt_03"
OUTPUT_DIR = ROOT / "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/attempt_04"
TRAINING_RUN_ID = "d88d5bfc2b29488884a88512152815d6"
BOOTSTRAP_COUNT = 2_000
BOOTSTRAP_SEED = 202611020970
CLASS_COUNT = 10
ROWS_PER_CLASS = 200
REFERENCE_PER_CLASS = 100
CALIBRATION_PER_CLASS = 100
GENERATED_PER_CLASS = 200
GRID_PRODUCTION = 2048
GRID_REFERENCE = 4096
CROSS_GRID_TOLERANCE = 1.0e-4
CACHE_TOLERANCE = 1.0e-6
BATCH_SIZE = 10
PHASE_STEPS = (0, 2_000)
CONDITION_PHASE_STEPS = {
    "E0G0": (0, 0), "E1G1": (2_000, 2_000),
    "E0G1": (0, 2_000), "E1G0": (2_000, 0),
}


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_torch(path: Path, value: Any, *, new: bool = False) -> None:
    if new:
        _require(not path.exists(), f"refusing to replace evidence: {path}", FileExistsError)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _same_state(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return (left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _load_pt(path: Path, *, mmap: bool = False) -> Any:
    _require(path.is_file(), f"explicit input file is missing: {path}", FileNotFoundError)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)


def _read_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == CONFIG_PATH.resolve(), "only the registered OLS-support config is allowed",
             ValueError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict)
             and config.get("schema") == "lightgen_train_generation_ols_support.v1"
             and config.get("task_id") == TASK_ID
             and config.get("evidence_tier") == "diagnostic_only",
             "registered OLS-support task identity changed", ValueError)
    _require(config.get("sources", {}).get("expected_training_run_id") == TRAINING_RUN_ID
             and config["sources"].get("expected_population_seed_base") == 202610300930
             and config.get("population") == {
                 "role": "consumed_train_diagnostic",
                 "selection": "first_200_distinct_train_rows_per_class_in_saved_schedule_order",
                 "rows": 2_000, "rows_per_class": 200, "reference_per_class": 100,
                 "calibration_per_class": 100, "generated_per_class": 200,
                 "official_test_reads": 0,
             }, "population identity or roles changed", ValueError)
    expected_sources = {
        "source_config": "configs/research/lightgen/recent_optical_fullpath_semantic_v1.yaml",
        "training_run": "outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/attempt_01",
        "training_backup": "outputs/20261030_lightgen_recent_optical_fullpath_semantic_v1/runs/autodl_backup_attempt_01",
        "expected_training_run_id": TRAINING_RUN_ID,
        "attribution_run": "outputs/20261031_lightgen_fullpath_zero_update_attribution_v1/runs/attempt_05",
        "attribution_backup": "outputs/20261031_lightgen_fullpath_zero_update_attribution_v1/runs/autodl_backup_attempt_05",
        "preflight_controls_run": "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/attempt_02",
        "preflight_controls_backup": "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/autodl_backup_attempt_02",
        "measurement_source_run": "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/attempt_03",
        "measurement_source_backup": "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/autodl_backup_attempt_03",
        "output_backup": "outputs/20261102_lightgen_train_generation_ols_support_v1/runs/autodl_backup_attempt_04",
        "train_reference_run": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/attempt_01/prior_evidence",
        "train_reference_backup": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/prior_evidence",
        "expected_population_seed_base": 202610300930,
    }
    _require(config.get("sources") == expected_sources,
             "explicit source and recovery paths changed", ValueError)
    _require(config.get("measurement") == {
        "endpoints": [0, 2_000], "production_grid": GRID_PRODUCTION,
        "independent_reference_grid": GRID_REFERENCE, "production_forwards": 4_000,
        "production_forwards_new": 40, "production_forwards_reused": 3_960,
        "cross_grid_reference_forwards": 40,
        "cross_grid_reference_forwards_new": 40,
        "cross_grid_reference_forwards_reused": 0,
        "positive_controls_per_class": 2,
        "cross_grid_tolerance": CROSS_GRID_TOLERANCE,
        "train_cache_relative_tolerance": CACHE_TOLERANCE,
        "inference_batch_size": BATCH_SIZE, "no_grad": True,
        "measurement_source": "attempt_03_noncontrols_only; remeasure_40_controls",
        "input_identity": "source_train_row_bound_before_native_slm_input_and_forward",
        "step_0_cache_policy": "all_2000_correct_rows_must_pass_1e-6",
    }, "registered optical measurement budget changed", ValueError)
    _require(config.get("metric", {}).get("bootstrap_resamples") == BOOTSTRAP_COUNT
             and config["metric"].get("bootstrap_seed") == BOOTSTRAP_SEED
             and config["metric"].get("confidence_level") == 0.95
             and config["metric"].get("reference_set") == "fixed_100_per_class"
             and config["metric"].get("pairing")
             == "same_generated_and_calibration_rows_across_endpoints",
             "registered distance or bootstrap definition changed", ValueError)
    _require(config["metric"].get("name")
             == "nearest_same_class_scale_whitened_unit_complex_global_phase_invariant_distance"
             and config["metric"].get("decision_routes") == {
                 "difference_expanded": "terminal_and_delta_ci95_lower_gt_zero",
                 "difference_present_not_confirmed_expanded": "terminal_ci95_lower_gt_zero",
                 "evidence_insufficient": "terminal_ci95_lower_not_gt_zero",
             }, "registered scientific decision rule changed", ValueError)
    _require(config.get("output") == {
        "run_dir": OUTPUT_DIR.relative_to(ROOT).as_posix(),
        "optimizer_updates": 0, "optimizer_instances": 0, "prior_refits": 0,
        "prior_samples": 0, "official_test_reads": 0,
        "preserve_all_source_artifacts": True, "stop_after_one_route_decision": True,
    }, "registered zero-update stop contract changed", ValueError)
    return config


def _training_dir(config: Mapping[str, Any], *, local: bool) -> Path:
    key = "training_backup" if local else "training_run"
    return (ROOT / str(config["sources"][key])).resolve()


def _attribution_dir(config: Mapping[str, Any], *, local: bool) -> Path:
    key = "attribution_backup" if local else "attribution_run"
    return (ROOT / str(config["sources"][key])).resolve()


def _map_remote_source_path(value: str, expected: Path, *, local: bool) -> bool:
    if value == str(expected):
        return True
    if not local:
        return False
    remote_prefix = "/root/autodl-tmp/codex-original/"
    if not value.startswith(remote_prefix):
        return False
    return (ROOT / Path(value[len(remote_prefix):])).resolve() == expected.resolve()


def _optimizer_steps_without_instantiation(state: Mapping[str, Any]) -> tuple[int, int]:
    groups = state.get("param_groups")
    slots = state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1
             and isinstance(slots, Mapping) and len(groups[0].get("params", [])) == 2,
             "saved optical Adam state no longer contains exactly the two trained phase tensors")
    ids = groups[0]["params"]
    _require(len(set(ids)) == 2 and all(key in slots for key in ids),
             "saved optical Adam phase slots are incomplete")
    steps = []
    for identifier in ids:
        slot = slots[identifier]
        _require(isinstance(slot, Mapping) and "step" in slot
                 and "exp_avg" in slot and "exp_avg_sq" in slot,
                 "saved optical Adam moment state is incomplete")
        step = int(torch.as_tensor(slot["step"]).item())
        _require(tuple(slot["exp_avg"].shape) == (2, 1, 100, 100)
                 and tuple(slot["exp_avg_sq"].shape) == (2, 1, 100, 100),
                 "saved optical Adam moments do not map to the registered 2+2 phase shapes")
        steps.append(step)
    return steps[0], steps[1]


def _load_research_context(config: Mapping[str, Any], *, device: torch.device,
                           local_sources: bool) -> dict[str, Any]:
    source_config_path = ROOT / str(config["sources"]["source_config"])
    previous_config = recent._read_config(source_config_path)
    runtime = recent._load_inputs(previous_config, device)
    train_dir = _training_dir(config, local=local_sources)
    attempt_dir = train_dir / "scientific_attempt"
    state = json.loads((attempt_dir / "task_state.json").read_text(encoding="utf-8"))
    summary = json.loads((attempt_dir / "training_summary.json").read_text(encoding="utf-8"))
    schedule_payload = _load_pt(attempt_dir / "training_schedule.pt")
    schedule = schedule_payload.get("schedule") if isinstance(schedule_payload, Mapping) else schedule_payload
    _require(state.get("run_id") == TRAINING_RUN_ID
             and state.get("authorized_updates") == 2_000
             and state.get("completed_updates") == 2_000
             and summary.get("task_id") == recent.TASK_ID
             and summary.get("selected_endpoint") == "D0"
             and summary.get("completed_updates") == 2_000
             and summary.get("completed_global_step") == 52_000
             and torch.is_tensor(schedule) and tuple(schedule.shape) == (2_000, 10),
             "explicit source training run, endpoint, step count, or schedule is incomplete")
    expected_schedule = recent._build_train_schedule(
        runtime["labels"], runtime["train_indices"], 202610300950, updates=2_000
    )
    _require(torch.equal(schedule, expected_schedule),
             "saved training schedule differs from the registered 2,000-step TRAIN schedule")

    endpoint_spec = previous_config["sources"]["endpoints"]["D0"]
    endpoint_path = (ROOT / endpoint_spec["checkpoint"]).resolve()
    source_endpoint = _load_pt(endpoint_path)
    recent._validate_endpoint_identity("D0", endpoint_spec, source_endpoint)
    expected_adam = (int(endpoint_spec["encoder_adam_step"]),
                     int(endpoint_spec["generator_adam_step"]))
    _require(_optimizer_steps_without_instantiation(source_endpoint["optimizer"]) == expected_adam,
             "source D0 Adam counters do not match the registered endpoint")
    selected = {"name": "D0", "path": endpoint_path, "checkpoint": source_endpoint,
                "optimizer_steps": expected_adam}

    checkpoints: dict[int, dict[str, Any]] = {}
    for updates in PHASE_STEPS:
        checkpoint_path = attempt_dir / f"checkpoint_step_{updates:05d}.pt"
        checkpoint = _load_pt(checkpoint_path)
        source_path = checkpoint.get("source_checkpoint")
        _require(isinstance(source_path, str)
                 and _map_remote_source_path(source_path, endpoint_path, local=local_sources),
                 f"step {updates} checkpoint source does not resolve to the explicit D0 endpoint")
        if local_sources and source_path != str(endpoint_path):
            checkpoint = dict(checkpoint)
            checkpoint["source_checkpoint"] = str(endpoint_path)
        _require(recent._validate_training_checkpoint(
            checkpoint, selected, schedule, previous_config, TRAINING_RUN_ID
        ) == updates, f"source optical checkpoint is not the exact step {updates} state")
        _require(_optimizer_steps_without_instantiation(checkpoint["optimizer"])
                 == (expected_adam[0] + updates, expected_adam[1] + updates),
                 f"step {updates} Adam phase counters are inconsistent")
        checkpoints[updates] = checkpoint

    terminal = _load_pt(attempt_dir / "terminal_checkpoint.pt")
    terminal_source_path = terminal.get("source_checkpoint")
    _require(isinstance(terminal_source_path, str)
             and _map_remote_source_path(terminal_source_path, endpoint_path, local=local_sources),
             "terminal checkpoint source does not resolve to the explicit D0 endpoint")
    if local_sources and terminal_source_path != str(endpoint_path):
        terminal = dict(terminal)
        terminal["source_checkpoint"] = str(endpoint_path)
    _require(recent._validate_training_checkpoint(
        terminal, selected, schedule, previous_config, TRAINING_RUN_ID
    ) == 2_000
             and recent._states_close(terminal["model"], checkpoints[2_000]["model"])
             and recent._states_close(terminal["optimizer"], checkpoints[2_000]["optimizer"]),
             "terminal checkpoint differs from the exact 2,000-update endpoint")
    _require(recent._states_close(checkpoints[0]["model"], source_endpoint["model"])
             and recent._states_close(checkpoints[0]["optimizer"], source_endpoint["optimizer"]),
             "saved step 0 does not exactly inherit the selected D0 model and Adam state")
    for key in ("weights", "structural_weight", "scale", "power_reference", "camera_gain",
                "model_identity"):
        _require(recent._states_close(checkpoints[0].get(key), checkpoints[2_000].get(key)),
                 f"fixed training reference changed between endpoints: {key}")
    for key in set(checkpoints[0]["model"]) - {
            "runtime.encoder.phase", "runtime.generator.phase"}:
        _require(torch.equal(checkpoints[0]["model"][key], checkpoints[2_000]["model"][key]),
                 f"non-phase optical state changed during the prior training: {key}")
    _require(not torch.equal(checkpoints[0]["model"]["runtime.encoder.phase"],
                             checkpoints[2_000]["model"]["runtime.encoder.phase"])
             and not torch.equal(checkpoints[0]["model"]["runtime.generator.phase"],
                                 checkpoints[2_000]["model"]["runtime.generator.phase"]),
             "the two endpoints do not contain the registered Encoder and Generator updates")

    cache_key = "train_reference_backup" if local_sources else "train_reference_run"
    cache_dir = ROOT / str(config["sources"][cache_key])
    cache = _load_pt(cache_dir / "raw_physical_phase_train_coefficients.pt", mmap=True)
    _require(cache.get("schema") == f"{frame.TASK_ID}.raw_train_coefficients.v1"
             and cache.get("task_id") == frame.TASK_ID
             and cache.get("representation")
             == "C45k raw physical phase OLS coefficients; no per-sample gauge rotation"
             and tuple(cache.get("coefficients", torch.empty(0)).shape) == (55_000, 100)
             and torch.equal(cache.get("rows"), runtime["train_indices"])
             and torch.equal(cache.get("labels"), runtime["labels"][runtime["train_indices"]]),
             "55,000-row matrix-reencoded TRAIN cache identity or role changed")
    replay100 = _load_pt(cache_dir / "source_matrix_replay_100.pt", mmap=True)
    replay_error = torch.linalg.vector_norm(
        replay100["frozen_matrix_coefficients"] - replay100["production_optical_coefficients"],
        dim=1,
    ) / torch.linalg.vector_norm(replay100["production_optical_coefficients"], dim=1).clamp_min(1.0e-30)
    _require(replay100.get("task_id") == frame.TASK_ID
             and replay100.get("production_response") == "C45k image adapter to SLM amplitude to Encoder/OLS"
             and tuple(replay100.get("source_train_rows", torch.empty(0)).shape) == (100,)
             and float(replay_error.max()) <= CACHE_TOLERANCE
             and torch.allclose(replay_error, replay100["relative_error_per_row"],
                                rtol=1.0e-12, atol=1.0e-15),
             "historical 100-row full-optics/cache cross-check failed its 1e-6 contract")
    return {
        "previous_config": previous_config, "runtime": runtime, "training_dir": train_dir,
        "attempt_dir": attempt_dir, "schedule": schedule, "selected": selected,
        "source_endpoint": source_endpoint, "checkpoints": checkpoints,
        "terminal": terminal, "cache": cache, "replay100": replay100,
        "cache_replay_max_relative_error": float(replay_error.max()),
        "local_sources": local_sources,
    }


def _sample_manifest(schedule: torch.Tensor, labels: torch.Tensor,
                     train_indices: torch.Tensor) -> list[dict[str, Any]]:
    train_set = set(int(value) for value in train_indices.tolist())
    records: list[dict[str, Any]] = []
    for label in range(CLASS_COUNT):
        seen: set[int] = set()
        class_rows: list[tuple[int, int]] = []
        for step, row in enumerate(schedule[:, label].tolist()):
            row = int(row)
            _require(row in train_set and int(labels[row]) == label,
                     "saved schedule contains a non-TRAIN row or wrong class")
            if row not in seen:
                seen.add(row)
                class_rows.append((row, step))
                if len(class_rows) == ROWS_PER_CLASS:
                    break
        _require(len(class_rows) == ROWS_PER_CLASS,
                 f"saved schedule lacks 200 distinct TRAIN rows for class {label}")
        for class_position, (row, step) in enumerate(class_rows):
            records.append({
                "population_index": len(records), "label": label,
                "class_position": class_position, "source_train_row": row,
                "schedule_step": step, "schedule_flat_position": step * CLASS_COUNT + label,
                "role": "reference" if class_position % 2 == 0 else "calibration",
                "positive_control": class_position < 2,
            })
    _require(len(records) == 2_000 and len({row["source_train_row"] for row in records}) == 2_000,
             "selected TRAIN rows are incomplete or repeated")
    _require(sum(row["role"] == "reference" for row in records) == 1_000
             and sum(row["role"] == "calibration" for row in records) == 1_000,
             "TRAIN reference and calibration assignments are unbalanced")
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_sample_manifest(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        actual = list(csv.DictReader(stream))
    expected = [{key: str(value) for key, value in row.items()} for row in records]
    _require(actual == expected, "saved TRAIN sample manifest has missing, changed, or reordered rows")


def _validate_preflight_source_status(status: Mapping[str, Any]) -> None:
    _require(status.get("task_id") == TASK_ID
             and status.get("status") == "INVALID_HARNESS"
             and status.get("stage") == "representative_preflight"
             and "step-0 cache preflight failed" in str(status.get("error", ""))
             and status.get("production_forwards_attempted") == 40
             and status.get("production_forwards_saved") == 40
             and status.get("cross_grid_reference_forwards_attempted") == 40
             and status.get("optimizer_updates") == 0
             and status.get("official_test_reads") == 0,
             "attempt_02 is not the explicit cache-mismatch-only preflight source")


def _validate_reused_cross_grid_report(report: Mapping[str, Any],
                                       rows: Sequence[Mapping[str, Any]]) -> None:
    _require(report.get("task_id") == TASK_ID and report.get("status") == "PASSED"
             and report.get("production_rows_included") == 40
             and report.get("independent_reference_rows") == 40
             and report.get("rows_per_class_per_endpoint") == 2
             and report.get("production_grid") == GRID_PRODUCTION
             and report.get("reference_grid") == GRID_REFERENCE
             and report.get("tolerance") == CROSS_GRID_TOLERANCE
             and report.get("semantic_evidence") is False
             and report.get("hardware_evidence") is False
             and len(rows) == 40 and len(report.get("rows", [])) == 40
             and all(row.get("passed_1e-4") is True for row in rows)
             and _numeric_tree_close(report.get("rows"), rows)
             and math.isclose(float(report.get("maximum_coefficient_relative_error", math.nan)),
                              max(float(row["coefficient_relative_error"]) for row in rows),
                              rel_tol=1.0e-12, abs_tol=1.0e-15)
             and math.isclose(float(report.get("maximum_raw_cmos_relative_error", math.nan)),
                              max(float(row["raw_cmos_relative_error"]) for row in rows),
                              rel_tol=1.0e-12, abs_tol=1.0e-15),
             "attempt_02 saved cross-grid controls do not independently reproduce its passing report")


def _load_reused_preflight_controls(config: Mapping[str, Any], context: Mapping[str, Any],
                                    manifest: Sequence[Mapping[str, Any]], *,
                                    local_sources: bool
                                    ) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    key = "preflight_controls_backup" if local_sources else "preflight_controls_run"
    source_dir = (ROOT / str(config["sources"][key])).resolve()
    status = json.loads((source_dir / "measurement_status.json").read_text(encoding="utf-8"))
    _validate_preflight_source_status(status)
    source_identity = json.loads((source_dir / "source_identity.json").read_text(encoding="utf-8"))
    expected_model_identity = context["checkpoints"][0]["model_identity"]
    _require(source_identity.get("task_id") == TASK_ID
             and source_identity.get("source_training_run_id") == TRAINING_RUN_ID
             and source_identity.get("source_endpoint") == "D0"
             and source_identity.get("model_identity") == expected_model_identity
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("global_step") == 50_000
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("encoder_adam_step") == 45_000
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("generator_adam_step") == 50_000
             and source_identity.get("measured_endpoints", {}).get("2000", {}).get("global_step") == 52_000
             and source_identity.get("measured_endpoints", {}).get("2000", {}).get("encoder_adam_step") == 47_000
             and source_identity.get("measured_endpoints", {}).get("2000", {}).get("generator_adam_step") == 52_000
             and source_identity.get("official_test_read") is False,
             "attempt_02 control endpoints do not match the registered D0 optical checkpoints")
    _validate_sample_manifest(source_dir / "train_sample_manifest.csv", manifest)
    control_indices = torch.tensor([int(row["population_index"]) for row in manifest
                                    if row["positive_control"]], dtype=torch.long)
    controls = _load_pt(source_dir / "cross_grid_controls.pt")
    _require(isinstance(controls, Mapping) and set(controls) == set(PHASE_STEPS),
             "attempt_02 cross-grid controls omit an endpoint")
    report = json.loads((source_dir / "cross_grid_reference.json").read_text(encoding="utf-8"))
    recomputed_rows: list[dict[str, Any]] = []
    normalized: dict[int, dict[str, torch.Tensor]] = {}
    for step in PHASE_STEPS:
        item = controls[step]
        required_shapes = {
            "population_indices": (20,), "coefficients_2048": (20, 100),
            "raw_cmos_2048": (20, 1, 200, 200), "plane_powers_2048": (20, 5),
            "raw_summaries_2048": (20, 5), "coefficients_4096": (20, 100),
            "raw_cmos_4096": (20, 1, 200, 200),
        }
        _require(isinstance(item, Mapping)
                 and set(item) == set(required_shapes)
                 and all(torch.is_tensor(item.get(name))
                         and tuple(item[name].shape) == shape
                         for name, shape in required_shapes.items())
                 and torch.equal(item["population_indices"].to(torch.long), control_indices),
                 f"attempt_02 step {step} control row identities or tensor shapes are invalid")
        _require(torch.is_complex(item["coefficients_2048"])
                 and torch.is_complex(item["coefficients_4096"])
                 and bool((item["raw_cmos_2048"] >= 0.0).all().item())
                 and bool((item["raw_cmos_4096"] >= 0.0).all().item())
                 and bool((item["plane_powers_2048"] >= 0.0).all().item()),
                 f"attempt_02 step {step} controls violate optical value constraints")
        for name, value in item.items():
            _require(torch.is_tensor(value)
                     and (torch.isfinite(value.real).all().item()
                          and torch.isfinite(value.imag).all().item()
                          if torch.is_complex(value) else torch.isfinite(value).all().item()),
                     f"attempt_02 step {step} control {name} is nonfinite or not a tensor")
        normalized[step] = {name: value.detach().cpu().clone() for name, value in item.items()}
        for position, population_index in enumerate(control_indices.tolist()):
            coefficient_error = projected._phase_error(
                item["coefficients_2048"][position], item["coefficients_4096"][position]
            )
            cmos_error = projected._raw_error(
                item["raw_cmos_2048"][position], item["raw_cmos_4096"][position]
            )
            recomputed_rows.append({
                "endpoint_step": step, "population_index": population_index,
                "source_train_row": int(manifest[population_index]["source_train_row"]),
                "label": int(manifest[population_index]["label"]),
                "coefficient_relative_error": coefficient_error,
                "raw_cmos_relative_error": cmos_error,
                "passed_1e-4": coefficient_error <= CROSS_GRID_TOLERANCE
                and cmos_error <= CROSS_GRID_TOLERANCE,
            })
    _validate_reused_cross_grid_report(report, recomputed_rows)
    reused_report = {
        **report,
        "source_attempt": "attempt_02",
        "source_run_dir": config["sources"]["preflight_controls_run"],
        "source_backup_dir": config["sources"]["preflight_controls_backup"],
        "production_rows_reused": 40,
        "reference_rows_reused": 40,
        "production_forwards_attempted_this_attempt": 0,
        "reference_forwards_attempted_this_attempt": 0,
        "source_report_recomputed": True,
        "semantic_evidence": False,
        "hardware_evidence": False,
    }
    return normalized, reused_report


def _validate_generation_commands(population_commands: torch.Tensor,
                                  attribution_commands: torch.Tensor, *,
                                  expected_rows: int = 2_000,
                                  command_shape: tuple[int, int, int] = (1, 200, 200)) -> None:
    _require(torch.is_tensor(population_commands) and torch.is_tensor(attribution_commands)
             and tuple(population_commands.shape) == tuple(attribution_commands.shape)
             == (expected_rows, *command_shape)
             and population_commands.dtype == attribution_commands.dtype == torch.float32
             and torch.isfinite(population_commands).all().item()
             and torch.isfinite(attribution_commands).all().item()
             and float(population_commands.min()) >= 0.0 and float(population_commands.max()) <= 1.0,
             "saved generation commands are incomplete or outside legal SLM amplitude range")
    for start in range(0, expected_rows, 100):
        stop = min(start + 100, expected_rows)
        _require(torch.equal(population_commands[start:stop],
                             attribution_commands[start:stop]),
                 "saved generation SLM command rows differ from the original fixed commands")


def _validate_condition_phase(name: str, provenance: Mapping[str, Any],
                              states: Mapping[str, Any], checkpoints: Mapping[int, Any]) -> None:
    _require(name in CONDITION_PHASE_STEPS, f"unregistered condition phase source: {name}")
    encoder_step, generator_step = CONDITION_PHASE_STEPS[name]
    _require(provenance == {
        "encoder_step": encoder_step, "generator_step": generator_step,
        "source_training_run_id": TRAINING_RUN_ID,
    }, f"saved {name} phase provenance is missing or wrong")
    expected_encoder = checkpoints[encoder_step]["model"]["runtime.encoder.phase"]
    expected_generator = checkpoints[generator_step]["model"]["runtime.generator.phase"]
    _require(_same_state(states.get("encoder"), expected_encoder)
             and _same_state(states.get("generator"), expected_generator),
             f"saved {name} phase tensors do not match the registered checkpoints")


def _load_generation_assets(context: Mapping[str, Any], config: Mapping[str, Any], *,
                            local_sources: bool) -> dict[str, Any]:
    training_dir = context["training_dir"]
    attr_dir = _attribution_dir(config, local=local_sources)
    population = _load_pt(training_dir / "final_population_and_commands.pt", mmap=True)
    source_outputs = _load_pt(training_dir / "final_reconsumed_outputs.pt", mmap=True)
    bundle = _load_pt(attr_dir / "condition_outputs.pt", mmap=True)
    source_csv = attr_dir / "per_sample_results.csv"
    _require(population.get("task_id") == recent.TASK_ID
             and population.get("schema") == f"{recent.TASK_ID}.final_population.v1"
             and population.get("seed_base") == config["sources"]["expected_population_seed_base"]
             and population.get("cohort", {}).get("role")
             == "new_final_population_not_consumed_during_route_or_training",
             "historical generation population identity changed")
    cohort = population["cohort"]
    labels, proposal_ids = cohort.get("labels"), cohort.get("proposal_ids")
    commands = population.get("commands")
    _require(torch.is_tensor(labels) and tuple(labels.shape) == (2_000,)
             and torch.equal(labels, torch.arange(10).repeat_interleave(200))
             and torch.is_tensor(proposal_ids) and tuple(proposal_ids.shape) == (2_000,)
             and torch.is_tensor(commands) and tuple(commands.shape) == (2_000, 1, 200, 200)
             and torch.is_tensor(cohort.get("coefficients"))
             and tuple(cohort["coefficients"].shape) == (2_000, 100)
             and torch.is_complex(cohort["coefficients"])
             and torch.isfinite(cohort["coefficients"].real).all().item()
             and torch.isfinite(cohort["coefficients"].imag).all().item()
             and commands.dtype == torch.float32 and torch.isfinite(commands).all().item()
             and float(commands.min()) >= 0.0 and float(commands.max()) <= 1.0,
             "saved generated targets, labels, proposal ids, or legal SLM commands are incomplete")
    _require(bundle.get("task_id") == "LIGHTGEN_FULLPATH_ZERO_UPDATE_ATTRIBUTION_V1"
             and bundle.get("population_role") == "consumed_diagnostic_population"
             and set(bundle.get("conditions", {})) == set(CONDITION_PHASE_STEPS)
             and torch.equal(bundle.get("labels"), labels)
             and torch.equal(bundle.get("proposal_ids"), proposal_ids),
             "saved attribution population identity or four-condition output set changed")
    _validate_generation_commands(commands, bundle.get("commands"))
    _require(source_outputs.get("task_id") == recent.TASK_ID
             and set(source_outputs.get("conditions", {}))
             == {"C45k", "selected_initial", "final_candidate"},
             "original reconsumed generation output bundle is incomplete")

    phase_models = context["checkpoints"]
    for name in CONDITION_PHASE_STEPS:
        provenance = bundle.get("phase_sources", {}).get(name, {})
        states = bundle.get("phase_states", {}).get(name, {})
        _validate_condition_phase(name, provenance, states, phase_models)
        output = bundle["conditions"][name]
        _require(tuple(output.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
                 and tuple(output.get("raw_cmos", torch.empty(0)).shape) == (2_000, 1, 200, 200)
                 and tuple(output.get("readouts", torch.empty(0)).shape) == (2_000, 1, 28, 28),
                 f"saved {name} generated optical outputs are incomplete")
        for key in ("coefficients", "raw_cmos", "readouts"):
            tensor = output[key]
            _require(torch.isfinite(tensor.real if torch.is_complex(tensor) else tensor).all().item()
                     and (not torch.is_complex(tensor)
                          or torch.isfinite(tensor.imag).all().item()),
                     f"saved {name} generated output contains nonfinite values: {key}")

    for source_name, condition_name in (("selected_initial", "E0G0"),
                                        ("final_candidate", "E1G1")):
        for key in ("coefficients", "raw_cmos", "readouts"):
            first = source_outputs["conditions"][source_name][key]
            second = bundle["conditions"][condition_name][key]
            _require(tuple(first.shape) == tuple(second.shape),
                     f"saved generation output shape changed: {condition_name}/{key}")
            for start in range(0, 2_000, 100):
                _require(torch.equal(first[start:start + 100], second[start:start + 100]),
                         f"{condition_name} no longer reuses the saved {source_name} {key}")

    predictions: list[dict[str, int]] = []
    with source_csv.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    _require(len(csv_rows) == 2_000,
             "existing evaluator result rows do not cover the saved generated population")
    for index, row in enumerate(csv_rows):
        label, proposal_id = int(row["label"]), int(row["proposal_id"])
        _require(int(row["population_index"]) == index
                 and label == int(labels[index]) and proposal_id == int(proposal_ids[index]),
                 "saved evaluator rows are misaligned with generation labels or proposal ids")
        pred = {key: int(value) for key, value in row.items() if key.endswith("_prediction")}
        _require(len(pred) == 8 and all(0 <= value < CLASS_COUNT for value in pred.values()),
                 "saved frozen E_D/E_C predictions are incomplete or malformed")
        predictions.append(pred)
    return {
        "population": population, "bundle": bundle, "source_outputs": source_outputs,
        "labels": labels.detach().cpu().long(),
        "proposal_ids": proposal_ids.detach().cpu().long(),
        "target_coefficients": cohort["coefficients"].detach().cpu().to(torch.complex128),
        "commands": commands, "predictions": predictions,
        "attribution_dir": attr_dir,
    }


def _cache_map(context: Mapping[str, Any]) -> dict[int, int]:
    return {int(row): index for index, row in enumerate(context["cache"]["rows"].tolist())}


def _relative_coefficient_error(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - reference, dim=1) / torch.linalg.vector_norm(
        actual, dim=1
    ).clamp_min(1.0e-30)


def _validate_cache_diagnostic_record(record: Mapping[str, Any], errors: torch.Tensor, *,
                                      rows_compared: int, source_attempt: str) -> bool:
    passed = bool(float(errors.max()) <= CACHE_TOLERANCE)
    _require(record.get("task_id") == TASK_ID
             and record.get("status") == ("PASSED" if passed else "DIAGNOSTIC_MISMATCH")
             and record.get("source_attempt") == source_attempt
             and record.get("rows_compared") == rows_compared
             and record.get("tolerance") == CACHE_TOLERANCE
             and record.get("passed") is passed
             and record.get("blocking_gate") is False
             and record.get("mismatch_policy") == "report_as_diagnostic_limitation"
             and record.get("user_approved_nonblocking_treatment") is True
             and record.get("step_2000_cache_used_as_reference") is False
             and math.isclose(float(record.get("maximum_relative_error", math.nan)),
                              float(errors.max()), rel_tol=1.0e-12, abs_tol=1.0e-15)
             and math.isclose(float(record.get("p95_relative_error", math.nan)),
                              float(torch.quantile(errors, 0.95)),
                              rel_tol=1.0e-12, abs_tol=1.0e-15),
             "saved cache comparison is not the measured diagnostic record")
    return passed


def _new_model(checkpoint: Mapping[str, Any], grid: int,
               runtime: Mapping[str, Any]) -> torch.nn.Module:
    model = recent._new_endpoint_model(checkpoint, grid, runtime)
    _require(not any(parameter.requires_grad for parameter in model.parameters()),
             "optical measurement model unexpectedly has trainable parameters")
    return model


def _raw_output_summary(raw: torch.Tensor, sensor_w_pixel: torch.Tensor,
                        coefficients: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200)
             and sensor_w_pixel.shape == raw.shape
             and coefficients.shape == (len(raw), 100),
             "raw CMOS summary inputs have incompatible shapes")
    flat = raw.reshape(len(raw), -1)
    columns = (
        flat.mean(1), torch.sqrt(flat.square().mean(1)), flat.max(1).values,
        sensor_w_pixel.reshape(len(raw), -1).sum(1),
        coefficients.abs().square().sum(1),
    )
    summary = torch.stack(tuple(value.detach().to(device="cpu", dtype=torch.float64)
                                for value in columns), dim=1)
    _require(tuple(summary.shape) == (len(raw), 5)
             and torch.isfinite(summary).all().item(),
             "raw CMOS numerical summary is incomplete or nonfinite")
    return summary


def _forward_input_identity(rows: torch.Tensor, runtime: Mapping[str, Any], *,
                            expected_source_rows: torch.Tensor,
                            expected_labels: torch.Tensor) -> dict[str, torch.Tensor]:
    actual_rows = torch.as_tensor(rows, dtype=torch.long).detach().cpu().reshape(-1)
    expected_rows = torch.as_tensor(expected_source_rows, dtype=torch.long).detach().cpu().reshape(-1)
    expected_label_values = torch.as_tensor(expected_labels, dtype=torch.long).detach().cpu().reshape(-1)
    _require(len(actual_rows) > 0 and len(actual_rows) == len(expected_rows)
             and len(actual_rows) == len(expected_label_values)
             and len(set(actual_rows.tolist())) == len(actual_rows)
             and torch.equal(actual_rows, expected_rows),
             "actual image rows differ from the manifest source_train_row sequence")
    pixels = runtime["pixels"]
    labels = runtime["labels"]
    train_indices = runtime["train_indices"]
    _require(torch.is_tensor(pixels) and pixels.dtype == torch.uint8
             and pixels.ndim == 3 and tuple(pixels.shape[1:]) == (28, 28)
             and torch.is_tensor(labels) and labels.dtype == torch.long
             and tuple(labels.shape) == (len(pixels),)
             and torch.is_tensor(train_indices) and train_indices.dtype == torch.long,
             "native image source or dataset-role tensors are malformed")
    _require(bool(torch.isin(actual_rows, train_indices.detach().cpu()).all().item()),
             "actual source image row is outside TRAIN")
    actual_labels = labels.index_select(0, actual_rows)
    _require(torch.equal(actual_labels, expected_label_values),
             "actual source image labels differ from the manifest labels")
    source_pixels = pixels.index_select(0, actual_rows).clone()
    commands = image_batch(pixels, actual_rows, "small", runtime["device"])
    _require(tuple(commands.shape) == (len(actual_rows), 1, 200, 200)
             and commands.dtype == torch.float32
             and torch.isfinite(commands).all().item()
             and float(commands.min()) >= 0.0 and float(commands.max()) <= 1.0,
             "native SLM commands are incomplete or outside the legal amplitude range")
    return {
        "source_rows": actual_rows,
        "labels": actual_labels.detach().cpu().clone(),
        "source_pixels": source_pixels,
        "slm_commands": commands,
    }


def _positive_control_identity(manifest: Sequence[Mapping[str, Any]]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    controls = [row for row in manifest if row.get("positive_control")]
    indices = torch.tensor([int(row["population_index"]) for row in controls], dtype=torch.long)
    expected_indices = torch.tensor(
        [label * ROWS_PER_CLASS + offset for label in range(CLASS_COUNT) for offset in (0, 1)],
        dtype=torch.long,
    )
    _require(len(controls) == 20 and torch.equal(indices, expected_indices),
             "positive controls do not match the two fixed first rows per TRAIN class")
    rows = torch.tensor([int(row["source_train_row"]) for row in controls], dtype=torch.long)
    labels = torch.tensor([int(row["label"]) for row in controls], dtype=torch.long)
    _require(len(set(rows.tolist())) == len(rows)
             and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(2)),
             "positive-control manifest rows or labels are duplicated or misordered")
    return indices, rows, labels


def _validate_control_input_evidence(controls: Mapping[int, Mapping[str, torch.Tensor]],
                                     manifest: Sequence[Mapping[str, Any]],
                                     runtime: Mapping[str, Any], *,
                                     command_device: torch.device | None = None) -> dict[str, Any]:
    indices, expected_rows, expected_labels = _positive_control_identity(manifest)
    _require(set(controls) == set(PHASE_STEPS),
             "corrected positive controls omit a measured endpoint")
    _require(bool(torch.isin(expected_rows, runtime["train_indices"].detach().cpu()).all().item()),
             "positive-control manifest contains a source row outside TRAIN")
    _require(torch.equal(runtime["labels"].index_select(0, expected_rows), expected_labels),
             "positive-control manifest labels do not match the source TRAIN images")
    replay_device = command_device or runtime["device"]
    maximum_command_error = 0.0
    for step in PHASE_STEPS:
        item = controls[step]
        _require(torch.equal(item.get("population_indices"), indices),
                 f"step {step} controls have missing, duplicate, or reordered population rows")
        for grid in ("2048", "4096"):
            suffix = f"_{grid}"
            saved_rows = item.get(f"source_rows{suffix}")
            saved_labels = item.get(f"labels{suffix}")
            saved_pixels = item.get(f"source_pixels{suffix}")
            saved_commands = item.get(f"slm_commands{suffix}")
            _require(torch.is_tensor(saved_rows) and torch.equal(saved_rows, expected_rows)
                     and torch.is_tensor(saved_labels) and torch.equal(saved_labels, expected_labels),
                     f"step {step} {grid} input evidence is bound to the wrong source row or label")
            _require(torch.is_tensor(saved_pixels)
                     and tuple(saved_pixels.shape) == (20, 28, 28)
                     and saved_pixels.dtype == torch.uint8,
                     f"step {step} {grid} original source pixels are incomplete")
            expected_pixels = runtime["pixels"].index_select(0, expected_rows)
            _require(torch.equal(saved_pixels, expected_pixels),
                     f"step {step} {grid} source pixels do not match the declared TRAIN rows")
            expected_command = image_batch(
                runtime["pixels"], expected_rows, "small", replay_device
            ).detach().cpu()
            _require(torch.is_tensor(saved_commands)
                     and tuple(saved_commands.shape) == (20, 1, 200, 200)
                     and saved_commands.dtype == torch.float32
                     and torch.isfinite(saved_commands).all().item(),
                     f"step {step} {grid} measured SLM command is incomplete")
            for key, shape in ((f"coefficients{suffix}", (20, 100)),
                               (f"plane_powers_w{suffix}", (20, 5)),
                               (f"raw_summaries{suffix}", (20, 5)),
                               (f"raw_cmos{suffix}", (20, 1, 200, 200))):
                value = item.get(key)
                _require(torch.is_tensor(value) and tuple(value.shape) == shape
                         and torch.isfinite(value.real if torch.is_complex(value) else value).all().item()
                         and (not torch.is_complex(value) or torch.isfinite(value.imag).all().item()),
                         f"step {step} {grid} optical output is missing or invalid: {key}")
            command_error = float((saved_commands - expected_command).abs().max())
            maximum_command_error = max(maximum_command_error, command_error)
            _require(command_error <= 1.0e-6,
                     f"step {step} {grid} SLM command does not reproduce from its source pixels")
    return {
        "status": "PASSED", "rows_per_endpoint": 20,
        "endpoint_grid_pairs": 4, "source_rows_and_labels_checked": True,
        "original_pixels_checked": True, "actual_slm_commands_recomputed": True,
        "command_recompute_device": str(replay_device),
        "maximum_slm_command_absolute_error": maximum_command_error,
        "slm_command_absolute_tolerance": 1.0e-6,
    }


def _measure_rows(model: torch.nn.Module, rows: torch.Tensor,
                  runtime: Mapping[str, Any], *, expected_source_rows: torch.Tensor,
                  expected_labels: torch.Tensor, keep_raw: bool,
                  on_forward_batch: Callable[[int], None] | None = None
                  ) -> dict[str, torch.Tensor]:
    model_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    pieces: dict[str, list[torch.Tensor]] = {
        "coefficients": [], "plane_powers_w": [], "raw_summaries": [],
        "raw_cmos": [], "source_rows": [], "labels": [],
        "source_pixels": [], "slm_commands": [],
    }
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            batch_expected_rows = expected_source_rows[start:start + BATCH_SIZE]
            batch_expected_labels = expected_labels[start:start + BATCH_SIZE]
            identity = _forward_input_identity(
                batch_rows, runtime,
                expected_source_rows=batch_expected_rows,
                expected_labels=batch_expected_labels,
            )
            result = base._forward(model, identity["slm_commands"])
            if on_forward_batch is not None:
                on_forward_batch(len(batch_rows))
            coefficients = result["ols_coefficients"].detach().to(torch.complex128).cpu()
            planes = result["plane_powers_w"].detach().to(torch.float64).cpu()
            raw = result["raw_intensity_w_per_m2"].detach().to(torch.float64)
            sensor_w_pixel = result["physical_w_per_pixel"].detach().to(torch.float64)
            _require(tuple(coefficients.shape) == (len(batch_rows), 100)
                     and tuple(planes.shape) == (len(batch_rows), 5)
                     and tuple(raw.shape) == (len(batch_rows), 1, 200, 200)
                     and tuple(sensor_w_pixel.shape) == tuple(raw.shape)
                     and torch.isfinite(coefficients.real).all().item()
                     and torch.isfinite(coefficients.imag).all().item()
                     and torch.isfinite(planes).all().item()
                     and torch.isfinite(raw).all().item()
                     and bool((raw >= 0.0).all().item()),
                     "production image-to-raw-CMOS output is incomplete or nonfinite")
            raw_summary = _raw_output_summary(raw, sensor_w_pixel, coefficients)
            pieces["coefficients"].append(coefficients)
            pieces["plane_powers_w"].append(planes)
            pieces["raw_summaries"].append(raw_summary)
            pieces["source_rows"].append(identity["source_rows"])
            pieces["labels"].append(identity["labels"])
            if keep_raw:
                pieces["raw_cmos"].append(raw.detach().cpu())
                pieces["source_pixels"].append(identity["source_pixels"])
                pieces["slm_commands"].append(identity["slm_commands"].detach().cpu())
    _require(_same_state(model.state_dict(), model_before),
             "frozen optical model changed during inference")
    return {key: torch.cat(values) if values else torch.empty(0)
            for key, values in pieces.items()}


def _measure_endpoint(step: int, context: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]],
                      control: Mapping[str, Any], runtime: Mapping[str, Any],
                      output_dir: Path,
                      on_forward_batch: Callable[[int], None] | None = None
                      ) -> dict[str, torch.Tensor]:
    rows = torch.tensor([int(item["source_train_row"]) for item in manifest], dtype=torch.long)
    control_indices = control["population_indices"].long()
    control_rows = rows.index_select(0, control_indices)
    expected_labels = torch.tensor([int(item["label"]) for item in manifest], dtype=torch.long)
    _require(torch.equal(control.get("source_rows_2048"), control_rows)
             and torch.equal(control.get("labels_2048"), expected_labels.index_select(0, control_indices)),
             "control evidence is not bound to its manifest source rows and labels")
    result = {
        "source_rows": rows.clone(),
        "labels": torch.tensor([int(item["label"]) for item in manifest], dtype=torch.long),
        "coefficients": torch.empty((2_000, 100), dtype=torch.complex128),
        "plane_powers_w": torch.empty((2_000, 5), dtype=torch.float64),
        "raw_summaries": torch.empty((2_000, 5), dtype=torch.float64),
        "control_population_indices": control_indices.clone(),
        "raw_cmos_controls_2048": control["raw_cmos_2048"].clone(),
    }
    result["coefficients"][control_indices] = control["coefficients_2048"]
    result["plane_powers_w"][control_indices] = control["plane_powers_2048"]
    result["raw_summaries"][control_indices] = control["raw_summaries_2048"]
    remaining = torch.tensor([index for index in range(2_000)
                              if index not in set(control_indices.tolist())], dtype=torch.long)
    _require(len(remaining) == 1_980, "production preflight rows were not counted inside 2,000")
    model = _new_model(context["checkpoints"][step], GRID_PRODUCTION, runtime)
    total = len(remaining)
    for start in range(0, total, BATCH_SIZE):
        positions = remaining[start:start + BATCH_SIZE]
        batch_rows = rows.index_select(0, positions)
        measured = _measure_rows(
            model, batch_rows, runtime,
            expected_source_rows=batch_rows,
            expected_labels=expected_labels.index_select(0, positions),
            keep_raw=False,
                                 on_forward_batch=on_forward_batch)
        result["coefficients"][positions] = measured["coefficients"]
        result["plane_powers_w"][positions] = measured["plane_powers_w"]
        result["raw_summaries"][positions] = measured["raw_summaries"]
        if (start + len(positions)) % 200 == 0 or start + len(positions) == total:
            _write_json(output_dir / "measurement_progress.json", {
                "task_id": TASK_ID, "endpoint_step": step,
                "completed_production_rows_this_endpoint": 20 + start + len(positions),
                "authorized_production_rows_per_endpoint": 2_000,
                "optimizer_updates": 0, "official_test_reads": 0,
            })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _require(torch.isfinite(result["coefficients"].real).all().item()
             and torch.isfinite(result["coefficients"].imag).all().item()
             and torch.isfinite(result["plane_powers_w"]).all().item()
             and torch.isfinite(result["raw_summaries"]).all().item(),
             f"step {step} saved full-path measurement contains nonfinite rows")
    _save_torch(output_dir / f"train_optical_outputs_step_{step:05d}.pt", result, new=True)
    return result


def _write_training_per_sample(path: Path,
                               endpoints: Mapping[int, Mapping[str, torch.Tensor]],
                               manifest: Sequence[Mapping[str, Any]]) -> None:
    plane_names = ("encoder_1_w", "encoder_2_w", "generator_entry_w",
                   "decoder_1_w", "decoder_2_w")
    rows: list[dict[str, Any]] = []
    for step in PHASE_STEPS:
        endpoint = endpoints[step]
        for index, sample in enumerate(manifest):
            summary = endpoint["raw_summaries"][index].tolist()
            plane = endpoint["plane_powers_w"][index].tolist()
            row: dict[str, Any] = {
                "endpoint_step": step,
                "population_index": index,
                "label": int(sample["label"]),
                "class_position": int(sample["class_position"]),
                "source_train_row": int(sample["source_train_row"]),
                "schedule_step": int(sample["schedule_step"]),
                "role": str(sample["role"]),
                "positive_control": bool(sample["positive_control"]),
                "raw_cmos_mean_w_per_m2": summary[0],
                "raw_cmos_rms_w_per_m2": summary[1],
                "raw_cmos_max_w_per_m2": summary[2],
                "sensor_total_power_w": summary[3],
                "total_modal_coefficient_power": summary[4],
            }
            row.update({name: float(value) for name, value in zip(plane_names, plane, strict=True)})
            rows.append(row)
    fields = ["endpoint_step", "population_index", "label", "class_position",
              "source_train_row", "schedule_step", "role", "positive_control",
              "raw_cmos_mean_w_per_m2", "raw_cmos_rms_w_per_m2", "raw_cmos_max_w_per_m2",
              "sensor_total_power_w", "total_modal_coefficient_power", *plane_names]
    _write_csv(path, rows, fields)


def _validate_training_per_sample(path: Path,
                                  endpoints: Mapping[int, Mapping[str, torch.Tensor]],
                                  manifest: Sequence[Mapping[str, Any]]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 4_000,
             "saved TRAIN per-sample CSV does not contain both complete endpoints")
    plane_names = ("encoder_1_w", "encoder_2_w", "generator_entry_w",
                   "decoder_1_w", "decoder_2_w")
    cursor = 0
    for step in PHASE_STEPS:
        endpoint = endpoints[step]
        for index, sample in enumerate(manifest):
            row = rows[cursor]
            cursor += 1
            expected = {"endpoint_step": step, "population_index": index,
                        "label": int(sample["label"]),
                        "source_train_row": int(sample["source_train_row"]),
                        "role": str(sample["role"])}
            _require(all(row[key] == str(value) for key, value in expected.items()),
                     "TRAIN per-sample output has a missing, reordered, or mislabeled row")
            values = endpoint["raw_summaries"][index].tolist()
            names = ("raw_cmos_mean_w_per_m2", "raw_cmos_rms_w_per_m2",
                     "raw_cmos_max_w_per_m2", "sensor_total_power_w",
                     "total_modal_coefficient_power")
            for name, value in zip(names, values, strict=True):
                _require(math.isclose(float(row[name]), float(value), rel_tol=1.0e-12,
                                      abs_tol=1.0e-30),
                         f"TRAIN per-sample saved {name} differs from its production tensor")
            for name, value in zip(plane_names, endpoint["plane_powers_w"][index].tolist(), strict=True):
                _require(math.isclose(float(row[name]), float(value), rel_tol=1.0e-12,
                                      abs_tol=1.0e-30),
                         f"TRAIN per-sample saved plane power differs: {name}")


def _whiten_unit_torch(coefficients: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = coefficients.to(torch.complex128)
    s = scale.detach().cpu().to(torch.float64)
    _require(z.ndim == 2 and z.shape[1] == 100 and s.shape == (100,)
             and torch.isfinite(z.real).all().item() and torch.isfinite(z.imag).all().item()
             and torch.isfinite(s).all().item() and bool((s > 0.0).all().item()),
             "complex coefficients or frozen modal scale are invalid")
    whitened = z / torch.sqrt(s).to(torch.complex128)
    norms = torch.linalg.vector_norm(whitened, dim=1)
    _require(torch.isfinite(norms).all().item() and bool((norms > 0.0).all().item()),
             "zero or nonfinite whitened complex field")
    return whitened / norms[:, None]


def _nearest_torch(query: torch.Tensor, reference: torch.Tensor, scale: torch.Tensor,
                   reference_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    q = _whiten_unit_torch(query, scale)
    r = _whiten_unit_torch(reference, scale)
    inner = q.conj() @ r.T
    complex_sq = (2.0 - 2.0 * inner.abs()).clamp_min(0.0)
    q_amp, r_amp = q.abs(), r.abs()
    amplitude_sq = (2.0 - 2.0 * (q_amp @ r_amp.T)).clamp_min(0.0)
    index = complex_sq.argmin(dim=1)
    arange = torch.arange(len(q))
    nearest_complex_sq = complex_sq[arange, index]
    nearest_amplitude_sq = amplitude_sq[arange, index]
    phase_sq = (nearest_complex_sq - nearest_amplitude_sq).clamp_min(0.0)
    return {
        "distance": torch.sqrt(nearest_complex_sq),
        "amplitude_distance": torch.sqrt(nearest_amplitude_sq),
        "relative_phase_penalty": torch.sqrt(phase_sq),
        "nearest_reference_index": index,
        "nearest_reference_row": reference_rows[index],
    }


def _bootstrap_torch_distances(distances: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]]
                               ) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_COUNT, 3), dtype=np.float64)
    for replicate in range(BOOTSTRAP_COUNT):
        generated_means = {step: [] for step in PHASE_STEPS}
        calibration_means = {step: [] for step in PHASE_STEPS}
        for label in range(CLASS_COUNT):
            generated_index = rng.integers(0, GENERATED_PER_CLASS, size=GENERATED_PER_CLASS)
            calibration_index = rng.integers(0, CALIBRATION_PER_CLASS,
                                             size=CALIBRATION_PER_CLASS)
            for step in PHASE_STEPS:
                generated = distances["generated"][str(step)][label].numpy()
                calibration = distances["calibration"][str(step)][label].numpy()
                generated_means[step].append(float(generated[generated_index].mean()))
                calibration_means[step].append(float(calibration[calibration_index].mean()))
        g0 = float(np.mean(generated_means[0]) - np.mean(calibration_means[0]))
        g1 = float(np.mean(generated_means[2_000]) - np.mean(calibration_means[2_000]))
        draws[replicate] = (g0, g1, g1 - g0)
    point = {}
    for step in PHASE_STEPS:
        point[str(step)] = float(np.mean([
            distances["generated"][str(step)][label].mean().item()
            - distances["calibration"][str(step)][label].mean().item()
            for label in range(CLASS_COUNT)
        ]))
    point["delta_2000_minus_0"] = point["2000"] - point["0"]
    intervals = {
        name: {"lower": float(np.quantile(draws[:, index], 0.025, method="linear")),
               "upper": float(np.quantile(draws[:, index], 0.975, method="linear"))}
        for name, index in (("G_0", 0), ("G_2000", 1), ("DeltaG", 2))
    }
    return {
        "point_estimates": {"G_0": point["0"], "G_2000": point["2000"],
                            "DeltaG": point["delta_2000_minus_0"]},
        "confidence_intervals_95": intervals,
        "bootstrap_resamples": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED,
        "unit_of_resampling": "within-class generated and calibration rows independently",
        "paired_across_endpoints": True,
        "fixed_reference_per_class": REFERENCE_PER_CLASS,
        "interval_scope": "conditional on the fixed 100-row TRAIN reference set per class",
        "draws": torch.from_numpy(draws),
    }


def _route_decision(intervals: Mapping[str, Mapping[str, float]]) -> str:
    terminal_lower = float(intervals["G_2000"]["lower"])
    delta_lower = float(intervals["DeltaG"]["lower"])
    if terminal_lower > 0.0 and delta_lower > 0.0:
        return "DIFFERENCE_EXPANDED"
    if terminal_lower > 0.0:
        return "DIFFERENCE_PRESENT_NOT_CONFIRMED_EXPANDED"
    return "EVIDENCE_INSUFFICIENT"


def _numpy_whiten_unit(coefficients: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = np.asarray(coefficients, dtype=np.complex128)
    s = np.asarray(scale, dtype=np.float64)
    _require(z.ndim == 2 and z.shape[1] == 100 and s.shape == (100,)
             and np.isfinite(z.real).all() and np.isfinite(z.imag).all()
             and np.isfinite(s).all() and (s > 0.0).all(),
             "independent NumPy consumer received invalid coefficients or scale")
    whitened = z / np.sqrt(s)[None, :]
    norms = np.linalg.norm(whitened, axis=1)
    _require(np.isfinite(norms).all() and (norms > 0.0).all(),
             "independent NumPy consumer found zero/nonfinite fields")
    return whitened / norms[:, None]


def _nearest_numpy(query: np.ndarray, reference: np.ndarray, scale: np.ndarray,
                   reference_rows: np.ndarray) -> dict[str, np.ndarray]:
    q = _numpy_whiten_unit(query, scale)
    r = _numpy_whiten_unit(reference, scale)
    inner = np.conjugate(q) @ r.T
    complex_sq = np.maximum(2.0 - 2.0 * np.abs(inner), 0.0)
    q_amp, r_amp = np.abs(q), np.abs(r)
    amplitude_sq = np.maximum(2.0 - 2.0 * (q_amp @ r_amp.T), 0.0)
    indices = np.argmin(complex_sq, axis=1)
    rows = np.arange(len(q))
    complex_distance_sq = complex_sq[rows, indices]
    amplitude_distance_sq = amplitude_sq[rows, indices]
    return {
        "distance": np.sqrt(complex_distance_sq),
        "amplitude_distance": np.sqrt(amplitude_distance_sq),
        "relative_phase_penalty": np.sqrt(np.maximum(complex_distance_sq - amplitude_distance_sq, 0.0)),
        "nearest_reference_index": indices,
        "nearest_reference_row": reference_rows[indices],
    }


def _bootstrap_numpy(distances: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]]
                     ) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((BOOTSTRAP_COUNT, 3), dtype=np.float64)
    for replicate in range(BOOTSTRAP_COUNT):
        gen_class = {step: np.empty(CLASS_COUNT, dtype=np.float64) for step in PHASE_STEPS}
        cal_class = {step: np.empty(CLASS_COUNT, dtype=np.float64) for step in PHASE_STEPS}
        for label in range(CLASS_COUNT):
            gen_index = rng.integers(0, GENERATED_PER_CLASS, size=GENERATED_PER_CLASS)
            cal_index = rng.integers(0, CALIBRATION_PER_CLASS, size=CALIBRATION_PER_CLASS)
            for step in PHASE_STEPS:
                gen_class[step][label] = distances["generated"][str(step)][label][gen_index].mean()
                cal_class[step][label] = distances["calibration"][str(step)][label][cal_index].mean()
        g0 = float(gen_class[0].mean() - cal_class[0].mean())
        g1 = float(gen_class[2_000].mean() - cal_class[2_000].mean())
        draws[replicate] = (g0, g1, g1 - g0)
    point = {}
    for step in PHASE_STEPS:
        point[str(step)] = float(np.mean([
            distances["generated"][str(step)][label].mean()
            - distances["calibration"][str(step)][label].mean()
            for label in range(CLASS_COUNT)
        ]))
    intervals = {
        name: {"lower": float(np.quantile(draws[:, index], 0.025, method="linear")),
               "upper": float(np.quantile(draws[:, index], 0.975, method="linear"))}
        for name, index in (("G_0", 0), ("G_2000", 1), ("DeltaG", 2))
    }
    result = {
        "point_estimates": {"G_0": point["0"], "G_2000": point["2000"],
                            "DeltaG": point["2000"] - point["0"]},
        "confidence_intervals_95": intervals,
        "bootstrap_resamples": BOOTSTRAP_COUNT, "seed": BOOTSTRAP_SEED,
        "unit_of_resampling": "within-class generated and calibration rows independently",
        "paired_across_endpoints": True,
        "fixed_reference_per_class": REFERENCE_PER_CLASS,
        "interval_scope": "conditional on the fixed 100-row TRAIN reference set per class",
    }
    return result, draws


def _collect_distance_analysis(train_outputs: Mapping[int, Mapping[str, torch.Tensor]],
                               generation: Mapping[str, Any], scale: torch.Tensor,
                               manifest: Sequence[Mapping[str, Any]], *, numpy_only: bool
                               ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = np.asarray([int(item["source_train_row"]) for item in manifest], dtype=np.int64)
    labels = np.asarray([int(item["label"]) for item in manifest], dtype=np.int64)
    roles = np.asarray([str(item["role"]) for item in manifest])
    s = scale.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(scale) else np.asarray(scale)
    distances: dict[str, dict[str, dict[int, Any]]] = {
        "generated": {"0": {}, "2000": {}},
        "calibration": {"0": {}, "2000": {}},
    }
    records: list[dict[str, Any]] = []
    class_metrics: dict[str, Any] = {}
    target = generation["target_coefficients"].detach().cpu().numpy()
    predicted = generation["predictions"]
    for step in PHASE_STEPS:
        key = str(step)
        train_z = train_outputs[step]["coefficients"].detach().cpu().numpy()
        gen_z = generation["actual_coefficients"][step].detach().cpu().numpy()
        _require(train_z.shape == (2_000, 100) and gen_z.shape == (2_000, 100),
                 f"step {step} coefficient arrays have missing or extra rows")
        endpoint_classes = {}
        for label in range(CLASS_COUNT):
            reference_mask = (labels == label) & (roles == "reference")
            calibration_mask = (labels == label) & (roles == "calibration")
            generated_mask = generation["labels"].numpy() == label
            reference_rows = rows[reference_mask]
            reference = train_z[reference_mask]
            calibration_rows = rows[calibration_mask]
            calibration = train_z[calibration_mask]
            generated = gen_z[generated_mask]
            if numpy_only:
                cal_nearest = _nearest_numpy(calibration, reference, s, reference_rows)
                gen_nearest = _nearest_numpy(generated, reference, s, reference_rows)
            else:
                cal_nearest_t = _nearest_torch(torch.from_numpy(calibration), torch.from_numpy(reference),
                                               torch.from_numpy(s), torch.from_numpy(reference_rows))
                gen_nearest_t = _nearest_torch(torch.from_numpy(generated), torch.from_numpy(reference),
                                               torch.from_numpy(s), torch.from_numpy(reference_rows))
                cal_nearest = {name: value.detach().cpu().numpy() for name, value in cal_nearest_t.items()}
                gen_nearest = {name: value.detach().cpu().numpy() for name, value in gen_nearest_t.items()}
            _require(len(reference) == 100 and len(calibration) == 100 and len(generated) == 200,
                     f"class {label} reference/calibration/generation counts changed")
            distances["calibration"][key][label] = np.asarray(cal_nearest["distance"], dtype=np.float64)
            distances["generated"][key][label] = np.asarray(gen_nearest["distance"], dtype=np.float64)
            train_power = np.abs(train_z[labels == label]) ** 2
            gen_power = np.abs(generated) ** 2
            prior_target = target[generation["labels"].numpy() == label]
            prior_delta = generated - prior_target
            norm_target = np.linalg.norm(prior_target, axis=1)
            relative_target_error = np.linalg.norm(prior_delta, axis=1) / np.maximum(norm_target, 1.0e-30)
            phase_target_error = np.asarray([
                projected._phase_error(generated[index], prior_target[index])
                for index in range(len(generated))
            ], dtype=np.float64)
            sem = []
            for index in np.flatnonzero(generation["labels"].numpy() == label):
                prediction = predicted[int(index)]
                ed_key, ec_key = f"{('E0G0' if step == 0 else 'E1G1')}_ED_prediction", f"{('E0G0' if step == 0 else 'E1G1')}_EC_prediction"
                ed, ec = prediction[ed_key], prediction[ec_key]
                sem.append({"ed_correct": ed == label, "ec_correct": ec == label,
                            "evaluators_disagree": ed != ec})
            endpoint_classes[str(label)] = {
                "reference_rows": reference_rows.tolist(), "calibration_rows": calibration_rows.tolist(),
                "generated_count": int(len(generated)),
                "G_class_contribution": float(np.mean(gen_nearest["distance"])
                                               - np.mean(cal_nearest["distance"])),
                "generated_nearest_distance_mean": float(np.mean(gen_nearest["distance"])),
                "generated_nearest_distance_median": float(np.median(gen_nearest["distance"])),
                "generated_nearest_distance_p95": float(np.quantile(gen_nearest["distance"], 0.95)),
                "calibration_nearest_distance_mean": float(np.mean(cal_nearest["distance"])),
                "calibration_nearest_distance_median": float(np.median(cal_nearest["distance"])),
                "calibration_nearest_distance_p95": float(np.quantile(cal_nearest["distance"], 0.95)),
                "generated_amplitude_distance_mean": float(np.mean(gen_nearest["amplitude_distance"])),
                "generated_relative_phase_penalty_mean": float(np.mean(gen_nearest["relative_phase_penalty"])),
                "train_modal_coefficient_power_mean": float(train_power.sum(1).mean()),
                "generated_modal_coefficient_power_mean": float(gen_power.sum(1).mean()),
                "train_modal_amplitude_mean_by_mode": np.mean(np.abs(train_z[labels == label]), axis=0).tolist(),
                "generated_modal_amplitude_mean_by_mode": np.mean(np.abs(generated), axis=0).tolist(),
                "prior_target_relative_complex_error_mean": float(relative_target_error.mean()),
                "prior_target_phase_aligned_complex_error_mean": float(phase_target_error.mean()),
                "E_D_correct_count": int(sum(item["ed_correct"] for item in sem)),
                "E_C_correct_count": int(sum(item["ec_correct"] for item in sem)),
                "E_D_E_C_disagreement_count": int(sum(item["evaluators_disagree"] for item in sem)),
            }
            for sample_index, distance_block, kind, query_rows in (
                (np.flatnonzero(generated_mask), gen_nearest, "generated", np.full(200, -1)),
                (np.flatnonzero(calibration_mask), cal_nearest, "train_calibration", calibration_rows),
            ):
                for local, index in enumerate(sample_index):
                    row: dict[str, Any] = {
                        "endpoint_step": step, "population_role": kind,
                        "label": label, "population_index": int(index),
                        "source_train_row": int(query_rows[local]) if kind == "train_calibration" else "",
                        "nearest_reference_train_row": int(distance_block["nearest_reference_row"][local]),
                        "complex_distance": float(distance_block["distance"][local]),
                        "amplitude_distance": float(distance_block["amplitude_distance"][local]),
                        "relative_phase_penalty": float(distance_block["relative_phase_penalty"][local]),
                    }
                    if kind == "generated":
                        prediction = predicted[int(index)]
                        condition = "E0G0" if step == 0 else "E1G1"
                        row.update({
                            "proposal_id": int(generation["proposal_ids"][index]),
                            "E_D_prediction": prediction[f"{condition}_ED_prediction"],
                            "E_C_prediction": prediction[f"{condition}_EC_prediction"],
                            "E_D_correct": prediction[f"{condition}_ED_prediction"] == label,
                            "E_C_correct": prediction[f"{condition}_EC_prediction"] == label,
                            "evaluators_disagree": prediction[f"{condition}_ED_prediction"]
                            != prediction[f"{condition}_EC_prediction"],
                            "prior_target_relative_complex_error": float(relative_target_error[local]),
                            "prior_target_phase_aligned_error": float(phase_target_error[local]),
                            "total_modal_coefficient_power": float(gen_power[local].sum()),
                        })
                    records.append(row)
        class_metrics[key] = endpoint_classes
    return class_metrics, distances, records


def _modal_amplitude_rows(train_outputs: Mapping[int, Mapping[str, torch.Tensor]],
                          generation: Mapping[str, Any], cache: Mapping[str, Any],
                          manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = np.asarray([int(item["label"]) for item in manifest], dtype=np.int64)
    roles = np.asarray([str(item["role"]) for item in manifest])
    output: list[dict[str, Any]] = []
    for step in PHASE_STEPS:
        z = train_outputs[step]["coefficients"].detach().cpu().numpy()
        gen = generation["actual_coefficients"][step].detach().cpu().numpy()
        for label in range(CLASS_COUNT):
            role_groups = {
                "train_reference": z[(labels == label) & (roles == "reference")],
                "train_calibration": z[(labels == label) & (roles == "calibration")],
                "prior_generation": gen[generation["labels"].numpy() == label],
            }
            cache_mask = cache["labels"].numpy() == label
            role_groups["matrix_reencoded_train_cache_55000"] = cache["coefficients"][cache_mask].numpy()
            for group, group_z in role_groups.items():
                amplitude = np.abs(group_z)
                for mode in range(100):
                    output.append({
                        "endpoint_step": step, "label": label, "population": group,
                        "mode_index": mode, "sample_count": int(len(group_z)),
                        "amplitude_mean": float(amplitude[:, mode].mean()),
                        "amplitude_q01": float(np.quantile(amplitude[:, mode], 0.01)),
                        "amplitude_median": float(np.median(amplitude[:, mode])),
                        "amplitude_q99": float(np.quantile(amplitude[:, mode], 0.99)),
                    })
    return output


def _generation_raw_summaries(generation: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    bundle = generation["bundle"]
    summaries: dict[int, dict[str, np.ndarray]] = {}
    for step, condition in ((0, "E0G0"), (2_000, "E1G1")):
        raw = bundle["conditions"][condition]["raw_cmos"]
        rows = []
        for start in range(0, len(raw), 100):
            values = raw[start:start + 100].to(torch.float64).reshape(-1, raw[0].numel())
            rows.append(torch.stack((values.mean(1), values.square().mean(1).sqrt(),
                                     values.max(1).values), dim=1))
        summaries[step] = {"raw_cmos_summary": torch.cat(rows).detach().cpu().numpy()}
    delta = []
    raw0 = bundle["conditions"]["E0G0"]["raw_cmos"]
    raw1 = bundle["conditions"]["E1G1"]["raw_cmos"]
    _require(len(raw0) == len(raw1) and len(raw0) > 0,
             "generation endpoint raw CMOS populations are not row aligned")
    for start in range(0, len(raw0), 100):
        a = raw0[start:start + 100].to(torch.float64).reshape(-1, raw0[0].numel())
        b = raw1[start:start + 100].to(torch.float64).reshape(-1, raw1[0].numel())
        delta.append((b - a).square().mean(1).sqrt())
    delta_values = torch.cat(delta).detach().cpu().numpy()
    summaries[0]["terminal_raw_cmos_delta_rms"] = delta_values
    summaries[2_000]["terminal_raw_cmos_delta_rms"] = delta_values
    return summaries


def _cache_crosscheck(context: Mapping[str, Any], train_outputs: Mapping[int, Mapping[str, torch.Tensor]],
                      manifest: Sequence[Mapping[str, Any]]
                      ) -> dict[str, Any]:
    cache = context["cache"]
    mapping = _cache_map(context)
    chosen = list(range(len(manifest)))
    source_rows = [int(manifest[index]["source_train_row"]) for index in chosen]
    cache_indices = torch.tensor([mapping[row] for row in source_rows], dtype=torch.long)
    measured = train_outputs[0]["coefficients"][chosen]
    cached = cache["coefficients"].index_select(0, cache_indices)
    errors = _relative_coefficient_error(measured, cached)
    passed = bool(float(errors.max()) <= CACHE_TOLERANCE)
    summary = {
        "source_role": "matrix_reencoding_of_55000_TRAIN_rows_with_existing_100_full_optics_validation",
        "rows_compared": len(chosen), "relative_error_definition":
        "norm(matrix_reencoded_minus_full_optics)/norm(full_optics)",
        "tolerance": CACHE_TOLERANCE,
        "maximum_relative_error": float(errors.max()),
        "p95_relative_error": float(torch.quantile(errors, 0.95)),
        "passed": passed,
        "status": "PASSED" if passed else "DIAGNOSTIC_MISMATCH",
        "blocking_gate": False,
        "decision_use": "diagnostic_only; this cache is not a step-2000 TRAIN reference",
        "mismatch_policy": "report_as_diagnostic_limitation",
        "step_2000_cache_used_as_reference": False,
        "historical_100_row_full_optics_max_relative_error": context["cache_replay_max_relative_error"],
    }
    return summary


def _write_asset_table(config: Mapping[str, Any], context: Mapping[str, Any],
                       output_dir: Path) -> None:
    entries = [
        {"asset": "D0 step 0 and 2,000 optical checkpoints plus saved TRAIN schedule",
         "source": config["sources"]["training_run"],
         "use": "fixed full-path TRAIN image measurements; no optimizer object or update; 40 preflight rows reused from attempt_02"},
        {"asset": "attempt_02 saved 2048/4096 representative controls",
         "source": config["sources"]["preflight_controls_run"] + "/cross_grid_controls.pt",
         "use": "revalidated against attempt_02 source identity, exact TRAIN manifest, and 1e-4 report; reused without rerunning"},
        {"asset": "saved final prior targets, legal SLM commands, and generated optical outputs",
         "source": config["sources"]["training_run"] + "/final_population_and_commands.pt",
         "use": "reuse consumed 2,000-row generation population and validate command identity"},
        {"asset": "four-condition generation outputs, phase provenance, and frozen E_D/E_C predictions",
         "source": config["sources"]["attribution_run"],
         "use": "reuse E0G0/E1G1 coefficients and existing evaluator outcomes; no evaluator rerun"},
        {"asset": "55,000-row physical-phase TRAIN coefficient cache",
         "source": config["sources"]["train_reference_backup" if context["local_sources"]
                             else "train_reference_run"] + "/raw_physical_phase_train_coefficients.pt",
         "use": "matrix-reencoded C45k context and step-0 cross-check only; not step-2,000 TRAIN data"},
        {"asset": "existing 100-row full-optics versus matrix replay",
         "source": config["sources"]["train_reference_backup" if context["local_sources"]
                             else "train_reference_run"] + "/source_matrix_replay_100.pt",
         "use": "verify historical 100-row replay only; the separate attempt_02 cache comparison mismatch remains a diagnostic limitation"},
        {"asset": "modal scale, Gaussian OLS, SLM input adapter, physical constants, fixed camera/evaluators",
         "source": config["sources"]["source_config"],
         "use": "frozen runtime identity and metric whitening scale"},
    ]
    _write_json(output_dir / "asset_reuse_table.json", {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "population_role_now": "consumed_train_diagnostic",
        "entries": entries,
    })


def _analyze(context: Mapping[str, Any], generation: Mapping[str, Any],
             train_outputs: Mapping[int, Mapping[str, torch.Tensor]],
             manifest: Sequence[Mapping[str, Any]], output_dir: Path,
             *, numpy_only: bool) -> dict[str, Any]:
    scale = context["runtime"]["scale"].detach().cpu().to(torch.float64)
    gen_coefficients = {
        0: generation["bundle"]["conditions"]["E0G0"]["coefficients"].detach().cpu().to(torch.complex128),
        2_000: generation["bundle"]["conditions"]["E1G1"]["coefficients"].detach().cpu().to(torch.complex128),
    }
    generation = dict(generation)
    generation["actual_coefficients"] = gen_coefficients
    class_metrics, distance_sets, distance_rows = _collect_distance_analysis(
        train_outputs, generation, scale, manifest, numpy_only=numpy_only
    )
    if numpy_only:
        bootstrap, bootstrap_draws = _bootstrap_numpy(distance_sets)
    else:
        torch_distances = {
            cohort: {step: {label: torch.from_numpy(values)
                            for label, values in per_class.items()}
                     for step, per_class in by_step.items()}
            for cohort, by_step in distance_sets.items()
        }
        bootstrap = _bootstrap_torch_distances(torch_distances)
        bootstrap_draws = bootstrap.pop("draws").numpy()
    decision = _route_decision(bootstrap["confidence_intervals_95"])
    raw_summary = _generation_raw_summaries(generation)
    generated_labels = generation["labels"].numpy()
    for step in PHASE_STEPS:
        for label in range(CLASS_COUNT):
            selected = generated_labels == label
            raw = raw_summary[step]["raw_cmos_summary"][selected]
            class_metrics[str(step)][str(label)].update({
                "generated_raw_cmos_mean_w_per_m2": float(raw[:, 0].mean()),
                "generated_raw_cmos_rms_w_per_m2": float(np.sqrt(np.mean(raw[:, 1] ** 2))),
                "generated_raw_cmos_max_w_per_m2": float(raw[:, 2].max()),
                "terminal_minus_start_raw_cmos_delta_rms_mean_w_per_m2": float(
                    raw_summary[step]["terminal_raw_cmos_delta_rms"][selected].mean()
                ),
            })
    sample_rows: list[dict[str, Any]] = []
    for row in distance_rows:
        if row["population_role"] != "generated":
            sample_rows.append(row)
            continue
        index = int(row["population_index"])
        step = int(row["endpoint_step"])
        raw_stats = raw_summary[step]["raw_cmos_summary"][index]
        row["raw_cmos_mean_w_per_m2"] = float(raw_stats[0])
        row["raw_cmos_rms_w_per_m2"] = float(raw_stats[1])
        row["raw_cmos_max_w_per_m2"] = float(raw_stats[2])
        row["terminal_raw_cmos_delta_rms_w_per_m2"] = float(
            raw_summary[step]["terminal_raw_cmos_delta_rms"][index]
        )
        row["original_population_role"] = "consumed_diagnostic_population"
        sample_rows.append(row)
    modal_rows = _modal_amplitude_rows(train_outputs, generation, context["cache"], manifest)
    summary = {
        "task_id": TASK_ID,
        "status": "COMPLETE_DIAGNOSTIC",
        "population_role": "consumed_train_diagnostic",
        "generation_population_role": "consumed_diagnostic_population",
        "distance_definition": {
            "whitening": "z/sqrt(frozen_positive_modal_scale)",
            "normalization": "unit_complex_L2_after_whitening",
            "distance": "sqrt(max(2-2*abs(vdot(u,v)),0))",
            "invariance": "one global complex phase and total amplitude scale",
            "preserves": "modal amplitude profile and inter-port relative phase",
            "reference": "nearest among fixed same-class 100 TRAIN rows at each endpoint",
        },
        "primary": bootstrap,
        "decision_route": decision,
        "route_interpretation": {
            "DIFFERENCE_EXPANDED": "G_2000 and DeltaG 95% lower bounds are both positive; supports designing a separate frozen-adapter/terminal-Encoder single-variable study.",
            "DIFFERENCE_PRESENT_NOT_CONFIRMED_EXPANDED": "G_2000 lower bound is positive but expansion is unconfirmed; retain support difference without attributing it to the 2,000 optical updates.",
            "EVIDENCE_INSUFFICIENT": "G_2000 lower bound is not positive; do not assign this explanation to a module or infer equal distributions.",
        }[decision],
        "per_class": class_metrics,
        "measurement": {
            "train_rows": 2_000, "train_rows_per_class": 200,
            "train_reference_rows": 1_000, "train_calibration_rows": 1_000,
            "generated_rows": 2_000, "endpoints": list(PHASE_STEPS),
            "production_full_path_rows": 4_000, "production_forwards_new": 40,
            "production_forwards_reused": 3_960, "cross_grid_rows": 40,
            "cross_grid_reference_forwards_new": 40,
            "optimizer_updates": 0, "optimizer_instances": 0,
            "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        },
        "interpretation_limits": [
            "Bootstrap intervals are conditional on the fixed 100-row same-class TRAIN reference set.",
            "The 55,000-row cache is matrix-reencoded C45k context with an existing 100-row full-optics validation; it is not a measured step-2,000 TRAIN distribution.",
            "Existing E_D/E_C correctness and disagreement are associations, not causal evidence.",
            "This numerical diagnostic does not establish human readability, model adoption, or hardware transfer.",
        ],
    }
    _write_json(output_dir / "input_distribution_analysis.json", summary)
    _save_torch(output_dir / "bootstrap_draws.pt", torch.from_numpy(bootstrap_draws), new=True)
    fields = sorted({key for row in sample_rows for key in row})
    _write_csv(output_dir / "per_sample_distance_results.csv", sample_rows, fields)
    _write_csv(output_dir / "modal_amplitude_distribution.csv", modal_rows,
               ["endpoint_step", "label", "population", "mode_index", "sample_count",
                "amplitude_mean", "amplitude_q01", "amplitude_median", "amplitude_q99"])
    return summary


def _snapshot_runtime(context: Mapping[str, Any]) -> dict[str, Any]:
    runtime = context["runtime"]
    keys = ("prior", "projector", "mapper_checkpoint", "matrix", "inverse",
            "scale", "camera_gain", "pixels", "labels", "train_indices", "dev_indices")
    return {
        "runtime": copy.deepcopy({key: runtime[key] for key in keys}) | {
            "response_matrix": copy.deepcopy(runtime["input_bundle"]["response_matrix"])
        },
        "checkpoints": {step: {"model": copy.deepcopy(context["checkpoints"][step]["model"]),
                               "optimizer": copy.deepcopy(context["checkpoints"][step]["optimizer"])}
                        for step in PHASE_STEPS},
        "evaluators": {name: copy.deepcopy(model.state_dict())
                       for name, model in (("ED", runtime["input_bundle"]["evaluator_ed"]),
                                           ("EC", runtime["input_bundle"]["evaluator_ec"]))},
    }


def _assert_snapshot_unchanged(context: Mapping[str, Any], snapshot: Mapping[str, Any],
                               models: Mapping[int, torch.nn.Module] | None = None) -> None:
    for step in PHASE_STEPS:
        _require(_same_state(context["checkpoints"][step]["model"], snapshot["checkpoints"][step]["model"])
                 and _same_state(context["checkpoints"][step]["optimizer"], snapshot["checkpoints"][step]["optimizer"]),
                 f"saved source model or Adam state changed at step {step}")
        if models is not None and step in models:
            _require(_same_state(models[step].state_dict(), snapshot["checkpoints"][step]["model"]),
                     f"inference model copy changed at step {step}")
    for key, value in snapshot["runtime"].items():
        actual = (context["runtime"]["input_bundle"]["response_matrix"]
                  if key == "response_matrix" else context["runtime"][key])
        _require(_same_state(actual, value),
                 f"frozen source input changed during numerical forwards: {key}")
    for name, model in (("ED", context["runtime"]["input_bundle"]["evaluator_ed"]),
                        ("EC", context["runtime"]["input_bundle"]["evaluator_ec"])):
        _require(_same_state(model.state_dict(), snapshot["evaluators"][name]),
                 f"frozen evaluator {name} state changed")


def _task_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID, "risk": "high",
        "decision_consumer": "input_distribution_analysis.json after Windows independent recomputation",
        "behavioral_consequence": "A positive support-gap result can prioritize one later adapter/Encoder study; this task does not train or alter any model.",
        "scope": {
            "repository_changes": [CONFIG_PATH.relative_to(ROOT).as_posix(),
                                   Path(__file__).relative_to(ROOT).as_posix()],
            "run_output": OUTPUT_DIR.relative_to(ROOT).as_posix(),
            "existing_source_artifacts_modified": False,
        },
        "invariants": [
            "Full image input path is native SLM input to Encoder to Gaussian OLS to Generator to raw CMOS.",
            "Only two existing D0 checkpoints and the original 2,000-row saved TRAIN schedule are used.",
            "The existing generation commands, outputs, labels, phase sources, prior, mapper, projector, scale, gain, and evaluators remain fixed.",
            "No optimizer is instantiated; zero parameter updates, zero prior refits/samples, and zero official TEST reads.",
            "Attempt_03's 3,960 non-control outputs are reused unchanged; its 40 incorrectly bound controls remain historical failure evidence and are excluded from the repaired population.",
            "The 40 corrected controls bind manifest population positions to source_train_row before image selection, and record actual labels, original pixels, and the SLM command consumed by the optical forward.",
            "All 2,000 corrected step-0 coefficients must pass the unchanged 1e-6 cache comparison; any failure invalidates the measurement and is not downgraded to a diagnostic limitation.",
            "Exactly 40 new 2048-grid and 40 new 4096-grid forwards repair the controls; no other production forwards are run.",
            "The final route uses the saved measured coefficients and fixed 2,000-resample paired bootstrap contract.",
        ],
        "verification": [
            "The actual source row, dataset label, raw pixels, and consumed native SLM command are checked before every repaired control forward and independently reconsumed on Windows.",
            "The 40 repaired controls pass the unchanged 1e-4 cross-grid comparisons; the other 3,960 production outputs are verified bit-identical to attempt_03.",
            "All 2,000 corrected step-0 production coefficients pass the unchanged 1e-6 cache comparison or the measurement stops as invalid.",
            "A Windows NumPy consumer reloads sources and outputs, independently reselects rows, recomputes distances, bootstrap intervals, and decision.",
            "The consumer rejects incomplete/misaligned rows, commands, labels, or phase provenance.",
        ],
        "stop": "One input-distribution route decision; no training, loss changes, prior refit, population refresh, model expansion, or hardware operation.",
    }


def _write_closeout(summary: Mapping[str, Any], review: Mapping[str, Any] | None,
                    output_dir: Path) -> None:
    route = str(summary["decision_route"])
    primary = summary["primary"]
    intervals = primary["confidence_intervals_95"]
    estimates = primary["point_estimates"]
    measurement = summary["measurement"]
    verified = review is not None
    result = "完成 / 科学路线结论有效" if verified else "测量完成 / 等待 Windows 独立复算"
    science = {
        "DIFFERENCE_EXPANDED": "在固定TRAIN参考集合下，终点生成输入比TRAIN校准分布更远，且差距相对step 0扩大；支持单独设计冻结输入适配器与终点Encoder配合关系的单变量研究。",
        "DIFFERENCE_PRESENT_NOT_CONFIRMED_EXPANDED": "在固定TRAIN参考集合下支持终点分布差异，但未证实2,000步更新扩大差异；不将差异归因于本次更新。",
        "EVIDENCE_INSUFFICIENT": "当前固定参考和人口未支持终点分布差异；这不证明两个分布相同，也不指定责任模块。",
    }[route]
    next_step = {
        "DIFFERENCE_EXPANDED": "建议另行设计冻结mapper和投影、只改变Encoder端点的一变量复核；本轮没有授权执行该后续实验。",
        "DIFFERENCE_PRESENT_NOT_CONFIRMED_EXPANDED": "建议如需继续，先设计不更新模型的输入适配器兼容性检查；不把本次变化作为训练责任证据。后续工作需单独规划。",
        "EVIDENCE_INSUFFICIENT": "不依据本次结果启动模块训练或改损失；保留现有权重与人口身份，若继续须先提出能区分其他解释的单变量问题。",
    }[route]
    cache = (review or {}).get("step_0_cache_all_2000_recomputed")
    if cache is None and (output_dir / "cache_crosscheck.json").is_file():
        cache = json.loads((output_dir / "cache_crosscheck.json").read_text(encoding="utf-8"))
    cache_max = float(cache["maximum_relative_error"]) if cache else math.nan
    cross = review or {}
    cross_coeff = float(cross.get("maximum_cross_grid_coefficient_relative_error", math.nan))
    cross_cmos = float(cross.get("maximum_cross_grid_raw_cmos_relative_error", math.nan))
    input_identity = cross.get("actual_control_source_rows_labels_pixels_and_commands_revalidated", {})
    identity_status = input_identity.get("status", "PENDING")
    if review and review.get("passed"):
        engineering = (
            "旧 attempt_03 正控将 population_index 当作原图行号；40 条错误正控已排除并保留原 attempt_03 作为故障证据。"
            f"其余 {measurement['production_forwards_reused']:,} 条2048输出按保存数值原样复用；"
            f"本轮新增 {measurement['production_forwards_new']:,} 条2048正控及 "
            f"{measurement['cross_grid_reference_forwards_new']:,} 条4096参考。"
            f"Windows核对源行、标签、原始像素和SLM命令通过（重算设备 {input_identity.get('command_recompute_device', 'UNKNOWN')}，"
            f"命令最大绝对差 {input_identity.get('maximum_slm_command_absolute_error', math.nan):.3g}）；"
            f"跨网格复系数/raw CMOS最大相对误差为 {cross_coeff:.3g}/{cross_cmos:.3g}，"
            f"step 0全部2,000行对照缓存最大相对误差 {cache_max:.3g}。"
            "Windows独立NumPy距离、bootstrap和路线复算通过。参数/Adam、mapper、投影、prior、物理参数及评价器未更新；"
            "优化器实例、先验重拟合/采样和官方TEST读取均为0。"
        )
    else:
        engineering = (
            f"本轮新增 {measurement.get('production_forwards_new', 0)} 条2048及 "
            f"{measurement.get('cross_grid_reference_forwards_new', 0)} 条4096正控；"
            f"Windows输入身份复核状态为 {identity_status}，缓存最大相对误差为 {cache_max:.3g}。"
            "未完成独立消费者复核前，不采信路线结论。"
        )
    lines = [
        "# LightGen 训练—生成 OLS 输入分布定位收尾",
        "",
        "## 结果与状态",
        "",
        f"{result}。唯一路线为 `{route}`。`G_0={estimates['G_0']:.8g}`，"
        f"`G_2000={estimates['G_2000']:.8g}`（95%区间[{intervals['G_2000']['lower']:.8g}, "
        f"{intervals['G_2000']['upper']:.8g}]），`DeltaG={estimates['DeltaG']:.8g}` "
        f"（95%区间[{intervals['DeltaG']['lower']:.8g}, {intervals['DeltaG']['upper']:.8g}]）。"
        "参数更新0、优化器实例0、先验重拟合/采样0、官方TEST读取0。",
        "",
        "## 科学认识",
        "",
        science,
        "",
        "区间以固定的每类100张TRAIN参考行作为条件；55,000行旧缓存只用于step 0交叉核对和辅助描述。E_D/E_C逐样本正确性与分歧只作关联，不作因果解释。",
        "",
        "## 目标缺口",
        "",
        "结论仅限已消费先验人口、固定TRAIN诊断样本、冻结仿真模型与评价器。全体生成分布、独立新人口、人工可读性及实物SLM/器件/OLS/CMOS链路仍未验证。",
        "",
        "## 工程完成与验证",
        "",
        engineering,
        f"Windows独立复算状态：`{'PASSED' if verified and review.get('passed') else 'PENDING'}`。详见`production_preflight.json`、`cross_grid_reference.json`、`cache_crosscheck.json`、`windows_review_02/independent_review.json`及`raw_cmos_train_controls_comparison.png`。",
        "",
        "## 决策影响与下一步",
        "",
        next_step,
        "",
        "该建议不构成训练、改损失、重拟合先验、重采样、扩容、量化或硬件操作授权。",
        "",
    ]
    (output_dir / "closeout_report.md").write_text("\n".join(lines), encoding="utf-8")


def _run(config_path: Path) -> dict[str, Any]:
    config = _read_config(config_path)
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for full-path production forwards")
    _require(not OUTPUT_DIR.exists(), f"refusing to reuse an existing output directory: {OUTPUT_DIR}",
             FileExistsError)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "run_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "runner_source.py").write_text(Path(__file__).read_text(encoding="utf-8"),
                                                 encoding="utf-8")
    _write_json(OUTPUT_DIR / "task_contract.json", _task_contract(config))
    context = _load_research_context(config, device=torch.device("cuda:0"), local_sources=False)
    runtime = context["runtime"]
    manifest = _sample_manifest(context["schedule"], runtime["labels"], runtime["train_indices"])
    _write_csv(OUTPUT_DIR / "train_sample_manifest.csv", manifest,
               ["population_index", "label", "class_position", "source_train_row",
                "schedule_step", "schedule_flat_position", "role", "positive_control"])
    generation = _load_generation_assets(context, config, local_sources=False)
    _write_asset_table(config, context, OUTPUT_DIR)
    source_identity = {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "source_endpoint": "D0", "source_global_step": 50_000,
        "measured_endpoints": {
            "0": {"completed_updates": 0, "global_step": 50_000,
                  "encoder_adam_step": 45_000, "generator_adam_step": 50_000},
            "2000": {"completed_updates": 2_000, "global_step": 52_000,
                     "encoder_adam_step": 47_000, "generator_adam_step": 52_000},
        },
        "model_identity": context["checkpoints"][0]["model_identity"],
        "training_schedule_rows": [2_000, 10],
        "official_test_read": False,
        "generation_population_role_now": "consumed_diagnostic_population",
        "fixed_input_assets": ["mapper", "fixed_projection", "physical_phase_prior",
                                "positive_modal_scale", "camera_gain", "physical_optics", "E_D", "E_C"],
    }
    _write_json(OUTPUT_DIR / "source_identity.json", source_identity)
    snapshot = _snapshot_runtime(context)
    try:
        control_outputs, cross_report = _load_reused_preflight_controls(
            config, context, manifest, local_sources=False
        )
    except Exception as exc:
        _write_json(OUTPUT_DIR / "measurement_status.json", {
            "task_id": TASK_ID, "status": "INVALID_HARNESS",
            "stage": "reused_preflight_validation", "error_type": type(exc).__name__,
            "error": str(exc), "production_forwards_attempted": 0,
            "cross_grid_reference_forwards_attempted": 0,
            "optimizer_updates": 0, "optimizer_instances": 0,
            "official_test_reads": 0,
        })
        raise
    _save_torch(OUTPUT_DIR / "cross_grid_controls.pt", control_outputs, new=True)
    _write_json(OUTPUT_DIR / "cross_grid_reference.json", cross_report)
    forward_counts = {"production_2048": 0, "reference_4096": 0}

    def record_forward(category: str) -> Callable[[int], None]:
        def increment(rows: int) -> None:
            forward_counts[category] += rows
        return increment

    control_indices = control_outputs[0]["population_indices"].long()
    cache_map = _cache_map(context)
    control_rows = [int(manifest[index]["source_train_row"])
                    for index in control_indices.tolist()]
    cache_indices = torch.tensor([cache_map[row] for row in control_rows], dtype=torch.long)
    cache_errors = _relative_coefficient_error(
        control_outputs[0]["coefficients_2048"],
        context["cache"]["coefficients"].index_select(0, cache_indices),
    )
    cache_passed = bool(float(cache_errors.max()) <= CACHE_TOLERANCE)
    cache_preflight = {
        "task_id": TASK_ID,
        "status": "PASSED" if cache_passed else "DIAGNOSTIC_MISMATCH",
        "source_attempt": "attempt_02_saved_controls_reused",
        "rows_compared": 20,
        "relative_error_definition": "norm(matrix_reencoded_minus_full_optics)/norm(full_optics)",
        "tolerance": CACHE_TOLERANCE,
        "maximum_relative_error": float(cache_errors.max()),
        "p95_relative_error": float(torch.quantile(cache_errors, 0.95)),
        "passed": cache_passed,
        "blocking_gate": False,
        "mismatch_policy": "report_as_diagnostic_limitation",
        "user_approved_nonblocking_treatment": True,
        "decision_use": "diagnostic_only; the old matrix cache is not the measured step-2000 TRAIN distribution",
        "historical_100_row_full_optics_max_relative_error": context["cache_replay_max_relative_error"],
        "step_2000_cache_used_as_reference": False,
    }
    _write_json(OUTPUT_DIR / "cache_preflight.json", cache_preflight)
    _require(forward_counts == {"production_2048": 0, "reference_4096": 0},
             f"attempt_03 unexpectedly reran reused preflight rows: {forward_counts}")
    _write_json(OUTPUT_DIR / "production_preflight.json", {
        "task_id": TASK_ID, "status": "PASSED", "production_grid": GRID_PRODUCTION,
        "independent_reference_grid": GRID_REFERENCE,
        "production_rows_included_in_total": 40,
        "cross_grid_reference_rows": 40, "cross_grid_passed": True,
        "preflight_controls_source_attempt": "attempt_02",
        "preflight_controls_independently_recomputed": True,
        "production_forwards_reused": 40,
        "cross_grid_reference_forwards_reused": 40,
        "production_forwards_authorized": 4_000,
        "production_forwards_remaining": 3_960,
        "production_forwards_attempted": 0,
        "cross_grid_reference_forwards_attempted": 0,
        "cache_preflight_status": cache_preflight["status"],
        "cache_preflight_passed": cache_preflight["passed"],
        "cache_mismatch_nonblocking_by_explicit_approval": True,
        "optimizer_updates": 0, "optimizer_instances": 0,
    })
    _write_json(OUTPUT_DIR / "measurement_status.json", {
        "task_id": TASK_ID, "status": "PREFLIGHT_PASSED_FULL_MEASUREMENT_PENDING",
        "stage": "reused_preflight_validated",
        "production_forwards_reused": 40,
        "cross_grid_reference_forwards_reused": 40,
        "production_forwards_attempted": 0,
        "cross_grid_reference_forwards_attempted": 0,
        "optimizer_updates": 0, "optimizer_instances": 0,
        "official_test_reads": 0,
        "old_cache_mismatch_status": cache_preflight["status"],
        "old_cache_mismatch_is_nonblocking_diagnostic": True,
    })

    train_outputs: dict[int, dict[str, torch.Tensor]] = {}
    try:
        for step in PHASE_STEPS:
            control = control_outputs[step]
            train_outputs[step] = _measure_endpoint(
                step, context, manifest, control, runtime, OUTPUT_DIR,
                on_forward_batch=record_forward("production_2048"),
            )
    except Exception as exc:
        _write_json(OUTPUT_DIR / "measurement_status.json", {
            "task_id": TASK_ID, "status": "INVALID_HARNESS",
            "stage": "full_population_measurement", "error_type": type(exc).__name__,
            "error": str(exc), "production_forwards_attempted": forward_counts["production_2048"],
            "cross_grid_reference_forwards_attempted": forward_counts["reference_4096"],
            "optimizer_updates": 0, "optimizer_instances": 0,
            "official_test_reads": 0,
        })
        raise
    _write_training_per_sample(OUTPUT_DIR / "train_fullpath_per_sample.csv",
                               train_outputs, manifest)
    cache = _cache_crosscheck(context, train_outputs, manifest)
    _write_json(OUTPUT_DIR / "cache_crosscheck.json", cache)
    training_payload = {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "population_role": "consumed_train_diagnostic",
        "grid": GRID_PRODUCTION, "rows": 2_000, "rows_per_class": 200,
        "reference_positions": "even class positions 0,2,...,198",
        "calibration_positions": "odd class positions 1,3,...,199",
        "modal_scale": runtime["scale"].detach().cpu().to(torch.float64),
        "endpoints": train_outputs,
    }
    _save_torch(OUTPUT_DIR / "training_fullpath_outputs.pt", training_payload, new=True)
    generation_payload = {
        "task_id": TASK_ID, "population_role": "consumed_diagnostic_population",
        "source_run_id": TRAINING_RUN_ID, "labels": generation["labels"],
        "proposal_ids": generation["proposal_ids"],
        "saved_prior_targets": generation["target_coefficients"],
        "actual_E0G0_coefficients": generation["bundle"]["conditions"]["E0G0"]["coefficients"].detach().cpu(),
        "actual_E1G1_coefficients": generation["bundle"]["conditions"]["E1G1"]["coefficients"].detach().cpu(),
        "commands_reused_in_place": True,
        "phase_sources_revalidated": True,
    }
    _save_torch(OUTPUT_DIR / "generation_coefficients.pt", generation_payload, new=True)
    _assert_snapshot_unchanged(context, snapshot)
    cache_summary = _cache_crosscheck(context, train_outputs, manifest)
    _write_json(OUTPUT_DIR / "cache_crosscheck.json", cache_summary)
    summary = _analyze(context, generation, train_outputs, manifest, OUTPUT_DIR, numpy_only=False)
    _write_json(OUTPUT_DIR / "measurement_status.json", {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "train_rows": 2_000, "production_full_path_rows": 4_000,
        "cross_grid_reference_rows": 40, "generated_rows_reused": 2_000,
        "production_forwards_reused": 40,
        "cross_grid_reference_forwards_reused": 40,
        "production_forwards_attempted": forward_counts["production_2048"],
        "cross_grid_reference_forwards_attempted": forward_counts["reference_4096"],
        "production_forwards_total": 4_000,
        "cross_grid_reference_forwards_total": 40,
        "old_cache_mismatch_status": cache_summary["status"],
        "old_cache_mismatch_is_nonblocking_diagnostic": True,
        "optimizer_updates": 0, "optimizer_instances": 0,
        "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        "source_model_adam_frozen_assets_and_evaluators_unchanged": True,
        "decision_route": summary["decision_route"],
        "windows_independent_review": "PENDING",
    })
    _write_closeout(summary, None, OUTPUT_DIR)
    return summary


def _load_attempt03_noncontrol_outputs(config: Mapping[str, Any], context: Mapping[str, Any],
                                       manifest: Sequence[Mapping[str, Any]],
                                       source_dir: Path, *, local_sources: bool = False
                                       ) -> tuple[dict[int, dict[str, torch.Tensor]],
                                                  dict[int, Mapping[str, torch.Tensor]],
                                                  dict[str, Any]]:
    expected_source = SOURCE_ATTEMPT_BACKUP_DIR if local_sources else SOURCE_ATTEMPT_DIR
    _require(source_dir.resolve() == expected_source.resolve() and source_dir.is_dir(),
             "attempt_03 reuse must use its explicit cloud run or Windows backup", ValueError)
    status = json.loads((source_dir / "measurement_status.json").read_text(encoding="utf-8"))
    source_identity = json.loads((source_dir / "source_identity.json").read_text(encoding="utf-8"))
    _require(status.get("task_id") == TASK_ID and status.get("status") == "COMPLETE_DIAGNOSTIC"
             and status.get("production_forwards_attempted") == 3_960
             and status.get("production_forwards_reused") == 40
             and status.get("cross_grid_reference_forwards_attempted") == 0
             and status.get("optimizer_updates") == 0
             and status.get("official_test_reads") == 0
             and source_identity.get("task_id") == TASK_ID
             and source_identity.get("source_training_run_id") == TRAINING_RUN_ID
             and source_identity.get("source_endpoint") == "D0"
             and source_identity.get("model_identity") == context["checkpoints"][0]["model_identity"]
             and source_identity.get("measured_endpoints", {}).get("0", {}) == {
                 "completed_updates": 0, "global_step": 50_000,
                 "encoder_adam_step": 45_000, "generator_adam_step": 50_000,
             }
             and source_identity.get("measured_endpoints", {}).get("2000", {}) == {
                 "completed_updates": 2_000, "global_step": 52_000,
                 "encoder_adam_step": 47_000, "generator_adam_step": 52_000,
             },
             "attempt_03 is not the registered zero-update measurement source")
    _validate_sample_manifest(source_dir / "train_sample_manifest.csv", manifest)

    old_runner = (source_dir / "runner_source.py").read_text(encoding="utf-8")
    _require('rows = torch.tensor([int(item["source_train_row"]) for item in manifest], dtype=torch.long)'
             in old_runner and "batch_rows = rows.index_select(0, positions)" in old_runner,
             "attempt_03 source does not establish the correct non-control TRAIN-row path")
    control_source_key = "preflight_controls_backup" if local_sources else "preflight_controls_run"
    old_control_dir = (ROOT / str(config["sources"][control_source_key])).resolve()
    old_control_runner = (old_control_dir / "runner_source.py").read_text(encoding="utf-8")
    _require('int(row["population_index"]) for row in manifest if row["positive_control"]'
             in old_control_runner
             and "model_2048, control_indices, runtime, keep_raw=True" in old_control_runner
             and "model_4096, control_indices, runtime, keep_raw=True" in old_control_runner,
             "attempt_02 source no longer reproduces the population-index input-binding defect")

    indices, expected_rows, expected_labels = _positive_control_identity(manifest)
    training = _load_pt(source_dir / "training_fullpath_outputs.pt", mmap=True)
    _require(training.get("task_id") == TASK_ID
             and training.get("source_training_run_id") == TRAINING_RUN_ID
             and training.get("population_role") == "consumed_train_diagnostic"
             and training.get("grid") == GRID_PRODUCTION
             and training.get("rows") == 2_000
             and training.get("rows_per_class") == ROWS_PER_CLASS
             and torch.equal(training.get("modal_scale"),
                             context["runtime"]["scale"].detach().cpu().to(torch.float64)),
             "attempt_03 full-path population or frozen modal scale changed")
    endpoints = training.get("endpoints", {})
    _require(set(endpoints) == set(PHASE_STEPS),
             "attempt_03 omits a measured optical endpoint")
    source_rows = torch.tensor([int(row["source_train_row"]) for row in manifest], dtype=torch.long)
    labels = torch.tensor([int(row["label"]) for row in manifest], dtype=torch.long)
    noncontrols = torch.tensor([index for index in range(2_000)
                                if index not in set(indices.tolist())], dtype=torch.long)
    controls = _load_pt(source_dir / "cross_grid_controls.pt", mmap=True)
    reused_attempt02 = _load_pt(old_control_dir / "cross_grid_controls.pt", mmap=True)
    _require(set(controls) == set(PHASE_STEPS) and _same_state(controls, reused_attempt02),
             "attempt_03 controls do not exactly match the controls produced by the faulty attempt_02 source")

    result: dict[int, dict[str, torch.Tensor]] = {}
    cache_map = _cache_map(context)
    cache_rows = torch.tensor([cache_map[int(source_rows[index])] for index in noncontrols],
                              dtype=torch.long)
    step0_noncontrol_error = _relative_coefficient_error(
        endpoints[0]["coefficients"].index_select(0, noncontrols),
        context["cache"]["coefficients"].index_select(0, cache_rows),
    )
    _require(float(step0_noncontrol_error.max()) <= CACHE_TOLERANCE,
             "attempt_03 non-control step-0 rows fail the unchanged 1e-6 cache check")

    for step in PHASE_STEPS:
        endpoint = endpoints[step]
        endpoint_file = _load_pt(source_dir / f"train_optical_outputs_step_{step:05d}.pt", mmap=True)
        _require(_same_state(endpoint, endpoint_file)
                 and torch.equal(endpoint.get("source_rows"), source_rows)
                 and torch.equal(endpoint.get("labels"), labels)
                 and tuple(endpoint.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
                 and tuple(endpoint.get("plane_powers_w", torch.empty(0)).shape) == (2_000, 5)
                 and tuple(endpoint.get("raw_summaries", torch.empty(0)).shape) == (2_000, 5)
                 and torch.equal(endpoint.get("control_population_indices"), indices)
                 and torch.equal(endpoint["coefficients"].index_select(0, indices),
                                 controls[step]["coefficients_2048"])
                 and torch.equal(endpoint["raw_cmos_controls_2048"],
                                 controls[step]["raw_cmos_2048"])
                 and torch.isfinite(endpoint["coefficients"].real).all().item()
                 and torch.isfinite(endpoint["coefficients"].imag).all().item()
                 and torch.isfinite(endpoint["plane_powers_w"]).all().item()
                 and torch.isfinite(endpoint["raw_summaries"]).all().item(),
                 f"attempt_03 step {step} outputs are not aligned with the saved TRAIN population")
        result[step] = {
            key: value.index_select(0, noncontrols).clone()
            for key, value in endpoint.items()
            if key in {"coefficients", "plane_powers_w", "raw_summaries"}
        }

    legacy_actual_rows = indices.clone()
    legacy_actual_labels = context["runtime"]["labels"].index_select(0, legacy_actual_rows)
    actual_train_mask = torch.isin(legacy_actual_rows, context["runtime"]["train_indices"])
    _require(int(actual_train_mask.sum()) == 16,
             "attempt_02 reconstructed wrong-row TRAIN membership differs from the recorded diagnosis")
    old_train_rows = legacy_actual_rows[actual_train_mask]
    old_train_positions = torch.nonzero(actual_train_mask, as_tuple=False).reshape(-1)
    old_cache_rows = torch.tensor([cache_map[int(row)] for row in old_train_rows], dtype=torch.long)
    legacy_cache_error = _relative_coefficient_error(
        controls[0]["coefficients_2048"].index_select(0, old_train_positions),
        context["cache"]["coefficients"].index_select(0, old_cache_rows),
    )
    _require(float(legacy_cache_error.max()) <= CACHE_TOLERANCE,
             "legacy controls no longer match their actual source rows for the 16 TRAIN members")
    return result, controls, {
        "status": "REPRODUCED_AND_EXCLUDED",
        "source_attempt_03": str(source_dir),
        "legacy_controls_source_attempt": str(old_control_dir),
        "legacy_source_code_passed_population_index_as_image_row": True,
        "legacy_source_code_reused_by_attempt_03": True,
        "legacy_population_indices": indices.tolist(),
        "legacy_actual_image_rows": legacy_actual_rows.tolist(),
        "legacy_expected_source_train_rows": expected_rows.tolist(),
        "legacy_actual_labels": legacy_actual_labels.tolist(),
        "legacy_expected_labels": expected_labels.tolist(),
        "legacy_actual_rows_in_train": int(actual_train_mask.sum()),
        "legacy_actual_label_mismatches": int((legacy_actual_labels != expected_labels).sum()),
        "legacy_actual_train_row_cache_max_relative_error": float(legacy_cache_error.max()),
        "legacy_controls_incorrect_and_excluded": 40,
        "noncontrol_outputs_reused_unchanged": 3_960,
        "step_0_noncontrol_cache_max_relative_error": float(step0_noncontrol_error.max()),
        "step_0_noncontrol_cache_tolerance": CACHE_TOLERANCE,
        "historical_controls_preserved_at_source": True,
    }


def _repair_controls(config: Mapping[str, Any], source_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the 40 corrected full-path controls")
    _require(source_dir.resolve() == SOURCE_ATTEMPT_DIR.resolve(),
             "--repair-controls requires the exact cloud attempt_03 artifact directory", ValueError)
    _require(not OUTPUT_DIR.exists(), f"refusing to replace the attempt_04 output: {OUTPUT_DIR}",
             FileExistsError)
    context = _load_research_context(config, device=torch.device("cuda:0"), local_sources=False)
    runtime = context["runtime"]
    manifest = _sample_manifest(context["schedule"], runtime["labels"], runtime["train_indices"])
    reused_noncontrols, legacy_controls, defect = _load_attempt03_noncontrol_outputs(
        config, context, manifest, source_dir
    )
    generation = _load_generation_assets(context, config, local_sources=False)
    source_generation = _load_pt(source_dir / "generation_coefficients.pt", mmap=True)
    _require(source_generation.get("task_id") == TASK_ID
             and torch.equal(source_generation.get("labels"), generation["labels"])
             and torch.equal(source_generation.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(source_generation.get("saved_prior_targets"), generation["target_coefficients"])
             and torch.equal(source_generation.get("actual_E0G0_coefficients"),
                             generation["bundle"]["conditions"]["E0G0"]["coefficients"])
             and torch.equal(source_generation.get("actual_E1G1_coefficients"),
                             generation["bundle"]["conditions"]["E1G1"]["coefficients"]),
             "attempt_03 generated targets, commands, or outputs differ from the registered saved population")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "run_config.yaml").write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "runner_source.py").write_text(Path(__file__).read_text(encoding="utf-8"),
                                                 encoding="utf-8")
    _write_json(OUTPUT_DIR / "task_contract.json", _task_contract(config))
    _write_csv(OUTPUT_DIR / "train_sample_manifest.csv", manifest,
               ["population_index", "label", "class_position", "source_train_row",
                "schedule_step", "schedule_flat_position", "role", "positive_control"])
    _write_json(OUTPUT_DIR / "input_identity_repair.json", defect)
    _write_json(OUTPUT_DIR / "source_identity.json", {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "source_endpoint": "D0", "source_global_step": 50_000,
        "measured_endpoints": {
            "0": {"completed_updates": 0, "global_step": 50_000,
                  "encoder_adam_step": 45_000, "generator_adam_step": 50_000},
            "2000": {"completed_updates": 2_000, "global_step": 52_000,
                     "encoder_adam_step": 47_000, "generator_adam_step": 52_000},
        },
        "model_identity": context["checkpoints"][0]["model_identity"],
        "training_schedule_rows": [2_000, 10], "official_test_read": False,
        "generation_population_role_now": "consumed_diagnostic_population",
        "fixed_input_assets": ["mapper", "fixed_projection", "physical_phase_prior",
                                "positive_modal_scale", "camera_gain", "physical_optics", "E_D", "E_C"],
    })
    _write_json(OUTPUT_DIR / "asset_reuse_table.json", {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "preserved_source_artifacts": True,
        "assets": [
            {"asset": "attempt_03 step-0 and step-2000 TRAIN full-path outputs",
             "source": config["sources"]["measurement_source_run"],
             "use": "reuse exactly the 1,980 non-control rows per endpoint; historical controls excluded"},
            {"asset": "corrected positive-control raw TRAIN rows and actual 2048 SLM commands",
             "source": "new attempt_04 measurement",
             "use": "40 new 2048 full-path forwards plus 40 corresponding 4096 references"},
            {"asset": "saved prior targets, legal commands, generated coefficients and evaluator results",
             "source": config["sources"]["training_run"] + " and " + config["sources"]["attribution_run"],
             "use": "reuse the consumed 2,000-row generated population with no resampling or evaluator rerun"},
            {"asset": "55,000-row physical-phase TRAIN coefficient cache",
             "source": config["sources"]["train_reference_run"] + "/raw_physical_phase_train_coefficients.pt",
             "use": "unchanged 1e-6 step-0 integrity comparison only"},
            {"asset": "frozen mapper, projection, prior, mode scale, camera gain and optics",
             "source": config["sources"]["source_config"],
             "use": "input adapter and optical parameters are read-only throughout this repair"},
        ],
    })
    snapshot = _snapshot_runtime(context)
    control_indices, control_rows, control_labels = _positive_control_identity(manifest)
    forward_counts = {"production_2048": 0, "reference_4096": 0}
    controls: dict[int, dict[str, torch.Tensor]] = {}

    def record_forward(category: str) -> Callable[[int], None]:
        def increment(rows: int) -> None:
            forward_counts[category] += rows
        return increment

    _write_json(OUTPUT_DIR / "measurement_status.json", {
        "task_id": TASK_ID, "status": "CORRECTED_CONTROL_MEASUREMENT_PENDING",
        "stage": "attempt_03_noncontrols_validated",
        "production_forwards_reused": 3_960, "production_forwards_attempted": 0,
        "cross_grid_reference_forwards_reused": 0,
        "cross_grid_reference_forwards_attempted": 0,
        "optimizer_updates": 0, "optimizer_instances": 0, "official_test_reads": 0,
    })
    try:
        for step in PHASE_STEPS:
            model_2048 = _new_model(context["checkpoints"][step], GRID_PRODUCTION, runtime)
            measured_2048 = _measure_rows(
                model_2048, control_rows, runtime,
                expected_source_rows=control_rows, expected_labels=control_labels,
                keep_raw=True, on_forward_batch=record_forward("production_2048"),
            )
            del model_2048
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_4096 = _new_model(context["checkpoints"][step], GRID_REFERENCE, runtime)
            measured_4096 = _measure_rows(
                model_4096, control_rows, runtime,
                expected_source_rows=control_rows, expected_labels=control_labels,
                keep_raw=True, on_forward_batch=record_forward("reference_4096"),
            )
            del model_4096
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            item: dict[str, torch.Tensor] = {"population_indices": control_indices.clone()}
            for grid, measured in (("2048", measured_2048), ("4096", measured_4096)):
                for key in ("source_rows", "labels", "source_pixels", "slm_commands",
                            "coefficients", "plane_powers_w", "raw_summaries", "raw_cmos"):
                    item[f"{key}_{grid}"] = measured[key]
            controls[step] = item
            _save_torch(OUTPUT_DIR / "corrected_controls_partial.pt", controls)
            _write_json(OUTPUT_DIR / "measurement_status.json", {
                "task_id": TASK_ID, "status": "CORRECTED_CONTROL_MEASUREMENT_IN_PROGRESS",
                "stage": f"step_{step}_controls_saved",
                "production_forwards_reused": 3_960,
                "production_forwards_attempted": forward_counts["production_2048"],
                "cross_grid_reference_forwards_reused": 0,
                "cross_grid_reference_forwards_attempted": forward_counts["reference_4096"],
                "optimizer_updates": 0, "optimizer_instances": 0, "official_test_reads": 0,
            })
    except Exception as exc:
        _write_json(OUTPUT_DIR / "measurement_status.json", {
            "task_id": TASK_ID, "status": "INVALID_HARNESS",
            "stage": "corrected_control_forward", "error_type": type(exc).__name__,
            "error": str(exc), "production_forwards_reused": 3_960,
            "production_forwards_attempted": forward_counts["production_2048"],
            "cross_grid_reference_forwards_reused": 0,
            "cross_grid_reference_forwards_attempted": forward_counts["reference_4096"],
            "optimizer_updates": 0, "optimizer_instances": 0, "official_test_reads": 0,
        })
        raise

    _require(forward_counts == {"production_2048": 40, "reference_4096": 40},
             f"corrected control forward budget differs from the approved 40+40: {forward_counts}")
    input_identity = _validate_control_input_evidence(controls, manifest, runtime)
    _write_json(OUTPUT_DIR / "control_input_identity_validation.json", input_identity)

    cross_rows: list[dict[str, Any]] = []
    for step in PHASE_STEPS:
        item = controls[step]
        for index, population_index in enumerate(control_indices.tolist()):
            coefficient_error = projected._phase_error(
                item["coefficients_2048"][index], item["coefficients_4096"][index]
            )
            raw_error = projected._raw_error(
                item["raw_cmos_2048"][index], item["raw_cmos_4096"][index]
            )
            cross_rows.append({
                "endpoint_step": step, "population_index": population_index,
                "source_train_row": int(control_rows[index]), "label": int(control_labels[index]),
                "coefficient_relative_error": coefficient_error,
                "raw_cmos_relative_error": raw_error,
                "passed_1e-4": coefficient_error <= CROSS_GRID_TOLERANCE
                and raw_error <= CROSS_GRID_TOLERANCE,
            })
    cross_report = {
        "task_id": TASK_ID, "status": "PASSED" if all(row["passed_1e-4"] for row in cross_rows)
        else "FAILED",
        "source_attempt": "attempt_04_corrected_controls",
        "source_attempt_03_noncontrols_reused": True,
        "production_rows_included": 40, "independent_reference_rows": 40,
        "rows_per_class_per_endpoint": 2, "production_grid": GRID_PRODUCTION,
        "reference_grid": GRID_REFERENCE, "tolerance": CROSS_GRID_TOLERANCE,
        "cross_grid_error_definition": "phase-aligned complex relative L2 and raw CMOS relative L2",
        "production_forwards_attempted_this_attempt": forward_counts["production_2048"],
        "reference_forwards_attempted_this_attempt": forward_counts["reference_4096"],
        "rows": cross_rows,
        "maximum_coefficient_relative_error": max(row["coefficient_relative_error"] for row in cross_rows),
        "maximum_raw_cmos_relative_error": max(row["raw_cmos_relative_error"] for row in cross_rows),
        "semantic_evidence": False, "hardware_evidence": False,
    }
    _write_json(OUTPUT_DIR / "cross_grid_reference.json", cross_report)
    _require(cross_report["status"] == "PASSED",
             "corrected TRAIN controls fail the unchanged 1e-4 cross-grid numerical validation")
    _save_torch(OUTPUT_DIR / "cross_grid_controls.pt", controls, new=True)

    source_training = _load_pt(source_dir / "training_fullpath_outputs.pt", mmap=True)
    source_endpoints = source_training["endpoints"]
    train_outputs: dict[int, dict[str, torch.Tensor]] = {}
    for step in PHASE_STEPS:
        endpoint = {key: value.clone() for key, value in source_endpoints[step].items()}
        for key, control_key in (("coefficients", "coefficients_2048"),
                                 ("plane_powers_w", "plane_powers_w_2048"),
                                 ("raw_summaries", "raw_summaries_2048")):
            endpoint[key][control_indices] = controls[step][control_key]
        endpoint["raw_cmos_controls_2048"] = controls[step]["raw_cmos_2048"].clone()
        train_outputs[step] = endpoint

    cache_preflight_errors = _relative_coefficient_error(
        controls[0]["coefficients_2048"],
        context["cache"]["coefficients"].index_select(
            0, torch.tensor([_cache_map(context)[int(row)] for row in control_rows], dtype=torch.long)
        ),
    )
    cache_preflight = {
        "task_id": TASK_ID, "status": "PASSED" if float(cache_preflight_errors.max()) <= CACHE_TOLERANCE
        else "FAILED", "source_attempt": "attempt_04_corrected_controls",
        "rows_compared": 20, "relative_error_definition":
        "norm(matrix_reencoded_minus_full_optics)/norm(full_optics)",
        "tolerance": CACHE_TOLERANCE,
        "maximum_relative_error": float(cache_preflight_errors.max()),
        "p95_relative_error": float(torch.quantile(cache_preflight_errors, 0.95)),
        "passed": bool(float(cache_preflight_errors.max()) <= CACHE_TOLERANCE),
        "measurement_validity_required": True,
        "step_2000_cache_used_as_reference": False,
    }
    _write_json(OUTPUT_DIR / "cache_preflight.json", cache_preflight)
    cache_summary = _cache_crosscheck(context, train_outputs, manifest)
    cache_summary["measurement_validity_required"] = True
    _write_json(OUTPUT_DIR / "cache_crosscheck.json", cache_summary)
    _require(cache_preflight["passed"] and cache_summary["passed"],
             "corrected step-0 outputs fail the unchanged 1e-6 cache check; no route decision is valid")
    _write_json(OUTPUT_DIR / "production_preflight.json", {
        "task_id": TASK_ID, "status": "PASSED", "production_grid": GRID_PRODUCTION,
        "independent_reference_grid": GRID_REFERENCE,
        "production_rows_included_in_total": 40, "cross_grid_reference_rows": 40,
        "cross_grid_passed": True, "positive_controls_source_attempt": "attempt_04_corrected_controls",
        "actual_source_rows_labels_pixels_and_slm_commands_checked": True,
        "production_forwards_new": 40, "production_forwards_reused": 3_960,
        "cross_grid_reference_forwards_new": 40, "cross_grid_reference_forwards_reused": 0,
        "step_0_cache_all_2000_passed_1e-6": True,
        "optimizer_updates": 0, "optimizer_instances": 0,
    })

    _validate_control_input_evidence(controls, manifest, runtime)
    control_identity, _, _ = _positive_control_identity(manifest)
    _require(torch.equal(control_identity, control_indices), "control population order changed during repair")
    _write_training_per_sample(OUTPUT_DIR / "train_fullpath_per_sample.csv", train_outputs, manifest)
    training_payload = {
        "task_id": TASK_ID, "source_training_run_id": TRAINING_RUN_ID,
        "population_role": "consumed_train_diagnostic", "grid": GRID_PRODUCTION,
        "rows": 2_000, "rows_per_class": 200,
        "reference_positions": "even class positions 0,2,...,198",
        "calibration_positions": "odd class positions 1,3,...,199",
        "modal_scale": runtime["scale"].detach().cpu().to(torch.float64),
        "endpoints": train_outputs,
    }
    for step in PHASE_STEPS:
        _save_torch(OUTPUT_DIR / f"train_optical_outputs_step_{step:05d}.pt",
                    train_outputs[step], new=True)
    _save_torch(OUTPUT_DIR / "training_fullpath_outputs.pt", training_payload, new=True)
    generation_payload = {
        "task_id": TASK_ID, "population_role": "consumed_diagnostic_population",
        "source_run_id": TRAINING_RUN_ID, "labels": generation["labels"],
        "proposal_ids": generation["proposal_ids"],
        "saved_prior_targets": generation["target_coefficients"],
        "actual_E0G0_coefficients": generation["bundle"]["conditions"]["E0G0"]["coefficients"].detach().cpu(),
        "actual_E1G1_coefficients": generation["bundle"]["conditions"]["E1G1"]["coefficients"].detach().cpu(),
        "commands_reused_in_place": True, "phase_sources_revalidated": True,
    }
    _save_torch(OUTPUT_DIR / "generation_coefficients.pt", generation_payload, new=True)
    _assert_snapshot_unchanged(context, snapshot)
    summary = _analyze(context, generation, train_outputs, manifest, OUTPUT_DIR, numpy_only=False)
    _write_json(OUTPUT_DIR / "measurement_progress.json", {
        "task_id": TASK_ID, "endpoint_step": 2_000,
        "completed_production_rows_this_endpoint": 2_000,
        "authorized_production_rows_per_endpoint": 2_000,
        "production_forwards_new": 40, "production_forwards_reused": 3_960,
        "official_test_reads": 0, "optimizer_updates": 0,
    })
    _write_json(OUTPUT_DIR / "measurement_status.json", {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "train_rows": 2_000, "production_full_path_rows": 4_000,
        "production_forwards_reused": 3_960, "production_forwards_attempted": 40,
        "production_forwards_total": 4_000,
        "correct_controls_remeasured": 40, "rejected_attempt03_controls": 40,
        "cross_grid_reference_rows": 40,
        "cross_grid_reference_forwards_reused": 0,
        "cross_grid_reference_forwards_attempted": 40,
        "cross_grid_reference_forwards_total": 40,
        "step_0_cache_check_all_rows_passed": True,
        "optimizer_updates": 0, "optimizer_instances": 0,
        "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        "source_model_adam_frozen_assets_and_evaluators_unchanged": True,
        "decision_route": summary["decision_route"],
        "windows_independent_review": "PENDING",
    })
    _write_closeout(summary, None, OUTPUT_DIR)
    return summary


def _finalize_saved_measurement(config: Mapping[str, Any], artifact_dir: Path) -> dict[str, Any]:
    _require(artifact_dir.resolve() == OUTPUT_DIR.resolve() and artifact_dir.is_dir(),
             "analysis recovery requires the exact attempt_03 production output directory", ValueError)
    status_path = artifact_dir / "measurement_status.json"
    prior_status = json.loads(status_path.read_text(encoding="utf-8"))
    _require(prior_status.get("task_id") == TASK_ID
             and prior_status.get("status") == "PREFLIGHT_PASSED_FULL_MEASUREMENT_PENDING"
             and prior_status.get("production_forwards_reused") == 40
             and prior_status.get("cross_grid_reference_forwards_reused") == 40
             and prior_status.get("production_forwards_attempted") == 0
             and prior_status.get("cross_grid_reference_forwards_attempted") == 0
             and prior_status.get("optimizer_updates") == 0
             and prior_status.get("official_test_reads") == 0,
             "attempt_03 is not in the expected complete-forward, analysis-pending state")
    failure_log = artifact_dir / "analysis_failure_run.log"
    _require(failure_log.is_file()
             and "TypeError" in failure_log.read_text(encoding="utf-8"),
             "preserved first-run analysis traceback is missing from attempt_03")
    before_recovery_path = artifact_dir / "measurement_status_before_recovery.json"
    _require(not before_recovery_path.exists(), "refusing to replace prior recovery status evidence",
             FileExistsError)
    _write_json(before_recovery_path, prior_status)

    context = _load_research_context(config, device=torch.device("cpu"), local_sources=False)
    source_identity = json.loads((artifact_dir / "source_identity.json").read_text(encoding="utf-8"))
    _require(source_identity.get("task_id") == TASK_ID
             and source_identity.get("source_training_run_id") == TRAINING_RUN_ID
             and source_identity.get("source_endpoint") == "D0"
             and source_identity.get("source_global_step") == 50_000
             and source_identity.get("model_identity") == context["checkpoints"][0]["model_identity"]
             and source_identity.get("official_test_read") is False,
             "saved attempt_03 endpoint identity differs from the explicit D0 checkpoints")
    manifest = _sample_manifest(context["schedule"], context["runtime"]["labels"],
                                context["runtime"]["train_indices"])
    _validate_sample_manifest(artifact_dir / "train_sample_manifest.csv", manifest)
    controls, expected_cross_report = _load_reused_preflight_controls(
        config, context, manifest, local_sources=False
    )
    saved_controls = _load_pt(artifact_dir / "cross_grid_controls.pt")
    cross_report = json.loads((artifact_dir / "cross_grid_reference.json").read_text(encoding="utf-8"))
    _require(_same_state(saved_controls, controls)
             and _numeric_tree_close(cross_report, expected_cross_report),
             "attempt_03 reused cross-grid controls differ from the independently checked attempt_02 source")

    training = _load_pt(artifact_dir / "training_fullpath_outputs.pt")
    _require(training.get("task_id") == TASK_ID
             and training.get("source_training_run_id") == TRAINING_RUN_ID
             and training.get("population_role") == "consumed_train_diagnostic"
             and training.get("grid") == GRID_PRODUCTION
             and training.get("rows") == 2_000
             and training.get("rows_per_class") == ROWS_PER_CLASS
             and tuple(training.get("modal_scale", torch.empty(0)).shape) == (100,)
             and torch.equal(training["modal_scale"], context["runtime"]["scale"].detach().cpu().to(torch.float64)),
             "saved full-path outputs have the wrong task identity, role, grid, or frozen scale")
    train_outputs = training.get("endpoints", {})
    _require(set(train_outputs) == set(PHASE_STEPS),
             "saved full-path outputs do not contain both endpoints")
    source_rows = torch.tensor([int(row["source_train_row"]) for row in manifest], dtype=torch.long)
    labels = torch.tensor([int(row["label"]) for row in manifest], dtype=torch.long)
    control_indices = controls[0]["population_indices"].long()
    for step in PHASE_STEPS:
        endpoint = train_outputs[step]
        _require(torch.equal(endpoint.get("source_rows"), source_rows)
                 and torch.equal(endpoint.get("labels"), labels)
                 and tuple(endpoint.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
                 and tuple(endpoint.get("plane_powers_w", torch.empty(0)).shape) == (2_000, 5)
                 and tuple(endpoint.get("raw_summaries", torch.empty(0)).shape) == (2_000, 5)
                 and torch.equal(endpoint.get("control_population_indices"), control_indices)
                 and torch.equal(endpoint.get("coefficients").index_select(0, control_indices),
                                 controls[step]["coefficients_2048"])
                 and torch.equal(endpoint.get("plane_powers_w").index_select(0, control_indices),
                                 controls[step]["plane_powers_2048"])
                 and torch.equal(endpoint.get("raw_summaries").index_select(0, control_indices),
                                 controls[step]["raw_summaries_2048"])
                 and torch.equal(endpoint.get("raw_cmos_controls_2048"),
                                 controls[step]["raw_cmos_2048"])
                 and torch.is_complex(endpoint["coefficients"])
                 and torch.isfinite(endpoint["coefficients"].real).all().item()
                 and torch.isfinite(endpoint["coefficients"].imag).all().item()
                 and torch.isfinite(endpoint["plane_powers_w"]).all().item()
                 and torch.isfinite(endpoint["raw_summaries"]).all().item(),
                 f"step {step} saved endpoint rows are incomplete or do not contain the reused controls")
        endpoint_file = _load_pt(artifact_dir / f"train_optical_outputs_step_{step:05d}.pt")
        _require(_same_state(endpoint_file, endpoint),
                 f"step {step} endpoint artifact differs from the combined full-path payload")
    _validate_training_per_sample(artifact_dir / "train_fullpath_per_sample.csv",
                                  train_outputs, manifest)
    progress = json.loads((artifact_dir / "measurement_progress.json").read_text(encoding="utf-8"))
    _require(progress.get("task_id") == TASK_ID
             and progress.get("endpoint_step") == 2_000
             and progress.get("completed_production_rows_this_endpoint") == 2_000
             and progress.get("authorized_production_rows_per_endpoint") == 2_000
             and progress.get("optimizer_updates") == 0
             and progress.get("official_test_reads") == 0,
             "saved production progress does not establish completion of both endpoint outputs")

    cache_summary = _cache_crosscheck(context, train_outputs, manifest)
    saved_cache = json.loads((artifact_dir / "cache_crosscheck.json").read_text(encoding="utf-8"))
    _require(_numeric_tree_close(cache_summary, saved_cache),
             "saved all-row cache diagnostic differs from independent recomputation")
    cache_preflight = json.loads((artifact_dir / "cache_preflight.json").read_text(encoding="utf-8"))
    cache_map = _cache_map(context)
    cache_indices = torch.tensor([cache_map[int(manifest[index]["source_train_row"])]
                                  for index in control_indices.tolist()], dtype=torch.long)
    preflight_errors = _relative_coefficient_error(
        controls[0]["coefficients_2048"],
        context["cache"]["coefficients"].index_select(0, cache_indices),
    )
    _validate_cache_diagnostic_record(
        cache_preflight, preflight_errors, rows_compared=20,
        source_attempt="attempt_02_saved_controls_reused",
    )
    production_preflight = json.loads((artifact_dir / "production_preflight.json").read_text(encoding="utf-8"))
    _require(production_preflight.get("status") == "PASSED"
             and production_preflight.get("cross_grid_passed") is True
             and production_preflight.get("production_rows_included_in_total") == 40
             and production_preflight.get("cross_grid_reference_rows") == 40
             and production_preflight.get("production_forwards_authorized") == 4_000
             and production_preflight.get("production_forwards_remaining") == 3_960
             and production_preflight.get("production_forwards_reused") == 40
             and production_preflight.get("cross_grid_reference_forwards_reused") == 40
             and production_preflight.get("production_forwards_attempted") == 0
             and production_preflight.get("cross_grid_reference_forwards_attempted") == 0
             and production_preflight.get("cache_mismatch_nonblocking_by_explicit_approval") is True,
             "saved production preflight does not match the approved reused-control budget")

    generation = _load_generation_assets(context, config, local_sources=False)
    saved_generation = _load_pt(artifact_dir / "generation_coefficients.pt")
    _require(saved_generation.get("task_id") == TASK_ID
             and saved_generation.get("source_run_id") == TRAINING_RUN_ID
             and saved_generation.get("population_role") == "consumed_diagnostic_population"
             and saved_generation.get("commands_reused_in_place") is True
             and saved_generation.get("phase_sources_revalidated") is True
             and torch.equal(saved_generation.get("labels"), generation["labels"])
             and torch.equal(saved_generation.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(saved_generation.get("saved_prior_targets"), generation["target_coefficients"])
             and torch.equal(saved_generation.get("actual_E0G0_coefficients"),
                             generation["bundle"]["conditions"]["E0G0"]["coefficients"])
             and torch.equal(saved_generation.get("actual_E1G1_coefficients"),
                             generation["bundle"]["conditions"]["E1G1"]["coefficients"]),
             "saved generation outputs differ from the reused source commands and phase states")

    snapshot = _snapshot_runtime(context)
    recovery_source_path = artifact_dir / "analysis_recovery_source.py"
    _require(not recovery_source_path.exists(), "refusing to replace analysis recovery source evidence",
             FileExistsError)
    recovery_source_path.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    try:
        summary = _analyze(context, generation, train_outputs, manifest, artifact_dir,
                           numpy_only=False)
        _assert_snapshot_unchanged(context, snapshot)
    except Exception as exc:
        _write_json(artifact_dir / "analysis_recovery_status.json", {
            "task_id": TASK_ID, "status": "INVALID_HARNESS",
            "error_type": type(exc).__name__, "error": str(exc),
            "new_optical_forwards": 0, "optimizer_updates": 0,
            "official_test_reads": 0,
        })
        raise
    recovery = {
        "task_id": TASK_ID, "status": "ANALYSIS_RECOVERED_FROM_SAVED_OUTPUTS",
        "prior_status": prior_status["status"],
        "prior_error": "NumPy analysis received Torch-backed raw CMOS summaries",
        "training_measurements_revalidated": True,
        "generated_population_and_commands_revalidated": True,
        "cross_grid_controls_reused_and_revalidated": True,
        "new_optical_forwards": 0, "new_4096_reference_forwards": 0,
        "optimizer_updates": 0, "official_test_reads": 0,
        "recovery_source": recovery_source_path.name,
        "decision_route": summary["decision_route"],
    }
    _write_json(artifact_dir / "analysis_recovery.json", recovery)
    _write_json(status_path, {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "train_rows": 2_000, "production_full_path_rows": 4_000,
        "cross_grid_reference_rows": 40, "generated_rows_reused": 2_000,
        "production_forwards_reused": 40,
        "cross_grid_reference_forwards_reused": 40,
        "production_forwards_attempted": 3_960,
        "cross_grid_reference_forwards_attempted": 0,
        "production_forwards_total": 4_000,
        "cross_grid_reference_forwards_total": 40,
        "analysis_recovered_from_saved_outputs": True,
        "new_optical_forwards_during_recovery": 0,
        "old_cache_mismatch_status": cache_summary["status"],
        "old_cache_mismatch_is_nonblocking_diagnostic": True,
        "optimizer_updates": 0, "optimizer_instances": 0,
        "prior_refits": 0, "prior_samples": 0, "official_test_reads": 0,
        "source_model_adam_frozen_assets_and_evaluators_unchanged": True,
        "decision_route": summary["decision_route"],
        "windows_independent_review": "PENDING",
    })
    _write_closeout(summary, None, artifact_dir)
    return summary


def _numeric_tree_close(left: Any, right: Any, tolerance: float = 1.0e-12) -> bool:
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and set(left) == set(right) and all(
            _numeric_tree_close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(
            _numeric_tree_close(a, b, tolerance) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, bool) or left is None or isinstance(left, str):
        return left == right
    if isinstance(left, (int, float)):
        return isinstance(right, (int, float)) and math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    return left == right


def _compare_csv_rows(left_path: Path, right_path: Path, *, tolerance: float = 1.0e-12) -> None:
    with left_path.open(newline="", encoding="utf-8") as stream:
        left = list(csv.DictReader(stream))
    with right_path.open(newline="", encoding="utf-8") as stream:
        right = list(csv.DictReader(stream))
    _require(len(left) == len(right) and (not left or set(left[0]) == set(right[0])),
             "independently recomputed per-sample CSV is incomplete")
    for row_index, (a, b) in enumerate(zip(left, right, strict=True)):
        for key in a:
            try:
                left_number, right_number = float(a[key]), float(b[key])
            except (TypeError, ValueError):
                _require(a[key] == b[key],
                         f"independent CSV categorical mismatch at row {row_index}: {key}")
            else:
                _require(math.isclose(left_number, right_number, rel_tol=tolerance,
                                      abs_tol=tolerance),
                         f"independent CSV numeric mismatch at row {row_index}: {key}")


def _write_cmos_control_montage(path: Path, controls: Mapping[int, Any],
                                manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    control_indices = [int(row["population_index"]) for row in manifest
                       if row["positive_control"]]
    _require(len(control_indices) == 20, "raw CMOS montage requires two fixed controls per class")
    arrays = [controls[step][f"raw_cmos_{grid}"] for step in PHASE_STEPS
              for grid in ("2048", "4096")]
    maximum = max(float(values.max()) for values in arrays)
    _require(math.isfinite(maximum) and maximum > 0.0,
             "raw CMOS controls are empty or nonpositive")
    figure, axes = plt.subplots(20, 4, figsize=(12.0, 34.0), squeeze=False,
                                constrained_layout=True)
    columns = ((0, "raw_cmos_2048", "step 0, 2048"),
               (2_000, "raw_cmos_2048", "step 2000, 2048"),
               (0, "raw_cmos_4096", "step 0, 4096"),
               (2_000, "raw_cmos_4096", "step 2000, 4096"))
    image = None
    for row_index, population_index in enumerate(control_indices):
        sample = manifest[population_index]
        for column_index, (step, tensor_key, title) in enumerate(columns):
            position = control_indices.index(population_index)
            values = controls[step][tensor_key][position, 0].detach().cpu().numpy()
            image = axes[row_index, column_index].imshow(
                values, cmap="inferno", vmin=0.0, vmax=maximum, interpolation="nearest"
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(title)
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(
                    f"class {sample['label']} / pos {sample['class_position']}", fontsize=7
                )
    _require(image is not None, "raw CMOS montage has no rendered panels")
    figure.suptitle("TRAIN raw CMOS controls; one shared linear scale, W/m²; diagnostic maps")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.5,
                    label="raw CMOS intensity (W/m²)")
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {
        "path": path.name, "rows": 20, "panels": 80,
        "samples_per_class": 2, "shared_linear_vmin_w_per_m2": 0.0,
        "shared_linear_vmax_w_per_m2": maximum,
        "display_is_raw_cmos_heatmap_not_semantic_image": True,
    }


def _verify_repaired_saved(config: Mapping[str, Any], artifact_dir: Path) -> dict[str, Any]:
    expected = (ROOT / config["sources"]["output_backup"]).resolve()
    _require(artifact_dir.resolve() == expected and artifact_dir.is_dir(),
             "Windows independent consumer requires the exact attempt_04 backup", ValueError)
    review_dir = artifact_dir.parent / "windows_review_02"
    partial_review_files = {
        "bootstrap_draws.pt", "input_distribution_analysis.json",
        "modal_amplitude_distribution.csv", "per_sample_distance_results.csv",
    }
    partial_review = review_dir.exists()
    if partial_review:
        _require(review_dir.is_dir()
                 and {path.name for path in review_dir.iterdir()} == partial_review_files,
                 f"refusing to replace or reuse a non-partial Windows review: {review_dir}",
                 FileExistsError)
    context = _load_research_context(config, device=torch.device("cpu"), local_sources=True)
    runtime = context["runtime"]
    manifest = _sample_manifest(context["schedule"], runtime["labels"], runtime["train_indices"])
    source_dir = (ROOT / str(config["sources"]["measurement_source_backup"])).resolve()
    reused_noncontrols, _, defect = _load_attempt03_noncontrol_outputs(
        config, context, manifest, source_dir, local_sources=True
    )
    _validate_sample_manifest(artifact_dir / "train_sample_manifest.csv", manifest)

    source_training = _load_pt(source_dir / "training_fullpath_outputs.pt", mmap=True)
    training = _load_pt(artifact_dir / "training_fullpath_outputs.pt", mmap=True)
    controls = _load_pt(artifact_dir / "cross_grid_controls.pt", mmap=True)
    command_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    control_identity = _validate_control_input_evidence(
        controls, manifest, runtime, command_device=command_device
    )
    saved_control_identity = json.loads(
        (artifact_dir / "control_input_identity_validation.json").read_text(encoding="utf-8")
    )
    producer_identity = {key: control_identity[key] for key in saved_control_identity}
    _require(set(saved_control_identity) == set(control_identity) - {"command_recompute_device"}
             and _numeric_tree_close(producer_identity, saved_control_identity),
             "Windows GPU replay of the source image-to-SLM command differs from the saved producer check")
    control_indices, control_rows, control_labels = _positive_control_identity(manifest)
    noncontrols = torch.tensor([index for index in range(2_000)
                                if index not in set(control_indices.tolist())], dtype=torch.long)
    source_rows = torch.tensor([int(row["source_train_row"]) for row in manifest], dtype=torch.long)
    labels = torch.tensor([int(row["label"]) for row in manifest], dtype=torch.long)
    _require(training.get("task_id") == TASK_ID
             and training.get("population_role") == "consumed_train_diagnostic"
             and training.get("grid") == GRID_PRODUCTION
             and training.get("rows") == 2_000
             and training.get("rows_per_class") == ROWS_PER_CLASS
             and torch.equal(training.get("modal_scale"), runtime["scale"].to(torch.float64))
             and set(training.get("endpoints", {})) == set(PHASE_STEPS),
             "attempt_04 population role, grid, scale, or endpoints changed")
    train_outputs = training["endpoints"]
    for step in PHASE_STEPS:
        endpoint = train_outputs[step]
        original = source_training["endpoints"][step]
        _require(torch.equal(endpoint.get("source_rows"), source_rows)
                 and torch.equal(endpoint.get("labels"), labels)
                 and tuple(endpoint.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
                 and tuple(endpoint.get("plane_powers_w", torch.empty(0)).shape) == (2_000, 5)
                 and tuple(endpoint.get("raw_summaries", torch.empty(0)).shape) == (2_000, 5)
                 and torch.equal(endpoint["coefficients"].index_select(0, noncontrols),
                                 original["coefficients"].index_select(0, noncontrols))
                 and torch.equal(endpoint["plane_powers_w"].index_select(0, noncontrols),
                                 original["plane_powers_w"].index_select(0, noncontrols))
                 and torch.equal(endpoint["raw_summaries"].index_select(0, noncontrols),
                                 original["raw_summaries"].index_select(0, noncontrols))
                 and torch.equal(endpoint["coefficients"].index_select(0, control_indices),
                                 controls[step]["coefficients_2048"])
                 and torch.equal(endpoint["plane_powers_w"].index_select(0, control_indices),
                                 controls[step]["plane_powers_w_2048"])
                 and torch.equal(endpoint["raw_summaries"].index_select(0, control_indices),
                                 controls[step]["raw_summaries_2048"])
                 and torch.equal(endpoint["raw_cmos_controls_2048"], controls[step]["raw_cmos_2048"])
                 and torch.isfinite(endpoint["coefficients"].real).all().item()
                 and torch.isfinite(endpoint["coefficients"].imag).all().item(),
                 f"step {step} attempt_04 population differs from the exact 3,960-row reuse plus corrected controls")
        endpoint_file = _load_pt(artifact_dir / f"train_optical_outputs_step_{step:05d}.pt")
        _require(_same_state(endpoint_file, endpoint),
                 f"step {step} endpoint file differs from the full attempt_04 payload")
    _validate_training_per_sample(artifact_dir / "train_fullpath_per_sample.csv",
                                  train_outputs, manifest)

    cross_rows: list[dict[str, Any]] = []
    for step in PHASE_STEPS:
        item = controls[step]
        for index, population_index in enumerate(control_indices.tolist()):
            coefficient_error = projected._phase_error(
                item["coefficients_2048"][index], item["coefficients_4096"][index]
            )
            raw_error = projected._raw_error(
                item["raw_cmos_2048"][index], item["raw_cmos_4096"][index]
            )
            cross_rows.append({
                "endpoint_step": step, "population_index": population_index,
                "source_train_row": int(control_rows[index]), "label": int(control_labels[index]),
                "coefficient_relative_error": coefficient_error,
                "raw_cmos_relative_error": raw_error,
                "passed_1e-4": coefficient_error <= CROSS_GRID_TOLERANCE
                and raw_error <= CROSS_GRID_TOLERANCE,
            })
    cross_report = json.loads((artifact_dir / "cross_grid_reference.json").read_text(encoding="utf-8"))
    _require(all(row["passed_1e-4"] for row in cross_rows)
             and cross_report.get("task_id") == TASK_ID
             and cross_report.get("status") == "PASSED"
             and cross_report.get("source_attempt") == "attempt_04_corrected_controls"
             and cross_report.get("production_rows_included") == 40
             and cross_report.get("independent_reference_rows") == 40
             and cross_report.get("production_forwards_attempted_this_attempt") == 40
             and cross_report.get("reference_forwards_attempted_this_attempt") == 40
             and _numeric_tree_close(cross_report.get("rows"), cross_rows)
             and math.isclose(float(cross_report["maximum_coefficient_relative_error"]),
                              max(row["coefficient_relative_error"] for row in cross_rows),
                              rel_tol=1.0e-12, abs_tol=1.0e-15)
             and math.isclose(float(cross_report["maximum_raw_cmos_relative_error"]),
                              max(row["raw_cmos_relative_error"] for row in cross_rows),
                              rel_tol=1.0e-12, abs_tol=1.0e-15),
             "Windows independent cross-grid recomputation differs or fails 1e-4")

    cache_summary = _cache_crosscheck(context, train_outputs, manifest)
    cache_summary["measurement_validity_required"] = True
    saved_cache = json.loads((artifact_dir / "cache_crosscheck.json").read_text(encoding="utf-8"))
    _require(cache_summary["passed"] is True
             and cache_summary["maximum_relative_error"] <= CACHE_TOLERANCE
             and _numeric_tree_close(cache_summary, saved_cache),
             "Windows independent all-row step-0 cache check fails the unchanged 1e-6 tolerance")
    cache_preflight = json.loads((artifact_dir / "cache_preflight.json").read_text(encoding="utf-8"))
    cache_map = _cache_map(context)
    control_cache_rows = torch.tensor([cache_map[int(row)] for row in control_rows], dtype=torch.long)
    control_cache_errors = _relative_coefficient_error(
        controls[0]["coefficients_2048"],
        context["cache"]["coefficients"].index_select(0, control_cache_rows),
    )
    _require(cache_preflight.get("status") == "PASSED"
             and cache_preflight.get("passed") is True
             and cache_preflight.get("source_attempt") == "attempt_04_corrected_controls"
             and cache_preflight.get("rows_compared") == 20
             and cache_preflight.get("tolerance") == CACHE_TOLERANCE
             and float(control_cache_errors.max()) <= CACHE_TOLERANCE
             and math.isclose(float(cache_preflight["maximum_relative_error"]),
                              float(control_cache_errors.max()), rel_tol=1.0e-12, abs_tol=1.0e-15),
             "Windows independently recomputed corrected-control cache preflight differs")

    generation = _load_generation_assets(context, config, local_sources=True)
    saved_generation = _load_pt(artifact_dir / "generation_coefficients.pt", mmap=True)
    _require(saved_generation.get("task_id") == TASK_ID
             and saved_generation.get("source_run_id") == TRAINING_RUN_ID
             and saved_generation.get("population_role") == "consumed_diagnostic_population"
             and saved_generation.get("commands_reused_in_place") is True
             and saved_generation.get("phase_sources_revalidated") is True
             and torch.equal(saved_generation.get("labels"), generation["labels"])
             and torch.equal(saved_generation.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(saved_generation.get("saved_prior_targets"), generation["target_coefficients"])
             and torch.equal(saved_generation.get("actual_E0G0_coefficients"),
                             generation["bundle"]["conditions"]["E0G0"]["coefficients"])
             and torch.equal(saved_generation.get("actual_E1G1_coefficients"),
                             generation["bundle"]["conditions"]["E1G1"]["coefficients"]),
             "attempt_04 generated population, prior targets, or actual coefficients changed")
    stored_analysis = json.loads((artifact_dir / "input_distribution_analysis.json").read_text(encoding="utf-8"))
    if partial_review:
        recomputed = json.loads(
            (review_dir / "input_distribution_analysis.json").read_text(encoding="utf-8")
        )
    else:
        review_dir.mkdir(parents=True, exist_ok=False)
        recomputed = _analyze(context, generation, train_outputs, manifest, review_dir, numpy_only=True)
    _require(_numeric_tree_close(recomputed, stored_analysis),
             "Windows NumPy distance, bootstrap, or route recomputation differs")
    _compare_csv_rows(artifact_dir / "per_sample_distance_results.csv",
                      review_dir / "per_sample_distance_results.csv")
    _compare_csv_rows(artifact_dir / "modal_amplitude_distribution.csv",
                      review_dir / "modal_amplitude_distribution.csv")
    saved_draws = _load_pt(artifact_dir / "bootstrap_draws.pt").numpy()
    independent_draws = _load_pt(review_dir / "bootstrap_draws.pt").numpy()
    _require(saved_draws.shape == independent_draws.shape == (BOOTSTRAP_COUNT, 3)
             and np.allclose(saved_draws, independent_draws, rtol=1.0e-12, atol=1.0e-12),
             "Windows bootstrap seed/resamples differ from the saved analysis")

    defect_report = json.loads((artifact_dir / "input_identity_repair.json").read_text(encoding="utf-8"))
    expected_defect = dict(defect)
    expected_defect["source_attempt_03"] = (
        "/root/autodl-tmp/codex-original/" + config["sources"]["measurement_source_run"]
    )
    expected_defect["legacy_controls_source_attempt"] = (
        "/root/autodl-tmp/codex-original/" + config["sources"]["preflight_controls_run"]
    )
    production = json.loads((artifact_dir / "production_preflight.json").read_text(encoding="utf-8"))
    status = json.loads((artifact_dir / "measurement_status.json").read_text(encoding="utf-8"))
    _require(defect_report == expected_defect
             and defect_report.get("legacy_controls_incorrect_and_excluded") == 40
             and defect_report.get("noncontrol_outputs_reused_unchanged") == 3_960
             and production.get("status") == "PASSED"
             and production.get("production_forwards_new") == 40
             and production.get("production_forwards_reused") == 3_960
             and production.get("cross_grid_reference_forwards_new") == 40
             and production.get("step_0_cache_all_2000_passed_1e-6") is True
             and status.get("status") == "COMPLETE_DIAGNOSTIC"
             and status.get("production_forwards_attempted") == 40
             and status.get("production_forwards_reused") == 3_960
             and status.get("cross_grid_reference_forwards_attempted") == 40
             and status.get("cross_grid_reference_forwards_reused") == 0
             and status.get("optimizer_updates") == 0
             and status.get("optimizer_instances") == 0
             and status.get("prior_refits") == 0 and status.get("prior_samples") == 0
             and status.get("official_test_reads") == 0,
             "attempt_04 status or repair-source evidence does not match the authorized budget")
    review = {
        "task_id": TASK_ID, "passed": True,
        "independent_consumer": "Windows source-row, raw-pixel, SLM-command, cache, and NumPy analysis reconsumption",
        "attempt03_controls_rejected_and_preserved_as_failure_evidence": True,
        "numpy_analysis_reused_from_verified_partial_review": partial_review,
        "attempt03_noncontrol_rows_reused_bit_identically": 3_960,
        "actual_control_source_rows_labels_pixels_and_commands_revalidated": control_identity,
        "corrected_2048_forwards": 40, "corrected_4096_reference_forwards": 40,
        "maximum_cross_grid_coefficient_relative_error": max(
            row["coefficient_relative_error"] for row in cross_rows),
        "maximum_cross_grid_raw_cmos_relative_error": max(
            row["raw_cmos_relative_error"] for row in cross_rows),
        "step_0_cache_all_2000_recomputed": cache_summary,
        "numpy_distance_and_bootstrap_match": True,
        "decision_route": recomputed["decision_route"],
        "optimizer_updates": 0, "official_test_reads": 0,
    }
    review["raw_cmos_control_montage"] = _write_cmos_control_montage(
        review_dir / "raw_cmos_train_controls_comparison.png", controls, manifest
    )
    _write_json(review_dir / "recomputed_analysis.json", recomputed)
    _write_json(review_dir / "independent_review.json", review)
    _write_closeout(recomputed, review, review_dir)
    return review


def _verify_saved(config: Mapping[str, Any], artifact_dir: Path) -> dict[str, Any]:
    expected = (ROOT / config["sources"]["output_backup"]).resolve()
    _require(artifact_dir.resolve() == expected and artifact_dir.is_dir(),
             "Windows independent consumer requires the exact new AutoDL backup directory", ValueError)
    review_dir = artifact_dir.parent / "windows_review_01"
    _require(not review_dir.exists(), f"refusing to replace an existing Windows review: {review_dir}",
             FileExistsError)
    context = _load_research_context(config, device=torch.device("cpu"), local_sources=True)
    manifest = _sample_manifest(context["schedule"], context["runtime"]["labels"],
                                context["runtime"]["train_indices"])
    reused_controls, reused_cross_report = _load_reused_preflight_controls(
        config, context, manifest, local_sources=True
    )
    _validate_sample_manifest(artifact_dir / "train_sample_manifest.csv", manifest)
    generation = _load_generation_assets(context, config, local_sources=True)
    training = _load_pt(artifact_dir / "training_fullpath_outputs.pt")
    _require(training.get("task_id") == TASK_ID
             and training.get("population_role") == "consumed_train_diagnostic"
             and training.get("grid") == GRID_PRODUCTION
             and training.get("rows") == 2_000
             and training.get("rows_per_class") == ROWS_PER_CLASS
             and training.get("reference_positions") == "even class positions 0,2,...,198"
             and training.get("calibration_positions") == "odd class positions 1,3,...,199"
             and tuple(training.get("modal_scale", torch.empty(0)).shape) == (100,)
             and torch.equal(training["modal_scale"], context["runtime"]["scale"].detach().cpu()),
             "saved full-path coefficients use wrong scale, role, or task identity")
    train_outputs = training.get("endpoints", {})
    _require(set(train_outputs) == set(PHASE_STEPS),
             "saved full-path measurements omit an endpoint")
    for step in PHASE_STEPS:
        endpoint = train_outputs[step]
        _require(torch.equal(endpoint.get("source_rows"), torch.tensor(
            [int(item["source_train_row"]) for item in manifest], dtype=torch.long))
            and torch.equal(endpoint.get("labels"), torch.tensor(
                [int(item["label"]) for item in manifest], dtype=torch.long))
            and tuple(endpoint.get("coefficients", torch.empty(0)).shape) == (2_000, 100)
            and tuple(endpoint.get("plane_powers_w", torch.empty(0)).shape) == (2_000, 5)
            and tuple(endpoint.get("raw_summaries", torch.empty(0)).shape) == (2_000, 5)
            and torch.isfinite(endpoint["coefficients"].real).all().item()
            and torch.isfinite(endpoint["coefficients"].imag).all().item(),
            f"saved endpoint {step} rows are missing, reordered, or nonfinite")
    _validate_training_per_sample(artifact_dir / "train_fullpath_per_sample.csv",
                                  train_outputs, manifest)
    # The controls are class-major at indices 0,1,200,201,...; verify both saved grids from raw tensors.
    controls = _load_pt(artifact_dir / "cross_grid_controls.pt")
    control_indices = torch.tensor([int(row["population_index"]) for row in manifest
                                    if row["positive_control"]], dtype=torch.long)
    _require(set(controls) == set(PHASE_STEPS)
             and _same_state(controls, reused_controls),
             "attempt_03 controls differ from the validated attempt_02 controls")
    independent_cross = []
    for step in PHASE_STEPS:
        item = controls[step]
        train_item = train_outputs[step]
        _require(torch.equal(item["population_indices"], control_indices)
                 and torch.equal(item["coefficients_2048"],
                                 train_item["coefficients"].index_select(0, control_indices))
                 and torch.equal(item["raw_cmos_2048"], train_item["raw_cmos_controls_2048"])
                 and tuple(item["coefficients_4096"].shape) == (20, 100)
                 and tuple(item["raw_cmos_2048"].shape) == (20, 1, 200, 200)
                 and tuple(item["raw_cmos_4096"].shape) == (20, 1, 200, 200),
                 f"step {step} cross-grid command/row identity is invalid")
        for index in range(20):
            coeff = projected._phase_error(item["coefficients_2048"][index],
                                            item["coefficients_4096"][index])
            cmos = projected._raw_error(item["raw_cmos_2048"][index],
                                        item["raw_cmos_4096"][index])
            independent_cross.append({"endpoint_step": step, "control_index": index,
                                      "coefficient_relative_error": coeff,
                                      "raw_cmos_relative_error": cmos,
                                      "passed": coeff <= CROSS_GRID_TOLERANCE
                                      and cmos <= CROSS_GRID_TOLERANCE})
    _require(all(row["passed"] for row in independent_cross),
             "Windows independently recomputed cross-grid check failed")
    cross_report = json.loads((artifact_dir / "cross_grid_reference.json").read_text(encoding="utf-8"))
    recomputed_cross_rows = []
    for step in PHASE_STEPS:
        for control_index, population_index in enumerate(control_indices.tolist()):
            recomputed_cross_rows.append({
                "endpoint_step": step,
                "population_index": population_index,
                "source_train_row": int(manifest[population_index]["source_train_row"]),
                "label": int(manifest[population_index]["label"]),
                "coefficient_relative_error": independent_cross[step // 2_000 * 20 + control_index][
                    "coefficient_relative_error"],
                "raw_cmos_relative_error": independent_cross[step // 2_000 * 20 + control_index][
                    "raw_cmos_relative_error"],
                "passed_1e-4": independent_cross[step // 2_000 * 20 + control_index]["passed"],
            })
    _require(cross_report.get("task_id") == TASK_ID
             and cross_report.get("status") == "PASSED"
             and cross_report.get("production_rows_included") == 40
             and cross_report.get("independent_reference_rows") == 40
             and cross_report.get("rows_per_class_per_endpoint") == 2
             and cross_report.get("production_grid") == GRID_PRODUCTION
             and cross_report.get("reference_grid") == GRID_REFERENCE
             and cross_report.get("tolerance") == CROSS_GRID_TOLERANCE
             and cross_report.get("source_attempt") == "attempt_02"
             and cross_report.get("source_report_recomputed") is True
             and cross_report.get("production_forwards_attempted_this_attempt") == 0
             and cross_report.get("reference_forwards_attempted_this_attempt") == 0
             and _numeric_tree_close(cross_report, reused_cross_report)
             and _numeric_tree_close(cross_report.get("rows"), recomputed_cross_rows)
             and math.isclose(float(cross_report["maximum_coefficient_relative_error"]),
                              max(row["coefficient_relative_error"] for row in independent_cross),
                              rel_tol=1.0e-12, abs_tol=1.0e-15)
             and math.isclose(float(cross_report["maximum_raw_cmos_relative_error"]),
                              max(row["raw_cmos_relative_error"] for row in independent_cross),
                              rel_tol=1.0e-12, abs_tol=1.0e-15),
             "saved cross-grid report differs from Windows recomputation")
    cache_summary = _cache_crosscheck(context, train_outputs, manifest)
    saved_cache = json.loads((artifact_dir / "cache_crosscheck.json").read_text(encoding="utf-8"))
    _require(_numeric_tree_close(cache_summary, saved_cache),
             "Windows independently recomputed step-0 cache comparison differs")
    cache_preflight = json.loads((artifact_dir / "cache_preflight.json").read_text(encoding="utf-8"))
    cache_map = _cache_map(context)
    preflight_rows = [int(manifest[index]["source_train_row"]) for index in control_indices]
    cache_indices = torch.tensor([cache_map[row] for row in preflight_rows], dtype=torch.long)
    preflight_errors = _relative_coefficient_error(
        reused_controls[0]["coefficients_2048"],
        context["cache"]["coefficients"].index_select(0, cache_indices),
    )
    _validate_cache_diagnostic_record(
        cache_preflight, preflight_errors, rows_compared=20,
        source_attempt="attempt_02_saved_controls_reused",
    )
    stored_analysis = json.loads((artifact_dir / "input_distribution_analysis.json").read_text(encoding="utf-8"))
    production_preflight = json.loads(
        (artifact_dir / "production_preflight.json").read_text(encoding="utf-8")
    )
    _require(production_preflight.get("status") == "PASSED"
             and production_preflight.get("production_rows_included_in_total") == 40
             and production_preflight.get("cross_grid_reference_rows") == 40
             and production_preflight.get("preflight_controls_source_attempt") == "attempt_02"
             and production_preflight.get("preflight_controls_independently_recomputed") is True
             and production_preflight.get("production_forwards_reused") == 40
             and production_preflight.get("cross_grid_reference_forwards_reused") == 40
             and production_preflight.get("production_forwards_authorized") == 4_000
             and production_preflight.get("production_forwards_remaining") == 3_960
             and production_preflight.get("production_forwards_attempted") == 0
             and production_preflight.get("cross_grid_reference_forwards_attempted") == 0
             and production_preflight.get("cache_preflight_status") == cache_preflight["status"]
             and production_preflight.get("cache_preflight_passed") is cache_preflight["passed"]
             and production_preflight.get("cache_mismatch_nonblocking_by_explicit_approval") is True
             and production_preflight.get("optimizer_updates") == 0
             and production_preflight.get("optimizer_instances") == 0,
             "saved representative production preflight is incomplete")
    source_identity = json.loads((artifact_dir / "source_identity.json").read_text(encoding="utf-8"))
    _require(source_identity.get("task_id") == TASK_ID
             and source_identity.get("source_training_run_id") == TRAINING_RUN_ID
             and source_identity.get("source_endpoint") == "D0"
             and source_identity.get("official_test_read") is False
             and source_identity.get("model_identity") == context["checkpoints"][0]["model_identity"]
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("global_step") == 50_000
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("encoder_adam_step") == 45_000
             and source_identity.get("measured_endpoints", {}).get("0", {}).get("generator_adam_step") == 50_000
             and source_identity.get("measured_endpoints", {}).get("2000", {}).get("global_step") == 52_000,
             "saved optical endpoint identity differs from explicit source checkpoints")
    _require(source_identity["measured_endpoints"]["2000"].get("encoder_adam_step") == 47_000
             and source_identity["measured_endpoints"]["2000"].get("generator_adam_step") == 52_000,
             "saved endpoint Adam counters differ from the registered D0 update lineage")
    saved_generation = _load_pt(artifact_dir / "generation_coefficients.pt")
    _require(saved_generation.get("task_id") == TASK_ID
             and saved_generation.get("source_run_id") == TRAINING_RUN_ID
             and saved_generation.get("population_role") == "consumed_diagnostic_population"
             and saved_generation.get("commands_reused_in_place") is True
             and saved_generation.get("phase_sources_revalidated") is True
             and torch.equal(saved_generation.get("labels"), generation["labels"])
             and torch.equal(saved_generation.get("proposal_ids"), generation["proposal_ids"])
             and torch.equal(saved_generation.get("saved_prior_targets"), generation["target_coefficients"])
             and torch.equal(saved_generation.get("actual_E0G0_coefficients"),
                             generation["bundle"]["conditions"]["E0G0"]["coefficients"])
             and torch.equal(saved_generation.get("actual_E1G1_coefficients"),
                             generation["bundle"]["conditions"]["E1G1"]["coefficients"]),
             "saved generation coefficients differ from the actual source commands and outputs")
    with (artifact_dir / "per_sample_distance_results.csv").open(
            newline="", encoding="utf-8") as stream:
        distance_csv = list(csv.DictReader(stream))
    _require(len(distance_csv) == 6_000,
             "saved per-sample distance table is incomplete")
    counts: dict[tuple[str, str], int] = {}
    for row in distance_csv:
        key = (row["endpoint_step"], row["population_role"])
        counts[key] = counts.get(key, 0) + 1
        label = int(row["label"])
        index = int(row["population_index"])
        _require(row["population_role"] in {"generated", "train_calibration"}
                 and 0 <= label < CLASS_COUNT and 0 <= index < 2_000,
                 "saved per-sample distance row has an invalid role or population index")
        if row["population_role"] == "generated":
            _require(label == int(generation["labels"][index])
                     and int(row["proposal_id"]) == int(generation["proposal_ids"][index]),
                     "saved generated distance row is misaligned with its label/proposal")
        else:
            _require(label == int(manifest[index]["label"])
                     and int(row["source_train_row"]) == int(manifest[index]["source_train_row"])
                     and manifest[index]["role"] == "calibration",
                     "saved TRAIN calibration distance row is misaligned")
    _require(counts == {("0", "generated"): 2_000, ("0", "train_calibration"): 1_000,
                        ("2000", "generated"): 2_000, ("2000", "train_calibration"): 1_000},
             "saved per-sample distance table has missing or duplicate cohort rows")
    review_dir.mkdir(parents=True, exist_ok=False)
    recomputed = _analyze(context, generation, train_outputs, manifest, review_dir,
                          numpy_only=True)
    _require(_numeric_tree_close(recomputed, stored_analysis),
             "Windows NumPy analysis differs from the measured result or route")
    _compare_csv_rows(artifact_dir / "per_sample_distance_results.csv",
                      review_dir / "per_sample_distance_results.csv")
    _compare_csv_rows(artifact_dir / "modal_amplitude_distribution.csv",
                      review_dir / "modal_amplitude_distribution.csv")
    draws = _load_pt(artifact_dir / "bootstrap_draws.pt").numpy()
    independent_draws = _load_pt(review_dir / "bootstrap_draws.pt").numpy()
    _require(draws.shape == independent_draws.shape == (BOOTSTRAP_COUNT, 3)
             and np.allclose(draws, independent_draws, rtol=1.0e-12, atol=1.0e-12),
             "Windows independent bootstrap resamples differ")
    status = json.loads((artifact_dir / "measurement_status.json").read_text(encoding="utf-8"))
    recovery = json.loads((artifact_dir / "analysis_recovery.json").read_text(encoding="utf-8"))
    recovery_before = json.loads(
        (artifact_dir / "measurement_status_before_recovery.json").read_text(encoding="utf-8")
    )
    recovery_source = recovery.get("recovery_source")
    _require(status.get("status") == "COMPLETE_DIAGNOSTIC"
             and status.get("production_full_path_rows") == 4_000
             and status.get("cross_grid_reference_rows") == 40
             and status.get("production_forwards_reused") == 40
             and status.get("cross_grid_reference_forwards_reused") == 40
             and status.get("production_forwards_attempted") == 3_960
             and status.get("cross_grid_reference_forwards_attempted") == 0
             and status.get("production_forwards_total") == 4_000
             and status.get("cross_grid_reference_forwards_total") == 40
             and status.get("old_cache_mismatch_status") == cache_summary["status"]
             and status.get("old_cache_mismatch_is_nonblocking_diagnostic") is True
             and status.get("analysis_recovered_from_saved_outputs") is True
             and status.get("new_optical_forwards_during_recovery") == 0
             and status.get("optimizer_updates") == 0
             and status.get("optimizer_instances") == 0
             and status.get("prior_samples") == 0 and status.get("official_test_reads") == 0,
             "measurement status is incomplete or outside the zero-update contract")
    _require(recovery.get("status") == "ANALYSIS_RECOVERED_FROM_SAVED_OUTPUTS"
             and recovery.get("prior_status") == "PREFLIGHT_PASSED_FULL_MEASUREMENT_PENDING"
             and recovery.get("training_measurements_revalidated") is True
             and recovery.get("generated_population_and_commands_revalidated") is True
             and recovery.get("cross_grid_controls_reused_and_revalidated") is True
             and recovery.get("new_optical_forwards") == 0
             and recovery.get("new_4096_reference_forwards") == 0
             and recovery.get("optimizer_updates") == 0
             and recovery.get("official_test_reads") == 0
             and recovery_before.get("status") == "PREFLIGHT_PASSED_FULL_MEASUREMENT_PENDING"
             and recovery_source == "analysis_recovery_source.py"
             and (artifact_dir / recovery_source).is_file()
             and (artifact_dir / "analysis_failure_run.log").is_file()
             and "TypeError" in (artifact_dir / "analysis_failure_run.log").read_text(encoding="utf-8"),
             "saved analysis recovery evidence is incomplete or does not preserve the original failure")
    review = {
        "task_id": TASK_ID, "passed": True,
        "independent_consumer": "Windows NumPy plus explicit source and output reconsumption",
        "sample_manifest_recomputed": True,
        "attempt_02_preflight_controls_reused_and_revalidated": True,
        "analysis_recovered_from_saved_outputs": True,
        "generation_commands_labels_proposal_ids_and_phase_states_revalidated": True,
        "all_4000_training_rows_revalidated": True,
        "cross_grid_rows_recomputed": 40,
        "maximum_cross_grid_coefficient_relative_error": max(
            row["coefficient_relative_error"] for row in independent_cross),
        "maximum_cross_grid_raw_cmos_relative_error": max(
            row["raw_cmos_relative_error"] for row in independent_cross),
        "cache_crosscheck_recomputed": cache_summary,
        "new_2048_forwards": 3_960,
        "reused_2048_forwards": 40,
        "new_4096_forwards": 0,
        "reused_4096_forwards": 40,
        "numpy_distance_and_bootstrap_match": True,
        "decision_route": recomputed["decision_route"],
        "optimizer_updates": 0, "official_test_reads": 0,
    }
    _write_json(review_dir / "recomputed_analysis.json", recomputed)
    review["raw_cmos_control_montage"] = _write_cmos_control_montage(
        review_dir / "raw_cmos_train_controls_comparison.png", controls, manifest
    )
    _write_json(review_dir / "independent_review.json", review)
    _write_closeout(recomputed, review, review_dir)
    return review


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--repair-controls", action="store_true",
                        help="remeasure only the 40 incorrectly bound attempt_03 TRAIN controls")
    parser.add_argument("--verify-saved", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    config = _read_config(args.config.resolve())
    _require(not (args.repair_controls and args.verify_saved),
             "--repair-controls and --verify-saved are mutually exclusive", ValueError)
    _require(args.repair_controls or args.verify_saved,
             "select --repair-controls or --verify-saved; the superseded unrepaired measurement path is disabled",
             ValueError)
    _require(args.artifact_dir is not None,
             "--artifact-dir is required with the selected explicit repair/verification mode", ValueError)
    if args.repair_controls:
        summary = _repair_controls(config, args.artifact_dir.resolve())
        print(json.dumps({"task_id": TASK_ID, "status": "repaired_measurement_complete",
                          "new_2048_forwards": 40, "new_4096_forwards": 40,
                          "decision_route": summary["decision_route"],
                          "output_dir": OUTPUT_DIR.relative_to(ROOT).as_posix()}, ensure_ascii=False))
        return 0
    if args.verify_saved:
        review = _verify_repaired_saved(config, args.artifact_dir.resolve())
        print(json.dumps({"task_id": TASK_ID, "passed": review["passed"],
                          "decision_route": review["decision_route"]}, ensure_ascii=False))
        return 0
    raise AssertionError("argument validation failed to select a mode")


if __name__ == "__main__":
    raise SystemExit(main())
