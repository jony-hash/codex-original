"""Prepare and finalize a fixed-C native CMOS blind review for G1 versus G4."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.evidence import atomic_torch_save, atomic_write_json
from ldm_project.v2.application.research.lightgen.image_readout import resize_view


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1"
AUTHORITY = "USER_APPROVED_SMALL_G1_G4_NATIVE_READABILITY_20261022"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
REPLICATION_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
MODEL_NAME = "C_seed_2026101501"
SOURCE_STEP = 45_000
METHODS = ("G1", "G4")
CLASSES = tuple(range(10))
SAMPLE_COUNT_PER_CLASS = 1_000
SELECT_PER_CLASS = 10
BATCH_SIZE = 10
REPLAY_RTOL = 2e-6
REPLAY_ATOL = 5e-8
SELECTION_SEEDS = {"G1": 2026102271, "G4": 2026102272}
SHUFFLE_SEED = 2026102283
BOOTSTRAP_SEED = 2026102285
BOOTSTRAP_RESAMPLES = 2_000
REVIEW_SCHEMA = "lightgen_small_g1_g4_native_readability_review.v1"
RATINGS_SCHEMA = "lightgen_small_g1_g4_native_readability_ratings.v1"
REVIEW_TOTAL = 240
REVIEW_GENERATED = 200
REVIEW_DUPLICATES = 20
REVIEW_ANCHORS = 20
MIN_DUPLICATE_DISTANCE = 40
QUALITY_GATE = {
    "reconstruction_psnr_db_min": 20.0,
    "reconstruction_ssim_min": 0.80,
    "generation_macro_recall_min": 0.80,
    "generation_recall_per_class_min": 0.60,
    "proxy_clipping_fraction_max": 0.01,
    "generated_to_dev_tv_ratio_min": 0.25,
}
WATCHER_TELEMETRY_HEADER = (
    "timestamp_utc,run_id,run_state,progress_state,training_pid,training_rss_kib,"
    "gpu_utilization_percent,gpu_memory_used_mib,gpu_memory_total_mib,host_mem_available_kib,"
    "seed,arm,task_local_update,optimizer_step,sample_error"
)


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file(), f"explicit readability config is missing: {path}", FileNotFoundError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict)
             and config.get("schema") == "lightgen_small_g1_g4_native_readability.v1"
             and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY,
             "readability configuration identity changed", ValueError)
    source = config["source"]
    _require(source.get("task_id") == SOURCE_TASK_ID and source.get("branch") == "C"
             and source.get("seed") == 2026101501 and source.get("cumulative_step") == SOURCE_STEP
             and source.get("checkpoint") == (
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/"
                 "autodl_backup_attempt_01/C/checkpoint_step_010000.pt"
             )
             and source.get("baseline_evidence_dir") == (
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/"
                 "autodl_backup_attempt_01"
             )
             and source.get("replication_evidence_dir") == (
                 "outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/"
                 "runs/autodl_backup_attempt_01"
             )
             and source.get("replication_model") == MODEL_NAME
             and source.get("primary_sampling_stream") == 0,
             "fixed C endpoint or replication evidence paths changed", ValueError)
    optics = config["optics"]
    _require(optics.get("runtime_config") == "configs/research/lightgen/small_gaussian_system_baseline_v1.yaml"
             and optics.get("ols_basis") == "gaussian_s12_v1"
             and optics.get("propagation_grid") == 2048
             and optics.get("phase_shape") == [100, 100]
             and optics.get("global_gain") == "frozen_C_checkpoint"
             and optics.get("readout") == "global_gain_then_clamp_0_1_then_adaptive_average_pool_28"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "fixed Gaussian optics or production readout changed", ValueError)
    evaluation = config["evaluation"]
    _require(evaluation.get("samples_per_class_per_method") == SELECT_PER_CLASS
             and evaluation.get("selection_seed_G1") == SELECTION_SEEDS["G1"]
             and evaluation.get("selection_seed_G4") == SELECTION_SEEDS["G4"]
             and evaluation.get("batch_size") == BATCH_SIZE
             and evaluation.get("replay_rtol") == REPLAY_RTOL
             and evaluation.get("replay_atol") == REPLAY_ATOL
             and evaluation.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES
             and evaluation.get("bootstrap_seed") == BOOTSTRAP_SEED
             and evaluation.get("official_train_dev_test_reads") == 0
             and evaluation.get("new_encoder_forward") == 0
             and evaluation.get("new_evaluator_forward") == 0
             and evaluation.get("gradients") == 0
             and evaluation.get("optimizer_calls") == 0,
             "fixed replay population, thresholds, or zero-update boundaries changed", ValueError)
    human = config["human_review"]
    _require(human.get("reviewers") == 1 and human.get("cards_per_method") == 100
             and human.get("hidden_duplicates") == REVIEW_DUPLICATES
             and human.get("real_target_anchors") == REVIEW_ANCHORS
             and human.get("total_cards") == REVIEW_TOTAL
             and human.get("shuffle_seed") == SHUFFLE_SEED
             and human.get("minimum_duplicate_consistency") == 18
             and human.get("minimum_anchor_correct") == 18
             and human.get("segment_size") == 20
             and human.get("minimum_duplicate_card_distance") == MIN_DUPLICATE_DISTANCE,
             "fixed single-reviewer QC or card population changed", ValueError)
    output = config["output"]
    _require(output.get("experiment_root") == (
                 "outputs/20261022_lightgen_small_g1_g4_native_readability_v1/runs/attempt_01"
             ) and output.get("formal_result_budget_bytes") == 134_217_728,
             "fixed output path or 128 MiB budget changed", ValueError)
    runtime_path = _resolve(Path(optics["runtime_config"]))
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    _require(runtime.get("schema") == "lightgen_matched_small_vs_chen400.v1"
             and runtime["optics"]["small"].get("ols_basis") == "gaussian_s12_v1"
             and runtime["optics"]["small"].get("production_propagation_grid") == 2048
             and runtime["optics"]["small"].get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "runtime model factory config differs from the fixed C geometry", ValueError)
    return config, runtime


def _load_inputs(
    source_checkpoint_path: Path, baseline_dir: Path, replication_dir: Path,
    config: dict[str, Any], runtime: dict[str, Any],
) -> dict[str, Any]:
    expected_checkpoint = (baseline_dir / "C" / "checkpoint_step_010000.pt").resolve()
    _require(source_checkpoint_path.resolve() == expected_checkpoint,
             "source must be the explicit fixed C 45,000-step endpoint")
    _require(baseline_dir.is_dir() and replication_dir.is_dir(),
             "explicit source evidence directories are missing", FileNotFoundError)
    source_report = json.loads((baseline_dir / "machine_report.json").read_text(encoding="utf-8"))
    source_final = json.loads((baseline_dir / "final_report.json").read_text(encoding="utf-8"))
    source_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    endpoint = json.loads((baseline_dir / "C" / "endpoint_summary.json").read_text(encoding="utf-8"))
    _require(source_report.get("task_id") == SOURCE_TASK_ID
             and source_report.get("measurement_complete") is True
             and source_final.get("task_id") == SOURCE_TASK_ID
             and source_final.get("status") == "REVIEW_INCONCLUSIVE"
             and source_final.get("human_review", {}).get("quality_checks", {}).get(
                 "duplicate_consistency", {}).get("count") == 15
             and source_final.get("human_review", {}).get("quality_checks", {}).get(
                 "real_anchor_label_match", {}).get("count") == 19,
             "source C/V task has not been formally closed with its registered result")
    _require(endpoint.get("branch") == "C" and endpoint.get("global_step") == SOURCE_STEP
             and endpoint.get("status") == "VALID_FAILED"
             and endpoint.get("measurement_complete") is True,
             "historical C endpoint identity or quality decision changed")
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("task_id") == SOURCE_TASK_ID and checkpoint.get("branch") == "C"
             and checkpoint.get("completed_global_step") == SOURCE_STEP
             and checkpoint.get("seed") == 2026101501
             and checkpoint.get("config_identity") == source_config
             and checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             "loaded source checkpoint step, branch, seed, or config identity changed")
    identity = checkpoint.get("model_identity", {})
    _require(identity.get("ols") == "gaussian_s12_v1"
             and identity.get("ols_geometry", {}).get("transport") == "identity"
             and tuple(checkpoint["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(checkpoint["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "source model is not the fixed Gaussian encoder/decoder endpoint")
    optimizer_state = checkpoint.get("optimizer", {}).get("state", {})
    steps = [int(torch.as_tensor(state["step"]).item()) for state in optimizer_state.values()
             if isinstance(state, dict) and "step" in state]
    _require(len(steps) == 2 and all(value == SOURCE_STEP for value in steps),
             "source checkpoint Adam state does not match step 45,000")
    references = torch.load(baseline_dir / "source_references.pt", map_location="cpu", weights_only=False)
    for field in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(checkpoint[field].detach().cpu(), references[field].detach().cpu()),
                 f"frozen source {field} differs from its evidence reference")
    _require(checkpoint["weights"] == references["weights"],
             "frozen source loss weights differ from the registered reference")
    terminal = torch.load(baseline_dir / "C" / "terminal_model.pt", map_location="cpu", weights_only=False)
    _require(terminal.get("branch") == "C" and terminal.get("global_step") == SOURCE_STEP
             and terminal.get("run_id") == source_report.get("run_id")
             and base._assert_close(checkpoint["model"], terminal.get("model", {}), "C_terminal_model") == 0.0,
             "fixed source checkpoint differs from its saved C terminal model")
    small = source_config["optics"]["small"]
    _require(small.get("ols_basis") == "gaussian_s12_v1"
             and small.get("production_propagation_grid") == 2048
             and small.get("phase_pixel_pitch_m") == 2e-6
             and small.get("port_pitch_m") == 2e-5
             and small.get("field_radius_m", small.get("field_1e_radius_m")) == 3.5e-6
             and source_config["optics"]["wavelength_m"] == 5.32e-7
             and source_config["optics"]["input_power_w"] == 2.493e-4
             and source_config["optics"]["layer_power_transmission"] == 0.889,
             "source checkpoint geometry, power, or production grid changed")

    replication_report_path = replication_dir / "report.json"
    replication_report = json.loads(replication_report_path.read_text(encoding="utf-8"))
    _require(replication_report.get("task_id") == REPLICATION_TASK_ID
             and replication_report.get("status") == "COMPLETE_DIAGNOSTIC"
             and replication_report.get("decision", {}).get("result") ==
             "G4_PROXY_GAIN_REPEATED_ON_THREE_TRAINING_SCHEDULES"
             and replication_report.get("endpoints", {}).get(MODEL_NAME, {}).get("cumulative_step") == SOURCE_STEP,
             "three-trajectory replication report is incomplete or has the wrong C endpoint")
    _require((replication_dir / "generation" / MODEL_NAME / "G1_stream_0.pt").is_file()
             and (replication_dir / "generation" / MODEL_NAME / "G4_stream_0.pt").is_file()
             and (replication_dir / "reference_C" / "dev_examples_step_045000.pt").is_file(),
             "fixed primary cohorts or saved DEV target examples are missing", FileNotFoundError)
    cohorts: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = replication_dir / "generation" / MODEL_NAME / f"{method}_stream_0.pt"
        cohort = torch.load(path, map_location="cpu", weights_only=False)
        _require(cohort.get("task_id") == REPLICATION_TASK_ID
                 and cohort.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
                 and cohort.get("method") == method and cohort.get("model") == MODEL_NAME
                 and cohort.get("model_seed") == 2026101501 and cohort.get("stream") == 0
                 and cohort.get("global_step") == SOURCE_STEP
                 and cohort.get("classifier_based_sample_selection") is False
                 and cohort.get("per_image_normalization") is False,
                 f"saved {method} cohort identity or selection rule changed")
        expected = {
            "images": (10_000, 1, 28, 28), "ols_coefficients": (10_000, 100),
            "requested_labels": (10_000,), "component_ids": (10_000,), "proposal_ids": (10_000,),
            "ed_logits": (10_000, 10), "ec_logits": (10_000, 10),
            "ed_predictions": (10_000,), "ec_predictions": (10_000,),
            "ed_requested_margin": (10_000,), "ec_requested_margin": (10_000,),
        }
        _require(all(tuple(cohort.get(key, torch.empty(0)).shape) == shape for key, shape in expected.items()),
                 f"saved {method} cohort is incomplete or misaligned")
        labels = cohort["requested_labels"].long()
        _require(torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), SAMPLE_COUNT_PER_CLASS))
                 and torch.isfinite(cohort["images"]).all().item()
                 and torch.isfinite(cohort["ols_coefficients"].real).all().item()
                 and torch.isfinite(cohort["ols_coefficients"].imag).all().item()
                 and torch.isfinite(cohort["ed_logits"]).all().item()
                 and torch.isfinite(cohort["ec_logits"]).all().item(),
                 f"saved {method} cohort has missing classes or nonfinite values")
        _require(torch.equal(cohort["ed_logits"].argmax(1), cohort["ed_predictions"].long())
                 and torch.equal(cohort["ec_logits"].argmax(1), cohort["ec_predictions"].long()),
                 f"saved {method} logits and predictions disagree")
        pairs = list(zip(labels.tolist(), cohort["proposal_ids"].long().tolist(), strict=True))
        _require(len(set(pairs)) == 10_000, f"saved {method} accepted cohort repeats a class/proposal identity")
        cohorts[method] = cohort
    examples = torch.load(replication_dir / "reference_C" / "dev_examples_step_045000.pt",
                          map_location="cpu", weights_only=False)
    _require(examples.get("task_id") == REPLICATION_TASK_ID
             and examples.get("seed") == 2026101501 and examples.get("step") == SOURCE_STEP
             and tuple(examples.get("rows", torch.empty(0)).shape) == (20,)
             and tuple(examples.get("labels", torch.empty(0)).shape) == (20,)
             and tuple(examples.get("targets28", torch.empty(0)).shape) == (20, 1, 28, 28)
             and torch.equal(torch.bincount(examples["labels"].long(), minlength=10), torch.full((10,), 2))
             and torch.isfinite(examples["targets28"]).all().item()
             and float(examples["targets28"].min()) >= 0.0
             and float(examples["targets28"].max()) <= 1.0,
             "fixed 20-image DEV target reference is incomplete or invalid")
    _require(runtime["optics"]["small"].get("ols_basis") == "gaussian_s12_v1",
             "Gaussian model factory basis does not match the C endpoint")
    return {
        "source_report": source_report, "source_final_report": source_final,
        "source_config": source_config, "checkpoint": checkpoint, "references": references,
        "replication_report": replication_report, "cohorts": cohorts, "dev_examples": examples,
        "source_run_id": source_report["run_id"],
        "replication_dir": replication_dir,
    }


def _make_run_identity(
    config: dict[str, Any], config_path: Path, source_checkpoint: Path,
    baseline_dir: Path, replication_dir: Path, inputs: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_g1_g4_native_readability_run_identity.v1",
        "task_id": TASK_ID, "run_id": run_id, "config": config,
        "config_path": str(config_path.resolve()),
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_step": SOURCE_STEP, "source_run_id": inputs["source_run_id"],
        "baseline_evidence_dir": str(baseline_dir.resolve()),
        "replication_evidence_dir": str(replication_dir.resolve()),
        "replication_model": MODEL_NAME, "sampling_stream": 0,
        "model_identity": inputs["checkpoint"]["model_identity"],
        "source_quality_status_preserved": "VALID_FAILED",
        "new_training_updates": 0,
    }


def _progress(
    run_dir: Path, identity: dict[str, Any], status: str, phase: str,
    blocks: list[str], *, review_id: str | None = None, optical_samples: int = 0,
) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    atomic_write_json(run_dir / "progress.json", {
        "task_id": TASK_ID, "run_id": identity["run_id"], "status": status,
        "phase": phase, "completed_blocks": blocks,
        "recent_durable_write_utc": now, "heartbeat_utc": now,
        "current_seed": 2026101501, "current_arm": "fixed_C_G1_G4_native_readability",
        "current_task_local_update": 0, "current_optimizer_step": 0,
        "optical_forward_samples_total": optical_samples,
        "new_encoder_forward_samples": 0, "new_evaluator_forward_samples": 0,
        "gradients": 0, "optimizer_calls": 0,
        "review_id": review_id,
    })


def _build_selection_manifest(cohorts: dict[str, dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, int | str]] = []
    for method in METHODS:
        labels = cohorts[method]["requested_labels"].long().numpy()
        rng = np.random.default_rng(SELECTION_SEEDS[method])
        for label in CLASSES:
            pool = np.flatnonzero(labels == label)
            _require(pool.size == SAMPLE_COUNT_PER_CLASS, f"{method} class {label} sample pool is incomplete")
            selected = np.sort(rng.choice(pool, size=SELECT_PER_CLASS, replace=False))
            for row in selected.tolist():
                cohort = cohorts[method]
                items.append({
                    "method": method, "requested_label": label, "stream": 0,
                    "cohort_row": int(row), "proposal_id": int(cohort["proposal_ids"][row]),
                    "component_id": int(cohort["component_ids"][row]),
                })
    return {
        "schema": "lightgen_small_g1_g4_native_readability_selection.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"], "source_run_id": identity["source_run_id"],
        "source_step": SOURCE_STEP, "stream": 0, "selection_seeds": SELECTION_SEEDS,
        "classifier_brightness_or_appearance_used_for_selection": False,
        "per_method_per_class": SELECT_PER_CLASS, "items": items,
    }


def _validate_selection_manifest(
    manifest: dict[str, Any], cohorts: dict[str, dict[str, Any]], identity: dict[str, Any],
) -> None:
    _require(manifest.get("schema") == "lightgen_small_g1_g4_native_readability_selection.v1"
             and manifest.get("task_id") == TASK_ID and manifest.get("run_id") == identity["run_id"]
             and manifest.get("source_run_id") == identity["source_run_id"]
             and manifest.get("source_step") == SOURCE_STEP and manifest.get("stream") == 0
             and manifest.get("selection_seeds") == SELECTION_SEEDS
             and manifest.get("classifier_brightness_or_appearance_used_for_selection") is False,
             "saved selection manifest belongs to another source or task")
    items = manifest.get("items")
    _require(isinstance(items, list) and len(items) == REVIEW_GENERATED,
             "saved random selection manifest is incomplete")
    keys: set[tuple[str, int, int]] = set()
    for method in METHODS:
        for label in CLASSES:
            subset = [item for item in items if item.get("method") == method
                      and item.get("requested_label") == label]
            _require(len(subset) == SELECT_PER_CLASS,
                     f"saved selected population for {method} class {label} is incomplete")
            for item in subset:
                row = item.get("cohort_row")
                _require(isinstance(row, int) and 0 <= row < 10_000,
                         "saved selection row is invalid", ValueError)
                cohort = cohorts[method]
                _require(int(cohort["requested_labels"][row]) == label
                         and int(cohort["proposal_ids"][row]) == item.get("proposal_id")
                         and int(cohort["component_ids"][row]) == item.get("component_id")
                         and int(cohort["stream"]) == 0,
                         "saved selection row, class, proposal, or component does not match its cohort")
                key = (method, label, int(item["proposal_id"]))
                _require(key not in keys, "saved selection repeats a method/class/proposal identity")
                keys.add(key)


def _numpy_adaptive_pool_28(native: np.ndarray) -> np.ndarray:
    _require(native.shape == (200, 200) and np.isfinite(native).all(),
             "independent readout requires one finite native 200x200 image")
    output = np.empty((28, 28), dtype=np.float64)
    for out_y in range(28):
        y0 = math.floor(out_y * 200 / 28)
        y1 = math.ceil((out_y + 1) * 200 / 28)
        for out_x in range(28):
            x0 = math.floor(out_x * 200 / 28)
            x1 = math.ceil((out_x + 1) * 200 / 28)
            output[out_y, out_x] = native[y0:y1, x0:x1].mean(dtype=np.float64)
    return output


def _independent_readout(raw_w_per_pixel: np.ndarray, gain: float) -> tuple[np.ndarray, np.ndarray]:
    _require(np.isfinite(raw_w_per_pixel).all() and math.isfinite(gain) and gain > 0.0,
             "independent production readout input or frozen gain is invalid")
    gained = raw_w_per_pixel.astype(np.float64, copy=False) * gain
    native = np.clip(gained, 0.0, 1.0)
    return native, _numpy_adaptive_pool_28(native)


def _load_model(inputs: dict[str, Any], runtime: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = base._model("small", 2048, device, runtime)
    model.load_state_dict(inputs["checkpoint"]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()),
             "frozen C decoder must be in eval mode with gradients disabled")
    return model


def _decode_batch(
    model: torch.nn.Module, coefficients: torch.Tensor, gain: torch.Tensor,
    device: torch.device, *, keep_raw: bool,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        result = model.forward_from_ols(coefficients.to(device=device))
        raw = result["physical_w_per_pixel"].float()
        gained = raw * gain.to(device=device, dtype=torch.float32)
        native = gained.clamp(0.0, 1.0)
        view28 = resize_view(gained)
    _require(tuple(raw.shape) == (coefficients.shape[0], 1, 200, 200)
             and tuple(view28.shape) == (coefficients.shape[0], 1, 28, 28)
             and torch.isfinite(raw).all().item() and torch.isfinite(native).all().item()
             and torch.isfinite(view28).all().item(),
             "frozen production decoder returned incomplete or nonfinite readout")
    return {
        "native200": native.detach().cpu(), "view28": view28.detach().cpu(),
        "raw_w_per_pixel": raw.detach().cpu() if keep_raw else torch.empty(0),
        "capture_power_w": raw.double().sum((1, 2, 3)).cpu(),
        "raw_mean_w_per_pixel": raw.double().mean((1, 2, 3)).cpu(),
        "raw_peak_w_per_pixel": raw.double().amax((1, 2, 3)).cpu(),
        "view_mean_intensity": view28.double().mean((1, 2, 3)).cpu(),
        "proxy_clipping_area_fraction": (gained > 1.0).double().mean((1, 2, 3)).cpu(),
        "proxy_clipped_excess_fraction": (
            (gained.double() - 1.0).clamp_min(0.0).sum((1, 2, 3))
            / gained.double().sum((1, 2, 3)).clamp_min(1e-30)
        ).cpu(),
    }


def _assert_state_unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor], phase: str) -> float:
    after = base._state_snapshot(model)
    _require(before.keys() == after.keys() and all(torch.equal(before[key], after[key]) for key in before),
             f"frozen C parameters or buffers changed during {phase}")
    return base._assert_close(before, after, f"frozen_C_{phase}")


def _ensure_preflight(
    run_dir: Path, identity: dict[str, Any], model: torch.nn.Module,
    cohorts: dict[str, dict[str, Any]], checkpoint: dict[str, Any], device: torch.device,
    blocks: list[str], optical_samples_so_far: int,
) -> tuple[dict[str, Any], int]:
    preflight_path = run_dir / "preflight_replay.pt"
    receipt_path = run_dir / "preflight.json"
    readout_path = run_dir / "independent_readout_checks.pt"
    if preflight_path.is_file() and receipt_path.is_file() and readout_path.is_file():
        saved = torch.load(preflight_path, map_location="cpu", weights_only=False)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw_checks = torch.load(readout_path, map_location="cpu", weights_only=False)
        _require(saved.get("run_id") == identity["run_id"] and receipt.get("status") == "READY"
                 and receipt.get("run_id") == identity["run_id"]
                 and tuple(saved.get("view28", torch.empty(0)).shape) == (20, 1, 28, 28)
                 and tuple(saved.get("native200", torch.empty(0)).shape) == (20, 1, 200, 200)
                 and tuple(raw_checks.get("raw_w_per_pixel", torch.empty(0)).shape) == (20, 1, 200, 200),
                 "saved preflight outputs are incomplete or belong to another run")
        gain = float(checkpoint["camera_gain"])
        for index in range(20):
            native_np, view_np = _independent_readout(raw_checks["raw_w_per_pixel"][index, 0].numpy(), gain)
            _require(np.allclose(native_np, raw_checks["native200_production"][index, 0].numpy(),
                                 rtol=0.0, atol=2e-7)
                     and np.allclose(view_np, raw_checks["view28_production"][index, 0].numpy(),
                                     rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     "saved independent raw readout check failed numerical revalidation")
        for method_index, method in enumerate(METHODS):
            selected_rows = [label * SAMPLE_COUNT_PER_CLASS for label in CLASSES]
            expected = cohorts[method]["images"].index_select(0, torch.tensor(selected_rows)).float()
            start = method_index * 10
            _require(torch.allclose(saved["view28"][start:start + 10], expected,
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     f"saved {method} preflight output no longer matches its explicit cohort rows")
        return receipt, 20
    first_indices = [label * SAMPLE_COUNT_PER_CLASS for method in METHODS for label in CLASSES]
    methods = [method for method in METHODS for _label in CLASSES]
    labels = [label for _method in METHODS for label in CLASSES]
    coefficients = torch.stack([cohorts[method]["ols_coefficients"][index] for method, index in zip(
        methods, first_indices, strict=True)])
    expected = torch.stack([cohorts[method]["images"][index] for method, index in zip(
        methods, first_indices, strict=True)])
    before = base._state_snapshot(model)
    start = time.perf_counter()
    decoded_parts = [
        _decode_batch(model, coefficients[offset:offset + BATCH_SIZE],
                      checkpoint["camera_gain"], device, keep_raw=True)
        for offset in range(0, len(methods), BATCH_SIZE)
    ]
    decoded = {key: torch.cat([part[key] for part in decoded_parts]) for key in decoded_parts[0]}
    duration = time.perf_counter() - start
    errors: list[dict[str, Any]] = []
    for index in range(20):
        actual = decoded["view28"][index]
        target = expected[index]
        _require(torch.allclose(actual, target, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                 f"saved {methods[index]} class {labels[index]} production view failed preflight replay")
        raw = decoded["raw_w_per_pixel"][index, 0].numpy()
        native_np, view_np = _independent_readout(raw, float(checkpoint["camera_gain"]))
        _require(np.allclose(native_np, decoded["native200"][index, 0].numpy(), rtol=0.0, atol=2e-7)
                 and np.allclose(view_np, actual[0].numpy(), rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                 "independent NumPy gain/clamp/pool readout differs from the production path")
        errors.append({
            "method": methods[index], "label": labels[index], "cohort_row": first_indices[index],
            "view_max_abs_error": float((actual - target).abs().max()),
            "numpy_view_max_abs_error": float(np.max(np.abs(view_np - actual[0].numpy()))),
        })
    state_error = _assert_state_unchanged(model, before, "preflight")
    payload = {
        "schema": "lightgen_small_g1_g4_native_readability_preflight.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "methods": methods, "labels": labels, "cohort_rows": first_indices,
        "native200": decoded["native200"], "view28": decoded["view28"],
        "expected_view28": expected,
    }
    raw_payload = {
        "schema": "lightgen_small_g1_g4_independent_readout_checks.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "methods": methods, "labels": labels, "cohort_rows": first_indices,
        "raw_w_per_pixel": decoded["raw_w_per_pixel"],
        "native200_production": decoded["native200"], "view28_production": decoded["view28"],
        "independent_checks": errors,
    }
    atomic_torch_save(preflight_path, payload)
    atomic_torch_save(readout_path, raw_payload)
    receipt = {
        "task_id": TASK_ID, "run_id": identity["run_id"], "status": "READY",
        "source_checkpoint_step": SOURCE_STEP, "production_grid": 2048,
        "preflight_sample_count": 20, "batch_size": BATCH_SIZE,
        "saved_view_replay_rtol": REPLAY_RTOL, "saved_view_replay_atol": REPLAY_ATOL,
        "numpy_independent_readout_samples": 20,
        "optical_forward_sample_count_this_phase": 20,
        "new_encoder_forward_samples": 0, "new_evaluator_forward_samples": 0,
        "gradients": 0, "optimizer_calls": 0,
        "frozen_model_parameter_buffer_max_abs_delta": state_error,
        "elapsed_seconds": duration, "checks": errors,
    }
    atomic_write_json(receipt_path, receipt)
    optical_samples_so_far += 20
    if "preflight_20_samples" not in blocks:
        blocks.append("preflight_20_samples")
    _progress(run_dir, identity, "RUNNING", "PREFLIGHT_READY", blocks,
              optical_samples=optical_samples_so_far)
    return receipt, optical_samples_so_far


def _formal_block_path(run_dir: Path, method: str, block_index: int) -> Path:
    return run_dir / "replay_blocks" / f"{method}_{block_index:02d}.pt"


def _load_or_replay_formal(
    run_dir: Path, identity: dict[str, Any], manifest: dict[str, Any],
    cohorts: dict[str, dict[str, Any]], checkpoint: dict[str, Any], model: torch.nn.Module,
    device: torch.device, blocks: list[str], optical_samples: int,
) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    block_root = run_dir / "replay_blocks"
    block_root.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict[str, torch.Tensor]] = {}
    metric_rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_items = [item for item in manifest["items"] if item["method"] == method]
        method_native: list[torch.Tensor] = []
        method_views: list[torch.Tensor] = []
        method_rows: list[dict[str, Any]] = []
        for block_index, start in enumerate(range(0, len(method_items), BATCH_SIZE)):
            items = method_items[start:start + BATCH_SIZE]
            path = _formal_block_path(run_dir, method, block_index)
            expected_rows = [int(item["cohort_row"]) for item in items]
            if path.is_file():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == identity["run_id"]
                         and saved.get("method") == method and saved.get("cohort_rows") == expected_rows
                         and tuple(saved.get("native200", torch.empty(0)).shape) == (BATCH_SIZE, 1, 200, 200)
                         and tuple(saved.get("view28", torch.empty(0)).shape) == (BATCH_SIZE, 1, 28, 28)
                         and all(tuple(saved.get(key, torch.empty(0)).shape) == (BATCH_SIZE,) for key in (
                             "capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                             "view_mean_intensity", "proxy_clipping_area_fraction",
                             "proxy_clipped_excess_fraction",
                         ))
                         and torch.isfinite(saved["native200"]).all().item()
                         and torch.isfinite(saved["view28"]).all().item(),
                         f"saved {method} replay block {block_index} is incomplete or misidentified")
                expected = cohorts[method]["images"].index_select(0, torch.tensor(expected_rows)).float()
                _require(torch.allclose(saved["view28"], expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                         and float(saved["native200"].min()) >= 0.0
                         and float(saved["native200"].max()) <= 1.0,
                         f"saved {method} replay block no longer matches the registered views")
                optical_samples += int(saved.get("optical_forward_sample_count", 0))
            else:
                indices = torch.tensor(expected_rows, dtype=torch.long)
                coefficients = cohorts[method]["ols_coefficients"].index_select(0, indices)
                expected = cohorts[method]["images"].index_select(0, indices).float()
                before = base._state_snapshot(model)
                started = time.perf_counter()
                decoded = _decode_batch(model, coefficients, checkpoint["camera_gain"], device, keep_raw=False)
                duration = time.perf_counter() - started
                _require(torch.allclose(decoded["view28"], expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                         f"formal {method} selected 28x28 views failed the saved replay tolerance")
                state_error = _assert_state_unchanged(model, before, f"formal_{method}_{block_index}")
                saved = {
                    "task_id": TASK_ID, "run_id": identity["run_id"], "method": method,
                    "block_index": block_index, "cohort_rows": expected_rows,
                    "native200": decoded["native200"], "view28": decoded["view28"],
                    "capture_power_w": decoded["capture_power_w"],
                    "raw_mean_w_per_pixel": decoded["raw_mean_w_per_pixel"],
                    "raw_peak_w_per_pixel": decoded["raw_peak_w_per_pixel"],
                    "view_mean_intensity": decoded["view_mean_intensity"],
                    "proxy_clipping_area_fraction": decoded["proxy_clipping_area_fraction"],
                    "proxy_clipped_excess_fraction": decoded["proxy_clipped_excess_fraction"],
                    "saved_view_max_abs_error": float((decoded["view28"] - expected).abs().max()),
                    "elapsed_seconds": duration,
                    "frozen_model_state_max_abs_delta": state_error,
                    "optical_forward_sample_count": BATCH_SIZE,
                }
                atomic_torch_save(path, saved)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                _require(saved.get("cohort_rows") == expected_rows
                         and tuple(saved["native200"].shape) == (BATCH_SIZE, 1, 200, 200),
                         f"formal {method} replay block {block_index} failed read-back")
                optical_samples += BATCH_SIZE
                block_name = f"{method}_batch_{block_index:02d}"
            block_name = f"{method}_batch_{block_index:02d}"
            if block_name not in blocks:
                blocks.append(block_name)
            _progress(run_dir, identity, "RUNNING", "FORMAL_REPLAY", blocks,
                      optical_samples=optical_samples)
            method_native.append(saved["native200"])
            method_views.append(saved["view28"])
            cohort = cohorts[method]
            for local_index, item in enumerate(items):
                row = int(item["cohort_row"])
                raw_fields = {
                    "capture_power_w": float(saved["capture_power_w"][local_index]),
                    "raw_mean_w_per_pixel": float(saved["raw_mean_w_per_pixel"][local_index]),
                    "raw_peak_w_per_pixel": float(saved["raw_peak_w_per_pixel"][local_index]),
                    "view_mean_intensity": float(saved["view_mean_intensity"][local_index]),
                    "proxy_clipping_area_fraction": float(
                        saved["proxy_clipping_area_fraction"][local_index]),
                    "proxy_clipped_excess_fraction": float(
                        saved["proxy_clipped_excess_fraction"][local_index]),
                    "capture_power_source_difference": float(
                        saved["capture_power_w"][local_index] - cohort["capture_power_w"][row]),
                }
                ed_logits = cohort["ed_logits"][row].float()
                ec_logits = cohort["ec_logits"][row].float()
                ed_pred = int(ed_logits.argmax())
                ec_pred = int(ec_logits.argmax())
                ed_requested_margin = float(cohort["ed_requested_margin"][row])
                ec_requested_margin = float(cohort["ec_requested_margin"][row])
                method_rows.append({
                    **item, "proposal_id": int(item["proposal_id"]),
                    "saved_view_max_abs_error": float(saved["saved_view_max_abs_error"]),
                    "replayed_native_mean": float(saved["native200"][local_index].mean()),
                    "replayed_view_mean": float(saved["view28"][local_index].mean()),
                    "E_D_prediction_saved": ed_pred, "E_D_requested_margin_saved": ed_requested_margin,
                    "E_C_prediction_saved": ec_pred, "E_C_requested_margin_saved": ec_requested_margin,
                    **raw_fields,
                })
        selected[method] = {
            "cohort_rows": torch.tensor([int(item["cohort_row"]) for item in method_items]),
            "native200": torch.cat(method_native), "view28": torch.cat(method_views),
        }
        _require(tuple(selected[method]["native200"].shape) == (100, 1, 200, 200)
                 and tuple(selected[method]["view28"].shape) == (100, 1, 28, 28),
                 f"formal {method} selected replay population is incomplete")
        metric_rows.extend(method_rows)
    _require(len(metric_rows) == REVIEW_GENERATED, "formal selected metric table must contain 200 generated samples")
    return selected, optical_samples, metric_rows


def _shuffle_cards(cards: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    for attempt in range(1, 1_000_001):
        permutation = rng.permutation(len(cards)).tolist()
        ordered = [cards[index] for index in permutation]
        positions: dict[str, list[int]] = {}
        for position, card in enumerate(ordered):
            group = card.get("duplicate_group")
            if group is not None:
                positions.setdefault(str(group), []).append(position)
        if len(positions) == REVIEW_DUPLICATES and all(
            len(pair) == 2 and abs(pair[0] - pair[1]) >= MIN_DUPLICATE_DISTANCE
            for pair in positions.values()
        ):
            return ordered, attempt
    raise RuntimeError("fixed shuffle seed did not produce the registered duplicate separation")


def _render_review_page(review_id: str, exposed_cards: list[dict[str, str]]) -> str:
    cards_json = json.dumps(exposed_cards, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>手写数字图像评阅</title>
<style>body{{font:18px system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem;color:#18212b}}main{{text-align:center}}img{{width:400px;height:400px;object-fit:contain;image-rendering:pixelated;border:1px solid #999;background:#fff}}.choices{{display:grid;grid-template-columns:repeat(6,1fr);gap:.4rem;margin:1rem 0}}label{{padding:.5rem;border:1px solid #ccd1d1;border-radius:.4rem}}button{{font-size:1rem;padding:.6rem 1rem;margin:.3rem}}#save_status{{color:#405060}}</style>
<h1>手写数字图像评阅</h1><p>请依据图像本身选择数字；不能辨认时选择“不可辨认”。所有图像均按固定[0,1]灰度显示，不做逐图增强。每20张为一段，段末可休息；评分和当前位置会自动保存。</p>
<main><p id='progress'></p><img id='image' alt='当前评阅图'><p id='segment_note'></p><div id='choices' class='choices'></div>
<button id='prev'>上一张</button><button id='next'>下一张</button><button id='save'>导出全部评分</button><p id='save_status'></p></main>
<script>
const TASK_ID={json.dumps(TASK_ID)}, REVIEW_ID={json.dumps(review_id)}, CARDS={cards_json};
const STORE='lightgen-native-review-'+REVIEW_ID;let state={{}};let position=0;
try{{const saved=JSON.parse(localStorage.getItem(STORE)||'{{}}');state=saved.ratings||{{}};position=Math.min(Math.max(0,Number(saved.position)||0),CARDS.length-1);}}catch{{state={{}};position=0;}}
function persist(){{localStorage.setItem(STORE,JSON.stringify({{ratings:state,position}}));const segment=Math.floor(position/20)+1;const within=position%20+1;document.getElementById('progress').textContent=`第 ${{segment}} / 12 段 · 本段 ${{within}} / 20 张 · 总计 ${{position+1}} / 240 张 · 已完成 ${{Object.keys(state).length}} 张`;}}
function render(){{const c=CARDS[position];document.getElementById('image').src=c.image;document.getElementById('choices').innerHTML='';document.getElementById('save_status').textContent='';document.getElementById('segment_note').textContent=(position%20===19)?'本段结束，可休息；当前位置与评分已自动保存。':'';
const opts=[...Array(10)].map((_,i)=>[String(i),String(i)]).concat([['unreadable','不可辨认']]);
for(const [value,text] of opts){{const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='rating';radio.value=value;radio.checked=state[c.card_id]===value;radio.onchange=()=>{{state[c.card_id]=value;persist();render();}};label.append(radio,document.createTextNode(' '+text));document.getElementById('choices').append(label);}}
persist();}}
document.getElementById('prev').onclick=()=>{{position=Math.max(0,position-1);persist();render();}};
document.getElementById('next').onclick=()=>{{position=Math.min(CARDS.length-1,position+1);persist();render();}};
document.getElementById('save').onclick=()=>{{if(Object.keys(state).length!==CARDS.length){{document.getElementById('save_status').textContent='请先完成全部240张评分。';return;}}
const cards=CARDS.map(c=>{{const v=state[c.card_id];return {{card_id:c.card_id,rating:v==='unreadable'?'unreadable':Number(v)}};}});
const payload={{schema:{json.dumps(RATINGS_SCHEMA)},task_id:TASK_ID,review_id:REVIEW_ID,cards}};
const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='lightgen_readability_ratings_'+REVIEW_ID+'.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('save_status').textContent='评分文件已导出。';}};
render();</script></html>"""


def _validate_blind_page(run_dir: Path, key: dict[str, Any]) -> None:
    page_path = run_dir / "blind_review.html"
    _require(page_path.is_file(), "blind review page is missing", FileNotFoundError)
    page_text = page_path.read_text(encoding="utf-8")
    declarations = re.search(
        r'const TASK_ID=("(?:\\.|[^"\\])*")\s*,\s*REVIEW_ID=("(?:\\.|[^"\\])*")'
        r'\s*,\s*CARDS=', page_text,
    )
    _require(declarations is not None and 'width:400px;height:400px' in page_text
             and "image-rendering:pixelated" in page_text
             and "Math.floor(position/20)+1" in page_text
             and "localStorage.setItem" in page_text,
             "blind page is missing its fixed display, 20-card segments, or autosave behavior")
    page_task_id = json.loads(declarations.group(1))
    page_review_id = json.loads(declarations.group(2))
    page_cards, _ = json.JSONDecoder().raw_decode(page_text, declarations.end())
    expected = [{"card_id": card["card_id"], "image": card["image"]} for card in key["cards"]]
    _require(page_task_id == TASK_ID and page_review_id == key.get("review_id")
             and page_cards == expected,
             "blind page task/review identity or card mapping differs from the hidden key")
    for forbidden in ("requested_label", "proposal_id", "target_label", "method"):
        _require(forbidden not in page_text, f"blind page leaks hidden review field {forbidden}")


def _write_blind_review(
    run_dir: Path, identity: dict[str, Any], manifest: dict[str, Any],
    selected: dict[str, dict[str, torch.Tensor]], cohorts: dict[str, dict[str, Any]],
    examples: dict[str, Any], progress_record: dict[str, Any] | None,
    blocks: list[str], optical_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    key_path = run_dir / "blind_review_key.json"
    cards_dir = run_dir / "review_cards"
    selected_by_identity: dict[tuple[str, int, int], dict[str, Any]] = {}
    for method in METHODS:
        items = [item for item in manifest["items"] if item["method"] == method]
        rows = selected[method]["cohort_rows"].tolist()
        native = selected[method]["native200"]
        for index, item in enumerate(items):
            cohort = cohorts[method]
            selected_by_identity[(method, int(item["requested_label"]), int(item["proposal_id"]))] = {
                **item, "image": native[index],
                "ed_prediction": int(cohort["ed_predictions"][int(item["cohort_row"])]),
                "ec_prediction": int(cohort["ec_predictions"][int(item["cohort_row"])]),
                "cohort_row_check": int(rows[index]),
            }
    rows = examples["rows"].long().tolist()
    labels = examples["labels"].long().tolist()
    targets = examples["targets28"].float()
    upsampled = F.interpolate(targets, size=(200, 200), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    anchor_by_row: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        anchor_by_row[int(row)] = {"row": int(row), "target_label": int(labels[index]),
                                   "image": upsampled[index].cpu()}

    if key_path.is_file():
        key = json.loads(key_path.read_text(encoding="utf-8"))
        _require(key.get("schema") == REVIEW_SCHEMA and key.get("task_id") == TASK_ID
                 and key.get("run_id") == identity["run_id"] and len(key.get("cards", [])) == REVIEW_TOTAL,
                 "saved blind-review key is incomplete or belongs to another run")
        review_id = key["review_id"]
        if progress_record and progress_record.get("review_id") is not None:
            _require(progress_record["review_id"] == review_id,
                     "saved progress review ID differs from the hidden review key")
    else:
        review_id = (progress_record or {}).get("review_id") or uuid.uuid4().hex
        cards: list[dict[str, Any]] = []
        for method in METHODS:
            for label in CLASSES:
                group = [item for item in manifest["items"]
                         if item["method"] == method and item["requested_label"] == label]
                _require(len(group) == SELECT_PER_CLASS, "fixed selected cards are incomplete")
                for item in group:
                    source = selected_by_identity[(method, label, int(item["proposal_id"]))]
                    cards.append({
                        "kind": "generated", "method": method, "requested_label": label,
                        "stream": 0, "cohort_row": int(item["cohort_row"]),
                        "proposal_id": int(item["proposal_id"]), "component_id": int(item["component_id"]),
                        "image": source["image"], "duplicate_group": None,
                    })
        _require(len(cards) == REVIEW_GENERATED, "blind review must contain 100 unique images per method")
        for method in METHODS:
            for label in CLASSES:
                original = next(card for card in cards if card["kind"] == "generated"
                                and card["method"] == method and card["requested_label"] == label)
                group_name = f"repeat_{method}_{label}"
                original["duplicate_group"] = group_name
                cards.append({**original, "kind": "duplicate"})
        _require(len(anchor_by_row) == REVIEW_ANCHORS and len(set(rows)) == REVIEW_ANCHORS and [
            sum(item["target_label"] == label for item in anchor_by_row.values()) for label in CLASSES
        ] == [2] * 10, "fixed DEV target anchors are not two per class")
        for anchor in anchor_by_row.values():
            cards.append({"kind": "real_anchor", **anchor, "method": None, "requested_label": None,
                          "stream": None, "cohort_row": None, "proposal_id": None,
                          "component_id": None, "duplicate_group": None})
        _require(len(cards) == REVIEW_TOTAL, "blind review population must contain exactly 240 cards")
        ordered, shuffle_attempt = _shuffle_cards(cards, SHUFFLE_SEED)
        hidden_cards: list[dict[str, Any]] = []
        for position, card in enumerate(ordered, start=1):
            relative = f"review_cards/card_{position:04d}.png"
            hidden_cards.append({key: value for key, value in card.items() if key != "image"} |
                                {"card_id": f"R{position:04d}", "image": relative})
        key = {
            "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "run_id": identity["run_id"],
            "review_id": review_id, "registered_card_count": REVIEW_TOTAL,
            "generated_unique_card_count": REVIEW_GENERATED, "hidden_duplicate_count": REVIEW_DUPLICATES,
            "real_anchor_count": REVIEW_ANCHORS, "shuffle_seed": SHUFFLE_SEED,
            "shuffle_attempt": shuffle_attempt, "minimum_duplicate_card_distance": MIN_DUPLICATE_DISTANCE,
            "reviewer_count": 1,
            "display": {"source_image_pixels": 200, "css_display_pixels": 400,
                        "range": "frozen global gain then clamp [0,1]; fixed grayscale; no per-image normalization"},
            "anchor_source": "saved MNIST DEV targets bilinear-upsampled to 200x200, align_corners=False; not optical outputs",
            "cards": hidden_cards,
        }
        atomic_write_json(key_path, key)
        _progress(run_dir, identity, "RUNNING", "WRITING_REVIEW_CARDS", blocks,
                  review_id=review_id, optical_samples=optical_samples)

    _require(key.get("review_id") == review_id and len(key.get("cards", [])) == REVIEW_TOTAL,
             "review identity or hidden key population is incomplete")
    card_images: list[torch.Tensor] = []
    for card in key["cards"]:
        if card["kind"] in ("generated", "duplicate"):
            item_key = (str(card["method"]), int(card["requested_label"]), int(card["proposal_id"]))
            _require(item_key in selected_by_identity, "saved review key points to an unselected generated sample")
            card_images.append(selected_by_identity[item_key]["image"])
        else:
            row = int(card["row"])
            _require(row in anchor_by_row and int(card["target_label"]) == anchor_by_row[row]["target_label"],
                     "saved review key points to an unknown real-target row")
            card_images.append(anchor_by_row[row]["image"])
    staging = run_dir / f"review_cards_staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    for card, image in zip(key["cards"], card_images, strict=True):
        target = (run_dir / card["image"]).resolve()
        _require(target.parent == cards_dir.resolve(), "review card path escapes the fixed image directory")
        pixels = np.rint(np.clip(image.detach().cpu().numpy().squeeze(), 0.0, 1.0) * 255.0).astype(np.uint8)
        _require(pixels.shape == (200, 200), "blind card display tensor is not a single native 200x200 view")
        Image.fromarray(pixels).save(
            staging / target.name, format="PNG", optimize=True,
        )
    _require(len(list(staging.glob("*.png"))) == REVIEW_TOTAL,
             "blind image export does not contain exactly 240 PNGs")
    for path in staging.glob("*.png"):
        with Image.open(path) as image:
            _require(image.mode == "L" and image.size == (200, 200),
                     "blind image quantization or native display dimensions changed")
    if cards_dir.exists():
        _require(cards_dir.resolve().parent == run_dir.resolve(), "existing card directory escaped the task output")
        shutil.rmtree(cards_dir)
    os.replace(staging, cards_dir)
    exposed = [{"card_id": card["card_id"], "image": card["image"]} for card in key["cards"]]
    page = _render_review_page(review_id, exposed)
    temporary_page = run_dir / f"blind_review.html.tmp.{uuid.uuid4().hex}"
    temporary_page.write_text(page, encoding="utf-8")
    os.replace(temporary_page, run_dir / "blind_review.html")
    atomic_write_json(run_dir / "blind_review_public_manifest.json", {
        "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "run_id": identity["run_id"],
        "review_id": review_id, "card_count": REVIEW_TOTAL,
        "rating_values": list(range(10)) + ["unreadable"],
        "display_source_pixels": 200, "display_css_pixels": 400,
        "segment_size": 20, "segment_count": 12,
    })
    _validate_blind_page(run_dir, key)
    for group_name in {card.get("duplicate_group") for card in key["cards"] if card.get("duplicate_group")}: 
        paths = [run_dir / card["image"] for card in key["cards"]
                 if card.get("duplicate_group") == group_name]
        _require(len(paths) == 2 and paths[0].read_bytes() == paths[1].read_bytes(),
                 "hidden repeat PNGs are not identical")
    return key, {"review_id": review_id, "card_count": REVIEW_TOTAL,
                 "generated_unique_count": REVIEW_GENERATED, "hidden_duplicate_count": REVIEW_DUPLICATES,
                 "real_anchor_count": REVIEW_ANCHORS, "shuffle_attempt": key.get("shuffle_attempt"),
                 "blind_page": "blind_review.html", "hidden_key": "blind_review_key.json"}


def _rating_value(value: Any, card_id: str) -> int | str:
    _require(value == "unreadable" or (type(value) is int and 0 <= value <= 9),
             f"rating for {card_id} must be a digit 0-9 or unreadable", ValueError)
    return value


def _validate_ratings(run_dir: Path, ratings_path: Path) -> dict[str, Any]:
    _require(not (run_dir / "final_report.json").exists(),
             "human review is already finalized; archived ratings cannot be replaced", FileExistsError)
    machine_path = run_dir / "machine_report.json"
    key_path = run_dir / "blind_review_key.json"
    manifest_path = run_dir / "blind_review_public_manifest.json"
    for path in (machine_path, key_path, manifest_path, ratings_path):
        _require(path.is_file(), f"human-rating input is missing: {path}", FileNotFoundError)
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    public = json.loads(manifest_path.read_text(encoding="utf-8"))
    submission = json.loads(ratings_path.read_text(encoding="utf-8"))
    run_identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    _require(machine.get("task_id") == TASK_ID and machine.get("status") == "AWAITING_HUMAN_RATINGS"
             and machine.get("measurement_complete") is True
             and machine.get("source", {}).get("source_status_preserved") == "VALID_FAILED"
             and machine.get("source", {}).get("global_step") == SOURCE_STEP
             and machine.get("source", {}).get("run_id") == run_identity.get("source_run_id")
             and run_identity.get("source_step") == SOURCE_STEP
             and run_identity.get("source_quality_status_preserved") == "VALID_FAILED"
             and machine.get("quality_gate", {}).get("historical_system_status") == "VALID_FAILED"
             and machine.get("quality_gate", {}).get("status_changed") is False
             and run_identity.get("task_id") == TASK_ID
             and run_identity.get("run_id") == machine.get("run_id"),
             "machine measurement or run identity is incomplete or already finalized")
    review_id = key.get("review_id")
    _require(key.get("schema") == REVIEW_SCHEMA and key.get("task_id") == TASK_ID
             and key.get("run_id") == machine.get("run_id")
             and key.get("registered_card_count") == REVIEW_TOTAL
             and public.get("schema") == REVIEW_SCHEMA and public.get("task_id") == TASK_ID
             and public.get("run_id") == machine.get("run_id")
             and public.get("review_id") == review_id and public.get("card_count") == REVIEW_TOTAL,
             "review task, review ID, or card mapping identity changed")
    _require(submission.get("schema") == RATINGS_SCHEMA and submission.get("task_id") == TASK_ID
             and submission.get("review_id") == review_id and isinstance(submission.get("cards"), list),
             "submitted ratings belong to another task or review", ValueError)
    cards = key.get("cards")
    expected_ids = [f"R{index:04d}" for index in range(1, REVIEW_TOTAL + 1)]
    _require(isinstance(cards, list) and [card.get("card_id") for card in cards] == expected_ids,
             "hidden review key has missing, repeated, or unordered cards")
    rating_cards = submission["cards"]
    _require(len(rating_cards) == REVIEW_TOTAL
             and all(isinstance(card, dict) and set(card) == {"card_id", "rating"}
                     and isinstance(card.get("card_id"), str) for card in rating_cards),
             "ratings must contain one well-formed record for every card", ValueError)
    rating_ids = [card["card_id"] for card in rating_cards]
    _require(len(set(rating_ids)) == REVIEW_TOTAL and set(rating_ids) == set(expected_ids),
             "ratings contain duplicate, missing, or foreign card IDs", ValueError)
    ratings = {card["card_id"]: _rating_value(card["rating"], card["card_id"]) for card in rating_cards}
    card_kinds = {kind: [card for card in cards if card.get("kind") == kind]
                  for kind in ("generated", "duplicate", "real_anchor")}
    _require([len(card_kinds[kind]) for kind in ("generated", "duplicate", "real_anchor")]
             == [REVIEW_GENERATED, REVIEW_DUPLICATES, REVIEW_ANCHORS],
             "review source counts differ from the fixed design")
    identities = [(card.get("method"), card.get("requested_label"), card.get("stream"),
                   card.get("proposal_id")) for card in card_kinds["generated"]]
    _require(len(set(identities)) == REVIEW_GENERATED
             and all(method in METHODS and type(label) is int and label in CLASSES and stream == 0
                     for method, label, stream, _proposal in identities),
             "unique generated sample identity is duplicated or malformed")
    for method in METHODS:
        for label in CLASSES:
            _require(sum(card.get("method") == method and card.get("requested_label") == label
                         for card in card_kinds["generated"]) == SELECT_PER_CLASS,
                     f"review generated population for {method} class {label} is incomplete")
    group_map: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        if card.get("duplicate_group") is not None:
            group_map.setdefault(str(card["duplicate_group"]), []).append(card)
    _require(len(group_map) == REVIEW_DUPLICATES and all(len(group) == 2 for group in group_map.values()),
             "hidden repeat groups are incomplete")
    duplicate_matches = 0
    duplicate_pairs: list[dict[str, Any]] = []
    for group_name, group in sorted(group_map.items()):
        originals = [card for card in group if card.get("kind") == "generated"]
        copies = [card for card in group if card.get("kind") == "duplicate"]
        _require(len(originals) == len(copies) == 1,
                 f"hidden repeat group {group_name} does not contain one original and one copy")
        original, duplicate = originals[0], copies[0]
        fields = ("method", "requested_label", "stream", "cohort_row", "proposal_id", "component_id")
        _require(all(original.get(field) == duplicate.get(field) for field in fields),
                 f"hidden repeat {group_name} maps to another image")
        same = ratings[original["card_id"]] == ratings[duplicate["card_id"]]
        duplicate_matches += int(same)
        duplicate_pairs.append({"group": group_name, "original_card_id": original["card_id"],
                                "repeat_card_id": duplicate["card_id"], "consistent": same})
        first = (run_dir / original["image"]).read_bytes()
        second = (run_dir / duplicate["image"]).read_bytes()
        _require(first == second, f"hidden repeat {group_name} images are not pixel-identical")
    anchor_counts = [sum(card.get("target_label") == label for card in card_kinds["real_anchor"])
                     for label in CLASSES]
    _require(anchor_counts == [2] * 10, "real-target review anchors are not two per class")
    anchor_correct = sum(int(ratings[card["card_id"]] == card.get("target_label"))
                         for card in card_kinds["real_anchor"])
    ordered_positions: dict[str, list[int]] = {}
    for position, card in enumerate(cards):
        if card.get("duplicate_group") is not None:
            ordered_positions.setdefault(str(card["duplicate_group"]), []).append(position)
    _require(all(len(pair) == 2 and abs(pair[0] - pair[1]) >= MIN_DUPLICATE_DISTANCE
                 for pair in ordered_positions.values()),
             "hidden repeat card spacing is below the fixed minimum")
    _validate_blind_page(run_dir, key)
    cards_root = (run_dir / "review_cards").resolve()
    for card in cards:
        relative = card.get("image")
        _require(isinstance(relative, str) and relative.startswith("review_cards/"),
                 f"card image path for {card['card_id']} is malformed")
        image_path = (run_dir / relative).resolve()
        _require(image_path.is_relative_to(cards_root) and image_path.is_file(),
                 f"review card image {card['card_id']} is missing or escapes the card directory")
        with Image.open(image_path) as image:
            _require(image.mode == "L" and image.size == (200, 200),
                     f"review image {card['card_id']} changed display format")
    with (run_dir / "selected_per_image.csv").open(encoding="utf-8", newline="") as stream:
        selected_rows = list(csv.DictReader(stream))
    row_ids = [(row["method"], int(row["requested_label"]), int(row["stream"]),
                int(row["proposal_id"])) for row in selected_rows]
    expected_rows = {(card["method"], card["requested_label"], card["stream"], card["proposal_id"])
                     for card in card_kinds["generated"]}
    _require(len(row_ids) == REVIEW_GENERATED and len(set(row_ids)) == REVIEW_GENERATED
             and set(row_ids) == expected_rows,
             "saved classifier rows are missing, duplicated, or mismatched")
    _require(all(int(row[field]) in CLASSES for row in selected_rows
                 for field in ("E_D_prediction_saved", "E_C_prediction_saved")),
             "saved classifier predictions are outside the digit range")
    return {"machine": machine, "key": key, "ratings": submission, "cards": cards,
            "selected_rows": selected_rows,
            "rating_by_id": ratings, "duplicate_consistency_count": duplicate_matches,
            "duplicate_pairs": duplicate_pairs, "anchor_correct_count": anchor_correct}


def _method_metrics(cards: list[dict[str, Any]], ratings: dict[str, int | str], method: str) -> dict[str, Any]:
    per_class = []
    success_by_class: list[np.ndarray] = []
    unreadable_by_class: list[np.ndarray] = []
    other_by_class: list[np.ndarray] = []
    for label in CLASSES:
        selected = [card for card in cards if card.get("kind") == "generated"
                    and card.get("method") == method and card.get("requested_label") == label]
        _require(len(selected) == SELECT_PER_CLASS, f"{method} class {label} denominator is incomplete")
        values = [ratings[card["card_id"]] for card in selected]
        success = np.asarray([float(value == label) for value in values], dtype=np.float64)
        unreadable = np.asarray([float(value == "unreadable") for value in values], dtype=np.float64)
        other = np.asarray([float(type(value) is int and value != label) for value in values], dtype=np.float64)
        per_class.append({"requested_label": label, "count": SELECT_PER_CLASS,
                          "requested_digit_matches": int(success.sum()),
                          "unreadable_count": int(unreadable.sum()), "other_digit_count": int(other.sum()),
                          "requested_digit_rate": float(success.mean()),
                          "unreadable_rate": float(unreadable.mean()), "other_digit_rate": float(other.mean())})
        success_by_class.append(success)
        unreadable_by_class.append(unreadable)
        other_by_class.append(other)
    return {
        "method": method, "unique_generated_denominator": 100,
        "equal_class_requested_digit_rate": float(np.mean([item.mean() for item in success_by_class])),
        "equal_class_unreadable_rate": float(np.mean([item.mean() for item in unreadable_by_class])),
        "equal_class_other_digit_rate": float(np.mean([item.mean() for item in other_by_class])),
        "per_class": per_class, "_success_by_class": success_by_class,
    }


def _bootstrap_difference(g1: dict[str, Any], g4: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        means: dict[str, list[float]] = {"G1": [], "G4": []}
        for method, metrics in (("G1", g1), ("G4", g4)):
            for values in metrics["_success_by_class"]:
                means[method].append(float(values[rng.integers(0, values.size, size=values.size)].mean()))
        samples[index] = float(np.mean(means["G4"]) - np.mean(means["G1"]))
    observed = g4["equal_class_requested_digit_rate"] - g1["equal_class_requested_digit_rate"]
    return {
        "estimand": "G4 minus G1 equal-class rate of the single reviewer matching the requested digit",
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
        "resampling": "independent within each method and requested class; 10 unique images per class per method",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty per-image table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_pending_report(run_dir: Path, report: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "machine_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    page = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>固定C端点盲评</title><style>body{font:17px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#18212b}a{font-size:1.15rem}</style>
<h1>固定C端点生成图盲评</h1><p>机器重放和卡片校验已完成。请按图像本身评分；页面不会显示请求类别、方法或分类器结果。</p>
<p><a href='blind_review.html'>打开240张盲评页面</a></p><p>每20张为一段，评分会在浏览器本地自动保存。完成后导出JSON，并把文件交回本任务完成统计。</p></html>"""
    (run_dir / "report.html").write_text(page, encoding="utf-8")


def _finalize_ratings(run_dir: Path, ratings_path: Path, config: dict[str, Any]) -> None:
    started = time.perf_counter()
    validated = _validate_ratings(run_dir, ratings_path)
    cards = validated["cards"]
    ratings = validated["rating_by_id"]
    g1 = _method_metrics(cards, ratings, "G1")
    g4 = _method_metrics(cards, ratings, "G4")
    effect = _bootstrap_difference(g1, g4)
    human_cfg = config["human_review"]
    duplicate_ok = validated["duplicate_consistency_count"] >= human_cfg["minimum_duplicate_consistency"]
    anchor_ok = validated["anchor_correct_count"] >= human_cfg["minimum_anchor_correct"]
    qc_passed = duplicate_ok and anchor_ok
    low, high = effect["ci_95"]
    if not qc_passed:
        conclusion = "REVIEW_INCONCLUSIVE"
        status = "REVIEW_INCONCLUSIVE"
    else:
        conclusion = ("HUMAN_READABILITY_IMPROVEMENT_SUPPORTED" if low > 0.0 else
                      "HUMAN_READABILITY_WORSE" if high < 0.0 else
                      "HUMAN_READABILITY_INDEPENDENT_EFFECT_UNCLEAR")
        status = "COMPLETE_DIAGNOSTIC"
    card_rows: list[dict[str, Any]] = []
    for card in cards:
        value = ratings[card["card_id"]]
        card_rows.append({
            "card_id": card["card_id"], "kind": card["kind"], "method": card.get("method"),
            "requested_label": card.get("requested_label"), "proposal_id": card.get("proposal_id"),
            "stream": card.get("stream"), "human_rating": value,
            "matches_requested": (value == card.get("requested_label")
                                  if card["kind"] in ("generated", "duplicate") and value != "unreadable"
                                  else False if card["kind"] in ("generated", "duplicate") else None),
            "readable": value != "unreadable", "target_label": card.get("target_label"),
            "mnist_row": card.get("row"), "duplicate_group": card.get("duplicate_group"),
        })
    _write_csv(run_dir / "human_ratings_unblinded.csv", card_rows)
    raw_destination = run_dir / "human_ratings_submitted.json"
    raw_temporary = run_dir / f"human_ratings_submitted.json.tmp.{uuid.uuid4().hex}"
    raw_temporary.write_bytes(ratings_path.read_bytes())
    os.replace(raw_temporary, raw_destination)

    auxiliary: dict[str, Any] = {method: {"readable_digit_matches_saved_E_D": 0,
                                          "readable_digit_matches_saved_E_C": 0,
                                          "readable_generated_cards": 0,
                                          "unreadable_generated_cards": 0}
                                 for method in METHODS}
    rows = {(row["method"], int(row["requested_label"]), int(row["proposal_id"])): row
            for row in validated["selected_rows"]}
    for card in cards:
        if card["kind"] != "generated":
            continue
        value = ratings[card["card_id"]]
        record = rows[(card["method"], int(card["requested_label"]), int(card["proposal_id"]))]
        if value == "unreadable":
            auxiliary[card["method"]]["unreadable_generated_cards"] += 1
        else:
            auxiliary[card["method"]]["readable_generated_cards"] += 1
            auxiliary[card["method"]]["readable_digit_matches_saved_E_D"] += int(
                int(record["E_D_prediction_saved"]) == value)
            auxiliary[card["method"]]["readable_digit_matches_saved_E_C"] += int(
                int(record["E_C_prediction_saved"]) == value)

    human_review = {
        "status": status, "review_id": validated["key"]["review_id"], "reviewer_count": 1,
        "rating_count": REVIEW_TOTAL, "primary_unique_generated_count": REVIEW_GENERATED,
        "duplicate_count": REVIEW_DUPLICATES, "real_anchor_count": REVIEW_ANCHORS,
        "quality_checks": {
            "duplicate_consistency": {"count": validated["duplicate_consistency_count"],
                                       "total": REVIEW_DUPLICATES,
                                       "minimum": human_cfg["minimum_duplicate_consistency"],
                                       "passed": duplicate_ok,
                                       "pairs": validated["duplicate_pairs"]},
            "real_anchor_label_match": {"count": validated["anchor_correct_count"],
                                        "total": REVIEW_ANCHORS,
                                        "minimum": human_cfg["minimum_anchor_correct"],
                                        "passed": anchor_ok},
        },
        "primary_endpoint": "equal-class requested-digit match rate among the 200 unique generated images; unreadable counts as no match",
        "G1": {key: value for key, value in g1.items() if not key.startswith("_")},
        "G4": {key: value for key, value in g4.items() if not key.startswith("_")},
        "G4_minus_G1_primary_effect": effect,
        "saved_classifier_auxiliary_readings": auxiliary,
        "interpretation": conclusion,
        "limitations": [
            "fixed C endpoint, fixed saved primary stream, and one reviewer only",
            "ten generated images per class per method; per-class results are descriptive",
            "E_D/E_C are saved auxiliary predictions, not human truth or selection criteria",
            "this review does not establish physical CMOS readability or upgrade the historical system gate",
        ],
        "elapsed_seconds_finalization": time.perf_counter() - started,
    }
    report = validated["machine"]
    report["status"] = status
    report["final_status"] = status
    report["human_review_status"] = status
    report["decision"]["result"] = conclusion
    report["decision"]["human_rating_received"] = True
    report["decision"]["human_readability_conclusion"] = conclusion
    report["human_review"] = human_review
    report["artifacts"].update({
        "human_ratings": "human_ratings_submitted.json",
        "human_ratings_unblinded": "human_ratings_unblinded.csv",
        "results_html": "results.html",
    })
    results_html = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>G1/G4盲评揭盲结果</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#18212b}"
        "table{border-collapse:collapse}td,th{padding:.5rem .7rem;border:1px solid #ccd1d1;text-align:left}</style>",
        f"<h1>固定C端点盲评揭盲</h1><p>状态：{html.escape(status)}；解释：{html.escape(conclusion)}</p>",
        f"<p>重复一致：{validated['duplicate_consistency_count']}/20；真实目标参照正确：{validated['anchor_correct_count']}/20。</p>",
        "<table><tr><th>方法</th><th>请求类别匹配（等类宏平均）</th><th>不可辨认率</th><th>其他数字率</th></tr>",
        f"<tr><td>G1</td><td>{g1['equal_class_requested_digit_rate']:.4f}</td><td>{g1['equal_class_unreadable_rate']:.4f}</td><td>{g1['equal_class_other_digit_rate']:.4f}</td></tr>",
        f"<tr><td>G4</td><td>{g4['equal_class_requested_digit_rate']:.4f}</td><td>{g4['equal_class_unreadable_rate']:.4f}</td><td>{g4['equal_class_other_digit_rate']:.4f}</td></tr></table>",
        f"<p>G4−G1差值 {effect['observed_difference']:.4f}，95% CI [{low:.4f}, {high:.4f}]。</p>",
        "<p>每类只有10张；分类器只作为辅助读数。该结果只适用于固定端点、固定样本和本评阅者。</p></html>",
    ]
    (run_dir / "results.html").write_text("\n".join(results_html), encoding="utf-8")
    (run_dir / "report.html").write_text("\n".join(results_html), encoding="utf-8")
    output_bytes = base._directory_bytes(run_dir)
    within_budget = output_bytes <= int(config["output"]["formal_result_budget_bytes"])
    report["output"]["formal_result_bytes_before_final_report"] = output_bytes
    report["output"]["within_budget"] = within_budget
    if not within_budget:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["final_status"] = "MEASUREMENT_INCOMPLETE"
        report["human_review_status"] = "REVIEW_INCONCLUSIVE"
        report["measurement_complete"] = False
    atomic_write_json(run_dir / "final_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_json(run_dir / "progress.json", {
        "task_id": TASK_ID, "run_id": report["run_id"], "status": report["status"],
        "phase": "FINALIZED", "completed_blocks": report["execution"]["completed_blocks"],
        "recent_durable_write_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "heartbeat_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "current_seed": 2026101501, "current_arm": "fixed_C_G1_G4_native_readability",
        "current_task_local_update": 0, "current_optimizer_step": 0,
        "optical_forward_samples_total": report["execution"]["optical_forward_sample_count"],
        "new_encoder_forward_samples": 0, "new_evaluator_forward_samples": 0,
        "gradients": 0, "optimizer_calls": 0,
    })
    final_bytes = base._directory_bytes(run_dir)
    report["output"]["final_result_bytes"] = final_bytes
    if final_bytes > int(config["output"]["formal_result_budget_bytes"]):
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["final_status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
        report["output"]["within_budget"] = False
    atomic_write_json(run_dir / "final_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "conclusion": conclusion,
                      "report": str(run_dir / "final_report.json")}, ensure_ascii=False), flush=True)


def _write_awaiting_report(
    run_dir: Path, identity: dict[str, Any], config: dict[str, Any], inputs: dict[str, Any],
    preflight: dict[str, Any], manifest: dict[str, Any], replay: dict[str, Any],
    selection_metrics: list[dict[str, Any]], review_summary: dict[str, Any],
    elapsed: dict[str, float], blocks: list[str], optical_samples: int,
) -> dict[str, Any]:
    _write_csv(run_dir / "selected_per_image.csv", selection_metrics)
    classifier_payload: dict[str, Any] = {"task_id": TASK_ID, "run_id": identity["run_id"]}
    for method in METHODS:
        rows = [item for item in manifest["items"] if item["method"] == method]
        cohort = inputs["cohorts"][method]
        idx = torch.tensor([int(item["cohort_row"]) for item in rows], dtype=torch.long)
        classifier_payload[method] = {
            "requested_labels": cohort["requested_labels"].index_select(0, idx).long(),
            "proposal_ids": cohort["proposal_ids"].index_select(0, idx).long(),
            "E_D_logits_saved": cohort["ed_logits"].index_select(0, idx).float(),
            "E_C_logits_saved": cohort["ec_logits"].index_select(0, idx).float(),
            "E_D_predictions_saved": cohort["ed_predictions"].index_select(0, idx).long(),
            "E_C_predictions_saved": cohort["ec_predictions"].index_select(0, idx).long(),
            "E_D_requested_margin_saved": cohort["ed_requested_margin"].index_select(0, idx).float(),
            "E_C_requested_margin_saved": cohort["ec_requested_margin"].index_select(0, idx).float(),
        }
    atomic_torch_save(run_dir / "selected_saved_classifiers.pt", classifier_payload)
    report = {
        "schema": "lightgen_small_g1_g4_native_readability_report.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"],
        "status": "AWAITING_HUMAN_RATINGS", "measurement_complete": True,
        "scientific_question": "At the frozen C 45,000-step endpoint, does G4 improve this reviewer's requested-digit recognition on fixed native 200x200 simulated CMOS samples versus G1?",
        "source": {"task_id": SOURCE_TASK_ID, "branch": "C", "seed": 2026101501,
                   "global_step": SOURCE_STEP, "run_id": inputs["source_run_id"],
                   "checkpoint": identity["source_checkpoint"], "source_status_preserved": "VALID_FAILED"},
        "replication_source": {"task_id": REPLICATION_TASK_ID, "model": MODEL_NAME,
                               "stream": 0, "report_result": inputs["replication_report"]["decision"]["result"]},
        "population": {"selected_generated": REVIEW_GENERATED,
                       "per_method_per_class": SELECT_PER_CLASS,
                       "hidden_duplicates": REVIEW_DUPLICATES,
                       "real_target_anchors": REVIEW_ANCHORS,
                       "total_review_cards": REVIEW_TOTAL,
                       "anchor_definition": "saved MNIST DEV target, bilinear 28x28 to 200x200; not optical output"},
        "replay": replay, "preflight": preflight, "selection": {
            "selection_manifest": "selection_manifest.json", "selection_seeds": SELECTION_SEEDS,
            "classifier_brightness_or_appearance_used": False,
        },
        "human_review": {"status": "AWAITING_HUMAN_RATINGS", "review_id": review_summary["review_id"],
                         "reviewers": 1, "page": review_summary["blind_page"],
                         "hidden_key": review_summary["hidden_key"],
                         "primary_endpoint": "equal-class requested-digit match rate over 200 unique generated images"},
        "quality_gate": {"historical_system_status": "VALID_FAILED", "thresholds": QUALITY_GATE,
                         "status_changed": False},
        "execution": {"optical_forward_sample_count": optical_samples,
                      "expected_preflight_samples": 20, "expected_formal_selected_samples": 200,
                      "new_encoder_forward_samples": 0, "new_evaluator_forward_samples": 0,
                      "gradients": 0, "optimizer_calls": 0, "official_train_dev_test_reads": 0,
                      "frozen_model_parameter_buffer_unchanged": True,
                      "elapsed_seconds": elapsed, "completed_blocks": blocks},
        "output": {"formal_result_budget_bytes": config["output"]["formal_result_budget_bytes"],
                   "result_bytes": base._directory_bytes(run_dir), "within_budget": True},
        "decision": {"result": "AWAITING_HUMAN_RATINGS", "human_rating_received": False,
                     "system_quality_gate_upgraded": False},
        "artifacts": {"selected_native_views": "selected_replay.pt",
                      "readout_raw_checks": "independent_readout_checks.pt",
                      "selected_metrics": "selected_per_image.csv",
                      "saved_classifier_logits": "selected_saved_classifiers.pt",
                      "review_page": "blind_review.html", "review_key": "blind_review_key.json",
                      "public_manifest": "blind_review_public_manifest.json",
                      "report_html": "report.html",
                      "validation_receipt": "validation_receipt.json"},
    }
    atomic_write_json(run_dir / "validation_receipt.json", {
        "task_id": TASK_ID, "run_id": identity["run_id"], "status": "PASSED",
        "checks": {
            "fixed_C_45000_Gaussian_checkpoint_and_frozen_references": True,
            "G1_G4_primary_stream_and_class_proposal_identity": True,
            "sampling_manifest_saved_before_optical_replay": True,
            "production_replay_matches_saved_float_views": True,
            "independent_numpy_gain_clamp_adaptive_pool_readout": True,
            "frozen_decoder_parameters_and_buffers_unchanged": True,
            "new_encoder_evaluators_gradients_and_optimizer_calls_zero": True,
            "240_card_mapping_and_blind_page_valid": True,
            "hidden_repeat_spacing_at_least_40_cards": True,
        },
        "counts": {"preflight_samples": 20, "formal_selected_samples": REVIEW_GENERATED,
                   "total_optical_forward_samples": optical_samples,
                   "new_encoder_forward_samples": 0, "new_evaluator_forward_samples": 0,
                   "gradients": 0, "optimizer_calls": 0, "official_train_dev_test_reads": 0},
    })
    report_bytes = base._directory_bytes(run_dir)
    _require(report_bytes <= int(config["output"]["formal_result_budget_bytes"]),
             "formal result payload exceeded its fixed 128 MiB budget")
    report["output"]["result_bytes"] = report_bytes
    report["output"]["within_budget"] = True
    _write_pending_report(run_dir, report)
    _progress(run_dir, identity, "AWAITING_HUMAN_RATINGS", "AWAITING_HUMAN_RATINGS",
              blocks, review_id=review_summary["review_id"], optical_samples=optical_samples)
    actual_bytes = base._directory_bytes(run_dir)
    _require(actual_bytes <= int(config["output"]["formal_result_budget_bytes"]),
             "formal result payload exceeded its fixed 128 MiB budget")
    report["output"]["actual_result_bytes"] = actual_bytes
    atomic_write_json(run_dir / "machine_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    return report


def _prepare_or_resume(
    run_dir: Path, identity: dict[str, Any], resume_state: Path | None,
) -> tuple[dict[str, Any], bool]:
    identity_path = run_dir / "run_identity.json"
    if resume_state is None:
        if run_dir.exists():
            _require(run_dir.is_dir(), "attempt path exists but is not a directory")
            entries = list(run_dir.iterdir())
            if entries:
                telemetry_path = run_dir / "resource_telemetry.csv"
                _require(len(entries) == 1 and entries[0] == telemetry_path and telemetry_path.is_file(),
                         "existing attempt requires --resume-state; refusing to overwrite saved evidence")
                with telemetry_path.open("r", encoding="utf-8-sig", newline="") as stream:
                    _require(stream.readline().rstrip("\r\n") == WATCHER_TELEMETRY_HEADER,
                             "existing resource_telemetry.csv is not the AutoDL watcher prelaunch file")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(identity_path, identity)
        return identity, False
    requested_state = resume_state.resolve()
    _require(requested_state == (run_dir / "progress.json").resolve(),
             "--resume-state must explicitly name this attempt's progress.json")
    _require(identity_path.is_file() and requested_state.is_file(),
             "resume requires this attempt's saved identity and progress file", FileNotFoundError)
    saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    progress = json.loads(requested_state.read_text(encoding="utf-8"))
    _require(saved_identity == identity
             and progress.get("task_id") == TASK_ID and progress.get("run_id") == saved_identity.get("run_id"),
             "resume inputs differ from the saved task/source identity")
    identity = saved_identity
    return identity, True


def _run(args: argparse.Namespace) -> None:
    wall_started = time.perf_counter()
    config_path = args.config.resolve()
    config, runtime = _read_config(config_path)
    run_dir = _resolve(args.output_dir)
    expected_dir = _resolve(Path(config["output"]["experiment_root"]))
    source_checkpoint = _resolve(args.source_checkpoint)
    baseline_dir = _resolve(args.baseline_evidence_dir)
    replication_dir = _resolve(args.replication_dir)
    if args.ratings_file is not None:
        if run_dir != expected_dir:
            _require(run_dir == expected_dir.parent / "autodl_backup_attempt_01",
                     "ratings output is not the fixed attempt or its explicit local backup", ValueError)
            transfer = json.loads((run_dir / "bundle_transfer_receipt.json").read_text(encoding="utf-8"))
            extraction = json.loads((run_dir / "bundle_extraction_receipt.json").read_text(encoding="utf-8"))
            _require(transfer.get("status") == "COMPLETE"
                     and Path(transfer["destination"]).resolve() == run_dir
                     and str(transfer.get("remote_run_dir", "")).endswith("/" + expected_dir.name)
                     and bool(transfer.get("files"))
                     and all(item.get("status") == "COPIED" and item.get("scp_exit_code") == 0
                             for item in transfer["files"])
                     and extraction.get("status") == "VERIFIED"
                     and extraction.get("size_inventory_match") is True,
                     "local rating evidence lacks a completed transfer and verified extraction")
        identity_path = run_dir / "run_identity.json"
        _require(identity_path.is_file(), "human-rating finalization requires the original run identity", FileNotFoundError)
        saved_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        _require(saved_identity.get("task_id") == TASK_ID and saved_identity.get("config") == config
                 and source_checkpoint == _resolve(Path(config["source"]["checkpoint"]))
                 and baseline_dir == _resolve(Path(config["source"]["baseline_evidence_dir"]))
                 and replication_dir == _resolve(Path(config["source"]["replication_evidence_dir"]))
                 and source_checkpoint.is_file() and baseline_dir.is_dir() and replication_dir.is_dir(),
                 "human-rating invocation differs from the original fixed source/config identity")
        _finalize_ratings(run_dir, args.ratings_file.resolve(), config)
        return
    _require(run_dir == expected_dir, "output directory differs from the explicitly fixed attempt path", ValueError)
    inputs = _load_inputs(source_checkpoint, baseline_dir, replication_dir, config, runtime)
    if args.resume_state is None:
        run_id = uuid.uuid4().hex
        identity = _make_run_identity(config, config_path, source_checkpoint, baseline_dir,
                                      replication_dir, inputs, run_id)
    else:
        saved = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
        identity = _make_run_identity(config, config_path, source_checkpoint, baseline_dir,
                                      replication_dir, inputs, str(saved.get("run_id")))
    identity, resuming = _prepare_or_resume(run_dir, identity, args.resume_state)
    progress_record = json.loads((run_dir / "progress.json").read_text(encoding="utf-8")) if resuming else None
    if progress_record and progress_record.get("status") == "AWAITING_HUMAN_RATINGS":
        print(json.dumps({"status": "AWAITING_HUMAN_RATINGS", "run_id": identity["run_id"],
                          "review_page": str(run_dir / "blind_review.html")}, ensure_ascii=False), flush=True)
        return
    if args.preflight_only and resuming and (run_dir / "report.json").exists():
        raise ValueError("preflight-only cannot resume an attempt with a completed machine report")
    if not torch.cuda.is_available():
        raise RuntimeError("AutoDL CUDA is unavailable; the fixed optical replay cannot run on CPU")
    device = torch.device("cuda")
    manifest_path = run_dir / "selection_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _build_selection_manifest(inputs["cohorts"], identity)
        _validate_selection_manifest(manifest, inputs["cohorts"], identity)
        atomic_write_json(manifest_path, manifest)
    _validate_selection_manifest(manifest, inputs["cohorts"], identity)
    blocks = list(progress_record.get("completed_blocks", [])) if progress_record else []
    _progress(run_dir, identity, "RUNNING", "INPUTS_VALIDATED_SELECTION_LOCKED", blocks,
              review_id=(progress_record or {}).get("review_id"),
              optical_samples=int((progress_record or {}).get("optical_forward_samples_total", 0)))
    model = _load_model(inputs, runtime, device)
    model_before = base._state_snapshot(model)
    phase_started = time.perf_counter()
    preflight, optical_samples = _ensure_preflight(
        run_dir, identity, model, inputs["cohorts"], inputs["checkpoint"], device,
        blocks, int((progress_record or {}).get("optical_forward_samples_total", 0)),
    )
    preflight_seconds = time.perf_counter() - phase_started
    if args.preflight_only:
        _progress(run_dir, identity, "PREFLIGHT_READY", "PREFLIGHT_READY", blocks,
                  optical_samples=optical_samples)
        print(json.dumps({"status": "PREFLIGHT_READY", "run_id": identity["run_id"],
                          "receipt": str(run_dir / "preflight.json")}, ensure_ascii=False), flush=True)
        return
    phase_started = time.perf_counter()
    selected, optical_samples, metric_rows = _load_or_replay_formal(
        run_dir, identity, manifest, inputs["cohorts"], inputs["checkpoint"], model,
        device, blocks, optical_samples,
    )
    formal_seconds = time.perf_counter() - phase_started
    after_model_error = _assert_state_unchanged(model, model_before, "all_formal_replay")
    atomic_torch_save(run_dir / "selected_replay.pt", {
        "schema": "lightgen_small_g1_g4_native_selected_replay.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"], "source_step": SOURCE_STEP,
        "G1": selected["G1"], "G4": selected["G4"],
        "optical_forward_samples": optical_samples,
        "parameter_buffer_max_abs_delta": after_model_error,
    })
    saved_replay = torch.load(run_dir / "selected_replay.pt", map_location="cpu", weights_only=False)
    _require(saved_replay.get("run_id") == identity["run_id"]
             and tuple(saved_replay["G1"]["native200"].shape) == (100, 1, 200, 200)
             and tuple(saved_replay["G4"]["native200"].shape) == (100, 1, 200, 200),
             "selected native replay failed complete file read-back")
    review_id = (progress_record or {}).get("review_id")
    review_started = time.perf_counter()
    key, review_summary = _write_blind_review(
        run_dir, identity, manifest, selected, inputs["cohorts"], inputs["dev_examples"],
        progress_record, blocks, optical_samples,
    )
    review_seconds = time.perf_counter() - review_started
    _require(key.get("review_id") == review_summary["review_id"], "saved review mapping changed after creation")
    atomic_torch_save(run_dir / "selected_replay.pt", {
        "schema": "lightgen_small_g1_g4_native_selected_replay.v1",
        "task_id": TASK_ID, "run_id": identity["run_id"], "source_step": SOURCE_STEP,
        "G1": selected["G1"], "G4": selected["G4"],
        "optical_forward_samples": optical_samples,
        "parameter_buffer_max_abs_delta": after_model_error,
    })
    _require(_assert_state_unchanged(model, model_before, "review_package") == 0.0,
             "frozen C model changed during review package finalization")
    replay_summary = {
        "passed": True, "population": REVIEW_GENERATED,
        "methods": {
            method: {"population": 100,
                     "maximum_view_error_vs_saved_float_view": max(
                         float(row["saved_view_max_abs_error"]) for row in metric_rows if row["method"] == method),
                     "completed_batch_blocks": sum(1 for name in blocks if name.startswith(f"{method}_batch_"))}
            for method in METHODS
        },
        "batch_size": BATCH_SIZE, "saved_view_replay_rtol": REPLAY_RTOL,
        "saved_view_replay_atol": REPLAY_ATOL,
        "independent_numpy_readout_samples": 20,
        "frozen_model_parameter_buffer_max_abs_delta": after_model_error,
    }
    elapsed = {"preflight_seconds": preflight_seconds, "formal_replay_seconds": formal_seconds,
               "review_package_seconds": review_seconds,
               "machine_total_seconds": time.perf_counter() - wall_started}
    report = _write_awaiting_report(
        run_dir, identity, config, inputs, preflight, manifest, replay_summary,
        metric_rows, review_summary, elapsed, blocks, optical_samples,
    )
    _require(report.get("status") == "AWAITING_HUMAN_RATINGS", "machine review package did not reach its stop point")
    print(json.dumps({"status": report["status"], "run_id": identity["run_id"],
                      "optical_forward_samples": optical_samples,
                      "review_page": str(run_dir / "blind_review.html"),
                      "report": str(run_dir / "report.json")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-evidence-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--ratings-file", type=Path)
    args = parser.parse_args()
    _require(not (args.preflight_only and args.ratings_file is not None),
             "--preflight-only cannot be combined with --ratings-file", ValueError)
    try:
        _run(args)
    except Exception as error:
        output_dir = _resolve(args.output_dir)
        if (args.ratings_file is None and output_dir.is_dir()
                and not (output_dir / "final_report.json").is_file()):
            atomic_write_json(output_dir / "engineering_error.json", {
                "task_id": TASK_ID, "status": "ENGINEERING_INVALID",
                "error_type": type(error).__name__, "error": str(error),
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            })
            if (output_dir / "run_identity.json").is_file():
                saved_identity = json.loads((output_dir / "run_identity.json").read_text(encoding="utf-8"))
                progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8")) \
                    if (output_dir / "progress.json").is_file() else {}
                _progress(output_dir, saved_identity, "ENGINEERING_INVALID", "ENGINEERING_INVALID",
                          list(progress.get("completed_blocks", [])),
                          review_id=progress.get("review_id"),
                          optical_samples=int(progress.get("optical_forward_samples_total", 0)))
        raise


if __name__ == "__main__":
    main()
