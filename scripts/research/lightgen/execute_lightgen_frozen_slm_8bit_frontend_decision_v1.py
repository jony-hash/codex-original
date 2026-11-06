"""Frozen 8-bit input preparation, semantic review, and decision workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.shared.research_run_control import (
    PREFLIGHT_READY,
    RUN_INTERRUPTED,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_8bit_error_attribution_v1 as attribution
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_8bit_replay_v1 as replay
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reachability


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FROZEN_SLM_8BIT_FRONTEND_DECISION_V1"
METHODS = ("G1", "G4")
CONDITIONS = ("A8_amplitude_linear", "I8_intensity_linear")
REVIEW_CONDITIONS = ("continuous", *CONDITIONS)
GRIDS = (2048, 4096)
IMAGE_SHAPE = (200, 200)
PIXEL_COUNT = 40_000
PORTS = 100
MAX_SWEEPS = 20
COEFFICIENT_STOP = 1.0e-6
WITNESS_LIMIT = 1.0e-4
MATRIX_LIMIT = 1.0e-6
POWER_RELATIVE_LIMIT = 1.0e-10
POWER_ABSOLUTE_LIMIT_W = 1.0e-15
REVIEW_ID = "frozen-slm-8bit-frontend-decision-20261026"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/frozen_slm_8bit_frontend_decision_v1.yaml"
PREFLIGHT_DIR = ROOT / "outputs/20261026_lightgen_frozen_slm_8bit_frontend_decision_v1/runs/preflight_01"
ATTEMPT_DIR = ROOT / "outputs/20261026_lightgen_frozen_slm_8bit_frontend_decision_v1/runs/attempt_01"
POWER_FIELDS = tuple(sorted(replay.POWER_TRACE_FIELDS))


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temp)
    os.replace(temp, path)


def _config(config_path: Path) -> dict[str, Any]:
    _require(config_path.resolve() == DEFAULT_CONFIG.resolve(), "this task uses its fixed configuration", ValueError)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "configuration must be a mapping", ValueError)
    _validate_config(value)
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == "lightgen_frozen_slm_8bit_frontend_decision.v1"
             and config.get("task_id") == TASK_ID
             and config.get("evidence_tier") == "diagnostic_only",
             "configuration identity or evidence tier changed", ValueError)
    source = config["source"]
    fixed_source = {
        "replay_attempt_dir": "outputs/20261025_lightgen_frozen_slm_8bit_replay_v1/runs/attempt_01",
        "replay_run_id": "01a34fe66f3743b2a8eb7701c7a9e9fd",
        "reachability_attempt_dir": "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01",
        "reachability_run_id": "04390141af8c48278cfd3578f3688e0c",
        "attribution_attempt_dir": "outputs/20261025_lightgen_frozen_slm_8bit_error_attribution_v1/runs/attempt_01",
        "attribution_run_id": "acebbbb1ba6f41b9a2addbfa170fda8a",
        "native_review_attempt_dir": "outputs/20261022_lightgen_small_g1_g4_native_readability_v1/runs/autodl_backup_attempt_01",
        "native_review_run_id": "768fcdd57733430d8c290221f0a3142b",
    }
    _require(all(source.get(key) == value for key, value in fixed_source.items()),
             "fixed source paths or identities changed", ValueError)
    population = config["population"]
    _require(population == {
        "methods": list(METHODS), "samples_per_method": 100,
        "samples_per_class_per_method": 10, "identities": 200,
        "preflight_samples": 20, "G4_semantic_samples": 100,
    }, "fixed population changed", ValueError)
    optimizer = config["optimizer"]
    _require(optimizer.get("matrix_grid") == 2048
             and optimizer.get("maximum_full_coordinate_scans") == MAX_SWEEPS
             and optimizer.get("coordinate_order") == "odd_forward_even_reverse"
             and optimizer.get("checkpoint_coordinate_interval") == 5000
             and optimizer.get("stop_coefficient_relative_error") == COEFFICIENT_STOP
             and optimizer.get("preserve_original_verified_direct_witnesses") is True
             and optimizer.get("public_phase_only") is True
             and optimizer.get("target_absolute_amplitude") is True,
             "discrete optimizer or budget changed", ValueError)
    optics = config["optics"]
    _require(optics.get("checkpoint_step") == 45_000
             and math.isclose(float(optics.get("source_power_w", 0.0)), 2.493e-4, rel_tol=0.0, abs_tol=1e-15)
             and optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("OLS_basis") == "gaussian_s12_v1"
             and optics.get("propagation_grids") == list(GRIDS)
             and optics.get("global_gain") == "frozen_C_checkpoint"
             and optics.get("phase_masks") == "frozen"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "frozen optical source changed", ValueError)
    review_config = config["review"]
    _require(review_config.get("conditions") == list(REVIEW_CONDITIONS)
             and review_config.get("generated_per_condition") == 100
             and review_config.get("hidden_duplicates") == 20
             and review_config.get("hidden_duplicate_per_class") == 2
             and review_config.get("duplicate_condition_counts") == {
                 "continuous": 7, "A8_amplitude_linear": 7, "I8_intensity_linear": 6,
             }
             and review_config.get("real_anchors") == 20
             and review_config.get("total_cards") == 340
             and review_config.get("shuffle_seed") == 2026102601
             and review_config.get("segment_size") == 20
             and review_config.get("minimum_same_identity_card_distance") == 40
             and review_config.get("minimum_duplicate_consistency") == 18
             and review_config.get("minimum_anchor_correct") == 18
             and review_config.get("confidence_level") == 0.95
             and review_config.get("allowed_semantic_loss") == 0.05,
             "blind review design or predeclared semantic threshold changed", ValueError)
    output = config["output"]
    _require(output.get("experiment_root") == ATTEMPT_DIR.relative_to(ROOT).as_posix()
             and output.get("preflight_root") == PREFLIGHT_DIR.relative_to(ROOT).as_posix()
             and output.get("final_consumer") == "recompute_from_saved_gray_matrix_outputs_cards_and_raw_ratings",
             "fixed output roots or final consumer changed", ValueError)


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    attribution_dir = ROOT / config["source"]["attribution_attempt_dir"]
    _require(attribution_dir.is_dir(), "fixed closed attribution attempt is missing", FileNotFoundError)
    attribution_identity = _json(attribution_dir / "run_identity.json")
    attribution_final = _json(attribution_dir / "final_decision.json")
    attribution_consumed = _json(attribution_dir / "windows_reconsumption.json")
    expected_counts = {"continuous": 200, "A8_amplitude_linear": 2, "I8_intensity_linear": 0}
    _require(attribution_identity.get("task_id") == "LIGHTGEN_FROZEN_SLM_8BIT_ERROR_ATTRIBUTION_V1"
             and attribution_identity.get("run_id") == config["source"]["attribution_run_id"]
             and attribution_identity.get("status") == "CLOSED"
             and attribution_final.get("run_id") == attribution_identity["run_id"]
             and attribution_final.get("status") == "VALID_SAVED_EVIDENCE_ATTRIBUTION"
             and attribution_final.get("sample_count") == 200
             and attribution_final.get("attribution_record_count") == 400
             and attribution_final.get("power_record_count") == 800
             and attribution_final.get("original_precision_counts") == expected_counts
             and attribution_consumed.get("run_id") == attribution_identity["run_id"]
             and attribution_consumed.get("status") == "RECOMPUTED_AND_MATCHED"
             and attribution_consumed.get("attribution_records_recomputed") == 400
             and attribution_consumed.get("continuous_reference_rows_recomputed") == 200
             and attribution_consumed.get("saved_output_rows_revalidated") == 1200
             and attribution_consumed.get("power_rows_recomputed") == 800
             and attribution_consumed.get("original_precision_counts_recomputed") == expected_counts,
             "fixed closed attribution evidence is missing or no longer matches its consumer")
    attr_path = ROOT / config["source"]["attribution_config"]
    attr_config = yaml.safe_load(attr_path.read_text(encoding="utf-8"))
    attribution._validate_config(attr_config, attr_path)
    inputs = attribution._load_inputs(attr_config)
    identities = inputs["identities"]
    inputs["_verified_old_output_cache"] = {}
    _require(len(identities) == 200 and len({item["sample_key"] for item in identities}) == 200,
             "fixed source identity population is incomplete or duplicated")
    counts = Counter((item["method"], int(item["label"])) for item in identities)
    _require(set(counts) == {(method, label) for method in METHODS for label in range(10)}
             and all(counts[(method, label)] == 10 for method in METHODS for label in range(10)),
             "fixed population does not preserve ten rows per method and class")
    inputs["target"] = inputs["target"].to(torch.complex128).contiguous()
    inputs["matrix"] = inputs["matrix"].to(torch.complex128).contiguous()
    inputs["identity_index"] = {item["sample_key"]: i for i, item in enumerate(identities)}
    inputs["attribution_dir"] = attribution_dir
    inputs["attribution_identity"] = attribution_identity
    inputs["attribution_final"] = attribution_final
    native_dir = ROOT / config["source"]["native_review_attempt_dir"]
    native_identity = _json(native_dir / "run_identity.json")
    review_key = _json(native_dir / "blind_review_key.json")
    _require(native_identity.get("run_id") == config["source"]["native_review_run_id"]
             and native_identity.get("task_id") == "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1"
             and review_key.get("run_id") == native_identity["run_id"]
             and len(review_key.get("cards", [])) == 240,
             "fixed historical blind-review evidence is missing or changed")
    anchors = [card for card in review_key["cards"] if card.get("kind") == "real_anchor"]
    _require(len(anchors) == 20, "historical review does not contain the fixed twenty anchors")
    anchor_counts = Counter(int(item["target_label"]) for item in anchors)
    _require(all(anchor_counts[label] == 2 for label in range(10)),
             "historical anchor allocation is not two per class")
    for card in anchors:
        image_path = native_dir / card["image"]
        _require(image_path.is_file(), f"historical anchor image is missing: {image_path}", FileNotFoundError)
        with Image.open(image_path) as image:
            _require(image.mode == "L" and image.size == IMAGE_SHAPE,
                     "historical anchor image is not a native grayscale 200x200 card")
    inputs["native_review_dir"] = native_dir
    inputs["native_review_key"] = review_key
    inputs["anchor_cards"] = anchors
    return inputs


def _select_preflight(identities: Sequence[Mapping[str, Any]]) -> list[int]:
    first: dict[tuple[str, int], int] = {}
    for index, identity in enumerate(identities):
        first.setdefault((str(identity["method"]), int(identity["label"])), index)
    chosen = [first[(method, label)] for method in METHODS for label in range(10)]
    _require(len(chosen) == 20 and len(set(chosen)) == 20,
             "first-per-method-per-class preflight selection changed")
    return chosen


def _decode_gray(codes: np.ndarray | torch.Tensor, condition: str) -> np.ndarray:
    raw = np.asarray(codes)
    _require(raw.dtype == np.uint8, "gray commands must be saved as uint8 codes", ValueError)
    values = raw.astype(np.float64) / 255.0
    if condition == "A8_amplitude_linear":
        return values
    if condition == "I8_intensity_linear":
        return np.sqrt(values)
    raise ValueError(f"unsupported discrete encoding condition: {condition}")


def _best_coordinate_codes(current_codes: np.ndarray, cross: np.ndarray, column_norm_sq: float,
                           condition: str, objective_scale: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(current_codes)
    _require(current.dtype == np.uint8 and current.ndim == 1,
             "coordinate codes must be a uint8 sample vector", ValueError)
    _require(cross.shape == current.shape and objective_scale.shape == current.shape
             and np.isfinite(cross).all() and np.isfinite(objective_scale).all(),
             "coordinate response statistics are malformed", ValueError)
    if column_norm_sq <= 0.0:
        unchanged = current.copy()
        amplitude = _decode_gray(current, condition)
        return unchanged, amplitude, amplitude, np.zeros(len(current), dtype=bool)
    old_amplitude = _decode_gray(current, condition)
    optimum = np.clip(old_amplitude - cross / column_norm_sq, 0.0, 1.0)
    continuous_code = (optimum * 255.0 if condition == "A8_amplitude_linear"
                       else np.square(optimum) * 255.0)
    lower = np.floor(continuous_code).astype(np.int64).clip(0, 255)
    upper = np.ceil(continuous_code).astype(np.int64).clip(0, 255)
    # Stable candidate ordering: current first, then lower code, then upper code.
    candidate_codes = np.stack((current.astype(np.int64), lower, upper), axis=1).astype(np.uint8)
    candidate_amplitude = _decode_gray(candidate_codes, condition)
    delta = candidate_amplitude - old_amplitude[:, None]
    cost_delta = 2.0 * delta * cross[:, None] + np.square(delta) * column_norm_sq
    best = np.argmin(cost_delta, axis=1)
    chosen_codes = candidate_codes[np.arange(len(current)), best]
    chosen_amplitude = candidate_amplitude[np.arange(len(current)), best]
    best_delta = cost_delta[np.arange(len(current)), best]
    tolerance = 64.0 * np.finfo(np.float64).eps * np.maximum(objective_scale, np.finfo(np.float64).tiny)
    changed = (chosen_codes != current) & (best_delta < -tolerance)
    chosen_codes = np.where(changed, chosen_codes, current).astype(np.uint8)
    chosen_amplitude = np.where(changed, chosen_amplitude, old_amplitude)
    return chosen_codes, old_amplitude, chosen_amplitude, changed


def _intensity_from_amplitude(amplitude: np.ndarray) -> np.ndarray:
    _require(np.isfinite(amplitude).all() and float(amplitude.min()) >= 0.0
             and float(amplitude.max()) <= 1.0,
             "decoded amplitude is nonfinite or outside [0,1]", ValueError)
    return np.square(amplitude, dtype=np.float64)


def _relative(actual: np.ndarray | torch.Tensor, expected: np.ndarray | torch.Tensor) -> float:
    a = np.asarray(actual)
    e = np.asarray(expected)
    denom = float(np.linalg.norm(e.reshape(-1)))
    _require(math.isfinite(denom) and denom > 0.0, "relative-error reference is zero or nonfinite")
    return float(np.linalg.norm((a - e).reshape(-1)) / denom)


def _phase_error(actual: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> tuple[float, float]:
    a = np.asarray(actual, dtype=np.complex128).reshape(-1)
    t = np.asarray(target, dtype=np.complex128).reshape(-1)
    _require(a.shape == t.shape and np.isfinite(a.real).all() and np.isfinite(a.imag).all()
             and np.isfinite(t.real).all() and np.isfinite(t.imag).all(),
             "phase comparison has malformed coefficients")
    denom = float(np.linalg.norm(t))
    _require(denom > 0.0 and math.isfinite(denom), "coefficient target norm is zero or nonfinite")
    phase = float(np.angle(np.vdot(t, a)))
    error = float(np.linalg.norm(a - np.exp(1j * phase) * t) / denom)
    return phase, error


def _relative_close(actual: np.ndarray | torch.Tensor, expected: np.ndarray | torch.Tensor,
                    rtol: float = 2.0e-6, atol: float = 5.0e-8) -> bool:
    return bool(np.allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol))


def _old_output(inputs: Mapping[str, Any], index: int, condition: str, grid: int) -> dict[str, Any]:
    cache = inputs.get("_verified_old_output_cache")
    if not isinstance(cache, dict):
        return attribution._load_output(inputs, inputs["identities"][index], condition, grid)
    key = (index, condition, grid)
    if key not in cache:
        cache[key] = attribution._load_output(inputs, inputs["identities"][index], condition, grid)
    return cache[key]


def _recompute_original_counts(inputs: Mapping[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for condition in REVIEW_CONDITIONS:
        rows: list[dict[str, Any]] = []
        for index, identity in enumerate(inputs["identities"]):
            coefficient_errors: list[float] = []
            cmos_errors: list[float] = []
            for grid in GRIDS:
                output = _old_output(inputs, index, condition, grid)
                coefficient_errors.append(_phase_error(output["actual_coefficients"], inputs["target"][index].numpy())[1])
                cmos_errors.append(_relative(output["raw_cmos_intensity_w_per_m2"], inputs["direct_cmos"][str(grid)][index, 0]))
            cross_coeff = _relative(
                _old_output(inputs, index, condition, GRIDS[0])["actual_coefficients"],
                _old_output(inputs, index, condition, GRIDS[1])["actual_coefficients"],
            )
            cross_cmos = _relative(
                _old_output(inputs, index, condition, GRIDS[0])["raw_cmos_intensity_w_per_m2"],
                _old_output(inputs, index, condition, GRIDS[1])["raw_cmos_intensity_w_per_m2"],
            )
            rows.append({
                "sample_key": identity["sample_key"], "method": identity["method"],
                "label": int(identity["label"]),
                "coefficient_error_max": max(coefficient_errors),
                "native_cmos_error_max": max(cmos_errors),
                "cross_grid_coefficient_error": cross_coeff,
                "cross_grid_native_cmos_error": cross_cmos,
                "witness": max(coefficient_errors) <= WITNESS_LIMIT
                and max(cmos_errors) <= WITNESS_LIMIT
                and max(cross_coeff, cross_cmos) <= WITNESS_LIMIT,
            })
        results[condition] = {"passed": sum(row["witness"] for row in rows), "rows": rows}
    observed = {name: results[name]["passed"] for name in REVIEW_CONDITIONS}
    _require(observed == {"continuous": 200, "A8_amplitude_linear": 2, "I8_intensity_linear": 0},
             f"frozen original witness counts changed: {observed}")
    return {"sample_denominator_per_condition": 200, "conditions": results,
            "counts": observed, "source_attempts_read_only": True,
            "saved_attribution_counts_match": observed == inputs["attribution_final"]["original_precision_counts"]}


def _matrix_prediction(inputs: Mapping[str, Any], codes: np.ndarray, condition: str) -> np.ndarray:
    amplitude = _decode_gray(codes, condition)
    matrix = inputs["matrix"].numpy()
    _require(matrix.shape == (PORTS, PIXEL_COUNT) and matrix.dtype == np.complex128,
             "saved response matrix must be the fixed 2048 complex128 100x40000 matrix")
    return amplitude @ matrix.T


def _coordinate_optimize(
    matrix: np.ndarray,
    target: np.ndarray,
    initial_codes: np.ndarray,
    condition: str,
    identities: Sequence[Mapping[str, Any]],
    skip_reason: Sequence[str | None],
    state_path: Path,
    progress_path: Path,
    run_id: str,
    *,
    resume: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    _require(matrix.shape == (PORTS, PIXEL_COUNT) and matrix.dtype == np.complex128,
             "coordinate solver requires the saved complex128 2048 response matrix")
    raw_codes = np.asarray(initial_codes)
    _require(raw_codes.dtype == np.uint8,
             "initial commands must be the saved uint8 gray codes", ValueError)
    codes = raw_codes.copy()
    _require(target.dtype == np.complex128 and target.shape == (len(identities), PORTS)
             and np.isfinite(target.real).all() and np.isfinite(target.imag).all()
             and codes.shape == (len(identities), PIXEL_COUNT),
             "coordinate solver inputs have inconsistent population shapes")
    _require(len(skip_reason) == len(identities), "coordinate solver skip map has the wrong population")
    state: dict[str, Any] | None = None
    if resume and state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        _require(saved.get("schema") == "lightgen_frozen_slm_8bit_coordinate_state.v1"
                 and saved.get("run_id") == run_id and saved.get("condition") == condition
                 and saved.get("sample_keys") == [str(item["sample_key"]) for item in identities],
                 "saved coordinate state belongs to another condition or population")
        required_state = {
            "gray_u8": (torch.uint8, (len(identities), PIXEL_COUNT)),
            "completed": (torch.bool, (len(identities),)),
            "phase_rad": (torch.float64, (len(identities),)),
            "residual": (torch.complex128, (len(identities), PORTS)),
            "sweep_changed": (torch.bool, (len(identities),)),
            "coordinate_visits": (torch.int64, (len(identities),)),
        }
        for field, (dtype, shape) in required_state.items():
            tensor = saved.get(field)
            _require(isinstance(tensor, torch.Tensor) and tensor.dtype == dtype
                     and tuple(tensor.shape) == shape,
                     f"saved optimizer state has a malformed {field}")
        _require(isinstance(saved.get("sweep_index"), int)
                 and 0 <= saved["sweep_index"] <= MAX_SWEEPS
                 and isinstance(saved.get("cursor"), int)
                 and 0 <= saved["cursor"] <= PIXEL_COUNT
                 and (saved["sweep_index"] < MAX_SWEEPS or saved["cursor"] == 0),
                 "saved optimizer cursor is outside the fixed coordinate budget")
        reasons = saved.get("record_reasons")
        _require(isinstance(reasons, list) and len(reasons) == len(identities),
                 "saved optimizer termination reasons do not match the population")
        completed_saved = saved["completed"].cpu().numpy()
        visits_saved = saved["coordinate_visits"].cpu().numpy()
        _require(np.all(visits_saved >= 0) and np.all(visits_saved <= MAX_SWEEPS * PIXEL_COUNT)
                 and np.all(visits_saved[~completed_saved] == saved["sweep_index"] * PIXEL_COUNT + saved["cursor"]),
                 "saved optimizer visits do not match the resumable cursor")
        for index, reason in enumerate(skip_reason):
            if reason is not None:
                _require(bool(completed_saved[index]) and reasons[index] == reason,
                         f"saved optimizer skip reason changed for {identities[index]['sample_key']}")
            else:
                _require(reasons[index] in {
                    "optimized", "coefficient_tolerance", "full_sweep_no_gray_change",
                    "specified_scan_budget_exhausted",
                }, f"saved optimizer has an invalid termination reason for {identities[index]['sample_key']}")
        codes = saved["gray_u8"].cpu().numpy().copy()
        state = saved
    if state is None:
        complete = np.array([reason is not None for reason in skip_reason], dtype=bool)
        pred = _matrix_prediction({"matrix": torch.from_numpy(matrix)}, codes, condition)
        phase = np.angle(np.sum(np.conj(target) * pred, axis=1))
        residual = pred - np.exp(1j * phase)[:, None] * target
        state = {
            "schema": "lightgen_frozen_slm_8bit_coordinate_state.v1", "run_id": run_id,
            "condition": condition, "sample_keys": [str(item["sample_key"]) for item in identities],
            "gray_u8": torch.from_numpy(codes.copy()), "completed": torch.from_numpy(complete),
            "sweep_index": 0, "cursor": 0, "phase_rad": torch.from_numpy(phase.copy()),
            "residual": torch.from_numpy(residual.copy()),
            "sweep_changed": torch.zeros(len(identities), dtype=torch.bool),
            "coordinate_visits": torch.zeros(len(identities), dtype=torch.int64),
            "record_reasons": [reason or "optimized" for reason in skip_reason],
            "started_at_unix": time.time(),
        }
    else:
        codes = state["gray_u8"].cpu().numpy().copy()
        # Recover from the exact saved codes and verify the resumable residual before continuing.
        amp = _decode_gray(codes, condition)
        pred = amp @ matrix.T
        phase = np.asarray(state["phase_rad"], dtype=np.float64)
        expected_residual = pred - np.exp(1j * phase)[:, None] * target
        saved_residual = state["residual"].cpu().numpy().astype(np.complex128, copy=False)
        residual_error = float(np.linalg.norm(expected_residual - saved_residual)
                               / max(float(np.linalg.norm(expected_residual)),
                                     float(np.linalg.norm(saved_residual)), np.finfo(np.float64).tiny))
        _require(residual_error <= 1.0e-12,
                 f"saved coordinate residual does not reproduce from saved gray codes: {residual_error:.3g}")
        state["residual"] = torch.from_numpy(expected_residual.copy())

    norm_sq = np.sum(np.abs(matrix) ** 2, axis=0, dtype=np.float64)
    target_norm = np.linalg.norm(target, axis=1)
    _require(np.isfinite(norm_sq).all() and np.isfinite(target_norm).all() and np.all(target_norm > 0.0),
             "coordinate solver matrix or target contains invalid norms")
    cursor_interval = 5000
    visits_since_save = 0

    def checkpoint() -> None:
        state["gray_u8"] = torch.from_numpy(codes.copy())
        state["residual"] = torch.from_numpy(np.asarray(state["residual"]).copy())
        _atomic_torch(state_path, state)
        _atomic_json(progress_path, {
            "task_id": TASK_ID, "run_id": run_id, "phase": "discrete_coordinate_solve",
            "condition": condition, "sweep_index": int(state["sweep_index"]),
            "cursor": int(state["cursor"]), "coordinate_visits_per_sample": state["coordinate_visits"].tolist(),
            "completed_samples": int(np.asarray(state["completed"]).sum()),
            "sample_count": len(identities), "model_updates": 0, "adam_calls": 0,
            "prior_fit": 0, "new_sampling": 0, "one_shot_mapping_training": 0,
        })

    while int(state["sweep_index"]) < MAX_SWEEPS:
        completed = np.asarray(state["completed"], dtype=bool)
        active = np.flatnonzero(~completed)
        if len(active) == 0:
            break
        sweep_index = int(state["sweep_index"])
        cursor = int(state["cursor"])
        direction = 1 if sweep_index % 2 == 0 else -1
        residual = state["residual"].cpu().numpy().astype(np.complex128, copy=False)
        phase = state["phase_rad"].cpu().numpy().astype(np.float64, copy=False)
        changed = state["sweep_changed"].cpu().numpy().astype(bool, copy=True)
        visits = state["coordinate_visits"].cpu().numpy().astype(np.int64, copy=True)
        while cursor < PIXEL_COUNT:
            pixel = cursor if direction > 0 else PIXEL_COUNT - 1 - cursor
            column = matrix[:, pixel]
            column_norm = norm_sq[pixel]
            if column_norm > 0.0:
                active = np.flatnonzero(~np.asarray(state["completed"], dtype=bool))
                if len(active):
                    current_code = codes[active, pixel]
                    local_residual = residual[active]
                    cross = np.real(np.conj(local_residual) @ column)
                    residual_norm = np.linalg.norm(local_residual, axis=1)
                    scale = np.maximum(np.square(residual_norm), np.square(target_norm[active]))
                    chosen_code, current_amp, chosen_amp, take = _best_coordinate_codes(
                        current_code, cross, column_norm, condition, scale,
                    )
                    if np.any(take):
                        changed[active[take]] = True
                        codes[active[take], pixel] = chosen_code[take]
                        residual[active[take]] += (chosen_amp[take] - current_amp[take])[:, None] * column[None, :]
            visits[~np.asarray(state["completed"], dtype=bool)] += 1
            cursor += 1
            visits_since_save += 1
            state["cursor"] = cursor
            state["coordinate_visits"] = torch.from_numpy(visits.copy())
            state["sweep_changed"] = torch.from_numpy(changed.copy())
            state["phase_rad"] = torch.from_numpy(phase.copy())
            state["residual"] = torch.from_numpy(residual.copy())
            if visits_since_save >= cursor_interval:
                checkpoint()
                visits_since_save = 0

        amplitude = _decode_gray(codes, condition)
        prediction = amplitude @ matrix.T
        phase = np.angle(np.sum(np.conj(target) * prediction, axis=1))
        residual = prediction - np.exp(1j * phase)[:, None] * target
        current_error = np.linalg.norm(residual, axis=1) / target_norm
        completed = np.asarray(state["completed"], dtype=bool).copy()
        sweep_changed = changed.copy()
        active = np.flatnonzero(~completed)
        stop = (current_error <= COEFFICIENT_STOP) | (~sweep_changed)
        completed[active[stop[active]]] = True
        reason_values = list(state["record_reasons"])
        for index in active:
            if completed[index] and reason_values[index] == "optimized":
                reason_values[index] = "coefficient_tolerance" if current_error[index] <= COEFFICIENT_STOP else "full_sweep_no_gray_change"
        state["record_reasons"] = reason_values
        state["completed"] = torch.from_numpy(completed)
        state["phase_rad"] = torch.from_numpy(phase.copy())
        state["residual"] = torch.from_numpy(residual.copy())
        state["sweep_changed"] = torch.zeros(len(identities), dtype=torch.bool)
        state["sweep_index"] = sweep_index + 1
        state["cursor"] = 0
        checkpoint()
        visits_since_save = 0

    completed = np.asarray(state["completed"], dtype=bool).copy()
    for index, reason in enumerate(state["record_reasons"]):
        if not completed[index]:
            completed[index] = True
            state["record_reasons"][index] = "specified_scan_budget_exhausted"
    state["completed"] = torch.from_numpy(completed)
    final_amp = _decode_gray(codes, condition)
    final_pred = final_amp @ matrix.T
    final_phase = np.angle(np.sum(np.conj(target) * final_pred, axis=1))
    final_residual = final_pred - np.exp(1j * final_phase)[:, None] * target
    final_errors = np.linalg.norm(final_residual, axis=1) / target_norm
    original = np.asarray(initial_codes, dtype=np.uint8)
    record_rows: list[dict[str, Any]] = []
    visits = state["coordinate_visits"].cpu().numpy().astype(np.int64)
    reasons = list(state["record_reasons"])
    for index, identity in enumerate(identities):
        initial_amp = _decode_gray(original[index], condition)
        initial_pred = initial_amp @ matrix.T
        initial_phase = np.angle(np.vdot(target[index], initial_pred))
        initial_residual = initial_pred - np.exp(1j * initial_phase) * target[index]
        record_rows.append({
            "sample_key": str(identity["sample_key"]), "method": str(identity["method"]),
            "label": int(identity["label"]), "condition": condition,
            "initial_matrix_coefficient_error": float(np.linalg.norm(initial_residual) / target_norm[index]),
            "final_matrix_coefficient_error": float(final_errors[index]),
            "public_phase_rad": float(final_phase[index]),
            "gray_pixels_changed": int(np.count_nonzero(codes[index] != original[index])),
            "coordinate_visits": int(visits[index]), "full_scans": int(math.ceil(visits[index] / PIXEL_COUNT)),
            "termination": reasons[index], "skipped_input": skip_reason[index] is not None,
        })
    state["gray_u8"] = torch.from_numpy(codes.copy())
    state["phase_rad"] = torch.from_numpy(final_phase.copy())
    state["residual"] = torch.from_numpy(final_residual.copy())
    _atomic_torch(state_path, state)
    _atomic_json(progress_path, {
        "task_id": TASK_ID, "run_id": run_id, "phase": "discrete_coordinate_solve_complete",
        "condition": condition, "completed_samples": len(identities), "sample_count": len(identities),
        "coordinate_visits_per_sample": visits.tolist(), "model_updates": 0,
        "adam_calls": 0, "prior_fit": 0, "new_sampling": 0,
        "one_shot_mapping_training": 0,
    })
    return codes, record_rows


def _load_reachability_inputs(config: Mapping[str, Any], identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = ROOT / config["source"]["reachability_config"]
    source_config = reachability._load_yaml(path)
    reachability._validate_config(source_config, path)
    inputs = reachability._load_inputs(source_config)
    _require(inputs["checkpoint"].get("completed_global_step") == 45_000
             and inputs["checkpoint"].get("branch") == "C",
             "loaded optical checkpoint is not frozen C at 45,000 steps")
    reachability_rows = [
        (item.get("method"), item.get("cohort_row"), item.get("requested_label"),
         item.get("proposal_id"), item.get("component_id"))
        for item in inputs["entries"]
    ]
    replay_rows = [
        (item.get("method"), item.get("cohort_row"), item.get("label"),
         item.get("proposal_id"), item.get("component_id"))
        for item in identities
    ]
    _require(reachability_rows == replay_rows,
             "reachability and replay sample identity order differs")
    return inputs


def _new_models(checkpoint: Mapping[str, Any], device: torch.device) -> dict[int, torch.nn.Module]:
    models = {grid: reachability._new_model(checkpoint, grid, device) for grid in GRIDS}
    return models


def _run_forward(model: torch.nn.Module, intensity: np.ndarray, device: torch.device) -> dict[str, Any]:
    tensor = torch.from_numpy(np.asarray(intensity, dtype=np.float64)).reshape(1, 1, *IMAGE_SHAPE).to(device)
    return replay._runtime_forward(model, tensor)


def _compare_production_to_saved(actual: Mapping[str, Any], saved: Mapping[str, Any], sample_key: str,
                                 grid: int, condition: str) -> dict[str, float]:
    coeff_error = _relative(actual["actual_coefficients"], saved["actual_coefficients"])
    cmos_error = _relative(actual["raw_cmos_intensity_w_per_m2"], saved["raw_cmos_intensity_w_per_m2"])
    _require(_relative_close(actual["actual_coefficients"], saved["actual_coefficients"])
             and _relative_close(actual["raw_cmos_intensity_w_per_m2"],
                                 saved["raw_cmos_intensity_w_per_m2"]),
             f"frozen production replay differs from saved closed output: {condition}/{grid}/{sample_key}")
    return {"coefficient_replay_relative_error": coeff_error, "native_cmos_replay_relative_error": cmos_error}


def _snapshot_models(models: Mapping[int, torch.nn.Module]) -> dict[int, dict[str, torch.Tensor]]:
    return {grid: reachability._snapshot_state(model) for grid, model in models.items()}


def _assert_models_unchanged(models: Mapping[int, torch.nn.Module],
                             before: Mapping[int, Mapping[str, torch.Tensor]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for grid, model in models.items():
        after = reachability._snapshot_state(model)
        _require(before[grid].keys() == after.keys(), f"model state keys changed at grid {grid}")
        deltas = [float((after[key] - before[grid][key]).abs().max()) for key in after]
        _require(all(torch.equal(after[key], before[grid][key]) for key in after),
                 f"frozen model parameter or buffer changed at grid {grid}")
        result[grid] = max(deltas, default=0.0)
    return result


def _run_preflight(
    config: Mapping[str, Any], inputs: Mapping[str, Any], originals: Mapping[str, Any],
    reach_inputs: Mapping[str, Any], run_id: str,
) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized AutoDL preflight")
    device = torch.device("cuda:0")
    cuda_free_memory_before_bytes = int(torch.cuda.mem_get_info(device)[0])
    models = _new_models(reach_inputs["checkpoint"], device)
    state_before = _snapshot_models(models)
    selected = _select_preflight(inputs["identities"])
    selected_identities = [inputs["identities"][index] for index in selected]
    matrix = inputs["matrix"].numpy()
    target = inputs["target"].numpy()[selected]
    original_witness = {
        condition: np.array([originals["conditions"][condition]["rows"][index]["witness"]
                             for index in selected], dtype=bool)
        for condition in CONDITIONS
    }
    gray_by_condition = {
        "A8_amplitude_linear": inputs["gray_a"].numpy()[selected].copy(),
        "I8_intensity_linear": inputs["gray_i"].numpy()[selected].copy(),
    }
    matrix_records: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    output_records: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    run_dir = PREFLIGHT_DIR
    for index, identity in enumerate(selected_identities):
        for condition in REVIEW_CONDITIONS:
            if condition == "continuous":
                amplitude = inputs["amplitude"].numpy()[selected[index]]
            else:
                amplitude = _decode_gray(gray_by_condition[condition][index], condition)
            prediction = amplitude @ matrix.T
            observed = _old_output(inputs, selected[index], condition, 2048)["actual_coefficients"].numpy()
            error = _relative(prediction, observed)
            matrix_records.append({"sample_key": identity["sample_key"], "condition": condition,
                                  "matrix_prediction_relative_error": error})
            _require(error <= MATRIX_LIMIT,
                     f"2048 response matrix failed production forward closure: {condition}/{identity['sample_key']} {error:.3g}")
        for grid in GRIDS:
            for condition in REVIEW_CONDITIONS:
                if condition == "continuous":
                    amplitude = inputs["amplitude"].numpy()[selected[index]]
                else:
                    amplitude = _decode_gray(gray_by_condition[condition][index], condition)
                actual = _run_forward(models[grid], _intensity_from_amplitude(amplitude), device)
                saved = _old_output(inputs, selected[index], condition, grid)
                comparison = _compare_production_to_saved(actual, saved, str(identity["sample_key"]), grid, condition)
                _require(math.isclose(actual["power_trace_w"]["P_source_before_SLM_W"],
                                      float(config["optics"]["source_power_w"]), rel_tol=0.0, abs_tol=1e-12),
                         "source power changed during the preflight production call")
                slm_error = attribution._power_checks(
                    inputs, actual, torch.from_numpy(_intensity_from_amplitude(amplitude)), grid,
                    str(identity["sample_key"]),
                )
                replay_records.append({"sample_key": identity["sample_key"], "condition": condition,
                                       "grid": grid, **comparison, **slm_error})
    optimizer_records: dict[str, list[dict[str, Any]]] = {}
    optimized_gray: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        reasons = ["source_verified_original_witness" if value else None
                   for value in original_witness[condition]]
        codes, records = _coordinate_optimize(
            matrix, target, gray_by_condition[condition], condition, selected_identities, reasons,
            run_dir / f"optimizer_{condition}.pt", run_dir / f"progress_{condition}.json", run_id,
            resume=True,
        )
        optimized_gray[condition] = codes
        optimizer_records[condition] = records
        for local_index, identity in enumerate(selected_identities):
            if original_witness[condition][local_index]:
                _require(np.array_equal(codes[local_index], gray_by_condition[condition][local_index]),
                         "original verified direct witness was changed by the discrete solver")
    frozen_replay_max = max(max(row["coefficient_replay_relative_error"], row["native_cmos_replay_relative_error"])
                            for row in replay_records)
    optimized_matrix_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for local_index, identity in enumerate(selected_identities):
            amplitude = _decode_gray(optimized_gray[condition][local_index], condition)
            prediction = amplitude @ matrix.T
            for grid in GRIDS:
                actual = _run_forward(models[grid], _intensity_from_amplitude(amplitude), device)
                if grid == 2048:
                    matrix_error = _relative(prediction, actual["actual_coefficients"])
                    _require(matrix_error <= MATRIX_LIMIT,
                             f"optimized preflight code fails response matrix closure: {condition}/{identity['sample_key']}")
                    optimized_matrix_records.append({"sample_key": identity["sample_key"], "condition": condition,
                                                     "matrix_prediction_relative_error": matrix_error})
                power = attribution._power_checks(
                    inputs, actual, torch.from_numpy(_intensity_from_amplitude(amplitude)),
                    grid, str(identity["sample_key"]),
                )
                output_path = run_dir / "samples" / condition / str(grid) / f"{identity['sample_key']}.pt"
                _atomic_torch(output_path, {
                    "schema": "lightgen_frozen_slm_8bit_frontend_output.v1", "task_id": TASK_ID,
                    "run_id": run_id, "condition": condition, "grid": grid,
                    "sample_identity": dict(identity),
                    "actual_coefficients": actual["actual_coefficients"],
                    "raw_cmos_intensity_w_per_m2": actual["raw_cmos_intensity_w_per_m2"],
                    "power_trace_w": actual["power_trace_w"],
                    "camera_pixel_area_m2": actual["camera_pixel_area_m2"],
                    "camera_total_power_w": actual["camera_total_power_w"],
                    "optimizer_output": True,
                })
                replay_records.append({"sample_key": identity["sample_key"], "condition": condition,
                                       "grid": grid, "optimizer_output": True, **power})
                output_records[condition].append({"sample_key": identity["sample_key"], "grid": grid})
    model_state_delta = _assert_models_unchanged(models, state_before)
    _require(all(value == 0.0 for value in model_state_delta.values()),
             "preflight changed a frozen model parameter or buffer")
    matrix_max = max(row["matrix_prediction_relative_error"] for row in matrix_records)
    optimized_matrix_max = max(row["matrix_prediction_relative_error"] for row in optimized_matrix_records)
    resource = {
        "cuda_available": torch.cuda.is_available(), "cuda_device": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "cuda_free_memory_before_bytes": cuda_free_memory_before_bytes,
        "cuda_free_memory_after_preflight_bytes": int(torch.cuda.mem_get_info(device)[0]),
        "disk_path": str(PREFLIGHT_DIR.parent.resolve()),
        "disk_free_bytes": int(shutil.disk_usage(PREFLIGHT_DIR.parent.resolve()).free),
        "resource_values_are_observations_not_admission_quotas": True,
    }
    checks = {
        "fixed_closed_sources_and_original_200_2_0_counts": {"executed": True, "passed": True},
        "population_identity_and_ten_per_class": {"executed": True, "passed": len(selected) == 20},
        "2048_matrix_prediction_le_1e-6": {"executed": True, "passed": matrix_max <= MATRIX_LIMIT,
                                            "maximum_relative_error": matrix_max},
        "frozen_reference_replay_20_by_three_conditions_by_two_grids": {
            "executed": True, "passed": frozen_replay_max <= 1.0e-6,
            "maximum_relative_error": frozen_replay_max, "rows": 120,
        },
        "optimized_response_matrix_prediction_le_1e-6": {
            "executed": True, "passed": optimized_matrix_max <= MATRIX_LIMIT,
            "maximum_relative_error": optimized_matrix_max, "rows": len(optimized_matrix_records),
        },
        "source_and_cmos_power_integrals": {"executed": True,
                                             "passed": all(
                                                 row["analytic_slm_power_relative_error"] <= POWER_RELATIVE_LIMIT
                                                 or abs(row["P_after_SLM_W"] - row["P_after_SLM_expected_W"]) <= POWER_ABSOLUTE_LIMIT_W
                                                 for row in replay_records
                                             ) and all(row["camera_integral_power_relative_error"] <= POWER_RELATIVE_LIMIT
                                                       or abs(row["P_CMOS_integrated_W"] - row["P_CMOS_trace_W"]) <= POWER_ABSOLUTE_LIMIT_W
                                                       for row in replay_records)},
        "frozen_model_state_exactly_unchanged": {"executed": True,
                                                  "passed": all(v == 0.0 for v in model_state_delta.values()),
                                                  "max_state_delta": max(model_state_delta.values())},
        "discrete_optimizer_positive_control_and_budget": {
            "executed": True, "passed": all(len(optimizer_records[c]) == 20 for c in CONDITIONS),
            "encoding_rows": {c: len(optimizer_records[c]) for c in CONDITIONS},
            "optimization_success_not_a_preflight_requirement": True,
        },
    }
    _require(all(v["executed"] and v["passed"] for v in checks.values()),
             "production preflight failed; formal population will not start")
    _atomic_torch(run_dir / "optimized_preflight_gray_u8.pt", {
        "schema": "lightgen_frozen_slm_8bit_frontend_preflight_gray.v1", "task_id": TASK_ID,
        "run_id": run_id, "sample_identities": [dict(item) for item in selected_identities],
        "A8_gray_u8": torch.from_numpy(optimized_gray["A8_amplitude_linear"].copy()),
        "I8_gray_u8": torch.from_numpy(optimized_gray["I8_intensity_linear"].copy()),
    })
    result = {
        "schema": "lightgen_frozen_slm_8bit_frontend_preflight.v1", "task_id": TASK_ID,
        "run_id": run_id, "status": "READY", "selected_samples": [dict(item) for item in selected_identities],
        "source_run_ids": {
            "reachability": inputs["reach_identity"]["run_id"],
            "replay": inputs["replay_identity"]["run_id"],
            "attribution": inputs["attribution_identity"]["run_id"],
        },
        "condition_grid_outputs": sum(len(v) for v in output_records.values()),
        "matrix_records": matrix_records, "replay_records": replay_records,
        "optimizer_records": optimizer_records,
        "model_state_delta": model_state_delta, "resources_observed": resource,
        "checks": checks, "zero_update_scope": {
            "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
            "new_sampling": 0, "one_shot_mapping_training": 0,
        },
    }
    return result


def _binomial_cdf_le(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_n_factorial = math.lgamma(n + 1)
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    total = 0.0
    for value in range(k + 1):
        log_mass = (log_n_factorial - math.lgamma(value + 1) - math.lgamma(n - value + 1)
                    + value * log_p + (n - value) * log_q)
        total += math.exp(log_mass)
    return min(1.0, max(0.0, total))


def _binomial_sf_ge(k: int, n: int, probability: float) -> float:
    if k <= 0:
        return 1.0
    if k > n or probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    log_n_factorial = math.lgamma(n + 1)
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    total = 0.0
    for value in range(k, n + 1):
        log_mass = (log_n_factorial - math.lgamma(value + 1) - math.lgamma(n - value + 1)
                    + value * log_p + (n - value) * log_q)
        total += math.exp(log_mass)
    return min(1.0, max(0.0, total))


def _clopper_pearson_one_sided(errors: int, denominator: int, confidence: float = 0.95) -> dict[str, float | None]:
    _require(denominator >= 0 and 0 <= errors <= denominator,
             "binomial count is inconsistent", ValueError)
    _require(0.0 < confidence < 1.0, "confidence must lie between zero and one", ValueError)
    if denominator == 0:
        return {"lower": None, "upper": None, "confidence": confidence}
    alpha = 1.0 - confidence
    if errors == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binomial_sf_ge(errors, denominator, mid) < alpha:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0
    if errors == denominator:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binomial_cdf_le(errors, denominator, mid) > alpha:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return {"lower": lower, "upper": upper, "confidence": confidence}


def _semantic_loss_classification(errors: int, denominator: int,
                                  allowed_loss: float = 0.05) -> dict[str, Any]:
    interval = _clopper_pearson_one_sided(errors, denominator, 0.95)
    if denominator == 0:
        decision = "UNDECIDABLE_NO_BASELINE_CORRECT_SAMPLES"
    elif interval["upper"] is not None and interval["upper"] <= allowed_loss:
        decision = "SUPPORTS_SEMANTIC_RETENTION"
    elif interval["lower"] is not None and interval["lower"] > allowed_loss:
        decision = "SUPPORTS_SEMANTIC_LOSS_ABOVE_LIMIT"
    else:
        decision = "INCONCLUSIVE"
    return {"errors": errors, "denominator": denominator,
            "observed_loss_fraction": None if denominator == 0 else errors / denominator,
            "one_sided_95_percent_clopper_pearson": interval,
            "allowed_loss": allowed_loss, "decision": decision}


def _round_condition(base_condition: str, round_index: int) -> str:
    return REVIEW_CONDITIONS[(REVIEW_CONDITIONS.index(base_condition) + round_index) % len(REVIEW_CONDITIONS)]


def _review_schedule(inputs: Mapping[str, Any]) -> dict[str, Any]:
    seed = 2026102601
    rng = np.random.default_rng(seed)
    g4_indices = [i for i, item in enumerate(inputs["identities"]) if item["method"] == "G4"]
    _require(len(g4_indices) == 100, "semantic review must use exactly the fixed G4 population")
    class_indices = {
        label: [i for i in g4_indices if int(inputs["identities"][i]["label"]) == label]
        for label in range(10)
    }
    _require(all(len(items) == 10 for items in class_indices.values()),
             "G4 semantic population has changed class denominators")
    repeated: list[int] = []
    for label in range(10):
        repeated.extend(int(i) for i in rng.choice(class_indices[label], size=2, replace=False))
    repeated = [int(i) for i in rng.permutation(repeated)]
    remaining = [i for i in g4_indices if i not in set(repeated)]
    remaining = [int(i) for i in rng.permutation(remaining)]
    identity_order = repeated + remaining
    duplicate_conditions: list[str] = []
    for condition, count in (("continuous", 7), ("A8_amplitude_linear", 7), ("I8_intensity_linear", 6)):
        duplicate_conditions.extend([condition] * count)
    duplicate_conditions = [str(x) for x in rng.permutation(duplicate_conditions)]
    repeat_condition = {index: condition for index, condition in zip(repeated, duplicate_conditions, strict=True)}

    base_conditions = {index: repeat_condition[index] for index in repeated}
    remaining_conditions: list[str] = []
    for condition, total in (("continuous", 34), ("A8_amplitude_linear", 33), ("I8_intensity_linear", 33)):
        remaining_conditions.extend([condition] * (total - sum(v == condition for v in base_conditions.values())))
    _require(len(remaining_conditions) == len(remaining), "review condition allocation does not sum to 100")
    for index, condition in zip(remaining, rng.permutation(remaining_conditions), strict=True):
        base_conditions[index] = str(condition)

    repeated_order = identity_order[:20]
    other_order = identity_order[20:]
    round_orders = [
        identity_order,
        [*repeated_order[5:] + repeated_order[:5], *np.roll(other_order, 33).tolist()],
        [*repeated_order[10:] + repeated_order[:10], *np.roll(other_order, 66).tolist()],
    ]
    cards: list[dict[str, Any]] = []
    source_card_by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for round_index, order in enumerate(round_orders):
        for index in order:
            identity = inputs["identities"][index]
            condition = _round_condition(base_conditions[index], round_index)
            card = {
                "kind": "generated", "condition": condition,
                "sample_key": str(identity["sample_key"]), "method": "G4",
                "requested_label": int(identity["label"]), "proposal_id": int(identity["proposal_id"]),
                "component_id": int(identity["component_id"]), "round": round_index + 1,
                "identity_group": str(identity["sample_key"]),
                "duplicate_group": None, "image_source": (index, condition),
            }
            cards.append(card)
            if round_index == 0:
                source_card_by_pair[(index, condition)] = card
    anchor_order = [dict(card) for card in inputs["anchor_cards"]]
    anchor_order = [anchor_order[int(i)] for i in rng.permutation(len(anchor_order))]
    # The twenty fixed real anchors are split between the latter two rounds; no new data rows are read.
    anchors_after_round_2 = anchor_order[:10]
    anchors_after_round_3 = anchor_order[10:]
    cards[100:100] = [
        {"kind": "real_anchor", "condition": None, "sample_key": None, "method": None,
         "requested_label": None, "proposal_id": None, "component_id": None, "round": None,
         "identity_group": None, "duplicate_group": None,
         "anchor_row": int(card["row"]), "target_label": int(card["target_label"]),
         "image_source": ("anchor", str(card["image"]))}
        for card in anchors_after_round_2
    ]
    cards[210:210] = [
        {"kind": "real_anchor", "condition": None, "sample_key": None, "method": None,
         "requested_label": None, "proposal_id": None, "component_id": None, "round": None,
         "identity_group": None, "duplicate_group": None,
         "anchor_row": int(card["row"]), "target_label": int(card["target_label"]),
         "image_source": ("anchor", str(card["image"]))}
        for card in anchors_after_round_3
    ]
    duplicate_records: list[dict[str, Any]] = []
    for duplicate_index, sample_index in enumerate(repeated):
        condition = repeat_condition[sample_index]
        source_card = source_card_by_pair[(sample_index, condition)]
        repeat_card = dict(source_card)
        repeat_card["kind"] = "hidden_duplicate"
        repeat_card["duplicate_group"] = f"D{duplicate_index + 1:02d}"
        source_card["duplicate_group"] = repeat_card["duplicate_group"]
        repeat_card["duplicate_group"] = source_card["duplicate_group"]
        repeat_card["image_source"] = source_card["image_source"]
        duplicate_records.append(repeat_card)
    cards.extend(duplicate_records)
    _require(len(cards) == 340, f"blind review card construction returned {len(cards)} cards")
    for position, card in enumerate(cards):
        card["card_id"] = f"R{position + 1:04d}"
        card["image"] = f"review_cards/card_{position + 1:04d}.png"
    identity_positions: dict[str, list[int]] = defaultdict(list)
    duplicate_positions: dict[str, list[int]] = defaultdict(list)
    for position, card in enumerate(cards):
        if card.get("identity_group"):
            identity_positions[str(card["identity_group"])].append(position)
        if card.get("duplicate_group"):
            duplicate_positions[str(card["duplicate_group"])].append(position)
    _require(len(identity_positions) == 100 and all(
        len(positions) == (4 if key in {str(inputs["identities"][i]["sample_key"]) for i in repeated} else 3)
        and min(b - a for a, b in zip(positions, positions[1:])) >= 40
        for key, positions in identity_positions.items()),
        "same-identity presentation separation is below forty cards")
    _require(len(duplicate_positions) == 20 and all(len(v) == 2 and v[1] - v[0] >= 40
                                                     for v in duplicate_positions.values()),
             "hidden duplicate card pairs are incomplete or too close")
    repeat_counts = Counter(repeat_condition.values())
    _require(repeat_counts == {"continuous": 7, "A8_amplitude_linear": 7, "I8_intensity_linear": 6},
             "hidden duplicate condition allocation changed")
    per_class = Counter(int(inputs["identities"][index]["label"]) for index in repeated)
    _require(all(per_class[label] == 2 for label in range(10)), "hidden duplicates are not two per class")
    return {
        "cards": cards, "generated_count": 300, "real_anchor_count": 20,
        "hidden_duplicate_count": 20, "total_count": len(cards),
        "duplicate_condition_counts": dict(repeat_counts),
        "hidden_duplicate_identities": [str(inputs["identities"][index]["sample_key"]) for index in repeated],
        "round_condition_counts": {
            str(round_index + 1): dict(Counter(card["condition"] for card in cards
                                               if card.get("round") == round_index + 1))
            for round_index in range(3)
        },
    }


def _review_image_bytes(inputs: Mapping[str, Any], card: Mapping[str, Any]) -> bytes:
    source = card["image_source"]
    if source[0] == "anchor":
        return (inputs["native_review_dir"] / source[1]).read_bytes()
    index, condition = source
    old = _old_output(inputs, int(index), str(condition), 2048)
    native = (old["raw_cmos_intensity_w_per_m2"].double()
              * float(old["camera_pixel_area_m2"]) * float(inputs["camera_gain"])).clamp(0.0, 1.0)
    pixels = np.rint(native.numpy() * 255.0).astype(np.uint8)
    _require(pixels.shape == IMAGE_SHAPE and pixels.min() >= 0 and pixels.max() <= 255,
             "blind review native CMOS export is malformed")
    from io import BytesIO
    buffer = BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _review_page(review_id: str, public_cards: Sequence[Mapping[str, str]]) -> str:
    public_json = json.dumps(list(public_cards), ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>固定光路输入盲评</title>
<style>body{{font:18px system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem;color:#18212b}}main{{text-align:center}}img{{width:400px;height:400px;object-fit:contain;image-rendering:pixelated;border:1px solid #999;background:#fff}}.choices{{display:grid;grid-template-columns:repeat(6,1fr);gap:.4rem;margin:1rem 0}}label{{padding:.5rem;border:1px solid #ccd1d1;border-radius:.4rem}}button{{font-size:1rem;padding:.6rem 1rem;margin:.3rem}}#save_status{{color:#405060}}</style>
<h1>固定光路输入盲评</h1><p>请依据图像本身选择 0–9；不能辨认时选择“不可辨认”。所有图像按固定原显示规则呈现，不做逐图增强。每 20 张为一段；评分与当前位置自动保存在本机浏览器中。</p>
<main><p id='progress'></p><img id='image' alt='匿名图像'><p id='segment_note'></p><div id='choices' class='choices'></div>
<button id='prev'>上一张</button><button id='next'>下一张</button><button id='export' disabled>完成 340 张后导出评分</button><p id='save_status'></p></main>
<script>
const REVIEW_ID={json.dumps(review_id)}, CARDS={public_json};
const STORE='lightgen-frontend-review-'+REVIEW_ID;let state={{}};let position=0;
try{{const saved=JSON.parse(localStorage.getItem(STORE)||'{{}}');state=saved.ratings||{{}};position=Math.min(Math.max(0,Number(saved.position)||0),CARDS.length-1);}}catch{{state={{}};position=0;}}
function persist(){{localStorage.setItem(STORE,JSON.stringify({{ratings:state,position}}));const segment=Math.floor(position/20)+1;const within=position%20+1;const complete=CARDS.filter(c=>Object.prototype.hasOwnProperty.call(state,c.card_id)).length;document.getElementById('progress').textContent=`第 ${{segment}} / 17 段 · 本段 ${{within}} / 20 张 · 总计 ${{position+1}} / 340 张 · 已完成 ${{complete}} 张`;document.getElementById('export').disabled=complete!==340;}}
function render(){{const c=CARDS[position];document.getElementById('image').src=c.image;document.getElementById('choices').innerHTML='';document.getElementById('save_status').textContent='';document.getElementById('segment_note').textContent=(position%20===19)?'本段结束，可休息；评分和位置已自动保存。':'';
const options=[...Array(10)].map((_,i)=>[String(i),String(i)]).concat([['unreadable','不可辨认']]);for(const [value,labelText] of options){{const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='rating';radio.value=value;radio.checked=String(state[c.card_id])===value;radio.onchange=()=>{{state[c.card_id]=value;persist();}};label.append(radio,document.createTextNode(' '+labelText));document.getElementById('choices').append(label);}}persist();}}
document.getElementById('prev').onclick=()=>{{position=Math.max(0,position-1);render();}};document.getElementById('next').onclick=()=>{{position=Math.min(CARDS.length-1,position+1);render();}};
 document.getElementById('export').onclick=()=>{{if(CARDS.some(c=>!Object.prototype.hasOwnProperty.call(state,c.card_id)))return;const cards=CARDS.map(c=>({{card_id:c.card_id,rating:state[c.card_id]==='unreadable'?'unreadable':Number(state[c.card_id])}}));const blob=new Blob([JSON.stringify({{schema:'lightgen_frozen_slm_8bit_frontend_ratings.v1',task_id:'{TASK_ID}',review_id:REVIEW_ID,submitted_at:new Date().toISOString(),cards}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_ratings_submitted.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.getElementById('save_status').textContent='评分文件已导出，请保留原文件。';}};
render();
</script></html>"""


def _create_review_packet(run_dir: Path, inputs: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    schedule = _review_schedule(inputs)
    cards_dir = run_dir / "review_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for card in schedule["cards"]:
        path = run_dir / card["image"]
        expected_bytes = _review_image_bytes(inputs, card)
        if path.is_file():
            _require(path.read_bytes() == expected_bytes, f"existing blind card differs from fixed source: {card['card_id']}")
        else:
            temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
            temp.write_bytes(expected_bytes)
            os.replace(temp, path)
        with Image.open(path) as image:
            _require(image.mode == "L" and image.size == IMAGE_SHAPE,
                     f"blind card is not a native 200x200 grayscale PNG: {card['card_id']}")
    key = {
        "schema": "lightgen_frozen_slm_8bit_frontend_review_key.v1", "task_id": TASK_ID,
        "run_id": run_id, "review_id": REVIEW_ID, "shuffle_seed": 2026102601,
        "reviewer_count": 1, "registered_card_count": 340,
        "generated_unique_card_count": 300, "real_anchor_count": 20,
        "hidden_duplicate_count": 20, "minimum_same_identity_card_distance": 40,
        "segment_size": 20, "segment_count": 17,
        "cards": schedule["cards"], "schedule_summary": {
            key: value for key, value in schedule.items() if key != "cards"
        },
    }
    key = json.loads(json.dumps(key, ensure_ascii=False))
    key_path = run_dir / "blind_review_key.json"
    if key_path.is_file():
        _require(_json(key_path) == key, "saved blind review key differs from deterministic fixed population")
    else:
        _atomic_json(key_path, key)
    public_cards = [{"card_id": card["card_id"], "image": card["image"]} for card in schedule["cards"]]
    page = _review_page(REVIEW_ID, public_cards)
    page_path = run_dir / "blind_review.html"
    if page_path.is_file():
        _require(page_path.read_text(encoding="utf-8") == page,
                 "existing blind review page differs from the fixed public card list")
    else:
        temp = page_path.with_name(page_path.name + f".tmp.{uuid.uuid4().hex}")
        temp.write_text(page, encoding="utf-8")
        os.replace(temp, page_path)
    lowered = page.lower()
    _require(not any(secret in lowered for secret in (
        "a8_amplitude_linear", "i8_intensity_linear", "sample_key", "requested_label",
        "component_id", "proposal_id", "real_anchor", "condition",
    )), "blind public page exposes hidden condition or identity metadata")
    manifest = {
        "schema": "lightgen_frozen_slm_8bit_frontend_review_public_manifest.v1",
        "task_id": TASK_ID, "run_id": run_id, "review_id": REVIEW_ID,
        "card_count": 340, "segment_size": 20, "segment_count": 17,
        "rating_values": list(range(10)) + ["unreadable"],
        "display_source_pixels": 200, "display_css_pixels": 400,
        "conditions_exposed": False, "identity_mapping_exposed": False,
    }
    manifest_path = run_dir / "blind_review_public_manifest.json"
    if manifest_path.is_file():
        _require(_json(manifest_path) == manifest, "review public manifest identity changed")
    else:
        _atomic_json(manifest_path, manifest)
    return {"review_id": REVIEW_ID, "card_count": 340, "segment_count": 17,
            "generated_unique_count": 300, "real_anchor_count": 20,
            "hidden_duplicate_count": 20, "blind_page": "blind_review.html",
            "hidden_key": "blind_review_key.json"}


def _rating_value(value: Any) -> int | str:
    if value == "unreadable":
        return value
    _require(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 9,
             f"invalid blind review rating: {value!r}", ValueError)
    return value


def _consume_ratings(run_dir: Path, ratings_path: Path) -> dict[str, Any]:
    raw = _json(ratings_path)
    _require(raw.get("schema") == "lightgen_frozen_slm_8bit_frontend_ratings.v1"
             and raw.get("task_id") == TASK_ID and raw.get("review_id") == REVIEW_ID,
             "ratings file does not belong to the fixed blind review", ValueError)
    key = _json(run_dir / "blind_review_key.json")
    cards = key.get("cards")
    _require(isinstance(cards, list) and len(cards) == 340, "saved private review key is incomplete")
    expected_ids = [str(card["card_id"]) for card in cards]
    _require(len(set(expected_ids)) == 340, "saved review key contains duplicate card identities")
    submitted = raw.get("cards")
    _require(isinstance(submitted, list) and len(submitted) == 340,
             "ratings submission must include exactly 340 cards", ValueError)
    rating_map: dict[str, int | str] = {}
    for row in submitted:
        _require(isinstance(row, dict) and set(row) == {"card_id", "rating"},
                 "rating rows must contain only card_id and rating", ValueError)
        card_id = row.get("card_id")
        _require(isinstance(card_id, str) and card_id in expected_ids and card_id not in rating_map,
                 "ratings contain a missing, extra, or duplicated card identity", ValueError)
        rating_map[card_id] = _rating_value(row.get("rating"))
    _require(set(rating_map) == set(expected_ids), "ratings do not cover the full blind card population", ValueError)
    rating_bytes = ratings_path.read_bytes()
    archived = run_dir / "human_ratings_submitted.json"
    if archived.resolve() == ratings_path.resolve():
        pass
    elif archived.exists():
        _require(archived.read_bytes() == rating_bytes,
                 "a different raw rating submission is already archived; preserve both explicitly")
    else:
        temp = archived.with_name(archived.name + f".tmp.{uuid.uuid4().hex}")
        temp.write_bytes(rating_bytes)
        os.replace(temp, archived)

    by_id = {card["card_id"]: card for card in cards}
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        group = card.get("duplicate_group")
        if group is not None:
            duplicate_groups[str(group)].append(str(card["card_id"]))
    _require(len(duplicate_groups) == 20 and all(len(ids) == 2 for ids in duplicate_groups.values()),
             "saved hidden repeat mapping is malformed")
    duplicate_matches = sum(rating_map[ids[0]] == rating_map[ids[1]] for ids in duplicate_groups.values())
    anchors = [card for card in cards if card.get("kind") == "real_anchor"]
    _require(len(anchors) == 20, "saved anchor mapping is malformed")
    anchor_correct = sum(rating_map[str(card["card_id"])] == int(card["target_label"]) for card in anchors)
    quality = {
        "duplicate_pairs": len(duplicate_groups), "duplicate_consistent_pairs": duplicate_matches,
        "duplicate_minimum": 18, "duplicate_qc_passed": duplicate_matches >= 18,
        "anchor_count": len(anchors), "anchor_correct": anchor_correct,
        "anchor_minimum": 18, "anchor_qc_passed": anchor_correct >= 18,
    }
    g4 = [card for card in cards if card.get("kind") == "generated" and card.get("method") == "G4"]
    _require(len(g4) == 300, "blind key must contain 100 unique G4 outputs per direct encoding condition")
    unique_map: dict[tuple[str, str], dict[str, Any]] = {}
    for card in g4:
        key_tuple = (str(card["sample_key"]), str(card["condition"]))
        _require(key_tuple not in unique_map, f"duplicate G4 condition card: {key_tuple}")
        unique_map[key_tuple] = card
    _require(len(unique_map) == 300, "G4 semantic identity-condition map is incomplete")
    labels_by_sample = {str(card["sample_key"]): int(card["requested_label"]) for card in g4}
    _require(len(labels_by_sample) == 100, "G4 semantic identity denominator changed")
    semantic_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    transitions: dict[str, Counter[tuple[str, str]]] = {condition: Counter() for condition in CONDITIONS}
    summaries: dict[str, dict[str, Any]] = {}
    quality_passed = quality["duplicate_qc_passed"] and quality["anchor_qc_passed"]
    for sample_key, label in sorted(labels_by_sample.items()):
        ratings = {
            condition: rating_map[str(unique_map[(sample_key, condition)]["card_id"])]
            for condition in REVIEW_CONDITIONS
        }
        row = {"sample_key": sample_key, "label": label,
               **{f"{condition}_rating": ratings[condition] for condition in REVIEW_CONDITIONS}}
        for condition in CONDITIONS:
            row[f"{condition}_transition"] = f"{ratings['continuous']}->{ratings[condition]}"
            transitions[condition][(str(ratings["continuous"]), str(ratings[condition]))] += 1
        semantic_rows.append(row)
    for condition in CONDITIONS:
        absolute_correct = sum(row[f"{condition}_rating"] == row["label"] for row in semantic_rows)
        unreadable = sum(row[f"{condition}_rating"] == "unreadable" for row in semantic_rows)
        baseline_correct = [row for row in semantic_rows if row["continuous_rating"] == row["label"]]
        losses = sum(row[f"{condition}_rating"] != row["label"] for row in baseline_correct)
        recovered = sum(row["continuous_rating"] != row["label"] and row[f"{condition}_rating"] == row["label"]
                        for row in semantic_rows)
        loss_decision = _semantic_loss_classification(losses, len(baseline_correct), 0.05)
        if not quality_passed:
            loss_decision["decision"] = "REVIEW_QUALITY_INVALID"
        class_summary: dict[str, Any] = {}
        for label in range(10):
            group = [row for row in semantic_rows if row["label"] == label]
            base_ok = [row for row in group if row["continuous_rating"] == label]
            class_loss = sum(row[f"{condition}_rating"] != label for row in base_ok)
            class_summary[str(label)] = {
                "denominator": len(group),
                "absolute_correct": sum(row[f"{condition}_rating"] == label for row in group),
                "unreadable": sum(row[f"{condition}_rating"] == "unreadable" for row in group),
                "continuous_correct_denominator": len(base_ok), "losses_from_continuous_correct": class_loss,
            }
            class_rows.append({"condition": condition, "label": label, **class_summary[str(label)]})
        summaries[condition] = {
            "population_denominator": 100, "absolute_correct": absolute_correct,
            "absolute_accuracy": absolute_correct / 100.0, "unreadable": unreadable,
            "unreadable_fraction": unreadable / 100.0, "recovered_from_continuous_incorrect": recovered,
            "continuous_correct_denominator": len(baseline_correct),
            "losses_from_continuous_correct": losses,
            "semantic_loss_decision": loss_decision,
            "per_class_descriptive_only": class_summary,
            "transition_counts": {f"{left}->{right}": count
                                  for (left, right), count in sorted(transitions[condition].items())},
        }
    semantic = {
        "schema": "lightgen_frozen_slm_8bit_frontend_semantic_results.v1",
        "task_id": TASK_ID, "review_id": REVIEW_ID, "ratings_count": len(rating_map),
        "quality_control": quality, "quality_passed": quality_passed,
        "conditions": summaries, "transitions_per_sample": semantic_rows,
        "per_class_rows": class_rows,
        "inference_scope": "fixed G4 population and one reviewer under a binomial working model",
        "optimized_outputs_were_not_reviewed": True,
    }
    return semantic


def _final_route(semantic: Mapping[str, Any], machine: Mapping[str, Any]) -> str:
    if not semantic.get("quality_passed"):
        return "REVIEW_QUALITY_INVALID_NO_SEMANTIC_CONCLUSION"
    decisions = [semantic["conditions"][condition]["semantic_loss_decision"]["decision"]
                 for condition in CONDITIONS]
    if len(set(decisions)) != 1 or any(value in {"INCONCLUSIVE", "UNDECIDABLE_NO_BASELINE_CORRECT_SAMPLES"}
                                       for value in decisions):
        return "ENCODING_DEPENDENT_OR_SEMANTIC_EVIDENCE_INCONCLUSIVE"
    all_witness = all(machine["conditions"][condition]["witness_count"] == 200 for condition in CONDITIONS)
    if decisions[0] == "SUPPORTS_SEMANTIC_RETENTION":
        return "DESIGN_ONE_SHOT_MAPPING_NEXT" if all_witness else "SEMANTIC_RETENTION_WITH_DISCRETE_PREPARATION_GAP"
    if decisions[0] == "SUPPORTS_SEMANTIC_LOSS_ABOVE_LIMIT":
        return "QUANTIZED_INPUT_MAPPING_PRIORITY" if all_witness else "FROZEN_DISCRETE_PREPARATION_FIRST"
    return "ENCODING_DEPENDENT_OR_SEMANTIC_EVIDENCE_INCONCLUSIVE"


def _research_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id,
        trajectory_id="C45k-frozen-two-layer-Gaussian-OLS-8bit-input-decision",
        semantic_authority_id=str(config["semantic_authority_id"]),
        data_role="fixed saved G1/G4 200 identities; G4 100 direct-output semantic cards; 20 old anchors",
        config_identity={"config": dict(config), "population": 200, "preflight": 20,
                         "A8_I8_coordinate_scans_max": MAX_SWEEPS},
        model_identity={"branch": "C", "completed_global_step": 45_000,
                        "encoder_layers": 2, "decoder_layers": 2, "OLS": "gaussian_s12_v1"},
        optimizer_identity={"model_updates": 0, "adam_calls": 0, "optical_optimizer": "not_instantiated"},
        sampler_identity={"source": "fixed closed attempt_01 populations", "new_sampling": 0},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt_01",
    )


def _ready_receipt(config: Mapping[str, Any], run_id: str, preflight: Mapping[str, Any]) -> PreflightReceipt:
    checks = preflight.get("checks", {})
    _require(preflight.get("status") == "READY" and checks
             and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()),
             "formal attempt requires a complete passing production preflight")
    control = ResearchRunControl(_research_identity(config, run_id), authorization_id=f"{TASK_ID}:authorization:01")
    return control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "checks": checks,
                     "fixed_population_and_original_counts": preflight["checks"]["fixed_closed_sources_and_original_200_2_0_counts"]["passed"],
                     "response_matrix_and_production_replay": preflight["checks"]["2048_matrix_prediction_le_1e-6"]["passed"]
                     and preflight["checks"]["frozen_reference_replay_20_by_three_conditions_by_two_grids"]["passed"]},
        resource={"passed": True, "status": "PASS", **preflight["resources_observed"]},
        implementation={"passed": True, "status": "PASS",
                        "selected_engine": "autodl-monthly-cuda-float64-complex128",
                        "frozen_production_forward": True, "optimizer": "deterministic-coordinate-descent"},
        selected_engine="autodl-monthly-cuda-float64-complex128",
        resource_envelope=preflight["resources_observed"],
    )


def _preflight_command(config: Mapping[str, Any]) -> dict[str, Any]:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    input_authority = _load_inputs(config)
    originals = _recompute_original_counts(input_authority)
    reach_inputs = _load_reachability_inputs(config, input_authority["identities"])
    identity_path = PREFLIGHT_DIR / "preflight_identity.json"
    if identity_path.is_file():
        identity = _json(identity_path)
        _require(identity.get("task_id") == TASK_ID and identity.get("phase") == "production_preflight"
                 and isinstance(identity.get("run_id"), str),
                 "preflight directory belongs to another task")
        run_id = str(identity["run_id"])
    else:
        run_id = uuid.uuid4().hex
        identity = {"schema": "lightgen_frozen_slm_8bit_frontend_preflight_identity.v1",
                    "task_id": TASK_ID, "run_id": run_id, "phase": "production_preflight",
                    "status": "PREPARING", "source_run_ids": {
                        "replay": config["source"]["replay_run_id"],
                        "reachability": config["source"]["reachability_run_id"],
                        "attribution": config["source"]["attribution_run_id"],
                    }}
        _atomic_json(identity_path, identity)
    if (PREFLIGHT_DIR / "ready_preflight_receipt.json").is_file():
        preflight = _json(PREFLIGHT_DIR / "preflight.json")
        stored_receipt = _json(PREFLIGHT_DIR / "ready_preflight_receipt.json")
        receipt = PreflightReceipt.from_mapping(stored_receipt)
        _require(receipt.status == PREFLIGHT_READY
                 and receipt.source_semantics_id == TASK_ID
                 and receipt.authorization_id == f"{TASK_ID}:authorization:01"
                 and preflight.get("status") == "READY"
                 and all(item.get("executed") is True and item.get("passed") is True
                         for item in preflight.get("checks", {}).values()),
                 "stored READY receipt or fixed preflight checks are incomplete")
        return {"status": "READY", "run_id": run_id, "preflight": preflight,
                "receipt": stored_receipt, "resumed_existing_ready_preflight": True}
    try:
        preflight = _run_preflight(config, input_authority, originals, reach_inputs, run_id)
        preflight["original_witness_counts_recomputed"] = originals["counts"]
        preflight["source_identity_count"] = len(input_authority["identities"])
        _atomic_json(PREFLIGHT_DIR / "preflight.json", preflight)
        receipt = _ready_receipt(config, run_id, preflight)
        control = ResearchRunControl(_research_identity(config, run_id), authorization_id=f"{TASK_ID}:authorization:01")
        control.mark_preflight_ready(receipt)
        _atomic_json(PREFLIGHT_DIR / "ready_preflight_receipt.json", receipt.to_dict())
        identity.update({"status": PREFLIGHT_READY, "ready_receipt": "ready_preflight_receipt.json"})
        _atomic_json(identity_path, identity)
        return {"status": "READY", "run_id": run_id, "preflight": preflight,
                "receipt": receipt.to_dict(), "resumed_existing_ready_preflight": False}
    except Exception as exc:
        identity.update({"status": "PREFLIGHT_BLOCKED", "failure_type": type(exc).__name__, "failure": str(exc)})
        _atomic_json(identity_path, identity)
        _atomic_json(PREFLIGHT_DIR / "engineering_failure.json", {
            "task_id": TASK_ID, "run_id": run_id, "status": "INVALID_HARNESS",
            "exception_type": type(exc).__name__, "message": str(exc),
            "scientific_conclusion": "none; preflight incomplete",
        })
        raise


def _open_attempt(config: Mapping[str, Any], preflight: Mapping[str, Any], run_id: str) -> ResearchRunControl:
    receipt_path = PREFLIGHT_DIR / "ready_preflight_receipt.json"
    _require(receipt_path.is_file(), "formal attempt is blocked until a READY preflight receipt exists")
    receipt = PreflightReceipt.from_mapping(_json(receipt_path))
    _require(receipt.status == PREFLIGHT_READY and receipt.authorization_id == f"{TASK_ID}:authorization:01",
             "saved preflight receipt is not READY for this task authorization")
    identity = _research_identity(config, run_id)
    if ATTEMPT_DIR.exists():
        old = _json(ATTEMPT_DIR / "run_identity.json") if (ATTEMPT_DIR / "run_identity.json").is_file() else {}
        _require(old.get("task_id") == TASK_ID and old.get("run_id") == run_id,
                 "formal attempt output path contains another run identity")
        _require(old.get("status") != "CLOSED", "closed scientific attempt cannot be reopened")
        control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01",
                                     preflight_status=PREFLIGHT_READY, state=RUN_INTERRUPTED,
                                     receipt=receipt)
        control.open_scientific_attempt(ATTEMPT_DIR, f"{TASK_ID}:attempt_01", receipt, resume=True)
        old["status"] = "SCIENTIFIC_ACTIVE"
        _atomic_json(ATTEMPT_DIR / "run_identity.json", old)
    else:
        control = ResearchRunControl(identity, authorization_id=f"{TASK_ID}:authorization:01")
        control.mark_preflight_ready(receipt)
        control.start_execution()
        control.open_scientific_attempt(ATTEMPT_DIR, f"{TASK_ID}:attempt_01", receipt, resume=False)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", {
            "schema": "lightgen_frozen_slm_8bit_frontend_run_identity.v1", "task_id": TASK_ID,
            "run_id": run_id, "status": "SCIENTIFIC_ACTIVE", "evidence_tier": "diagnostic_only",
            "preflight_receipt": "../preflight_01/ready_preflight_receipt.json",
            "checkpoint_step": 45_000, "methods": list(METHODS), "sample_count": 200,
            "conditions": list(CONDITIONS), "grids": list(GRIDS),
            "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
            "new_sampling": 0, "one_shot_mapping_training": 0,
            "source_run_ids": dict(preflight.get("source_run_ids", {})),
        })
    _atomic_json(ATTEMPT_DIR / "lifecycle_receipt.json", receipt.to_dict())
    return control


def _validate_final_output(path: Path, run_id: str, identity: Mapping[str, Any],
                           condition: str, grid: int) -> dict[str, Any]:
    _require(path.is_file(), f"final full-path output is missing: {condition}/{grid}/{identity['sample_key']}",
             FileNotFoundError)
    output = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(output, dict)
             and output.get("schema") == "lightgen_frozen_slm_8bit_frontend_output.v1"
             and output.get("task_id") == TASK_ID and output.get("run_id") == run_id
             and output.get("condition") == condition and output.get("grid") == grid
             and output.get("sample_identity") == dict(identity),
             f"full-path output identity is invalid: {condition}/{grid}/{identity['sample_key']}")
    coeff = output.get("actual_coefficients")
    raw = output.get("raw_cmos_intensity_w_per_m2")
    _require(isinstance(coeff, torch.Tensor) and coeff.dtype == torch.complex128
             and tuple(coeff.shape) == (PORTS,)
             and isinstance(raw, torch.Tensor) and tuple(raw.shape) == IMAGE_SHAPE
             and torch.isfinite(coeff.real).all().item() and torch.isfinite(coeff.imag).all().item()
             and torch.isfinite(raw).all().item() and float(raw.min()) >= 0.0,
             f"full-path output contains malformed or nonfinite tensors: {condition}/{grid}/{identity['sample_key']}")
    powers = output.get("power_trace_w")
    _require(isinstance(powers, dict) and set(powers) == set(POWER_FIELDS)
             and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in powers.values())
             and isinstance(output.get("camera_pixel_area_m2"), (int, float))
             and math.isfinite(float(output["camera_pixel_area_m2"]))
             and float(output["camera_pixel_area_m2"]) > 0.0
             and isinstance(output.get("camera_total_power_w"), (int, float))
             and math.isfinite(float(output["camera_total_power_w"]))
             and float(output["camera_total_power_w"]) >= 0.0,
             f"full-path output is missing absolute plane powers: {condition}/{grid}/{identity['sample_key']}")
    return output


def _save_commands(run_dir: Path, inputs: Mapping[str, Any], run_id: str,
                   gray: Mapping[str, np.ndarray], optimizer_records: Mapping[str, list[dict[str, Any]]]) -> None:
    identities = inputs["identities"]
    a8 = np.asarray(gray["A8_amplitude_linear"], dtype=np.uint8)
    i8 = np.asarray(gray["I8_intensity_linear"], dtype=np.uint8)
    _require(a8.shape == i8.shape == (200, PIXEL_COUNT), "final command population has the wrong shape")
    amp_a = _decode_gray(a8, "A8_amplitude_linear")
    amp_i = _decode_gray(i8, "I8_intensity_linear")
    _atomic_torch(run_dir / "final_gray_and_decoded_amplitudes.pt", {
        "schema": "lightgen_frozen_slm_8bit_frontend_commands.v1", "task_id": TASK_ID,
        "run_id": run_id, "sample_identities": identities,
        "A8_gray_u8": torch.from_numpy(a8.reshape(200, 1, *IMAGE_SHAPE).copy()),
        "I8_gray_u8": torch.from_numpy(i8.reshape(200, 1, *IMAGE_SHAPE).copy()),
        "A8_decoded_amplitude_float64": torch.from_numpy(amp_a.reshape(200, 1, *IMAGE_SHAPE).copy()),
        "I8_decoded_amplitude_float64": torch.from_numpy(amp_i.reshape(200, 1, *IMAGE_SHAPE).copy()),
        "A8_production_intensity_is_amplitude_squared": True,
        "I8_production_intensity_is_gray_divided_by_255": True,
        "optimizer_records": {condition: optimizer_records[condition] for condition in CONDITIONS},
    })


def _validate_identity_population(expected: Sequence[Mapping[str, Any]],
                                  observed: Sequence[Mapping[str, Any]], expected_count: int = 200) -> None:
    expected_keys = [str(item.get("sample_key")) for item in expected]
    observed_keys = [str(item.get("sample_key")) for item in observed]
    _require(len(expected) == len(observed) == expected_count
             and len(set(expected_keys)) == expected_count
             and len(set(observed_keys)) == expected_count
             and observed_keys == expected_keys
             and list(expected) == list(observed),
             "saved sample identity population has a missing, duplicated, or changed row", ValueError)


def _validate_decoded_command(codes: np.ndarray | torch.Tensor,
                              amplitude: np.ndarray | torch.Tensor, condition: str) -> None:
    decoded = _decode_gray(codes, condition)
    observed = np.asarray(amplitude)
    _require(observed.dtype == np.float64 and observed.shape == decoded.shape
             and np.isfinite(observed).all() and np.array_equal(decoded, observed),
             f"saved decoded amplitude differs from uint8 gray command: {condition}", ValueError)


def _load_commands(run_dir: Path, inputs: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    path = run_dir / "final_gray_and_decoded_amplitudes.pt"
    _require(path.is_file(), "saved final gray command bundle is missing", FileNotFoundError)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    identities = inputs["identities"]
    _require(bundle.get("schema") == "lightgen_frozen_slm_8bit_frontend_commands.v1"
             and bundle.get("task_id") == TASK_ID and bundle.get("run_id") == run_id,
             "saved final gray command bundle has a wrong identity or population")
    observed_identities = bundle.get("sample_identities")
    _require(isinstance(observed_identities, list), "saved command identity list is missing")
    _validate_identity_population(identities, observed_identities)
    for code_key, amplitude_key, condition in (
        ("A8_gray_u8", "A8_decoded_amplitude_float64", "A8_amplitude_linear"),
        ("I8_gray_u8", "I8_decoded_amplitude_float64", "I8_intensity_linear"),
    ):
        codes = bundle.get(code_key)
        amplitude = bundle.get(amplitude_key)
        _require(isinstance(codes, torch.Tensor) and codes.dtype == torch.uint8
                 and tuple(codes.shape) == (200, 1, *IMAGE_SHAPE)
                 and isinstance(amplitude, torch.Tensor) and amplitude.dtype == torch.float64
                 and tuple(amplitude.shape) == (200, 1, *IMAGE_SHAPE),
                 f"saved command has wrong type or shape: {code_key}")
        _validate_decoded_command(codes.numpy().reshape(200, PIXEL_COUNT),
                                  amplitude.numpy().reshape(200, PIXEL_COUNT), condition)
    return bundle


def _output_path(run_dir: Path, condition: str, grid: int, sample_key: str) -> Path:
    return run_dir / "samples" / condition / str(grid) / f"{sample_key}.pt"


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty research table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0 and np.isfinite(array).all(), "statistics require finite nonempty values")
    return {"median": float(np.median(array)), "p95": float(np.quantile(array, 0.95)),
            "maximum": float(np.max(array))}


def _machine_consumer(run_dir: Path, inputs: Mapping[str, Any], run_id: str,
                      originals: Mapping[str, Any]) -> dict[str, Any]:
    commands = _load_commands(run_dir, inputs, run_id)
    conditions = {
        "A8_amplitude_linear": commands["A8_gray_u8"].numpy().reshape(200, PIXEL_COUNT),
        "I8_intensity_linear": commands["I8_gray_u8"].numpy().reshape(200, PIXEL_COUNT),
    }
    saved_optimizer_records = commands.get("optimizer_records")
    _require(isinstance(saved_optimizer_records, dict), "saved command bundle has no optimizer records")
    optimizer_by_condition: dict[str, dict[str, Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        rows = saved_optimizer_records.get(condition)
        _require(isinstance(rows, list) and len(rows) == 200,
                 f"saved optimizer records have the wrong denominator: {condition}")
        by_key = {str(row.get("sample_key")): row for row in rows}
        _require(len(by_key) == 200 and set(by_key) == {str(item["sample_key"]) for item in inputs["identities"]},
                 f"saved optimizer records contain missing or duplicate identities: {condition}")
        optimizer_by_condition[condition] = by_key
    matrix = inputs["matrix"].numpy()
    target = inputs["target"].numpy()
    metrics: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    power_rows: list[dict[str, Any]] = []
    max_matrix_error = 0.0
    max_cross_grid = 0.0
    max_power_rel = 0.0
    max_power_abs = 0.0
    power_invariants_passed = True
    output_count = 0
    for condition in CONDITIONS:
        folder_by_grid = {
            grid: sorted(path.name for path in (run_dir / "samples" / condition / str(grid)).glob("*.pt"))
            for grid in GRIDS
        }
        for grid in GRIDS:
            expected_names = sorted(f"{identity['sample_key']}.pt" for identity in inputs["identities"])
            _require(folder_by_grid[grid] == expected_names,
                     f"saved full-path outputs do not contain exactly 200 fixed identities: {condition}/{grid}")
        for index, identity in enumerate(inputs["identities"]):
            codes = conditions[condition][index]
            amplitude = _decode_gray(codes, condition)
            _require(np.isfinite(amplitude).all() and amplitude.min() >= 0.0 and amplitude.max() <= 1.0,
                     f"saved gray command is outside [0,1]: {condition}/{identity['sample_key']}")
            matrix_prediction = amplitude @ matrix.T
            final_matrix_target_error = _phase_error(matrix_prediction, target[index])[1]
            optimizer_row = optimizer_by_condition[condition][str(identity["sample_key"])]
            coeff_errors: list[float] = []
            cmos_errors: list[float] = []
            output_by_grid: dict[int, dict[str, Any]] = {}
            matrix_error = 0.0
            for grid in GRIDS:
                output = _validate_final_output(_output_path(run_dir, condition, grid, identity["sample_key"]),
                                                run_id, identity, condition, grid)
                output_by_grid[grid] = output
                output_count += 1
                target_coeff = target[index]
                coeff_errors.append(_phase_error(output["actual_coefficients"], target_coeff)[1])
                cmos_error = _relative(output["raw_cmos_intensity_w_per_m2"],
                                       inputs["direct_cmos"][str(grid)][index, 0])
                cmos_errors.append(cmos_error)
                if grid == 2048:
                    matrix_error = _relative(matrix_prediction, output["actual_coefficients"].numpy())
                    max_matrix_error = max(max_matrix_error, matrix_error)
                intensity = _intensity_from_amplitude(amplitude)
                power_check = attribution._power_checks(
                    inputs, output, torch.from_numpy(intensity), grid, str(identity["sample_key"]),
                )
                max_power_rel = max(max_power_rel, power_check["analytic_slm_power_relative_error"],
                                    power_check["camera_integral_power_relative_error"])
                max_power_abs = max(max_power_abs,
                                    abs(power_check["P_after_SLM_W"] - power_check["P_after_SLM_expected_W"]),
                                    abs(power_check["P_CMOS_integrated_W"] - power_check["P_CMOS_trace_W"]))
                power_invariants_passed &= (
                    (power_check["analytic_slm_power_relative_error"] <= POWER_RELATIVE_LIMIT
                     or abs(power_check["P_after_SLM_W"] - power_check["P_after_SLM_expected_W"])
                     <= POWER_ABSOLUTE_LIMIT_W)
                    and (power_check["camera_integral_power_relative_error"] <= POWER_RELATIVE_LIMIT
                         or abs(power_check["P_CMOS_integrated_W"] - power_check["P_CMOS_trace_W"])
                         <= POWER_ABSOLUTE_LIMIT_W)
                )
                power_rows.append({
                    "condition": condition, "sample_key": identity["sample_key"],
                    "method": identity["method"], "label": int(identity["label"]), "grid": grid,
                    "analytic_slm_power_relative_error": power_check["analytic_slm_power_relative_error"],
                    "camera_integral_power_relative_error": power_check["camera_integral_power_relative_error"],
                    **output["power_trace_w"],
                })
            cross_coeff = _relative(output_by_grid[2048]["actual_coefficients"],
                                    output_by_grid[4096]["actual_coefficients"])
            cross_cmos = _relative(output_by_grid[2048]["raw_cmos_intensity_w_per_m2"],
                                   output_by_grid[4096]["raw_cmos_intensity_w_per_m2"])
            max_cross_grid = max(max_cross_grid, cross_coeff, cross_cmos)
            original = _old_output(inputs, index, condition, 2048)
            initial_amplitude = _decode_gray(inputs["gray_a" if condition.startswith("A8") else "gray_i"][index].numpy(),
                                             condition)
            initial_prediction = initial_amplitude @ matrix.T
            initial_phase = float(np.angle(np.vdot(target[index], initial_prediction)))
            initial_matrix_error = float(np.linalg.norm(initial_prediction
                                                        - np.exp(1j * initial_phase) * target[index])
                                         / np.linalg.norm(target[index]))
            changed_pixels = int(np.count_nonzero(
                conditions[condition][index]
                != inputs["gray_a" if condition.startswith("A8") else "gray_i"][index].numpy()
            ))
            expected_visits = optimizer_row.get("coordinate_visits")
            expected_scans = optimizer_row.get("full_scans")
            _require(isinstance(expected_visits, int) and 0 <= expected_visits <= MAX_SWEEPS * PIXEL_COUNT
                     and isinstance(expected_scans, int)
                     and expected_scans == math.ceil(expected_visits / PIXEL_COUNT)
                     and optimizer_row.get("gray_pixels_changed") == changed_pixels
                     and math.isclose(float(optimizer_row.get("initial_matrix_coefficient_error", math.nan)),
                                      initial_matrix_error, rel_tol=1.0e-10, abs_tol=1.0e-12)
                     and math.isclose(float(optimizer_row.get("final_matrix_coefficient_error", math.nan)),
                                      final_matrix_target_error, rel_tol=1.0e-10, abs_tol=1.0e-12)
                     and optimizer_row.get("termination") in {
                         "source_verified_original_witness", "coefficient_tolerance",
                         "full_sweep_no_gray_change", "specified_scan_budget_exhausted",
                     }
                     and isinstance(optimizer_row.get("skipped_input"), bool),
                     f"saved optimizer evidence differs from independently recomputed values: {condition}/{identity['sample_key']}")
            initial_coeff_error = max(
                _phase_error(_old_output(inputs, index, condition, grid)["actual_coefficients"], target[index])[1]
                for grid in GRIDS
            )
            initial_cmos_error = max(
                _relative(_old_output(inputs, index, condition, grid)["raw_cmos_intensity_w_per_m2"],
                          inputs["direct_cmos"][str(grid)][index, 0])
                for grid in GRIDS
            )
            witness = (max(coeff_errors) <= WITNESS_LIMIT and max(cmos_errors) <= WITNESS_LIMIT
                       and max(cross_coeff, cross_cmos) <= WITNESS_LIMIT)
            metrics[condition].append({
                "sample_key": identity["sample_key"], "method": identity["method"],
                "label": int(identity["label"]), "target_coefficient_norm": float(np.linalg.norm(target[index])),
                "initial_matrix_coefficient_error": initial_matrix_error,
                "final_matrix_coefficient_error": final_matrix_target_error,
                "coordinate_visits": expected_visits, "full_scans": expected_scans,
                "optimizer_termination": optimizer_row["termination"],
                "initial_full_path_coefficient_error": initial_coeff_error,
                "initial_native_cmos_error": initial_cmos_error,
                "final_matrix_production_closure_error": matrix_error,
                "final_coefficient_error_max_two_grids": max(coeff_errors),
                "final_native_cmos_error_max_two_grids": max(cmos_errors),
                "cross_grid_coefficient_error": cross_coeff,
                "cross_grid_native_cmos_error": cross_cmos,
        "A8_or_I8_gray_pixels_changed": int(np.count_nonzero(
                    conditions[condition][index] != inputs["gray_a" if condition.startswith("A8") else "gray_i"][index].numpy()
                )),
                "original_direct_witness": bool(originals["conditions"][condition]["rows"][index]["witness"]),
                "witness": witness,
            })
    _require(output_count == 800, f"final consumer recomputed {output_count} full-path outputs instead of 800")
    _require(max_matrix_error <= MATRIX_LIMIT,
             f"final 2048 response matrix closure exceeded {MATRIX_LIMIT}: {max_matrix_error:.3g}")
    _require(max_cross_grid <= WITNESS_LIMIT,
             f"final two-grid numerical window exceeded {WITNESS_LIMIT}: {max_cross_grid:.3g}")
    _require(power_invariants_passed,
             "saved SLM/CMOS absolute-power evidence failed independent recomputation")
    summaries: dict[str, Any] = {}
    for condition, rows in metrics.items():
        by_method: dict[str, Any] = {}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            class_stats: dict[str, Any] = {}
            for label in range(10):
                group = [row for row in method_rows if row["label"] == label]
                _require(len(group) == 10, f"final numerical class denominator changed: {condition}/{method}/{label}")
                class_stats[str(label)] = {
                    "denominator": len(group), "witness_count": sum(row["witness"] for row in group),
                    "initial_matrix_coefficient_error": _stats([row["initial_matrix_coefficient_error"] for row in group]),
                    "final_matrix_coefficient_error": _stats([row["final_matrix_coefficient_error"] for row in group]),
                    "gray_pixels_changed": _stats([row["A8_or_I8_gray_pixels_changed"] for row in group]),
                    "coordinate_visits": _stats([row["coordinate_visits"] for row in group]),
                    "coefficient_error": _stats([row["final_coefficient_error_max_two_grids"] for row in group]),
                    "native_cmos_error": _stats([row["final_native_cmos_error_max_two_grids"] for row in group]),
                }
            by_method[method] = {
                "denominator": len(method_rows), "witness_count": sum(row["witness"] for row in method_rows),
                "initial_matrix_coefficient_error": _stats([row["initial_matrix_coefficient_error"] for row in method_rows]),
                "final_matrix_coefficient_error": _stats([row["final_matrix_coefficient_error"] for row in method_rows]),
                "gray_pixels_changed": _stats([row["A8_or_I8_gray_pixels_changed"] for row in method_rows]),
                "coordinate_visits": _stats([row["coordinate_visits"] for row in method_rows]),
                "coefficient_error": _stats([row["final_coefficient_error_max_two_grids"] for row in method_rows]),
                "native_cmos_error": _stats([row["final_native_cmos_error_max_two_grids"] for row in method_rows]),
                "per_class": class_stats,
            }
        summaries[condition] = {
            "denominator": len(rows), "witness_count": sum(row["witness"] for row in rows),
            "coefficient_error": _stats([row["final_coefficient_error_max_two_grids"] for row in rows]),
            "native_cmos_error": _stats([row["final_native_cmos_error_max_two_grids"] for row in rows]),
            "initial_witness_count_recomputed": sum(row["original_direct_witness"] for row in rows),
            "per_method": by_method,
        }
    result = {
        "schema": "lightgen_frozen_slm_8bit_frontend_machine_results.v1", "task_id": TASK_ID,
        "run_id": run_id, "status": "MACHINE_MEASUREMENT_COMPLETE",
        "source_run_ids": {
            "reachability": inputs["reach_identity"]["run_id"],
            "replay": inputs["replay_identity"]["run_id"],
            "attribution": inputs["attribution_identity"]["run_id"],
        },
        "sample_denominator_per_condition": 200, "condition_grid_output_count": output_count,
        "original_recomputed_witness_counts": {condition: originals["conditions"][condition]["passed"]
                                               for condition in REVIEW_CONDITIONS},
        "conditions": summaries, "maximum_matrix_prediction_relative_error": max_matrix_error,
        "maximum_cross_grid_relative_error": max_cross_grid,
        "maximum_power_relative_error": max_power_rel, "maximum_power_absolute_error_w": max_power_abs,
        "power_table_rows": len(power_rows), "zero_model_updates": True,
        "all_rows": metrics,
    }
    _atomic_json(run_dir / "machine_results.json", result)
    _csv_write(run_dir / "per_sample_numerical_metrics.csv",
               [row for condition in CONDITIONS for row in metrics[condition]])
    _csv_write(run_dir / "per_sample_optimizer_metrics.csv",
               [row for condition in CONDITIONS for row in saved_optimizer_records[condition]])
    _csv_write(run_dir / "per_plane_power_table.csv", power_rows)
    return result


def _write_calibration_preparation(run_dir: Path) -> None:
    content = """# 实际器件标定准备（本轮未执行实物测量）

本轮 A8 与 I8 是理想线性编码情景。当前没有可追溯的 HES6001 实际灰阶响应曲线，因此不能把任一情景称为器件实测模型。

## 待补设备与测量

| 项目 | 最低记录内容 | 缺失时失去的保证 |
|---|---|---|
| 灰阶—强度 LUT | 每个灰阶的归一化反射强度、曝光/相机线性化、照明与偏振条件、拟合与重复测量 | A8/I8 理想编码不能代表真实灰阶响应 |
| 残余相位 | 各灰阶相位变化、偏振、入射角、波长、空间采样和不确定度 | 纯振幅假设及复系数预测不成立 |
| 空间差异 | 全屏/项目有效区域的灰阶强度与相位平场，像素/区域位置 | 单一全局 LUT 无法覆盖像素非均匀性 |
| 时间与重复性 | 上升/下降、帧间、温度、刷新等待时间与多次重复 | 无法给出重复性、稳定时间或帧率保证 |
| OLS 端口复传输 | 每个输入端口到全部再发射端口的复振幅、相位参考、损耗、串扰、功率标定 | 数值 Gaussian OLS 不能替代真实端口算子 |
| 相机读出 | 暗场、平场、线性区、曝光、增益、饱和、像素响应和重复性 | 原生 CMOS 强度比较不能转为实机读数容差 |
| 几何与对准 | SLM 像素映射、relay 放大/畸变、孔径、相对位置、偏振和波前 | 当前物理坐标下的仿真输入不可直接迁移 |

## 建议的测量顺序

1. 固定波长、偏振、入射角、曝光和光束条件，先测暗场/平场及相机响应线性区。
2. 对项目有效区域以多个重复序列采集 256 灰阶，分别记录上升和下降扫描的强度；保存原始帧、ROI、曝光、时间戳和校准不确定度。
3. 用干涉或等效相位测量同步得到灰阶相关相位与空间差异，并记录测量装置的相位参考。
4. 对 OLS 的每个受控输入端口逐一激励并测量全部输出端口复场；保留绝对幅度、相位参考、损耗和串扰，不用单位矩阵代替实测。
5. 独立重复测量与相机响应核验后，冻结校准版本；再将该 LUT/复传输作为新的设备参数输入，重新设计面向实际器件的数值诊断。

本轮没有启动实物测量、校准拟合或设备部署；任何 LUT、残余相位或端口复传输仍为缺失输入。
"""
    path = run_dir / "calibration_preparation.md"
    if path.is_file():
        _require(path.read_text(encoding="utf-8") == content,
                 "calibration preparation document changed across resume")
    else:
        path.write_text(content, encoding="utf-8")


def _native_display(raw: torch.Tensor, pixel_area: float, gain: float) -> np.ndarray:
    view = (raw.double() * pixel_area * gain).clamp(0.0, 1.0).numpy()
    _require(view.shape == IMAGE_SHAPE and np.isfinite(view).all(), "native CMOS display transform is invalid")
    return np.rint(view * 255.0).astype(np.uint8)


def _write_comparison_atlas(run_dir: Path, inputs: Mapping[str, Any], selected: Sequence[int]) -> None:
    width, height = 300, 120
    image = Image.new("RGB", (width * 5, height * 4), "white")
    draw = ImageDraw.Draw(image)
    for position, index in enumerate(selected):
        identity = inputs["identities"][index]
        col, row = position % 5, position // 5
        left, top = col * width, row * height
        reference = inputs["direct_cmos"]["2048"][index, 0]
        ref_output = _old_output(inputs, index, "continuous", 2048)
        reference_px = _native_display(reference, float(ref_output["camera_pixel_area_m2"]),
                                       float(inputs["camera_gain"]))
        original = _old_output(inputs, index, "A8_amplitude_linear", 2048)
        final = torch.load(_output_path(run_dir, "A8_amplitude_linear", 2048,
                                       str(identity["sample_key"])), map_location="cpu", weights_only=False)
        panels = [
            (reference_px, "OLS direct reference"),
            (_native_display(original["raw_cmos_intensity_w_per_m2"], float(original["camera_pixel_area_m2"]),
                             float(inputs["camera_gain"])), "A8 original code"),
            (_native_display(final["raw_cmos_intensity_w_per_m2"], float(final["camera_pixel_area_m2"]),
                             float(inputs["camera_gain"])), "A8 optimized code"),
        ]
        draw.text((left + 3, top + 1), f"{identity['sample_key']} · label {identity['label']}", fill="black")
        for panel_index, (pixels, title) in enumerate(panels):
            x, y = left + 4 + panel_index * 96, top + 22
            thumb = Image.fromarray(pixels, mode="L").resize((88, 88), Image.Resampling.NEAREST).convert("RGB")
            image.paste(thumb, (x, y))
            draw.text((x, top + 108), title, fill="black")
    path = run_dir / "comparison_atlas_fixed20.png"
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    image.save(temp, format="PNG", optimize=True)
    os.replace(temp, path)


def _write_report(run_dir: Path, originals: Mapping[str, Any], machine: Mapping[str, Any],
                  semantic: Mapping[str, Any] | None, task_status: str,
                  route: str | None = None) -> None:
    original_counts = originals["counts"]
    numerical_lines = []
    for condition in CONDITIONS:
        summary = machine["conditions"][condition]
        numerical_lines.append(
            f"- `{condition}`：离散见证 {summary['witness_count']}/200；"
            f"G1 {summary['per_method']['G1']['witness_count']}/100，G4 {summary['per_method']['G4']['witness_count']}/100；"
            f"系数误差中位数/P95/最大值 {summary['coefficient_error']['median']:.4g}/"
            f"{summary['coefficient_error']['p95']:.4g}/{summary['coefficient_error']['maximum']:.4g}；"
            f"CMOS 强度误差 {summary['native_cmos_error']['median']:.4g}/"
            f"{summary['native_cmos_error']['p95']:.4g}/{summary['native_cmos_error']['maximum']:.4g}。"
        )
    class_lines = [
        "| 编码 | 方法 | 类别 | 分母 | 见证 | 改动灰阶像素 median/P95/max | 矩阵残差初始 median → 最终 median/P95/max | 系数误差 median/P95/max | CMOS 强度误差 median/P95/max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for method in METHODS:
            for label in range(10):
                row = machine["conditions"][condition]["per_method"][method]["per_class"][str(label)]
                initial = row["initial_matrix_coefficient_error"]["median"]
                final_matrix = row["final_matrix_coefficient_error"]
                changed = row["gray_pixels_changed"]
                coefficient = row["coefficient_error"]
                cmos = row["native_cmos_error"]
                class_lines.append(
                    f"| {condition} | {method} | {label} | {row['denominator']} | {row['witness_count']} | "
                    f"{changed['median']:.0f}/{changed['p95']:.0f}/{changed['maximum']:.0f} | "
                    f"{initial:.4g}/{final_matrix['median']:.4g}/{final_matrix['p95']:.4g}/{final_matrix['maximum']:.4g} | "
                    f"{coefficient['median']:.4g}/{coefficient['p95']:.4g}/{coefficient['maximum']:.4g} | "
                    f"{cmos['median']:.4g}/{cmos['p95']:.4g}/{cmos['maximum']:.4g} |"
                )
    if semantic is None:
        semantic_text = "人工盲评尚未提交；页面共 340 张，进度与评分保存在评阅者本机浏览器。当前不得形成语义结论。"
        semantic_decisions = "待评阅。"
    else:
        semantic_text = f"盲评质控：{'通过' if semantic['quality_passed'] else '失败'}；隐藏重复一致 {semantic['quality_control']['duplicate_consistent_pairs']}/20，真实参照正确 {semantic['quality_control']['anchor_correct']}/20。"
        lines = []
        for condition in CONDITIONS:
            row = semantic["conditions"][condition]
            loss = row["semantic_loss_decision"]
            interval = loss["one_sided_95_percent_clopper_pearson"]
            interval_text = "未定义" if interval["upper"] is None else f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
            lines.append(f"- `{condition}`：绝对正确 {row['absolute_correct']}/100，不可辨认 {row['unreadable']}/100；连续正确子集损失 {loss['errors']}/{loss['denominator']}，单侧95% CP区间 {interval_text}，判断 `{loss['decision']}`。")
        semantic_decisions = "\n".join(lines)
    route = route or "WAIT_FOR_HUMAN_RATINGS_BEFORE_FINAL_ROUTE"
    report = f"""# 冻结 8 位前端：语义保留与离散制备联合推进

## 结果与状态

- 当前状态：`{task_status}`。固定输入为 C 累计 45,000 步、G1/G4 各 100 条；模型更新、Adam、先验拟合、新采样、一次映射训练均为 0。
- 原始证据由保存输出重新计算：连续 {original_counts['continuous']}/200、A8 {original_counts['A8_amplitude_linear']}/200、I8 {original_counts['I8_intensity_linear']}/200。
{chr(10).join(numerical_lines)}
- 下表按编码、G1/G4 与类别列出分母、数值见证、2048 矩阵残差变化和完整光路误差。
{chr(10).join(class_lines)}
- {semantic_text}
- 当前综合路线：`{route}`。

## 科学认识

- 离散输入优化使用已保存的 2048 complex128 响应矩阵，保留目标绝对幅度，仅允许一个公共相位；完整光路由保存灰阶重放到 CMOS。
- 量化优化后的输出没有进入本轮人工盲评，因此数值制备见证不等于语义保持。
- {semantic_decisions}
- A8/I8 是两种理想线性校准情景。实际 LUT、残余相位、空间响应、OLS 端口复传输和相机校准仍缺失；不据此宣称硬件可达或器件性能。

## 目标缺口

- 本轮未测真实 SLM 灰阶响应、残余相位、实测 OLS 复传输或相机噪声/线性区。
- 单次前向输入映射尚未训练或验证；本轮结果只用于决定是否值得另行设计该工作。
- 语义统计仅覆盖固定 G4 100 条和一名评阅者，并依赖预设二项工作模型；不外推新人口或实物系统。

## 工程完成与验证

- 固定 20 条执行预检通过后，使用 AutoDL 冻结生产模型完成数值任务；连续/A8/I8 参考链、响应矩阵闭合、功率积分、模型状态、人口身份和保存输出由最终消费者重算。
- 最终输出：{machine['condition_grid_output_count']} 条条件×网格×样本记录，逐样本系数/CMOS 指标 {sum(len(machine['all_rows'][c]) for c in CONDITIONS)} 条，绝对功率表 {machine['power_table_rows']} 行。
- 人工评阅材料包括 300 张直接输出卡、20 张历史真实参照、20 张隐藏重复；原始评分以提交文件完整归档。量化优化结果不混入评阅卡。
- 校准准备方案已交付；本轮未做实物测量。

## 决策影响与下一步

- {route}。
- 如果两种编码都支持语义保留且都达到 200/200 数值见证，可另行设计一次映射训练与独立验证；本轮不自动启动。
- 若语义保留但离散见证不足，应先明确应用是否仍需原 1e-4 数值精度，再单独设计冻结输入制备改进。
- 若语义损失有支持且见证充分，优先设计量化后输入映射；若两者都不足，先设计更强冻结制备方法。
- 任何编码结果分歧、区间不确定或评阅质控失败，都保留不确定性；LUT 标定、量化感知训练、扩大人口、架构扩展和实物部署不在本轮授权内。

历史 `VALID_FAILED` 与 `PROJECT_GEOMETRY_UNMEASURED` 保持不变。本轮为 `diagnostic-only`。
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def _execute_full(config: Mapping[str, Any], inputs: Mapping[str, Any], originals: Mapping[str, Any],
                  reach_inputs: Mapping[str, Any], preflight: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    control = _open_attempt(config, preflight, run_id)
    review_packet = _create_review_packet(ATTEMPT_DIR, inputs, run_id)
    _write_calibration_preparation(ATTEMPT_DIR)
    preflight_gray = torch.load(PREFLIGHT_DIR / "optimized_preflight_gray_u8.pt",
                                map_location="cpu", weights_only=False)
    preflight_ids = preflight_gray.get("sample_identities")
    _require(preflight_gray.get("run_id") == run_id and preflight_ids == preflight["selected_samples"],
             "preflight optimized instructions no longer match the approved 20 identities")
    identity_index = inputs["identity_index"]
    selected_indices = [identity_index[item["sample_key"]] for item in preflight_ids]
    starting = {
        "A8_amplitude_linear": inputs["gray_a"].numpy().copy(),
        "I8_intensity_linear": inputs["gray_i"].numpy().copy(),
    }
    preflight_codes = {
        "A8_amplitude_linear": preflight_gray["A8_gray_u8"].numpy(),
        "I8_intensity_linear": preflight_gray["I8_gray_u8"].numpy(),
    }
    preflight_records = preflight.get("optimizer_records")
    _require(isinstance(preflight_records, dict), "READY preflight is missing optimizer evidence")
    for condition in CONDITIONS:
        _require(preflight_codes[condition].shape == (20, PIXEL_COUNT),
                 f"preflight {condition} command shape changed")
        starting[condition][selected_indices] = preflight_codes[condition]
    optimizer_records: dict[str, list[dict[str, Any]]] = {}
    final_gray: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        reason_by_key = {str(inputs["identities"][index]["sample_key"]): "reused_preflight"
                         for index in selected_indices}
        for index, row in enumerate(originals["conditions"][condition]["rows"]):
            if row["witness"]:
                reason_by_key.setdefault(str(inputs["identities"][index]["sample_key"]),
                                         "source_verified_original_witness")
        skip_reasons = [reason_by_key.get(str(identity["sample_key"])) for identity in inputs["identities"]]
        codes, records = _coordinate_optimize(
            inputs["matrix"].numpy(), inputs["target"].numpy(), starting[condition], condition,
            inputs["identities"], skip_reasons,
            ATTEMPT_DIR / f"optimizer_{condition}.pt", ATTEMPT_DIR / f"progress_{condition}.json",
            run_id, resume=True,
        )
        recorded_preflight = preflight_records.get(condition)
        _require(isinstance(recorded_preflight, list) and len(recorded_preflight) == 20,
                 f"READY preflight has incomplete optimizer records for {condition}")
        preflight_by_key = {str(row.get("sample_key")): row for row in recorded_preflight}
        _require(len(preflight_by_key) == 20, f"READY preflight optimizer identities are duplicated for {condition}")
        for index in selected_indices:
            identity = inputs["identities"][index]
            source_record = preflight_by_key.get(str(identity["sample_key"]))
            _require(source_record is not None and source_record.get("condition") == condition,
                     f"preflight optimizer record is missing for {condition}/{identity['sample_key']}")
            _require(np.array_equal(codes[index], preflight_codes[condition][selected_indices.index(index)]),
                     f"formal optimizer changed a preflight command: {condition}/{identity['sample_key']}")
            records[index] = {**source_record, "reused_from_preflight": True}
        for index, row in enumerate(originals["conditions"][condition]["rows"]):
            if row["witness"]:
                original_codes = inputs["gray_a" if condition.startswith("A8") else "gray_i"][index].numpy()
                _require(np.array_equal(codes[index], original_codes),
                         f"original numerical witness was not preserved: {condition}/{row['sample_key']}")
        final_gray[condition] = codes
        optimizer_records[condition] = records
    _save_commands(ATTEMPT_DIR, inputs, run_id, final_gray, optimizer_records)

    _require(torch.cuda.is_available(), "CUDA became unavailable before full replay")
    device = torch.device("cuda:0")
    models = _new_models(reach_inputs["checkpoint"], device)
    state_before = _snapshot_models(models)
    preflight_key_set = {str(identity["sample_key"]) for identity in preflight_ids}
    total = len(CONDITIONS) * len(GRIDS) * len(inputs["identities"])
    completed = 0
    for condition in CONDITIONS:
        for grid in GRIDS:
            folder = ATTEMPT_DIR / "samples" / condition / str(grid)
            folder.mkdir(parents=True, exist_ok=True)
            for index, identity in enumerate(inputs["identities"]):
                path = _output_path(ATTEMPT_DIR, condition, grid, str(identity["sample_key"]))
                if path.is_file():
                    _validate_final_output(path, run_id, identity, condition, grid)
                elif str(identity["sample_key"]) in preflight_key_set:
                    source = _output_path(PREFLIGHT_DIR, condition, grid, str(identity["sample_key"]))
                    _validate_final_output(source, run_id, identity, condition, grid)
                    shutil.copy2(source, path)
                    _validate_final_output(path, run_id, identity, condition, grid)
                else:
                    amplitude = _decode_gray(final_gray[condition][index], condition)
                    actual = _run_forward(models[grid], _intensity_from_amplitude(amplitude), device)
                    _atomic_torch(path, {
                        "schema": "lightgen_frozen_slm_8bit_frontend_output.v1", "task_id": TASK_ID,
                        "run_id": run_id, "condition": condition, "grid": grid,
                        "sample_identity": dict(identity),
                        "actual_coefficients": actual["actual_coefficients"],
                        "raw_cmos_intensity_w_per_m2": actual["raw_cmos_intensity_w_per_m2"],
                        "power_trace_w": actual["power_trace_w"],
                        "camera_pixel_area_m2": actual["camera_pixel_area_m2"],
                        "camera_total_power_w": actual["camera_total_power_w"],
                    })
                    _validate_final_output(path, run_id, identity, condition, grid)
                completed += 1
                if completed % 20 == 0 or completed == total:
                    _atomic_json(ATTEMPT_DIR / "progress.json", {
                        "task_id": TASK_ID, "run_id": run_id, "phase": "full_frozen_optical_replay",
                        "completed": completed, "total": total, "model_updates": 0, "adam_calls": 0,
                        "prior_fit": 0, "new_sampling": 0, "one_shot_mapping_training": 0,
                    })
    model_state_delta = _assert_models_unchanged(models, state_before)
    _require(all(value == 0.0 for value in model_state_delta.values()),
             "full optical replay changed frozen model parameters or buffers")
    machine = _machine_consumer(ATTEMPT_DIR, inputs, run_id, originals)
    machine["model_state_delta"] = model_state_delta
    _atomic_json(ATTEMPT_DIR / "machine_results.json", machine)
    _write_comparison_atlas(ATTEMPT_DIR, inputs, _select_preflight(inputs["identities"]))
    task_state = {
        "task_id": TASK_ID, "run_id": run_id, "status": "AWAITING_HUMAN_RATINGS",
        "machine_measurement_complete": True, "review_packet": review_packet,
        "human_ratings_received": False, "model_updates": 0, "adam_calls": 0,
        "prior_fit": 0, "new_sampling": 0, "one_shot_mapping_training": 0,
    }
    _atomic_json(ATTEMPT_DIR / "task_state.json", task_state)
    _write_report(ATTEMPT_DIR, originals, machine, None, "AWAITING_HUMAN_RATINGS")
    return {"status": "AWAITING_HUMAN_RATINGS", "run_id": run_id,
            "outputs": machine["condition_grid_output_count"], "review_packet": review_packet,
            "lifecycle_state": control.state}


def _verify_review_packet(run_dir: Path, inputs: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    schedule = _review_schedule(inputs)
    key = _json(run_dir / "blind_review_key.json")
    _require(key.get("task_id") == TASK_ID and key.get("run_id") == run_id
             and key.get("review_id") == REVIEW_ID and key.get("cards") == schedule["cards"]
             and key.get("registered_card_count") == 340,
             "saved blind-review key differs from deterministic fixed population")
    for card in schedule["cards"]:
        path = run_dir / card["image"]
        _require(path.is_file() and path.read_bytes() == _review_image_bytes(inputs, card),
                 f"saved blind card is missing or differs from the source output: {card['card_id']}")
    public_cards = [{"card_id": card["card_id"], "image": card["image"]} for card in schedule["cards"]]
    page = (run_dir / "blind_review.html").read_text(encoding="utf-8")
    _require(page == _review_page(REVIEW_ID, public_cards), "blind review public page changed or exposes metadata")
    manifest = _json(run_dir / "blind_review_public_manifest.json")
    expected_manifest = {
        "schema": "lightgen_frozen_slm_8bit_frontend_review_public_manifest.v1",
        "task_id": TASK_ID, "run_id": run_id, "review_id": REVIEW_ID,
        "card_count": 340, "segment_size": 20, "segment_count": 17,
        "rating_values": list(range(10)) + ["unreadable"],
        "display_source_pixels": 200, "display_css_pixels": 400,
        "conditions_exposed": False, "identity_mapping_exposed": False,
    }
    _require(manifest == expected_manifest,
             "blind review manifest count or privacy boundary changed")
    return {"cards_recomputed": 340, "generated": 300, "anchors": 20, "hidden_duplicates": 20,
            "same_identity_minimum_distance": 40, "page_identity_and_privacy_checked": True}


def _finalize_review(config: Mapping[str, Any], inputs: Mapping[str, Any], originals: Mapping[str, Any],
                     ratings_path: Path) -> dict[str, Any]:
    run_identity = _json(ATTEMPT_DIR / "run_identity.json")
    run_id = str(run_identity.get("run_id"))
    machine = _machine_consumer(ATTEMPT_DIR, inputs, run_id, originals)
    packet = _verify_review_packet(ATTEMPT_DIR, inputs, run_id)
    semantic = _consume_ratings(ATTEMPT_DIR, ratings_path)
    _atomic_json(ATTEMPT_DIR / "semantic_results.json", semantic)
    _csv_write(ATTEMPT_DIR / "semantic_transitions_per_sample.csv", semantic["transitions_per_sample"])
    _csv_write(ATTEMPT_DIR / "semantic_per_class_summary.csv", semantic["per_class_rows"])
    route = _final_route(semantic, machine)
    decision = {
        "schema": "lightgen_frozen_slm_8bit_frontend_final_decision.v1", "task_id": TASK_ID,
        "run_id": run_id, "status": "CLOSED_DIAGNOSTIC", "decision_route": route,
        "original_recomputed_witness_counts": machine["original_recomputed_witness_counts"],
        "machine_witness_counts": {condition: machine["conditions"][condition]["witness_count"]
                                    for condition in CONDITIONS},
        "semantic_qc_passed": semantic["quality_passed"],
        "semantic_decisions": {condition: semantic["conditions"][condition]["semantic_loss_decision"]
                                for condition in CONDITIONS},
        "review_packet_recomputed": packet, "machine_results_recomputed_from_saved_evidence": True,
        "zero_update_scope": {"model_updates": 0, "adam_calls": 0, "prior_fit": 0,
                              "new_sampling": 0, "one_shot_mapping_training": 0},
        "historical_valid_failed_preserved": True,
        "geometry_status_preserved": "PROJECT_GEOMETRY_UNMEASURED",
        "hardware_lut_and_physical_link_verified": False,
    }
    _atomic_json(ATTEMPT_DIR / "final_decision.json", decision)
    _write_report(ATTEMPT_DIR, originals, machine, semantic, "CLOSED_DIAGNOSTIC", route)
    control = _open_attempt(config, _json(PREFLIGHT_DIR / "preflight.json"), run_id)
    control.complete()
    run_identity["status"] = "CLOSED"
    run_identity["decision_route"] = route
    _atomic_json(ATTEMPT_DIR / "run_identity.json", run_identity)
    state = {
        "task_id": TASK_ID, "run_id": run_id, "status": "CLOSED_DIAGNOSTIC",
        "machine_measurement_complete": True, "human_ratings_received": True,
        "human_review_quality_passed": semantic["quality_passed"],
        "decision_route": route, "lifecycle_state": control.state,
    }
    _atomic_json(ATTEMPT_DIR / "task_state.json", state)
    return decision


def _verify_only(config: Mapping[str, Any], inputs: Mapping[str, Any], originals: Mapping[str, Any]) -> dict[str, Any]:
    _require(ATTEMPT_DIR.is_dir(), "formal attempt does not exist; run the production execution first")
    identity = _json(ATTEMPT_DIR / "run_identity.json")
    _require(identity.get("task_id") == TASK_ID and identity.get("status") in {"SCIENTIFIC_ACTIVE", "CLOSED"},
             "formal attempt identity or state is invalid")
    run_id = str(identity["run_id"])
    machine = _machine_consumer(ATTEMPT_DIR, inputs, run_id, originals)
    review = _verify_review_packet(ATTEMPT_DIR, inputs, run_id)
    archive = ATTEMPT_DIR / "human_ratings_submitted.json"
    if archive.is_file():
        semantic = _consume_ratings(ATTEMPT_DIR, archive)
        saved_semantic = _json(ATTEMPT_DIR / "semantic_results.json")
        _require(saved_semantic == semantic,
                 "saved semantic summary differs from rating-level independent recomputation")
        route = _final_route(semantic, machine)
        decision = _json(ATTEMPT_DIR / "final_decision.json")
        _require(decision.get("decision_route") == route
                 and decision.get("machine_witness_counts") == {
                     condition: machine["conditions"][condition]["witness_count"] for condition in CONDITIONS
                 }, "saved final route or witness counts differ from recomputed evidence")
        result = {"status": "CLOSED_DIAGNOSTIC", "run_id": run_id,
                  "decision_route": route, "machine_outputs_recomputed": 800,
                  "semantic_ratings_recomputed": 340, "review_packet": review}
        return result
    _write_report(ATTEMPT_DIR, originals, machine, None, "AWAITING_HUMAN_RATINGS")
    return {"status": "AWAITING_HUMAN_RATINGS", "run_id": run_id,
            "machine_outputs_recomputed": 800, "review_packet": review,
            "ratings_file": "human_ratings_submitted.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("preflight", "execute", "verify", "finalize-review"), required=True)
    parser.add_argument("--ratings-file", type=Path)
    args = parser.parse_args()
    try:
        config = _config(args.config.resolve())
        if args.mode == "preflight":
            result = _preflight_command(config)
        else:
            inputs = _load_inputs(config)
            originals = _recompute_original_counts(inputs)
            if args.mode == "execute":
                ready = _preflight_command(config)
                reach_inputs = _load_reachability_inputs(config, inputs["identities"])
                result = _execute_full(config, inputs, originals, reach_inputs, ready["preflight"],
                                       str(ready["run_id"]))
            elif args.mode == "verify":
                result = _verify_only(config, inputs, originals)
            else:
                _require(args.ratings_file is not None, "--finalize-review requires --ratings-file", ValueError)
                result = _finalize_review(config, inputs, originals, args.ratings_file.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        failure_dir = ATTEMPT_DIR if ATTEMPT_DIR.is_dir() else PREFLIGHT_DIR
        if failure_dir.is_dir():
            _atomic_json(failure_dir / "engineering_failure.json", {
                "task_id": TASK_ID, "status": "INVALID_HARNESS",
                "exception_type": type(exc).__name__, "message": str(exc),
                "scientific_conclusion": "none; evidence or execution is incomplete",
            })
        print(f"ENGINEERING_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()






