"""Prepare and score a fixed C45k phase-chart native-view review."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from scripts.research.lightgen import execute_lightgen_c45k_phase_chart_ab_v1 as phase
from ldm_project.v2.application.research.lightgen.image_readout import resize_view


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_C45K_PHASE_CHART_READABILITY_SPEC_V1"
SOURCE_RUN_ID = "0a0ea0e11b334e0ebfb05674abe4149b"
SOURCE_STEP = 45_000
METHODS = ("original", "template")
STREAMS = (0, 1, 2)
CLASSES = tuple(range(10))
RATING_SCHEMA = "lightgen_c45k_phase_chart_readability_ratings.v1"
CONFIG_SCHEMA = "lightgen_c45k_phase_chart_readability_spec.v1"
NUMERIC_FIELDS = (
    "capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
    "view_mean_intensity", "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
)


class ReviewError(RuntimeError):
    """A fixed input, consumer, or review contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _load_original_g4_cohorts(replication_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    report_path = replication_dir / "report.json"
    identity_path = replication_dir / "run_identity.json"
    row_report_path = replication_dir / "generation_per_image_metrics.csv"
    report = _read_json(report_path)
    identity = _read_json(identity_path)
    _require(report.get("task_id") == phase.replication.TASK_ID
             and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("decision", {}).get("complete_valid_measurement") is True
             and report.get("endpoints", {}).get("C_seed_2026101501", {}).get("cumulative_step") == SOURCE_STEP
             and report.get("endpoints", {}).get("C_seed_2026101501", {}).get("source_status") == "VALID_FAILED"
             and identity.get("task_id") == phase.replication.TASK_ID and identity.get("run_id"),
             "historical G4 source report or run identity is incomplete")
    paths = identity.get("input_paths", {})
    checkpoint_path = str(paths.get("source_checkpoint", "")).replace("\\", "/")
    prior_evidence_path = str(paths.get("prior_evidence_dir", "")).replace("\\", "/")
    _require(checkpoint_path.endswith(
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/"
                 "runs/autodl_backup_attempt_01/C/checkpoint_step_010000.pt")
             and prior_evidence_path.endswith(
                 "outputs/20261021_lightgen_small_conditional_mixture_prior_ab_v1/"
                 "runs/autodl_backup_attempt_01"),
             "historical G4 run identity does not cite the fixed C checkpoint and prior evidence")
    rows = phase._read_csv(row_report_path, phase.audit.GENERATION_FIELDS)
    c_rows = [row for row in rows if row["model"] == "C_seed_2026101501"]
    g4_rows = [row for row in c_rows if row["method"] == "G4"]
    _require(len(rows) == 180_000 and len(c_rows) == 60_000 and len(g4_rows) == 30_000,
             "historical generation report does not contain the registered C/G4 population")
    row_lookup: dict[tuple[int, int, int], dict[str, str]] = {}
    for row in g4_rows:
        key = (int(row["stream"]), int(row["requested_label"]), int(row["proposal_id"]))
        _require(key not in row_lookup, f"historical G4 report repeats a proposal identity: {key}")
        row_lookup[key] = row

    cohorts: dict[int, dict[str, Any]] = {}
    for stream_id in STREAMS:
        path = replication_dir / "generation" / "C_seed_2026101501" / f"G4_stream_{stream_id}.pt"
        _require(path.is_file(), f"saved original G4 cohort is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(isinstance(payload, dict), f"saved original G4 cohort is malformed: {path}")
        phase._validate_source_cohort(payload, "G4", stream_id)
        arrays = phase._saved_metric_arrays(payload)
        for index, label_value in enumerate(arrays["label"]):
            label = int(label_value)
            proposal = int(arrays["proposal"][index])
            identity_key = (stream_id, label, proposal)
            saved = row_lookup.get(identity_key)
            _require(saved is not None and int(saved["component_id"]) == int(arrays["component"][index]),
                     f"historical G4 row report omits or misaligns {identity_key}")
            expected = {"E_D_prediction": int(arrays["ed_prediction"][index]),
                        "E_C_prediction": int(arrays["ec_prediction"][index]),
                        "E_D_correct": int(arrays["ed_correct"][index]),
                        "E_C_correct": int(arrays["ec_correct"][index])}
            for field, value in expected.items():
                _require(int(saved[field]) == value,
                         f"historical G4 row report disagrees for {field} at {identity_key}")
            for field in NUMERIC_FIELDS:
                _require(math.isclose(float(saved[field]), float(arrays[field][index]),
                                      rel_tol=1e-12, abs_tol=1e-15),
                         f"historical G4 row report disagrees for {field} at {identity_key}")
        cohorts[stream_id] = payload
    return cohorts, report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    os.replace(temporary, path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    _require(result == root.resolve() or root.resolve() in result.parents,
             f"configured path escapes the repository: {value}")
    return result


def _load_config(path: Path, task_input_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Path]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(config.get("schema") == CONFIG_SCHEMA and config.get("task_id") == TASK_ID,
             "task configuration schema or identity changed")
    review = config.get("review", {})
    _require(review == {
        "unique_per_method": 600, "per_stream_per_class": 20,
        "hidden_duplicates": 20, "real_target_anchors": 20,
        "selection_seed": 2026102311, "shuffle_seed": 2026102312,
        "bootstrap_seed": 2026102313, "bootstrap_resamples": 10_000,
        "minimum_duplicate_consistency": 18, "minimum_anchor_correct": 18,
        "minimum_duplicate_card_distance": 40, "segment_size": 20, "reviewers": 1,
    }, "fixed human review population, seeds, or quality controls changed")
    execution = config.get("execution", {})
    _require(execution == {
        "device": "cuda", "batch_size": 10,
        "preflight_rows_per_method_stream_class": 2,
        "replay_rtol": 2e-6, "replay_atol": 5e-8,
    }, "fixed CUDA replay or numerical tolerances changed")
    paths = {name: _resolve(ROOT, value) for name, value in config.get("source", {}).items()}
    paths["phase_evidence_dir"] = _resolve(ROOT, config["source"]["phase_evidence_dir"])
    paths["baseline_dir"] = _resolve(ROOT, config["source"]["baseline_dir"])
    paths["replication_dir"] = _resolve(ROOT, config["source"]["replication_dir"])
    paths["runtime_config"] = _resolve(ROOT, config["source"]["runtime_config"])
    paths["prior_readability_key"] = _resolve(ROOT, config["source"]["prior_readability_key"])
    paths["decoder_readability_key"] = _resolve(ROOT, config["source"]["decoder_readability_key"])
    paths["output_dir"] = _resolve(ROOT, config["output"]["directory"])
    paths["phase_readback_receipt"] = paths["phase_evidence_dir"] / "commands" / "windows_local_readback_receipt.json"
    if task_input_root is not None:
        input_root = task_input_root.resolve()
        _require(input_root.is_dir(), f"task-specific input bundle is missing: {input_root}")
        paths["task_input_root"] = input_root
        paths["replication_dir"] = input_root / "replication"
        paths["prior_readability_key"] = input_root / "prior_readability_key.json"
        paths["decoder_readability_key"] = input_root / "decoder_readability_key.json"
        paths["phase_readback_receipt"] = input_root / "phase_readback_receipt.json"
    return config, paths


def _load_sources(paths: dict[str, Path]) -> dict[str, Any]:
    evidence = paths["phase_evidence_dir"]
    report = _read_json(evidence / "report.json")
    local_receipt = _read_json(paths["phase_readback_receipt"])
    run_config = _read_json(evidence / "run_config.json")
    _require(report.get("task_id") == "LIGHTGEN_C45K_PHASE_CHART_AB_V1"
             and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("scientific_status") == "PHASE_TEMPLATE_G4_PROXY_GAIN_NOT_SUPPORTED"
             and report.get("source_run_id") == SOURCE_RUN_ID
             and report.get("source_global_step") == SOURCE_STEP
             and report.get("source_status") == "VALID_FAILED",
             "phase-chart source report is incomplete or belongs to another endpoint")
    _require(local_receipt.get("status") == "PASSED"
             and local_receipt.get("new_generation_rows") == 60_000
             and local_receipt.get("original_saved_rows") == 60_000
             and local_receipt.get("method_stream_class_groups") == 120
             and local_receipt.get("explicit_source_lineage", {}).get("checkpoint_run_id") == SOURCE_RUN_ID
             and local_receipt.get("explicit_source_lineage", {}).get("checkpoint_global_step") == SOURCE_STEP,
             "Windows readback receipt does not prove the fixed phase-chart populations")
    _require(run_config.get("source_endpoint", {}).get("run_id") == SOURCE_RUN_ID
             and run_config.get("source_endpoint", {}).get("global_step") == SOURCE_STEP
             and run_config.get("populations", {}).get("new_generation_rows") == 60_000
             and run_config.get("populations", {}).get("old_saved_generation_rows") == 60_000,
             "phase-chart run configuration does not match the delivered receipt")

    checkpoint_path = paths["baseline_dir"] / "C" / "checkpoint_step_010000.pt"
    endpoint_path = paths["baseline_dir"] / "C" / "endpoint_summary.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    endpoint = _read_json(endpoint_path)
    _require(checkpoint.get("task_id") == "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
             and checkpoint.get("branch") == "C" and checkpoint.get("seed") == 2026101501
             and checkpoint.get("completed_global_step") == SOURCE_STEP
             and checkpoint.get("run_id") == SOURCE_RUN_ID,
             "C checkpoint metadata is not the fixed global 45k endpoint")
    _require(endpoint.get("branch") == "C" and endpoint.get("global_step") == SOURCE_STEP
             and endpoint.get("status") == "VALID_FAILED" and endpoint.get("measurement_complete") is True,
             "C endpoint summary is incomplete or changed status")

    replication_dir = paths["replication_dir"]
    saved_original_g4, replication_report = _load_original_g4_cohorts(replication_dir)
    _require(replication_report.get("decision", {}).get("result") ==
             "G4_PROXY_GAIN_REPEATED_ON_THREE_TRAINING_SCHEDULES",
             "saved original generation report does not record the fixed G4 source result")
    original_cohorts = {("original", stream): saved_original_g4[stream] for stream in STREAMS}
    template_cohorts: dict[tuple[str, int], dict[str, Any]] = {}
    for stream in STREAMS:
        payload = torch.load(evidence / "cohorts" / f"G4_stream_{stream}.pt",
                             map_location="cpu", weights_only=False)
        phase._validate_new_cohort(payload, "G4", stream, require_cuda_event=True)
        template_cohorts[("template", stream)] = payload
    cohorts = {(method, stream): original_cohorts[(method, stream)] if method == "original"
               else template_cohorts[(method, stream)] for method in METHODS for stream in STREAMS}

    metric_path = evidence / "generation_per_image_metrics.csv"
    _require(metric_path.is_file(), "full saved per-image generation metrics are missing")
    metric_rows: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    with metric_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_fields = {
            "representation", "prior", "stream", "requested_label", "component_id", "proposal_id",
            "E_D_prediction", "E_C_prediction", "E_D_correct", "E_C_correct", *NUMERIC_FIELDS,
        }
        _require(set(reader.fieldnames or ()) == expected_fields,
                 "saved generation metric columns do not match the registered consumer")
        for raw in reader:
            if raw["prior"] != "G4":
                continue
            method = raw["representation"]
            stream_id = int(raw["stream"])
            label = int(raw["requested_label"])
            proposal = int(raw["proposal_id"])
            key = (method, stream_id, label, proposal)
            _require(method in METHODS and stream_id in STREAMS and label in CLASSES and key not in metric_rows,
                     "saved G4 metric identity is invalid or duplicated")
            metric_rows[key] = {**raw, "component_id": int(raw["component_id"]),
                                "E_D_prediction": int(raw["E_D_prediction"]),
                                "E_C_prediction": int(raw["E_C_prediction"]),
                                "E_D_correct": bool(int(raw["E_D_correct"])),
                                "E_C_correct": bool(int(raw["E_C_correct"]))}
    _require(len(metric_rows) == 60_000, "G4 saved metric population is not 60,000 unique records")

    identity_to_row: dict[tuple[str, int, int, int], int] = {}
    for method in METHODS:
        for stream_id in STREAMS:
            cohort = cohorts[(method, stream_id)]
            labels = cohort["requested_labels"].long()
            proposal_ids = cohort["proposal_ids"].long()
            components = cohort["component_ids"].long()
            _require(tuple(labels.shape) == tuple(proposal_ids.shape) == tuple(components.shape) == (10_000,)
                     and torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 1_000)),
                     f"{method} G4 stream {stream_id} has a wrong row count or class denominator")
            pairs: set[tuple[int, int]] = set()
            for row in range(10_000):
                label = int(labels[row])
                proposal = int(proposal_ids[row])
                identity = (method, stream_id, label, proposal)
                _require(identity not in identity_to_row and (label, proposal) not in pairs,
                         f"{method} G4 stream {stream_id} repeats a proposal identity")
                identity_to_row[identity] = row
                pairs.add((label, proposal))
                saved = metric_rows.get(identity)
                _require(saved is not None and saved["component_id"] == int(components[row]),
                         f"per-image metric row is missing or misaligned for {identity}")
                for field, source_field in (("E_D_prediction", "ed_predictions"),
                                            ("E_C_prediction", "ec_predictions")):
                    _require(saved[field] == int(cohort[source_field][row]),
                             f"saved evaluator prediction differs for {identity} {field}")
                for field, source_field in (("E_D_correct", "ed_predictions"), ("E_C_correct", "ec_predictions")):
                    _require(saved[field] == (int(cohort[source_field][row]) == label),
                             f"saved evaluator correctness differs for {identity} {field}")
                for field in NUMERIC_FIELDS:
                    value = float(saved[field])
                    _require(math.isfinite(value) and math.isclose(value, float(cohort[field][row]),
                                                                    rel_tol=2e-7, abs_tol=1e-12),
                             f"saved per-image metric differs for {identity} {field}")

    examples = torch.load(replication_dir / "reference_C" / "dev_examples_step_045000.pt",
                          map_location="cpu", weights_only=False)
    _require(examples.get("task_id") == phase.replication.TASK_ID
             and examples.get("seed") == 2026101501 and examples.get("step") == SOURCE_STEP
             and tuple(examples.get("rows", torch.empty(0)).shape) == (20,)
             and tuple(examples.get("labels", torch.empty(0)).shape) == (20,)
             and tuple(examples.get("targets28", torch.empty(0)).shape) == (20, 1, 28, 28)
             and torch.equal(torch.bincount(examples["labels"].long(), minlength=10), torch.full((10,), 2)),
             "saved DEV target anchors do not contain exactly two examples per class")
    return {"checkpoint": checkpoint, "endpoint": endpoint, "runtime_config": yaml.safe_load(
                paths["runtime_config"].read_text(encoding="utf-8")), "cohorts": cohorts,
            "metric_rows": metric_rows, "identity_to_row": identity_to_row, "examples": examples,
            "source_report": report, "local_receipt": local_receipt, "replication_report": replication_report}


def _sample_exclusions(
    paths: dict[str, Path], cohorts: dict[tuple[str, int], dict[str, Any]],
    identity_to_row: dict[tuple[str, int, int, int], int],
) -> dict[tuple[str, int, int, int], set[str]]:
    excluded: dict[tuple[str, int, int, int], set[str]] = {}
    for method in METHODS:
        for stream_id in STREAMS:
            cohort = cohorts[(method, stream_id)]
            labels = cohort["requested_labels"].long()
            proposals = cohort["proposal_ids"].long()
            for row in cohort["native_capture_indices"].long().tolist():
                identity = (method, stream_id, int(labels[row]), int(proposals[row]))
                _require(identity in identity_to_row, f"saved preview identity is absent from the cohort: {identity}")
                excluded.setdefault(identity, set()).add("saved_native_preview")

    for name in ("prior_readability_key", "decoder_readability_key"):
        key = _read_json(paths[name])
        _require(isinstance(key.get("cards"), list) and key.get("review_id")
                 and key.get("task_id"), f"historical review key is incomplete: {name}")
        for card in key["cards"]:
            if card.get("kind") != "generated":
                continue
            if name == "prior_readability_key":
                if card.get("method") != "G4":
                    continue
            else:
                _require(card.get("branch") in {"D0", "Dgrad"},
                         "decoder review contains an unexpected generated branch")
            identity = ("original", int(card["stream"]), int(card["requested_label"]), int(card["proposal_id"]))
            _require(identity in identity_to_row, f"historically reviewed identity cannot be mapped: {identity}")
            _require(identity_to_row[identity] == int(card["cohort_row"]),
                     f"historical review row does not match its coefficient identity: {identity}")
            excluded.setdefault(identity, set()).add("historically_reviewed")
    return excluded


def _summarize_saved_evaluators(metric_rows: dict[tuple[str, int, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for (method, stream_id, label, _proposal), row in metric_rows.items():
        grouped.setdefault((method, stream_id, label), []).append(row)
    result: list[dict[str, Any]] = []
    for method in METHODS:
        for stream_id in STREAMS:
            for label in CLASSES:
                rows = grouped.get((method, stream_id, label), [])
                _require(len(rows) == 1_000,
                         f"saved evaluator stratum {method}/stream_{stream_id}/class_{label} is incomplete")
                result.append({"method": method, "stream": stream_id, "requested_label": label,
                               "count": len(rows),
                               "E_D_correct_rate": float(np.mean([row["E_D_correct"] for row in rows])),
                               "E_C_correct_rate": float(np.mean([row["E_C_correct"] for row in rows])),
                               "proxy_clipping_area_fraction_mean": float(np.mean([
                                   float(row["proxy_clipping_area_fraction"]) for row in rows])),
                               "proxy_clipped_excess_fraction_mean": float(np.mean([
                                   float(row["proxy_clipped_excess_fraction"]) for row in rows])),
                               "view_mean_intensity_mean": float(np.mean([
                                   float(row["view_mean_intensity"]) for row in rows]))})
    _require(len(result) == 60, "saved evaluator summary does not contain all 60 G4 strata")
    return result


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty saved-evaluator summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_population(
    cohorts: dict[tuple[str, int], dict[str, Any]],
    identity_to_row: dict[tuple[str, int, int, int], int],
    exclusions: dict[tuple[str, int, int, int], set[str]], seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for method in METHODS:
        for stream_id in STREAMS:
            cohort = cohorts[(method, stream_id)]
            labels = cohort["requested_labels"].long()
            proposals = cohort["proposal_ids"].long()
            components = cohort["component_ids"].long()
            for label in CLASSES:
                available = [row for row in torch.nonzero(labels == label, as_tuple=False).flatten().tolist()
                             if (method, stream_id, label, int(proposals[row])) not in exclusions]
                _require(len(available) >= 20,
                         f"not enough eligible samples in {method}/stream_{stream_id}/class_{label}")
                chosen = sorted(int(row) for row in rng.choice(available, size=20, replace=False).tolist())
                for stratum_rank, row in enumerate(chosen):
                    proposal = int(proposals[row])
                    identity = (method, stream_id, label, proposal)
                    _require(identity_to_row.get(identity) == row,
                             f"selected source row is not bound to its saved identity: {identity}")
                    selected.append({"method": method, "stream": stream_id, "requested_label": label,
                                     "cohort_row": row, "proposal_id": proposal,
                                     "component_id": int(components[row]),
                                     "sample_id": f"{method}:G4:{stream_id}:{label}:{proposal}",
                                     "preflight": stratum_rank < 2,
                                     "preflight_rank": stratum_rank})
    for display_index, row in enumerate(selected, start=1):
        row["display_id"] = f"generated_{display_index:04d}"
    _require(len(selected) == 1_200 and len({row["sample_id"] for row in selected}) == 1_200,
             "fixed 1,200-row selection is incomplete or duplicated")
    return selected


def _numpy_adaptive_pool(image: np.ndarray, size: int = 28) -> np.ndarray:
    height, width = image.shape
    pooled = np.empty((size, size), dtype=np.float64)
    for out_y in range(size):
        y0 = math.floor(out_y * height / size)
        y1 = math.ceil((out_y + 1) * height / size)
        for out_x in range(size):
            x0 = math.floor(out_x * width / size)
            x1 = math.ceil((out_x + 1) * width / size)
            pooled[out_y, out_x] = float(image[y0:y1, x0:x1].mean(dtype=np.float64))
    return pooled


def _save_png(path: Path, image: torch.Tensor) -> None:
    array = np.rint(image.squeeze().detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    _require(array.shape == (200, 200), "review display source is not a native 200x200 view")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    Image.fromarray(array, mode="L").save(temporary, optimize=True)
    os.replace(temporary, path)


def _run_cuda_preflight(
    rows: list[dict[str, Any]], cohorts: dict[tuple[str, int], dict[str, Any]],
    checkpoint: dict[str, Any], runtime_config: dict[str, Any], output_dir: Path, batch_size: int,
    rtol: float, atol: float,
) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for frozen Decoder review replay")
    device = torch.device("cuda")
    model = phase.base._model("small", 2048, device, runtime_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(not any(parameter.requires_grad for parameter in model.parameters()),
             "frozen C model contains trainable parameters")
    gain = torch.as_tensor(checkpoint["camera_gain"], device=device, dtype=torch.float32)
    source_states = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    selected_preflight = [row for row in rows if row["preflight"]]
    _require(len(selected_preflight) == 120, "same-entrypoint preflight must cover 120 rows")
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    max_view_error = 0.0
    max_numpy_view_error = 0.0
    max_scalar_error: dict[str, float] = {field: 0.0 for field in NUMERIC_FIELDS}
    chunks: list[list[dict[str, Any]]] = [selected_preflight[i:i + batch_size]
                                           for i in range(0, len(selected_preflight), batch_size)]
    for chunk in chunks:
        coefficients = torch.stack([
            cohorts[(row["method"], row["stream"])]["ols_coefficients"][row["cohort_row"]]
            for row in chunk
        ]).to(device)
        with torch.inference_mode():
            decoded = model.forward_from_ols(coefficients)
            raw = decoded["physical_w_per_pixel"].float()
            scaled = raw * gain
            views = resize_view(scaled)
            clipped = scaled.clamp(0.0, 1.0)
            fields = {
                "capture_power_w": raw.double().sum((1, 2, 3)),
                "raw_mean_w_per_pixel": raw.double().mean((1, 2, 3)),
                "raw_peak_w_per_pixel": raw.double().amax((1, 2, 3)),
                "view_mean_intensity": views.double().mean((1, 2, 3)),
                "proxy_clipping_area_fraction": ((scaled < 0.0) | (scaled > 1.0)).double().mean((1, 2, 3)),
                "proxy_clipped_excess_fraction": (scaled.double().sub(1.0).clamp_min(0.0).sum((1, 2, 3))
                                                    / scaled.double().sum((1, 2, 3)).clamp_min(1e-30)),
            }
        for index, row in enumerate(chunk):
            cohort = cohorts[(row["method"], row["stream"])]
            source_row = row["cohort_row"]
            saved_view = cohort["images"]
            if saved_view.shape[0] == 10_000:
                view_error = float((views[index].detach().cpu() - saved_view[source_row]).abs().max())
                max_view_error = max(max_view_error, view_error)
                _require(torch.allclose(views[index].detach().cpu(), saved_view[source_row], rtol=rtol, atol=atol),
                         f"CUDA replay does not reproduce the saved view for {row['sample_id']}")
            elif saved_view.shape[0] != 0:
                raise ReviewError(f"saved image tensor is incomplete for {row['sample_id']}")
            numpy_view = _numpy_adaptive_pool(clipped[index, 0].detach().cpu().numpy())
            cuda_view = views[index, 0].detach().cpu().numpy()
            numpy_error = float(np.max(np.abs(numpy_view - cuda_view)))
            max_numpy_view_error = max(max_numpy_view_error, numpy_error)
            _require(np.allclose(numpy_view, cuda_view, rtol=rtol, atol=atol),
                     f"independent CPU readout exceeds replay tolerance for {row['sample_id']}; "
                     f"max_abs_error={numpy_error:.9g}")
            for field in NUMERIC_FIELDS:
                observed = float(fields[field][index].detach().cpu())
                saved = float(cohort[field][source_row])
                error = abs(observed - saved)
                max_scalar_error[field] = max(max_scalar_error[field], error)
                field_atol = 1e-12 if field.endswith("_w") or field == "raw_peak_w_per_pixel" else 5e-8
                _require(math.isclose(observed, saved, rel_tol=2e-6, abs_tol=field_atol),
                         f"CUDA replay scalar differs from saved source evidence for {row['sample_id']} {field}")
            _save_png(output_dir / "review_cards" / f"{row['display_id']}.png", clipped[index])
    finished.record()
    torch.cuda.synchronize(device)
    elapsed_ms = float(started.elapsed_time(finished))
    _require(elapsed_ms > 0.0, "CUDA event does not prove a Decoder forward")
    state_unchanged = all(torch.equal(value, model.state_dict()[name].detach().cpu())
                          for name, value in source_states.items())
    _require(state_unchanged, "frozen C model state changed during Decoder replay")
    return {"status": "PASSED", "device": torch.cuda.get_device_name(device), "device_index": device.index,
            "cuda_runtime": torch.version.cuda, "torch_version": torch.__version__,
            "major_computation_device": "cuda", "major_cuda_stages": ["frozen_Decoder", "fixed_readout"],
            "rows": 120, "groups": 60, "rows_per_group": 2, "cuda_event_ms": elapsed_ms,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_saved_28_view_error": max_view_error, "max_numpy_readout_error": max_numpy_view_error,
            "max_saved_scalar_absolute_error": max_scalar_error, "model_state_unchanged": state_unchanged,
            "encoder_forward_images": 0, "evaluator_forward_images": 0, "optimizer_creations": 0,
            "official_test_reads": 0, "lpips_evaluations": 0}


def _build_page(review_id: str, cards: list[dict[str, str]]) -> str:
    visible = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    total = len(cards)
    segments = math.ceil(total / 20)
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>手写数字图像评阅</title>
<style>body{{font:18px system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem;color:#18212b}}main{{text-align:center}}img{{width:400px;height:400px;object-fit:contain;image-rendering:pixelated;border:1px solid #999;background:#fff}}.choices{{display:grid;grid-template-columns:repeat(6,1fr);gap:.4rem;margin:1rem 0}}label{{padding:.5rem;border:1px solid #ccd1d1;border-radius:.4rem}}button{{font-size:1rem;padding:.6rem 1rem;margin:.3rem}}#save_status{{color:#405060}}</style>
<h1>手写数字图像评阅</h1><p>请依据图像本身选择数字；不能辨认时选择“不可辨认”。图像使用固定灰度范围和统一显示，不做逐图增强。每20张为一段，评分和当前位置会自动保存。</p>
<main><p id='progress'></p><img id='image' alt='当前评阅图'><p id='segment_note'></p><div id='choices' class='choices'></div>
<button id='prev'>上一张</button><button id='next'>下一张</button><button id='missing'>显示漏评编号</button><button id='next_missing'>跳到下一张漏评</button><button id='save'>导出全部评分</button><p id='save_status'></p></main>
<script>
const TASK_ID={json.dumps(TASK_ID)}, REVIEW_ID={json.dumps(review_id)}, CARDS={visible};
const STORE='lightgen-native-review-'+REVIEW_ID;let state={{}};let position=0;
try{{const saved=JSON.parse(localStorage.getItem(STORE)||'{{}}');state=saved.ratings||{{}};position=Math.min(Math.max(0,Number(saved.position)||0),CARDS.length-1);}}catch{{state={{}};position=0;}}
function persist(){{localStorage.setItem(STORE,JSON.stringify({{ratings:state,position}}));const segment=Math.floor(position/20)+1;const within=position%20+1;document.getElementById('progress').textContent=`第 ${{segment}} / {segments} 段 · 本段 ${{within}} / 20 张 · 总计 ${{position+1}} / {total} 张 · 已完成 ${{Object.keys(state).length}} 张`;}}
function render(){{const c=CARDS[position];document.getElementById('image').src=c.image;document.getElementById('choices').innerHTML='';document.getElementById('save_status').textContent='';document.getElementById('segment_note').textContent=(position%20===19)?'本段结束，可休息；当前位置与评分已自动保存。':'';const opts=[...Array(10)].map((_,i)=>[String(i),String(i)]).concat([['unreadable','不可辨认']]);for(const [value,text] of opts){{const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='rating';radio.value=value;radio.checked=state[c.card_id]===value;radio.onchange=()=>{{state[c.card_id]=value;persist();render();}};label.append(radio,document.createTextNode(' '+text));document.getElementById('choices').append(label);}}persist();}}
document.getElementById('prev').onclick=()=>{{position=Math.max(0,position-1);persist();render();}};document.getElementById('next').onclick=()=>{{position=Math.min(CARDS.length-1,position+1);persist();render();}};
function missingPositions(){{return CARDS.map((c,i)=>Object.hasOwn(state,c.card_id)?-1:i).filter(i=>i>=0);}}
document.getElementById('missing').onclick=()=>{{const missing=missingPositions();document.getElementById('save_status').textContent=missing.length?`漏评 ${{missing.length}} 张：`+missing.map(i=>`${{CARDS[i].card_id}}（总第${{i+1}}张，第${{Math.floor(i/20)+1}}段第${{i%20+1}}张）`).join('；'):'全部评分已完成，可以导出。';}};
document.getElementById('next_missing').onclick=()=>{{const missing=missingPositions();if(!missing.length){{document.getElementById('save_status').textContent='全部评分已完成，可以导出。';return;}}position=missing.find(i=>i>=position)??missing[0];persist();render();}};
document.getElementById('save').onclick=()=>{{if(Object.keys(state).length!==CARDS.length){{document.getElementById('save_status').textContent=`请先完成全部 ${{CARDS.length}} 张评分。`;return;}}const answers=CARDS.map(c=>{{const v=state[c.card_id];return {{card_id:c.card_id,rating:v==='unreadable'?'unreadable':Number(v)}};}});const payload={{schema:{json.dumps(RATING_SCHEMA)},task_id:TASK_ID,review_id:REVIEW_ID,cards:answers}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='lightgen_readability_ratings_'+REVIEW_ID+'.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('save_status').textContent='评分文件已导出。';}};render();</script></html>"""


def _make_review_package(
    rows: list[dict[str, Any]], examples: dict[str, Any],
    metric_rows: dict[tuple[str, int, int, int], dict[str, Any]], output_dir: Path,
    config: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    require_by_stratum = {(method, stream_id, label): 0 for method in METHODS
                          for stream_id in STREAMS for label in CLASSES}
    unique_cards: list[dict[str, Any]] = []
    source_to_unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = f"review_cards/{row['display_id']}.png"
        _require((output_dir / path).is_file(), f"native replay image is missing: {row['sample_id']}")
        saved = metric_rows[(row["method"], row["stream"], row["requested_label"], row["proposal_id"])]
        unique_cards.append({"kind": "generated", **row, "saved_metrics": saved, "image": path})
        source_to_unique[row["sample_id"]] = unique_cards[-1]
        group = (row["method"], row["stream"], row["requested_label"])
        require_by_stratum[group] += 1
    _require(all(count == 20 for count in require_by_stratum.values()),
             "blind-review population is not balanced by method, stream, and class")

    duplicate_candidates = [row for row in rows if row["stream"] == 0 and row["preflight_rank"] == 0]
    _require(len(duplicate_candidates) == 20, "hidden-repeat population must contain 20 unique source images")
    cards = list(unique_cards)
    for index, original in enumerate(duplicate_candidates):
        source = source_to_unique[original["sample_id"]]
        duplicate_group = f"D{index:02d}"
        source["duplicate_group"] = duplicate_group
        duplicate_path = f"review_cards/repeat_{index:02d}.png"
        (output_dir / duplicate_path).write_bytes((output_dir / source["image"]).read_bytes())
        cards.append({"kind": "duplicate", "duplicate_group": duplicate_group,
                      "source_sample_id": original["sample_id"], "image": duplicate_path})

    anchors = examples["targets28"].float()
    anchors200 = F.interpolate(anchors, size=(200, 200), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    anchor_cards: list[dict[str, Any]] = []
    for index in range(20):
        image_path = f"review_cards/anchor_{index:02d}.png"
        _save_png(output_dir / image_path, anchors200[index])
        anchor_cards.append({"kind": "real_anchor", "target_label": int(examples["labels"][index]),
                             "source_row": int(examples["rows"][index]), "image": image_path})
    _require([sum(card["target_label"] == label for card in anchor_cards) for label in CLASSES] == [2] * 10,
             "saved real-target anchors are not two per class")
    cards.extend(anchor_cards)

    shuffle_seed = int(config["review"]["shuffle_seed"])
    min_distance = int(config["review"]["minimum_duplicate_card_distance"])
    for attempt in range(10_000):
        order = np.arange(len(cards))
        np.random.default_rng(shuffle_seed + attempt).shuffle(order)
        positions: dict[str, list[int]] = {}
        for position, card_index in enumerate(order.tolist()):
            group = cards[card_index].get("duplicate_group")
            if group:
                positions.setdefault(group, []).append(position)
        if all(len(values) == 2 and abs(values[0] - values[1]) >= min_distance
               for values in positions.values()):
            cards = [cards[index] for index in order.tolist()]
            break
    else:
        raise ReviewError("fixed shuffle seed did not separate all hidden repeats")

    review_id = uuid.uuid4().hex
    hidden_cards: list[dict[str, Any]] = []
    visible_cards: list[dict[str, str]] = []
    for index, card in enumerate(cards, start=1):
        card_id = f"R{index:04d}"
        hidden_cards.append({"card_id": card_id, **card})
        visible_cards.append({"card_id": card_id, "image": card["image"]})
    _require(len(hidden_cards) == 1_240 and len(unique_cards) == 1_200,
             "review package does not contain exactly 1,200 unique samples and 40 controls")
    public = {"schema": "lightgen_c45k_phase_chart_readability_public_manifest.v1",
              "task_id": TASK_ID, "run_id": run_id, "review_id": review_id,
              "card_count": 1_240, "unique_generated_count": 1_200,
              "hidden_duplicate_count": 20, "real_target_anchor_count": 20,
              "display": {"source_pixels": 200, "display_pixels": 400,
                          "range": "fixed global gain then clamp [0,1]; no per-image normalization"}}
    key = {"schema": "lightgen_c45k_phase_chart_readability_key.v1", "task_id": TASK_ID,
           "run_id": run_id, "review_id": review_id, "cards": hidden_cards,
           "quality_control": {"minimum_duplicate_consistency": 18,
                                "minimum_anchor_correct": 18,
                                "minimum_duplicate_card_distance": min_distance},
           "source_endpoint": {"branch": "C", "seed": 2026101501,
                               "global_step": SOURCE_STEP, "run_id": SOURCE_RUN_ID}}
    (output_dir / "blind_review.html").write_text(_build_page(review_id, visible_cards), encoding="utf-8")
    _write_json(output_dir / "blind_review_key.json", key)
    _write_json(output_dir / "blind_review_public_manifest.json", public)
    _validate_public_page(output_dir, key)
    return {"review_id": review_id, "public": public, "key": key}


def _validate_public_page(output_dir: Path, key: dict[str, Any]) -> None:
    page = (output_dir / "blind_review.html").read_text(encoding="utf-8")
    _require("width:400px;height:400px" in page and "image-rendering:pixelated" in page
             and "localStorage.setItem" in page and "第 " in page,
             "blind page is missing fixed-size display, autosave, or progress segments")
    start = page.index("const TASK_ID=")
    cursor = page.index(", CARDS=", start) + len(", CARDS=")
    visible, _ = json.JSONDecoder().raw_decode(page, cursor)
    expected = [{"card_id": card["card_id"], "image": card["image"]} for card in key["cards"]]
    _require(visible == expected, "public card list differs from the hidden mapping")
    for forbidden in ("requested_label", "proposal_id", "component_id", "duplicate_group",
                      "target_label", "E_D_prediction", "E_C_prediction", "original:G4",
                      "template:G4", "G4_stream_"):
        _require(forbidden not in page, f"blind page leaks hidden review field {forbidden}")


def _validate_review_package(output_dir: Path, key: dict[str, Any]) -> None:
    cards = key.get("cards", [])
    generated = [card for card in cards if card.get("kind") == "generated"]
    duplicates = [card for card in cards if card.get("kind") == "duplicate"]
    anchors = [card for card in cards if card.get("kind") == "real_anchor"]
    _require(len(cards) == 1_240 and len(generated) == 1_200 and len(duplicates) == 20
             and len(anchors) == 20, "hidden review key has an incomplete card population")
    card_ids = [card.get("card_id") for card in cards]
    image_paths = [card.get("image") for card in cards]
    _require(len(set(card_ids)) == 1_240 and len(set(image_paths)) == 1_240,
             "review card IDs or image paths are repeated")
    sample_ids = [card.get("sample_id") for card in generated]
    _require(len(set(sample_ids)) == 1_200, "review key repeats a unique generated sample")
    _require({method: sum(card["method"] == method for card in generated) for method in METHODS}
             == {"original": 600, "template": 600}, "review key is not balanced by method")
    for method in METHODS:
        for stream_id in STREAMS:
            for label in CLASSES:
                _require(sum(card["method"] == method and card["stream"] == stream_id
                             and card["requested_label"] == label for card in generated) == 20,
                         f"review key stratum is incomplete: {method}/{stream_id}/{label}")
    _require([sum(card["target_label"] == label for card in anchors) for label in CLASSES] == [2] * 10,
             "review key anchors are not two per class")
    by_sample = {card["sample_id"]: card for card in generated}
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in duplicates:
        groups.setdefault(card.get("duplicate_group"), []).append(card)
    _require(len(groups) == 20, "hidden duplicate groups are incomplete")
    positions: dict[str, list[int]] = {}
    for index, card in enumerate(cards):
        if card.get("kind") == "generated":
            positions.setdefault(card["sample_id"], []).append(index)
        elif card.get("kind") == "duplicate":
            positions.setdefault(card["source_sample_id"], []).append(index)
    minimum = int(key["quality_control"]["minimum_duplicate_card_distance"])
    for group, copies in groups.items():
        _require(len(copies) == 1, f"hidden duplicate group is malformed: {group}")
        duplicate = copies[0]
        source = by_sample.get(duplicate.get("source_sample_id"))
        _require(source is not None and source.get("duplicate_group") == group
                 and (output_dir / source["image"]).read_bytes() ==
                 (output_dir / duplicate["image"]).read_bytes(),
                 f"hidden duplicate does not reproduce its source image: {group}")
        pair_positions = positions[duplicate["source_sample_id"]]
        _require(len(pair_positions) == 2 and abs(pair_positions[0] - pair_positions[1]) >= minimum,
                 f"hidden duplicate spacing failed: {group}")
    _validate_public_page(output_dir, key)


def _write_exclusion_csv(path: Path, exclusions: dict[tuple[str, int, int, int], set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("representation", "stream", "requested_label", "proposal_id", "reason"))
        for identity, reasons in sorted(exclusions.items()):
            writer.writerow((*identity, ";".join(sorted(reasons))))


def _read_ratings(path: Path, key: dict[str, Any]) -> dict[str, str | int]:
    payload = _read_json(path)
    _require(payload.get("schema") == RATING_SCHEMA and payload.get("task_id") == TASK_ID
             and payload.get("review_id") == key.get("review_id") and isinstance(payload.get("cards"), list),
             "ratings file belongs to another task or review")
    expected_ids = [card["card_id"] for card in key["cards"]]
    answers = payload["cards"]
    _require(len(answers) == len(expected_ids)
             and [answer.get("card_id") for answer in answers] == expected_ids,
             "ratings contain missing, duplicated, reordered, or foreign card identities")
    result: dict[str, str | int] = {}
    for answer in answers:
        value = answer.get("rating")
        _require(value == "unreadable" or (type(value) is int and value in CLASSES),
                 f"rating is not 0-9 or unreadable: {answer.get('card_id')}")
        result[answer["card_id"]] = value
    return result


def _bootstrap_difference(key: dict[str, Any], ratings: dict[str, str | int], config: dict[str, Any]) -> dict[str, Any]:
    scores: dict[tuple[str, int, int], np.ndarray] = {}
    for method in METHODS:
        for stream_id in STREAMS:
            for label in CLASSES:
                values = []
                for card in key["cards"]:
                    if card.get("kind") == "generated" and card["method"] == method \
                            and card["stream"] == stream_id and card["requested_label"] == label:
                        values.append(float(ratings[card["card_id"]] == label))
                _require(len(values) == 20,
                         f"review scores are incomplete for {method}/stream_{stream_id}/class_{label}")
                scores[(method, stream_id, label)] = np.asarray(values, dtype=np.float64)
    observed = float(np.mean([scores[("template", stream_id, label)].mean()
                              - scores[("original", stream_id, label)].mean()
                              for stream_id in STREAMS for label in CLASSES]))
    rng = np.random.default_rng(int(config["review"]["bootstrap_seed"]))
    draws = np.zeros(int(config["review"]["bootstrap_resamples"]), dtype=np.float64)
    for stream_id in STREAMS:
        for label in CLASSES:
            for method, sign in (("template", 1.0), ("original", -1.0)):
                values = scores[(method, stream_id, label)]
                sample_indices = rng.integers(0, len(values), size=(len(draws), len(values)))
                # There are 30 stream/class pairs.  Each paired difference is
                # averaged once, so each method mean contributes 1/30.
                draws += sign * values[sample_indices].mean(1) / 30.0
    lower, upper = np.quantile(draws, [0.025, 0.975], method="linear").tolist()
    return {"estimand": "template G4 minus original G4 equal-stream equal-class requested-digit rate",
            "difference": observed, "ci_95": [float(lower), float(upper)],
            "bootstrap_resamples": len(draws), "bootstrap_seed": config["review"]["bootstrap_seed"],
            "strata": 60, "rows_per_method_stratum": 20,
            "rating_unit": "one reviewer; fixed saved samples and C endpoint"}


def _summarize_review(key: dict[str, Any], ratings: dict[str, str | int], config: dict[str, Any]) -> dict[str, Any]:
    unique = [card for card in key["cards"] if card.get("kind") == "generated"]
    qc = [card for card in key["cards"] if card.get("kind") == "duplicate"]
    anchors = [card for card in key["cards"] if card.get("kind") == "real_anchor"]
    duplicate_matches = 0
    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for card in qc:
        duplicate_groups.setdefault(card["duplicate_group"], []).append(card)
    _require(len(qc) == 20 and len(anchors) == 20 and len(duplicate_groups) == 20,
             "review quality-control cards are incomplete")
    # Each repeat card has a hidden source_sample_id pointing to one unique sample.
    unique_by_id = {card["sample_id"]: card for card in unique}
    for group in duplicate_groups.values():
        _require(len(group) == 1 and group[0]["source_sample_id"] in unique_by_id,
                 "hidden repeat mapping is incomplete")
        repeated = group[0]
        source = unique_by_id[repeated["source_sample_id"]]
        source_rating = next(ratings[card["card_id"]] for card in unique
                             if card["sample_id"] == source["sample_id"])
        duplicate_matches += int(source_rating == ratings[repeated["card_id"]])
    anchor_correct = sum(int(ratings[card["card_id"]] == card["target_label"]) for card in anchors)
    minimum = int(config["review"]["minimum_duplicate_card_distance"])
    positions: dict[str, list[int]] = {}
    for index, card in enumerate(key["cards"]):
        if card.get("kind") == "duplicate":
            # Recover the unique source card's position for the same spacing check.
            source_id = card["source_sample_id"]
            positions.setdefault(source_id, []).append(index)
        elif card.get("kind") == "generated":
            positions.setdefault(card["sample_id"], []).append(index)
    distances = []
    for card in qc:
        source_id = card["source_sample_id"]
        locations = positions[source_id]
        _require(len(locations) == 2, "hidden repeat source does not occur exactly twice")
        distances.append(abs(locations[0] - locations[1]))
    duplicate_ok = duplicate_matches >= int(config["review"]["minimum_duplicate_consistency"])
    anchor_ok = anchor_correct >= int(config["review"]["minimum_anchor_correct"])
    qc_passed = duplicate_ok and anchor_ok and min(distances) >= minimum
    per_method: dict[str, Any] = {}
    disagreements: dict[str, Any] = {}
    for method in METHODS:
        method_cards = [card for card in unique if card["method"] == method]
        requested = np.asarray([card["requested_label"] for card in method_cards], dtype=np.int64)
        ratings_array = np.asarray([ratings[card["card_id"]] for card in method_cards], dtype=object)
        correct = np.asarray([value == target for value, target in zip(ratings_array, requested, strict=True)])
        unreadable = ratings_array == "unreadable"
        other = ~(correct | unreadable)
        per_class = []
        for label in CLASSES:
            mask = requested == label
            per_class.append({"label": label, "count": int(mask.sum()),
                              "requested_digit_rate": float(correct[mask].mean()),
                              "unreadable_rate": float(unreadable[mask].mean()),
                              "other_digit_rate": float(other[mask].mean())})
        per_method[method] = {"count": len(method_cards), "equal_class_requested_digit_rate":
                              float(np.mean([row["requested_digit_rate"] for row in per_class])),
                              "equal_class_unreadable_rate": float(np.mean([row["unreadable_rate"] for row in per_class])),
                              "equal_class_other_digit_rate": float(np.mean([row["other_digit_rate"] for row in per_class])),
                              "per_class": per_class}

        ed_correct = []
        ec_correct = []
        ed_supported_human_failure = []
        ec_supported_human_failure = []
        human_correct_ed_wrong = []
        human_correct_ec_wrong = []
        for card in method_cards:
            human = ratings[card["card_id"]]
            target = card["requested_label"]
            row = card["saved_metrics"]
            ed_ok = row["E_D_prediction"] == target
            ec_ok = row["E_C_prediction"] == target
            human_ok = human == target
            human_failed = human == "unreadable" or human != target
            ed_correct.append(ed_ok)
            ec_correct.append(ec_ok)
            ed_supported_human_failure.append(ed_ok and human_failed)
            ec_supported_human_failure.append(ec_ok and human_failed)
            human_correct_ed_wrong.append(human_ok and not ed_ok)
            human_correct_ec_wrong.append(human_ok and not ec_ok)
        disagreements[method] = {
            "rated_unique_rows": len(method_cards),
            "E_D_correct_human_wrong_or_unreadable": float(np.mean(ed_supported_human_failure)),
            "E_C_correct_human_wrong_or_unreadable": float(np.mean(ec_supported_human_failure)),
            "human_correct_E_D_wrong": float(np.mean(human_correct_ed_wrong)),
            "human_correct_E_C_wrong": float(np.mean(human_correct_ec_wrong)),
            "saved_E_D_correct_rate": float(np.mean(ed_correct)),
            "saved_E_C_correct_rate": float(np.mean(ec_correct)),
        }
    effect = _bootstrap_difference(key, ratings, config)
    if not qc_passed:
        result = "REVIEW_INVALID_QC"
    elif effect["ci_95"][0] > 0.0:
        result = "TEMPLATE_HUMAN_GAIN_SUPPORTED"
    elif effect["ci_95"][1] < 0.0:
        result = "TEMPLATE_HUMAN_HARM_SUPPORTED"
    else:
        result = "TEMPLATE_HUMAN_GAIN_UNRESOLVED"
    return {"status": "COMPLETE" if qc_passed else "REVIEW_INVALID_QC", "decision": result,
            "primary_comparison": effect, "human_results": per_method,
            "evaluator_disagreements": disagreements,
            "quality_control": {"duplicate_consistency": {"count": duplicate_matches, "total": 20,
                                                             "minimum": 18, "passed": duplicate_ok},
                                "real_target_correct": {"count": anchor_correct, "total": 20,
                                                        "minimum": 18, "passed": anchor_ok},
                                "minimum_repeat_distance": min(distances), "required_repeat_distance": minimum,
                                "passed": qc_passed},
            "limitations": ["one reviewer and one frozen endpoint", "no hardware readability claim",
                            "the earlier proxy clipping criterion remains failed",
                            "an interval containing zero is inconclusive, not equivalence"]}


def _hardware_spec(paths: dict[str, Path], output_dir: Path) -> None:
    runtime = yaml.safe_load(paths["runtime_config"].read_text(encoding="utf-8"))
    optics = runtime["optics"]
    small = optics["small"]
    wavelength_nm = float(optics["wavelength_m"]) * 1e9
    pixel_pitch_um = float(small["phase_pixel_pitch_m"]) * 1e6
    aperture_mm = pixel_pitch_um * int(small["slm_shape"][0]) / 1e3
    layer_spacing_mm = float(small["layer_spacing_m"]) * 1e3
    port_pitch_um = float(small["port_pitch_m"]) * 1e6
    field_radius_um = float(small["field_1e_radius_m"]) * 1e6
    report = f"""# C45k生成输入与实物加工规格缺口

状态：下单前整理；未联系厂商、未下单、未声称硬件可达。

| 环节 | 当前模型值与证据 | 实测或厂商依据 | 未知项及其影响 | 下一验证动作 |
|---|---|---|---|---|
| 光学链路 | C端点为2层Encoder、100×100相位面、100通道Gaussian S12理想OLS、2层Decoder；波长 {wavelength_nm:.1f} nm，生产网格 {small['production_propagation_grid']}。 | 本项目冻结仿真配置；不是器件验收。 | Gaussian端口坐标、模式场、耦合和复传输尚无实测矩阵；当前`{small['geometry_status']}`。不能据此保证100维软件系数可由光路生成。 | 获取工厂端面图、模式参数和全复传输矩阵，再对目标输入人口做可达性核验。 |
| 系数输入接口 | 机器生成将100维复系数直接送入冻结Decoder入口；完整实物链还需确定实际输入面及如何由光场得到这些系数。 | 已完成的机器盲评重放；不包含实际Encoder入射。 | 复振幅调制器、相位／幅度控制、偏振、放大倍率及参考相位尚未定；两种输入路线的光学接口尚不可等同。 | 分别绘制“直接制备OLS复场”和“保留Encoder整链”接口与测量量，确认保留哪一条再冻结采购要求。 |
| 双光子打印 | 单层相位面为 {small['slm_shape'][0]}×{small['slm_shape'][1]}，采样间距 {pixel_pitch_um:g} μm，有效口径约 {aperture_mm:g} mm；层间距 {layer_spacing_mm:g} mm；模型层透过率 {optics['layer_power_transmission']:.4g}。 | 配置中的理想相位采样与统一透过率假设。 | 实际材料折射率、透过率、相位—高度曲线、打印收缩、最小特征、表面误差、基底和装调误差都未知；透过率不是打印材料实测。 | 向工厂索取{wavelength_nm:.0f} nm材料数据、测试片、相位高度标定、打印精度及层间装调公差。用实测折射率差计算包裹相位深度，不先假定材料。 |
| 工厂光纤 | 模型为 `{small['ols_basis']}`，{small['ols_ports']}端口，按10×10方格放置；各轴中心坐标为`(i−4.5)×{port_pitch_um:g} μm + plane_origin`（i=0…9），端口间距 {port_pitch_um:g} μm，场1/e半径 {field_radius_um:g} μm，OLS传输为`{small['ols_transport']}`。 | Gaussian S12仿真配置，不是端口实测图。旧合成诊断参数另见其原报告，不能当成本端点器件规格。 | 实际芯径、包层、MFD、NA、端面坐标及容差、弯曲、插损、串扰、相位稳定性、温漂及矩阵测量格式均未确认。 | 先让工厂确认可加工阵列和端面公差；交付后逐端口测幅相、插损和串扰，形成实际复矩阵。 |
| 相机接口 | 仿真输出为固定全局增益后裁剪并降采样到28×28的分类视图；原生仿真帧为200×200。 | 冻结读出定义与模拟增益。 | 相机型号、像素间距、成像倍率、量子效率、曝光、暗电流、满阱和绝对增益未知。模拟增益不等于相机标定。 | 确认CMOS型号与有效区域，采集暗场／平场、曝光响应和饱和曲线，再登记物面到像素的倍率。 |

此表用于定义询价和入厂验收所需信息。缺失参数保留未知；本表本身不构成加工图纸、采购授权或硬件通过结论。
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hardware_specification.md").write_text(report, encoding="utf-8")


def _write_report(output_dir: Path, report: dict[str, Any], preflight: dict[str, Any] | None,
                  final: dict[str, Any] | None) -> None:
    if final is None:
        effect = "等待人工评分，暂不判定"
    elif not final["quality_control"]["passed"]:
        qc = final["quality_control"]
        effect = ("评阅质控未通过，不作科学判定（描述性差值="
                  f"{final['primary_comparison']['difference']:.6f}，95%区间="
                  f"[{final['primary_comparison']['ci_95'][0]:.6f}, "
                  f"{final['primary_comparison']['ci_95'][1]:.6f}]；隐藏重复一致"
                  f"{qc['duplicate_consistency']['count']}/20，真实目标正确"
                  f"{qc['real_target_correct']['count']}/20）")
    else:
        effect = (f"差值={final['primary_comparison']['difference']:.6f}，95%区间="
                  f"[{final['primary_comparison']['ci_95'][0]:.6f}, "
                  f"{final['primary_comparison']['ci_95'][1]:.6f}]，判定={final['decision']}")
    rows = "未执行" if preflight is None else (
        f"CUDA={preflight['device']}；全量重放={report.get('cuda_full_replay_rows', 0)}行"
        f"（预检{preflight['rows']}行纳入其中）；预检CUDA事件={preflight['cuda_event_ms']:.3f} ms；"
        f"峰值显存={preflight['peak_allocated_bytes']} bytes"
    )
    status = report.get("status", "AWAITING_HUMAN_RATINGS")
    qc_status = ("尚未评分" if final is None else
                 "通过" if final["quality_control"]["passed"] else "未通过；不形成科学负结果")
    content = f"""# C45k相位表示独立可读性确认与硬件规格

## 结果与状态

状态：`{status}`。冻结C全局45k、模板G4与原G4各600张；本次唯一人眼比较为模板减原表示。人工主结果：{effect}。质控：{qc_status}。历史模板机器裁剪判据仍未通过；系统状态`VALID_FAILED`保持。

## 科学认识

盲评只说明固定端点、固定保存人口及单一评阅者的目标数字识别差异。评价器复盘是解释性描述。硬件规格表区分模型假设、已有依据和未知参数，不形成可达性结论。

## 目标缺口

本轮不证明完整链路可生成、跨训练泛化、实物可读或模型可采用。真实复传输矩阵、打印材料响应、相机标定和多评阅者证据仍缺。现有共同相位统计不代表实际端口相位控制能力。

## 工程完成与验证

CUDA重放：{rows}。配置身份、检查点全局步数、三流原始记录、模板记录、1,200个独立样本身份及保存评价行已关联核对。重新运行评价器、Encoder、训练、优化器、TEST和LPIPS均为0。预检收据和逐行回读收据另存本目录。

## 决策影响与下一步

{effect}。评分有效后，若区间下界大于0，保留模板方向作为后继候选；若区间包含0，收益尚未明确并降低该方向优先级；若区间上界小于0，停止该模板候选。任何结果均不采用模型、不改变历史Gate，也不自动追加训练、硬件实验或采购。本轮当前停止于等待评分。
"""
    (output_dir / "report.md").write_text(content, encoding="utf-8")
    _ = report


def prepare(config_path: Path, *, resume: bool, task_input_root: Path | None = None) -> None:
    config, paths = _load_config(config_path, task_input_root)
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _hardware_spec(paths, output_dir)
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        _require(resume, "attempt already exists; use --resume to continue its fixed population")
        identity = _read_json(identity_path)
        _require(identity.get("task_id") == TASK_ID and identity.get("config") == config,
                 "resume configuration differs from the registered attempt")
        _require(identity.get("task_input_root") == str(paths.get("task_input_root", "")) or
                 identity.get("task_input_root") is None and "task_input_root" not in paths,
                 "resume task-specific input bundle differs from the registered attempt")
        run_id = str(identity["run_id"])
    else:
        allowed_existing = {"hardware_specification.md", "specification_receipt.json"}
        _require({path.name for path in output_dir.iterdir()} <= allowed_existing,
                 "output attempt contains unregistered files")
        run_id = uuid.uuid4().hex
        _write_json(identity_path, {"schema": "lightgen_c45k_phase_chart_readability_run.v1",
                                    "task_id": TASK_ID, "run_id": run_id, "config": config,
                                    "task_input_root": str(paths.get("task_input_root", "")) or None,
                                    "source_endpoint": {"run_id": SOURCE_RUN_ID, "global_step": SOURCE_STEP},
                                    "review_population": 1_240})
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        _require(_read_json(config_path) == config, "saved run configuration changed")
    else:
        _write_json(config_path, config)
    input_data = _load_sources(paths)
    exclusions = _sample_exclusions(paths, input_data["cohorts"], input_data["identity_to_row"])
    exclusion_path = output_dir / "excluded_sample_identities.csv"
    _write_exclusion_csv(exclusion_path, exclusions)
    evaluator_summary = _summarize_saved_evaluators(input_data["metric_rows"])
    _write_rows_csv(output_dir / "saved_evaluator_population_summary.csv", evaluator_summary)
    selection_path = output_dir / "selection_manifest.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        _require(selection.get("task_id") == TASK_ID and selection.get("run_id") == run_id
                 and selection.get("selection_seed") == config["review"]["selection_seed"],
                 "saved selection manifest belongs to another attempt or seed")
        selected = selection.get("rows", [])
        _require(len(selected) == 1_200, "saved selection manifest is incomplete")
        for row in selected:
            identity = (row["method"], int(row["stream"]), int(row["requested_label"]), int(row["proposal_id"]))
            source_row = input_data["identity_to_row"].get(identity)
            _require(source_row == int(row["cohort_row"])
                     and identity not in exclusions,
                     f"saved selection row no longer satisfies source/exclusion identity: {identity}")
    else:
        selected = _select_population(input_data["cohorts"], input_data["identity_to_row"], exclusions,
                                      int(config["review"]["selection_seed"]))
        _write_json(selection_path, {"schema": "lightgen_c45k_phase_chart_readability_selection.v1",
                                     "task_id": TASK_ID, "run_id": run_id,
                                     "selection_seed": config["review"]["selection_seed"], "rows": selected})

    preflight_receipt_path = output_dir / "preflight_receipt.json"
    preflight = _read_json(preflight_receipt_path) if preflight_receipt_path.is_file() else None
    if preflight is None:
        started = time.perf_counter()
        preflight = _run_cuda_preflight(selected, input_data["cohorts"], input_data["checkpoint"],
                                        input_data["runtime_config"], output_dir,
                                        int(config["execution"]["batch_size"]),
                                        float(config["execution"]["replay_rtol"]),
                                        float(config["execution"]["replay_atol"]))
        preflight["wall_seconds"] = time.perf_counter() - started
        _write_json(preflight_receipt_path, preflight)
    _require(preflight.get("status") == "PASSED" and preflight.get("rows") == 120
             and preflight.get("major_computation_device") == "cuda",
             "representative CUDA preflight is absent or incomplete")

    # Replay the remaining fixed population in resumable batches; selected identities never change.
    device = torch.device("cuda")
    model = phase.base._model("small", 2048, device, input_data["runtime_config"])
    model.load_state_dict(input_data["checkpoint"]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gain = torch.as_tensor(input_data["checkpoint"]["camera_gain"], device=device, dtype=torch.float32)
    formal_source_states = {name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()}
    batch_size = int(config["execution"]["batch_size"])
    replay_manifest_path = output_dir / "replay_manifest.json"
    replay_manifest = _read_json(replay_manifest_path) if replay_manifest_path.is_file() else {
        "schema": "lightgen_c45k_phase_chart_readability_replay.v1", "task_id": TASK_ID,
        "run_id": run_id, "rows": [{"sample_id": row["sample_id"],
                                      "cuda_event_ms_per_row": preflight["cuda_event_ms"] / 120.0,
                                      "preflight": True} for row in selected if row["preflight"]]}
    if not replay_manifest_path.is_file():
        _write_json(replay_manifest_path, replay_manifest)
    _require(replay_manifest.get("task_id") == TASK_ID and replay_manifest.get("run_id") == run_id
             and isinstance(replay_manifest.get("rows"), list),
             "replay recovery manifest belongs to another run")
    completed = [row.get("sample_id") for row in replay_manifest["rows"]]
    _require(len(completed) == len(set(completed)), "replay recovery manifest duplicates completed rows")
    completed = set(completed)
    _require(completed <= {row["sample_id"] for row in selected},
             "replay recovery manifest contains a foreign selected row")
    remaining = [row for row in selected if row["sample_id"] not in completed]
    for start in range(0, len(remaining), batch_size):
        chunk = remaining[start:start + batch_size]
        coeff = torch.stack([input_data["cohorts"][(row["method"], row["stream"])]["ols_coefficients"][
            row["cohort_row"]] for row in chunk]).to(device)
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        with torch.inference_mode():
            decoded = model.forward_from_ols(coeff)
            scaled = decoded["physical_w_per_pixel"].float() * gain
            native = scaled.clamp(0.0, 1.0)
        cuda_end.record()
        torch.cuda.synchronize(device)
        block_ms = float(cuda_start.elapsed_time(cuda_end))
        _require(block_ms > 0.0 and tuple(native.shape) == (len(chunk), 1, 200, 200),
                 "formal Decoder replay block did not execute on CUDA or returned a wrong shape")
        for index, row in enumerate(chunk):
            _save_png(output_dir / "review_cards" / f"{row['display_id']}.png", native[index])
            completed.add(row["sample_id"])
            replay_manifest["rows"].append({"sample_id": row["sample_id"],
                                            "cuda_event_ms_per_row": block_ms / len(chunk),
                                            "preflight": False})
        _write_json(replay_manifest_path, replay_manifest)
        print(json.dumps({"phase": "CUDA_NATIVE_REPLAY", "completed_rows": len(completed),
                          "total_rows": 1_200}, ensure_ascii=False), flush=True)
    _require(len(completed) == len(selected) and len(replay_manifest["rows"]) == 1_200,
             "full selected native replay is incomplete")
    formal_state_unchanged = all(
        torch.equal(value, model.state_dict()[name].detach().cpu())
        for name, value in formal_source_states.items())
    _require(formal_state_unchanged, "frozen C model state changed during full Decoder replay")
    package_path = output_dir / "blind_review_key.json"
    if package_path.is_file():
        key = _read_json(package_path)
        _validate_review_package(output_dir, key)
        package = {"key": key, "public": _read_json(output_dir / "blind_review_public_manifest.json"),
                   "review_id": key["review_id"]}
    else:
        package = _make_review_package(selected, input_data["examples"], input_data["metric_rows"],
                                       output_dir, config, run_id)
    _write_json(output_dir / "selection_manifest.json", {"schema": "lightgen_c45k_phase_chart_readability_selection.v1",
                  "task_id": TASK_ID, "run_id": run_id, "selection_seed": config["review"]["selection_seed"],
                  "unique_rows": 1_200, "rows": selected})
    _write_json(output_dir / "cuda_execution_receipt.json", {
        **preflight, "status": "PASSED", "full_replay_rows": 1_200,
        "formal_decoder_rows": 1_080, "preflight_rows_reused": 120,
        "full_population_replayed": len(replay_manifest["rows"]) == 1_200,
        "formal_model_state_unchanged": formal_state_unchanged,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "evaluator_forward_images": 0, "encoder_forward_images": 0,
        "official_test_reads": 0, "lpips_evaluations": 0, "optimizer_creations": 0,
    })
    public_card_files = list((output_dir / "review_cards").glob("*.png"))
    _require(len(public_card_files) == 1_240, "native review card image count is not exactly 1,240")
    _validate_review_package(output_dir, package["key"])
    report = {"task_id": TASK_ID, "run_id": run_id, "status": "AWAITING_HUMAN_RATINGS",
              "measurement_complete": True, "source_step": SOURCE_STEP,
              "source_run_id": SOURCE_RUN_ID, "human_population": {"original_G4": 600,
              "template_G4": 600, "hidden_duplicates": 20, "real_target_anchors": 20,
              "total_cards": 1_240, "strata": 60, "rows_per_stratum": 20},
              "source_rows": 60_000, "source_metric_rows_verified": 60_000,
              "saved_evaluator_summary_strata": len(evaluator_summary),
              "windows_lineage_readback": "PASSED", "preflight": preflight,
              "cuda_device": torch.cuda.get_device_name(device), "cuda_full_replay_rows": 1_200,
              "new_sampling_rows": 0, "training_updates": 0, "optimizer_creations": 0,
              "encoder_forward_images": 0, "evaluator_forward_images": 0,
              "official_test_reads": 0, "lpips_evaluations": 0,
              "review_id": package["review_id"], "source_status_preserved": "VALID_FAILED"}
    _write_json(output_dir / "report.json", report)
    _write_report(output_dir, report, preflight, None)
    _write_json(output_dir / "validation_receipt.json", {
        "task_id": TASK_ID, "run_id": run_id, "status": "PASSED",
        "checks": {"source_endpoint_C_45k": True, "phase_chart_windows_readback": True,
                   "all_60k_G4_rows_mapped_to_saved_metrics": True,
                   "prior_reviews_and_previews_excluded": True,
                   "balanced_1200_row_selection": True, "independent_numpy_readout": True,
                   "replay_matches_saved_views_where_available": True,
                   "blind_page_contains_no_method_or_label_metadata": True,
                   "hidden_duplicate_spacing": True, "complete_1240_card_readback": True,
                   "training_encoder_evaluators_test_lpips_zero": True},
        "counts": {"source_G4_rows": 60_000, "selected_unique_rows": 1_200,
                   "preflight_rows": 120, "formal_replay_rows": 1_080,
                   "review_cards": 1_240, "png_readback": len(public_card_files)}})
    print(json.dumps({"status": report["status"], "run_id": run_id,
                      "review_page": str(output_dir / "blind_review.html"),
                      "hardware_specification": str(output_dir / "hardware_specification.md")},
                     ensure_ascii=False), flush=True)


def finalize(output_dir: Path, ratings_path: Path) -> None:
    key = _read_json(output_dir / "blind_review_key.json")
    public = _read_json(output_dir / "blind_review_public_manifest.json")
    report = _read_json(output_dir / "report.json")
    _require(report.get("status") == "AWAITING_HUMAN_RATINGS"
             and report.get("review_id") == key.get("review_id")
             and public.get("review_id") == key.get("review_id"),
             "review is not awaiting ratings or its identities disagree")
    _validate_review_package(output_dir, key)
    ratings = _read_ratings(ratings_path, key)
    result = _summarize_review(key, ratings, _read_json(output_dir / "run_config.json"))
    config = _read_json(output_dir / "run_config.json")
    independent_check = _independently_recompute_primary_review(key, ratings, config, result)
    _write_json(output_dir / "human_review_independent_count_check.json", independent_check)
    ratings_archive = output_dir / "human_ratings_submitted.json"
    if ratings_archive.exists():
        _require(_read_json(ratings_archive) == _read_json(ratings_path),
                 "archived human ratings already exist and differ from the submitted file")
    else:
        ratings_archive.write_bytes(ratings_path.read_bytes())
    _write_json(output_dir / "human_review_results.json", result)
    report.update({"status": result["status"], "decision": result["decision"],
                   "human_review_received": True, "human_review_qc_passed": result["quality_control"]["passed"],
                   "primary_comparison": result["primary_comparison"]})
    _write_json(output_dir / "report.json", report)
    _write_report(output_dir, report, _read_json(output_dir / "preflight_receipt.json"), result)
    _require(result["status"] in {"COMPLETE", "REVIEW_INVALID_QC"},
             "review consumer returned an unknown final state")


def _independently_recompute_primary_review(
    key: dict[str, Any], ratings: dict[str, str | int], config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the primary rate and stratified bootstrap using paired strata."""
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for stream_id in STREAMS:
        for label in CLASSES:
            arms: dict[str, np.ndarray] = {}
            for method in METHODS:
                values = [float(ratings[card["card_id"]] == label) for card in key["cards"]
                          if card.get("kind") == "generated" and card.get("method") == method
                          and int(card.get("stream", -1)) == stream_id
                          and int(card.get("requested_label", -1)) == label]
                _require(len(values) == 20,
                         f"independent score census is incomplete for {method}/{stream_id}/{label}")
                arms[method] = np.asarray(values, dtype=np.float64)
            strata.append((arms["template"], arms["original"]))
    observed = float(sum(float(template.mean() - original.mean())
                         for template, original in strata) / len(strata))
    rng = np.random.default_rng(int(config["review"]["bootstrap_seed"]))
    draws = np.zeros(int(config["review"]["bootstrap_resamples"]), dtype=np.float64)
    for template, original in strata:
        template_sample = rng.integers(0, template.size, size=(len(draws), template.size))
        original_sample = rng.integers(0, original.size, size=(len(draws), original.size))
        draws += (template[template_sample].mean(axis=1)
                  - original[original_sample].mean(axis=1)) / len(strata)
    interval = np.quantile(draws, [0.025, 0.975], method="linear").tolist()
    production = result["primary_comparison"]
    _require(math.isclose(observed, float(production["difference"]), rel_tol=0.0, abs_tol=1e-15)
             and np.allclose(interval, production["ci_95"], rtol=0.0, atol=1e-15),
             "independent human-rating count or bootstrap differs from the production result")
    return {
        "task_id": TASK_ID,
        "status": "PASSED",
        "generated_rows": 1_200,
        "strata": len(strata),
        "independently_recomputed_difference": observed,
        "independently_recomputed_ci_95": [float(interval[0]), float(interval[1])],
        "production_difference_and_interval_match": True,
        "bootstrap_seed": int(config["review"]["bootstrap_seed"]),
        "bootstrap_resamples": int(config["review"]["bootstrap_resamples"]),
    }


def _validate_return_package(output_dir: Path) -> dict[str, Any]:
    key = _read_json(output_dir / "blind_review_key.json")
    public = _read_json(output_dir / "blind_review_public_manifest.json")
    report = _read_json(output_dir / "report.json")
    receipt = _read_json(output_dir / "validation_receipt.json")
    replay = _read_json(output_dir / "cuda_execution_receipt.json")
    _require(receipt.get("status") == "PASSED" and replay.get("status") == "PASSED"
             and report.get("task_id") == TASK_ID and report.get("source_run_id") == SOURCE_RUN_ID
             and report.get("source_step") == SOURCE_STEP
             and replay.get("major_computation_device") == "cuda"
             and replay.get("full_replay_rows") == 1_200
             and public.get("card_count") == 1_240 and len(key.get("cards", [])) == 1_240,
             "final package receipt, CUDA evidence, or blind population is incomplete")
    with (output_dir / "saved_evaluator_population_summary.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        _require(sum(1 for _ in csv.DictReader(stream)) == 60,
                 "saved evaluator summary does not contain all 60 strata")
    _validate_review_package(output_dir, key)
    _validate_public_page(output_dir, key)
    images = sorted((output_dir / "review_cards").glob("*.png"))
    _require(len(images) == 1_240, "full native image package has an incomplete row count")
    for path in images:
        with Image.open(path) as image:
            _require(image.mode == "L" and image.size == (200, 200),
                     f"native review card is malformed: {path.name}")
    if (output_dir / "human_review_results.json").is_file():
        result = _read_json(output_dir / "human_review_results.json")
        independent = _read_json(output_dir / "human_review_independent_count_check.json")
        _require(report.get("decision") == result.get("decision")
                 and report.get("primary_comparison") == result.get("primary_comparison"),
                 "final human result and report do not agree")
        _require((output_dir / "human_ratings_submitted.json").is_file()
                 and independent.get("status") == "PASSED"
                 and independent.get("production_difference_and_interval_match") is True,
                 "final human ratings archive or independent statistic check is incomplete")
    return {"task_id": TASK_ID, "status": "PASSED", "review_cards": len(images),
        "unique_generated": 1_200, "per_method": 600, "per_method_stream_class": 20,
        "source_g4_rows": 60_000, "cuda_device": replay["device"],
        "cuda_major_computation_device": replay["major_computation_device"],
        "source_global_step": SOURCE_STEP, "source_run_id": SOURCE_RUN_ID,
        "human_ratings_received": (output_dir / "human_review_results.json").is_file(),
        "human_decision": report.get("decision", "AWAITING_HUMAN_RATINGS"),
        "system_status_preserved": "VALID_FAILED"}


def readback(output_dir: Path) -> None:
    receipt = _validate_return_package(output_dir)
    _write_json(output_dir / "windows_local_readback_receipt.json", {
        **receipt, "consumer_host": "Windows"})


def validate_package(output_dir: Path) -> None:
    receipt = _validate_return_package(output_dir)
    _write_json(output_dir / "package_validation_receipt.json", {
        **receipt, "consumer_host": "autodl-monthly"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("specification", "prepare", "finalize", "validate-package", "readback"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ratings-file", type=Path)
    parser.add_argument("--repository-root", type=Path,
                        help="canonical data/config root when this entrypoint is staged independently")
    parser.add_argument("--task-input-root", type=Path,
                        help="isolated Windows-readback and historical review evidence bundle")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repository_root is not None:
        global ROOT
        ROOT = args.repository_root.resolve()
        _require(ROOT.is_dir() and (ROOT / "src").is_dir(),
                 "repository root must contain the canonical src directory")
    if args.stage in {"specification", "prepare"}:
        _require(args.config is not None, "--config is required for this stage")
        if args.stage == "specification":
            config, paths = _load_config(args.config.resolve(), args.task_input_root)
            _hardware_spec(paths, paths["output_dir"])
            _write_json(paths["output_dir"] / "specification_receipt.json", {
                "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
                "source_config": config["source"]["runtime_config"],
                "specification": "hardware_specification.md"})
        else:
            prepare(args.config.resolve(), resume=args.resume, task_input_root=args.task_input_root)
    else:
        _require(args.output_dir is not None, "--output-dir is required for this stage")
        output_dir = args.output_dir.resolve()
        if args.stage == "finalize":
            _require(args.ratings_file is not None, "--ratings-file is required for finalization")
            finalize(output_dir, args.ratings_file.resolve())
        elif args.stage == "validate-package":
            validate_package(output_dir)
        else:
            readback(output_dir)


if __name__ == "__main__":
    main()
