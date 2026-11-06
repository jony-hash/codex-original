"""C3.5 energy-constrained and fabrication-aware HW207 numerical diagnostic.

This entrypoint reuses the C3.4 scalar Fresnel Model-B forward and its fixed
binary input proxy and ROI. It never loads the MNIST official test set and does
not represent a calibrated HW207 bench or a characterized printer/material.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as c34_runner
from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen import hw207_mnist4_c34_process_aware_v1 as c34_process
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_mnist4_c35_v1.json"
IDENTITY = "HW207_MNIST4_C35_ENERGY_AND_PROCESS_DIAGNOSTIC"
CHECKPOINT_IDENTITY = "C35_PHASE_OR_HEIGHT_CHECKPOINT"


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("identity") != IDENTITY:
        raise ValueError("unexpected C3.5 configuration identity")
    if config.get("wavelength_nm") != 532.0:
        raise ValueError("C3.5 is frozen to the 532 nm numerical scenario")
    if config["geometry"]["phase_grid"] != [100, 100] or config["geometry"]["phase_pitch_um"] != 2.0:
        raise ValueError("C3.5 phase grid or pitch changed")
    if config["geometry"]["aperture_diameter_mm"] != 2.0:
        raise ValueError("C3.5 training aperture must remain the C3.4 2 mm scenario")
    geometry = config["geometry"]
    expected_geometry = {
        "screen_to_lens_mm": c34.MODEL_B_D1_UM / 1000.0,
        "lens_focal_length_mm": c34.MODEL_B_F_UM / 1000.0,
        "lens_to_cmos_mm": c34.MODEL_B_D2_UM / 1000.0,
        "camera_shape": list(c34.CMOS_SHAPE),
        "camera_pixel_pitch_um": c34.CMOS_PITCH_UM,
        "lens_plane_pitch_um": 4.0,
    }
    for name, expected in expected_geometry.items():
        if geometry.get(name) != expected:
            raise ValueError(f"C3.5 geometry field {name} does not match the C3.4 forward model")
    return config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_optimizer_state(optimizer: torch.optim.Optimizer, saved_state: dict[str, Any]) -> None:
    optimizer.load_state_dict(copy.deepcopy(saved_state))


def _indices_and_counts(config: dict[str, Any], *, ledger_dir: Path | None = None,
                        c33_holdout_rows: list[int] | None = None) -> dict[str, Any]:
    index_path = ROOT / config["data"]["c0_indices_path"]
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    roles = {name: torch.as_tensor(value["original_indices"], dtype=torch.long)
             for name, value in index_data.items()}
    labels_path = ROOT / config["data"]["mnist_train_label_path"]
    raw = labels_path.read_bytes()
    if struct.unpack(">II", raw[:8]) != (2049, 60_000):
        raise ValueError("unexpected MNIST training label header")
    labels = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()
    expected_train = torch.nonzero(labels < 4).flatten()
    train_union = torch.unique(torch.cat([roles["train"], roles["validation"]]), sorted=True)
    official_test_path = ROOT / "data/MNIST/raw/t10k-labels-idx1-ubyte"
    test_raw = official_test_path.read_bytes()
    if struct.unpack(">II", test_raw[:8]) != (2049, 10_000):
        raise ValueError("unexpected MNIST official test label header")
    test_labels = torch.from_numpy(np.frombuffer(test_raw, dtype=np.uint8, offset=8).copy()).long()
    expected_test = torch.nonzero(test_labels < 4).flatten()
    expected_c0_counts = {"train": 22_279, "validation": 2_475, "test": 4_157}
    for role, expected_count in expected_c0_counts.items():
        if len(roles[role]) != expected_count or len(torch.unique(roles[role])) != expected_count:
            raise ValueError(f"C0 {role} index count or uniqueness does not match its audited population")
        selected_labels = labels[roles[role]] if role != "test" else test_labels[roles[role]]
        if (selected_labels > 3).any():
            raise ValueError(f"C0 {role} indices include a label outside 0-3")
    if not torch.equal(train_union, expected_train):
        raise ValueError("C0 train/validation indices do not cover the 0-3 MNIST train population exactly")
    if not torch.equal(torch.sort(roles["test"]).values, expected_test):
        raise ValueError("C0 test indices do not cover the 0-3 official test population exactly")
    train_val_disjoint = not bool(torch.isin(roles["train"], roles["validation"]).any())
    c34_split = c34.stratified_c34_split(c33.c0_train_indices(), labels)
    expected_c34_counts = {"train": 18_279, "validation": 2_000, "holdout": 2_000}
    c34_role_by_index: dict[int, str] = {}
    for role, indices in c34_split.items():
        if len(indices) != expected_c34_counts[role]:
            raise ValueError(f"C3.4 {role} split count differs from the audited population")
        for index in indices.tolist():
            if int(index) in c34_role_by_index:
                raise ValueError("C3.4 data roles overlap")
            c34_role_by_index[int(index)] = role
    c33_holdout = set(int(v) for v in (c33_holdout_rows or []))
    if len(c33_holdout) != len(c33_holdout_rows or []):
        raise ValueError("C3.3 historical holdout indices contain duplicates")
    if c33_holdout and any(index < 0 or index >= len(labels) or int(labels[index]) > 3
                           for index in c33_holdout):
        raise ValueError("C3.3 historical holdout contains an invalid training row")
    ledger_path: str | None = None
    if ledger_dir is not None:
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_file = ledger_dir / "c35_data_usage_ledger.csv"
        with ledger_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "original_index", "label", "c0_role", "c34_role",
                             "c33_phase_holdout", "prior_project_use", "independent_for_c35",
                             "other_c33_train_validation_role"))
            for role in ("train", "validation", "test"):
                source_labels = labels if role != "test" else test_labels
                for index in roles[role].tolist():
                    writer.writerow(("mnist_train" if role != "test" else "mnist_official_test",
                                     int(index), int(source_labels[index]), role,
                                     c34_role_by_index.get(int(index), ""),
                                     int(int(index) in c33_holdout) if role != "test" else 0,
                                     1, 0, "UNKNOWN_NOT_SAVED_IN_C33_SUMMARIES"))
        ledger_path = str(ledger_file)
    test_0_to_3 = len(expected_test)
    return {
        "index_path": str(index_path.relative_to(ROOT)),
        "counts": {key: int(len(value)) for key, value in roles.items()},
        "mnist_train_0_to_3": int(len(expected_train)),
        "mnist_official_test_0_to_3": test_0_to_3,
        "train_validation_disjoint": train_val_disjoint,
        "c0_train_plus_validation_cover_all_train_0_to_3": True,
        "c0_test_covers_all_official_test_0_to_3": True,
        "unused_0_to_3_samples": 0,
        "c34_role_counts": expected_c34_counts,
        "c33_phase_holdout_rows_audited": len(c33_holdout),
        "row_level_ledger_path": ledger_path,
        "row_level_ledger_rows": sum(len(value) for value in roles.values()) if ledger_path else None,
        "c33_train_validation_row_roles": "NOT_RECONSTRUCTED; archived C3.3 summaries retain counts but not exact row lists",
        "independent_eval_status": "INDEPENDENT_EVAL_BLOCKED",
        "official_test_labels_read_for_index_census": True,
        "official_test_images_evaluated": False,
    }


def audit_saved_evidence(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Audit threshold provenance, saved energy definitions, and data exposure."""
    p3_summary_path = ROOT / config["sources"]["c34_p3_summary"]
    p3_summary = json.loads(p3_summary_path.read_text(encoding="utf-8"))
    p3_preflight_path = ROOT / "outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_closeout_01a18b9f/p3_aux/training_preflight.json"
    p3_preflight = json.loads(p3_preflight_path.read_text(encoding="utf-8"))
    frozen_path = ROOT / "outputs/20261104_hw207_mnist4_c34_v1/runs/autodl_backup_c34_p0_p2_closeout_01a18b9f/p2_frozen_inference/frozen_phase_comparison.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    attempt_path = ROOT / config["sources"]["c34_attempt05_summary"]
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    threshold = float(p3_summary["throughput_floor_from_train_zero_phase_reference"])
    config_floor = float(p3_preflight["throughput_floor"]["efficiency_vs_input"])
    zero_phase_base = config_floor / float(p3_preflight["throughput_floor"]["fraction_of_train_zero_phase_reference"])
    p2_model = frozen["model_b_estimated_geometry"]
    c33_metrics = p2_model["aperture_assumptions_mm"]["2.0"]
    zero_metrics = p2_model["no_phase_d2mm_metrics"]
    p3_holdout = p3_summary["holdout_metrics"]
    attempt_d05 = next(item for item in attempt["validation_comparisons"]
                       if float(item["delta_n_assumption"]) == 0.5)["process_aware_height_under_hard_process"]
    attempt_d06 = next(item for item in attempt["validation_comparisons"]
                       if float(item["delta_n_assumption"]) == 0.6)["process_aware_height_under_hard_process"]
    population = _indices_and_counts(
        config, ledger_dir=output_dir,
        c33_holdout_rows=frozen["population"]["original_indices"])
    audit = {
        "identity": "C3.5_P0_ENERGY_AND_EXPOSURE_AUDIT",
        "status": "PASS_WITH_HISTORICAL_ROW_RECALCULATION_LIMITATION",
        "threshold": {
            "value": threshold,
            "percent": 100.0 * threshold,
            "source": "C3.4 training preflight: 0.1 times the mean training-set zero-phase Model-B efficiency",
            "zero_phase_train_reference_efficiency": zero_phase_base,
            "definition": "project engineering reference derived from a numerical training baseline",
            "physical_or_literature_basis_found": False,
            "status": "EFFICIENCY_THRESHOLD_UNVERIFIED",
            "use_in_C3_5": "provisional Pareto reference only; not a physical acceptance threshold",
        },
        "metric_definition": {
            "name": "eta_phase_to_cmos",
            "numerator": "sum(I_cmos over full 2048x2048 sensor) * 6.5^2 um^2",
            "denominator": "sum(abs(E_phase_input)^2 over 100x100 plane) * 2^2 um^2",
            "pupil_energy": "sum(abs(E_after_pupil)^2 over sampled pupil) * lens_pitch_um^2",
            "roi_energy": "sum of four fixed C3.3 ROI integrals; not used as the efficiency numerator",
            "source_function": "scripts/research/lightgen/hw207_mnist4_c34_v1.py:model_b_forward",
            "source_lines": "217-274",
            "includes_upstream_laser_and_relay_losses": False,
        },
        "historical_saved_values": {
            "no_phase_model_b_2mm": {
                "samples": int(zero_metrics["sample_count"]),
                "accuracy": float(zero_metrics["accuracy_all_samples"]),
                "mean_detected_energy_um2": float(zero_metrics["mean_detected_energy_um2"]),
                "mean_eta_phase_to_cmos": float(zero_metrics["mean_full_sensor_efficiency_vs_input"]),
                "source": str(frozen_path.relative_to(ROOT)),
                "population_role": "C3.3 historical holdout, previously consumed",
            },
            "c33_frozen_phase_model_b_2mm": {
                "samples": int(c33_metrics["sample_count"]),
                "accuracy": float(c33_metrics["accuracy_all_samples"]),
                "mean_detected_energy_um2": float(c33_metrics["mean_detected_energy_um2"]),
                "mean_eta_phase_to_cmos": float(c33_metrics["mean_full_sensor_efficiency_vs_input"]),
                "source": str(frozen_path.relative_to(ROOT)),
                "population_role": "same C3.3 historical holdout as no-phase comparison",
            },
            "c34_p3_geometry_adapted_phase": {
                "samples": int(p3_holdout["sample_count"]),
                "accuracy": float(p3_holdout["accuracy_all_samples"]),
                "mean_detected_energy_um2": float(p3_holdout["mean_detected_energy_um2"]),
                "mean_eta_phase_to_cmos": float(p3_holdout["mean_full_sensor_efficiency_vs_input"]),
                "source": str(p3_summary_path.relative_to(ROOT)),
                "population_role": "C3.4 holdout; prior selection and diagnostic use recorded",
            },
            "c34_attempt05_process_aware_delta_n_0.5": {
                "mean_accuracy": float(attempt_d05["mean_accuracy_all_samples"]),
                "mean_eta_phase_to_cmos": float(attempt_d05["mean_full_sensor_efficiency_vs_input"]),
                "source": str(attempt_path.relative_to(ROOT)),
                "population_role": "C3.4 validation under assumed process realizations",
                "process_assumption_only": True,
            },
            "c34_attempt05_process_aware_delta_n_0.6": {
                "mean_accuracy": float(attempt_d06["mean_accuracy_all_samples"]),
                "mean_eta_phase_to_cmos": float(attempt_d06["mean_full_sensor_efficiency_vs_input"]),
                "source": str(attempt_path.relative_to(ROOT)),
                "population_role": "C3.4 validation under assumed process realizations",
                "process_assumption_only": True,
            },
        },
        "independent_rowwise_historical_recalculation": {
            "status": "NOT_AVAILABLE_FROM_ARCHIVED_C3_4_RAW_OUTPUTS",
            "reason": "C3.3/P3 summaries lack matching per-row input, pupil and full-sensor energy vectors; attempt05 arrays store detected energy but omit input/pupil vectors.",
            "action": "C3.5 control replay will recompute all four models on the same C3.4 validation rows through the verified Model-B forward and save each energy term.",
        },
        "population_audit": population,
        "source_files": [
            str(p3_summary_path.relative_to(ROOT)), str(p3_preflight_path.relative_to(ROOT)),
            str(frozen_path.relative_to(ROOT)), str(attempt_path.relative_to(ROOT)),
        ],
    }
    _write_json(output_dir / "c35_energy_audit.json", audit)
    _write_json(output_dir / "c35_independent_eval.json", {
        "identity": "C3.5_P4_INDEPENDENT_EVALUATION_AUDIT",
        "status": "INDEPENDENT_EVAL_BLOCKED",
        "population_audit": population,
        "permitted_action": "continue development-only train/validation diagnostics; no official test replay",
    })
    return audit


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[int(label), int(prediction)] += 1
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, confusion.sum(0), out=np.zeros(4), where=confusion.sum(0) > 0)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(4), where=confusion.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(4), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / max(1, confusion.sum())),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
        "sample_count": int(confusion.sum()),
    }


def _shuffled_label_negative_control(labels: np.ndarray, predictions: np.ndarray,
                                     seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if (labels.ndim != 1 or predictions.shape != labels.shape or len(labels) == 0
            or np.any((labels < 0) | (labels > 3))
            or np.any((predictions < 0) | (predictions > 3))):
        raise ValueError("shuffled-label negative control requires matching four-class vectors")
    shuffled = labels[np.random.default_rng(int(seed)).permutation(len(labels))]
    if not np.array_equal(np.bincount(labels, minlength=4), np.bincount(shuffled, minlength=4)):
        raise RuntimeError("shuffled-label negative control changed the class marginals")
    return shuffled, _classification_metrics(shuffled, predictions)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _eta_phase_to_cmos(cmos_energy: torch.Tensor | np.ndarray,
                       input_energy: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Full-sensor CMOS energy divided by phase-screen input-plane energy."""
    if isinstance(cmos_energy, torch.Tensor) or isinstance(input_energy, torch.Tensor):
        detected = torch.as_tensor(cmos_energy)
        incident = torch.as_tensor(input_energy, device=detected.device)
        if not torch.isfinite(detected).all() or not torch.isfinite(incident).all() or (incident <= 0).any():
            raise ValueError("eta_phase_to_cmos requires finite energies and positive input energy")
        return detected / incident
    detected_np = np.asarray(cmos_energy, dtype=np.float64)
    incident_np = np.asarray(input_energy, dtype=np.float64)
    if (not np.isfinite(detected_np).all() or not np.isfinite(incident_np).all()
            or np.any(incident_np <= 0)):
        raise ValueError("eta_phase_to_cmos requires finite energies and positive input energy")
    return detected_np / incident_np


def _resolve_phase(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        for key in ("phase", "phase_rad", "effective"):
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                value = candidate
                break
    phase = torch.as_tensor(value, dtype=torch.float32).squeeze()
    if phase.shape != (100, 100) or not torch.isfinite(phase).all():
        raise ValueError("loaded phase is not a finite 100x100 tensor")
    return phase.detach().cpu()


def _load_control_phases(config: dict[str, Any]) -> dict[str, torch.Tensor]:
    source_paths = {key: ROOT / value for key, value in config["sources"].items()
                    if key in {"c33_frozen_phase", "c33_bandlimited_phase", "c34_p3_phase", "c34_attempt05_height"}}
    c33_state = torch.load(source_paths["c33_frozen_phase"], map_location="cpu", weights_only=False)
    c33_phase = _resolve_phase(c33_state)
    c33_bandlimited_state = torch.load(source_paths["c33_bandlimited_phase"], map_location="cpu", weights_only=False)
    c33_bandlimited_phase = _resolve_phase(c33_bandlimited_state)
    p3_phase = _resolve_phase(np.load(source_paths["c34_p3_phase"], allow_pickle=False))
    height_state = torch.load(source_paths["c34_attempt05_height"], map_location="cpu", weights_only=False)
    height = torch.as_tensor(height_state["height_um"], dtype=torch.float32).squeeze()
    if height.shape != (100, 100) or not torch.isfinite(height).all():
        raise ValueError("C3.4 attempt05 height checkpoint is invalid")
    attempt05_phase = _resolve_phase(c34.phase_from_height(height, 0.5))
    return {
        "no_phase": torch.zeros((100, 100), dtype=torch.float32),
        "c33_frozen": c33_phase,
        "c33_bandlimited": c33_bandlimited_phase,
        "c34_p3": p3_phase,
        "c34_attempt05_height_as_phase_dn0.5": attempt05_phase,
    }


def _load_context(config: dict[str, Any], work_dir: Path, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C3.5 production training and optical propagation require CUDA")
    if os.name == "posix":
        expected_root = Path(config["run"]["autodl_system_work_dir"]).resolve().parent
        if not work_dir.resolve().is_relative_to(expected_root):
            raise ValueError("C3.5 run outputs must remain on the AutoDL system disk")
    images, labels, split = c34_runner._split_and_data()
    if (len(split["train"]) != config["data"]["train_count"]
            or len(split["validation"]) != config["data"]["validation_count"]):
        raise ValueError("C3.5 train/validation rows differ from the frozen configuration")
    if len(split["holdout"]) != config["data"]["previously_used_holdout_count"]:
        raise ValueError("C3.5 historical holdout row count changed")
    if torch.isin(split["train"], split["validation"]).any():
        raise ValueError("C3.5 train and validation rows overlap")
    if torch.isin(split["validation"], split["holdout"]).any():
        raise ValueError("C3.5 validation and prior holdout rows overlap")
    if (labels[torch.cat((split["train"], split["validation"]))] > 3).any():
        raise ValueError("C3.5 data contains labels outside 0-3")
    input_configuration = c33.build_configuration(52_000.0, str(device))
    input_configuration.base.screen.requires_grad_(False)
    required_rows = torch.sort(torch.cat([split["train"], split["validation"]])).values
    cache_path = work_dir / "cache" / "input_field_cache.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        cache_rows = torch.as_tensor(cache_payload.get("rows"), dtype=torch.long)
        fields_cpu = cache_payload.get("fields")
        if (cache_payload.get("identity") != "C35_FIXED_BINARY_INPUT_CACHE"
                or not torch.equal(cache_rows, required_rows)
                or not isinstance(fields_cpu, torch.Tensor)
                or fields_cpu.shape != (len(required_rows), 100, 100)
                or fields_cpu.dtype != torch.complex64 or not torch.isfinite(fields_cpu).all()):
            raise ValueError("saved C3.5 input cache does not match current train/validation rows")
        fields = fields_cpu.to(device)
        cache_metadata = dict(cache_payload["metadata"])
    else:
        cache_rows, fields, cache_metadata = c34_runner._prepare_field_cache(
            images, required_rows, input_configuration, device, chunk_size=16)
        payload = {"identity": "C35_FIXED_BINARY_INPUT_CACHE", "rows": cache_rows.cpu(),
                   "fields": fields.detach().cpu(), "metadata": cache_metadata}
        _atomic_torch_save(cache_path, payload)
    return {"images": images, "labels": labels, "split": split, "device": device,
            "input_configuration": input_configuration, "cache": (required_rows.to(device), fields),
            "cache_path": cache_path, "cache_metadata": cache_metadata,
            "lens_pitch_um": float(config["geometry"]["lens_plane_pitch_um"]),
            "screen_to_lens_um": float(config["geometry"]["screen_to_lens_mm"]) * 1000.0,
            "lens_to_cmos_um": float(config["geometry"]["lens_to_cmos_mm"]) * 1000.0,
            "aperture_mm": float(config["geometry"]["aperture_diameter_mm"]),
            "propagation_cache": {}, "work_dir": work_dir}


def _evaluate_phase(context: dict[str, Any], phase: torch.Tensor, rows: torch.Tensor,
                    tag: str, output_path: Path, *, aperture_mm: float = 2.0,
                    class_average: bool = False, eval_batch: int = 8,
                    save_records: bool = True) -> dict[str, Any]:
    device = context["device"]
    phase = phase.to(device=device, dtype=torch.float32).detach()
    labels_all = context["labels"]
    score_parts: list[np.ndarray] = []
    input_parts: list[np.ndarray] = []
    pupil_parts: list[np.ndarray] = []
    cmos_parts: list[np.ndarray] = []
    crop_parts: list[np.ndarray] = []
    outside_parts: list[np.ndarray] = []
    class_sums = torch.zeros((4, 2048, 2048), dtype=torch.float64) if class_average else None
    class_counts = torch.zeros(4, dtype=torch.long) if class_average else None
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), eval_batch):
            chunk_rows = rows[start:start + eval_batch].to(device=device, dtype=torch.long)
            fields = c34_runner._fields_for_rows(context["cache"], chunk_rows)
            result = c34.model_b_forward(
                fields, phase, aperture_mm, lens_pitch_um=context["lens_pitch_um"],
                plan_cache=context["propagation_cache"], return_camera=class_average,
                screen_to_lens_um=context["screen_to_lens_um"],
                lens_to_cmos_um=context["lens_to_cmos_um"])
            scores = result["roi_scores"].detach().cpu().double()
            score_parts.append(scores.numpy())
            input_parts.append(result["input_energy"].detach().cpu().double().numpy())
            pupil_parts.append(result["lens_energy"].detach().cpu().double().numpy())
            cmos_parts.append(result["detected_energy"].detach().cpu().double().numpy())
            crop_parts.append(result["comparison_crop_energy"].detach().cpu().double().numpy())
            outside_parts.append(result["outside_roi_full_cmos_energy"].detach().cpu().double().numpy())
            if class_average:
                intensities = result["camera_intensity"].detach().cpu().double()
                batch_labels = labels_all[chunk_rows.cpu()]
                for cls in range(4):
                    mask = batch_labels == cls
                    if mask.any():
                        class_sums[cls] += intensities[mask].sum(0)
                        class_counts[cls] += mask.sum()
            if start == 0 or (start + eval_batch) % 256 == 0 or start + eval_batch >= len(rows):
                print(f"eval {tag}: rows={min(start + eval_batch, len(rows))}/{len(rows)}", flush=True)
    scores_np = np.concatenate(score_parts)
    input_np = np.concatenate(input_parts)
    pupil_np = np.concatenate(pupil_parts)
    cmos_np = np.concatenate(cmos_parts)
    crop_np = np.concatenate(crop_parts)
    outside_np = np.concatenate(outside_parts)
    rows_np = rows.detach().cpu().numpy().astype(np.int64)
    labels_np = labels_all[rows.detach().cpu()].numpy().astype(np.int64)
    predictions = scores_np.argmax(1).astype(np.int64)
    eta = _eta_phase_to_cmos(cmos_np, input_np)
    pupil_eta = _eta_phase_to_cmos(pupil_np, input_np)
    detected_roi = scores_np.sum(1)
    correct_fraction = scores_np[np.arange(len(rows_np)), labels_np] / np.maximum(cmos_np, 1e-30)
    wrong = scores_np.copy()
    wrong[np.arange(len(rows_np)), labels_np] = -np.inf
    strongest_wrong_fraction = wrong.max(1) / np.maximum(cmos_np, 1e-30)
    roi_outside_fraction = outside_np / np.maximum(cmos_np, 1e-30)
    metrics = _classification_metrics(labels_np, predictions)
    metrics.update({
        "coverage": 1.0,
        "eta_phase_to_cmos": _distribution(eta),
        "pupil_eta_vs_input": _distribution(pupil_eta),
        "mean_input_energy_um2": float(input_np.mean()),
        "mean_pupil_energy_um2": float(pupil_np.mean()),
        "mean_cmos_energy_um2": float(cmos_np.mean()),
        "mean_roi_energy_um2": scores_np.mean(0).tolist(),
        "mean_roi_outside_cmos_energy_um2": float(outside_np.mean()),
        "roi_outside_cmos_fraction_mean": float(roi_outside_fraction.mean()),
        "correct_roi_energy_fraction_mean": float(correct_fraction.mean()),
        "strongest_wrong_roi_energy_fraction_mean": float(strongest_wrong_fraction.mean()),
        "correct_minus_strongest_wrong_fraction_mean": float((correct_fraction - strongest_wrong_fraction).mean()),
        "class_eta_mean": {str(cls): float(eta[labels_np == cls].mean()) for cls in range(4)},
        "class_correct_roi_fraction_mean": {str(cls): float(correct_fraction[labels_np == cls].mean()) for cls in range(4)},
        "all_zero_roi_scores": int(np.sum(detected_roi == 0)),
        "tied_maximum_roi_rows": int(np.sum((scores_np == scores_np.max(1, keepdims=True)).sum(1) > 1)),
        "decision": "argmax of four fixed ROI integrals; ties select the lowest ROI index",
        "elapsed_seconds": float(time.perf_counter() - start_time),
    })
    payload = {
        "original_rows": rows_np, "labels": labels_np, "predictions": predictions,
        "roi_scores": scores_np.astype(np.float32), "input_energy_um2": input_np,
        "pupil_energy_um2": pupil_np, "cmos_energy_um2": cmos_np,
        "crop_energy_um2": crop_np, "roi_outside_cmos_energy_um2": outside_np,
        "eta_phase_to_cmos": eta, "pupil_eta_vs_input": pupil_eta,
        "correct_roi_fraction": correct_fraction,
        "strongest_wrong_roi_fraction": strongest_wrong_fraction,
        "roi_outside_cmos_fraction": roi_outside_fraction,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_records:
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(output_path)
    if class_average:
        means = class_sums.numpy() / class_counts.numpy()[:, None, None]
        class_path = output_path.with_name(output_path.stem + "_class_average_cmos.npy")
        class_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(class_path, means.astype(np.float32), allow_pickle=False)
        metrics["class_average_cmos_path"] = str(class_path)
        try:
            import matplotlib.pyplot as plt

            displayed = np.log1p(np.maximum(means, 0.0))
            figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
            image = None
            for cls, axis in enumerate(axes.flat):
                image = axis.imshow(displayed[cls], cmap="magma", vmin=float(displayed.min()),
                                    vmax=float(displayed.max()))
                axis.set_title(f"Class {cls} mean CMOS intensity (log1p)")
                axis.set_axis_off()
            figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.72,
                            label="log(1 + mean intensity)")
            figure_path = output_path.with_name(output_path.stem + "_class_average_cmos.png")
            figure.savefig(figure_path, dpi=150)
            plt.close(figure)
            metrics["class_average_cmos_figure_path"] = str(figure_path)
        except ImportError as error:
            raise RuntimeError("matplotlib is required to deliver C3.5 CMOS intensity figures") from error
    _write_json(output_path.with_suffix(".json"), {"identity": "C3.5_FIXED_ROI_OPTICAL_READOUT", "tag": tag,
                                                       "metrics": metrics,
                                                       "row_records": output_path.name if save_records else None,
                                                       "full_sensor_energy_saved": True})
    return metrics


def _effective_phase(variable: torch.Tensor, mode: str, config: dict[str, Any], *,
                     noise_seed: int, edge_bias_nm: float, training: bool) -> torch.Tensor:
    if mode == "phase":
        return variable
    if mode != "height":
        raise ValueError(f"unsupported trainable variable mode {mode}")
    process_cfg = config["fabrication_assumptions"]
    noise = c34_process._noise_field(variable, noise_seed, float(process_cfg["height_noise_nm"]))
    height = c34_process.process_height(
        variable, psf_sigma_px=float(process_cfg["psf_sigma_px"]),
        minimum_feature_radius_px=int(process_cfg["minimum_feature_radius_px"]),
        height_levels=int(process_cfg["height_levels"]),
        maximum_height_um=float(process_cfg["maximum_height_um"]),
        noise_nm=float(process_cfg["height_noise_nm"]), noise_seed=int(noise_seed),
        edge_height_bias_nm=float(edge_bias_nm), noise_field=noise,
        straight_through_quantization=training)
    return c34.phase_from_height(height, float(process_cfg["delta_n_scenarios"][0]))


def _objective(result: dict[str, torch.Tensor | None], labels: torch.Tensor,
               phase: torch.Tensor, target_efficiency: float, multiplier: float,
               rho: float, config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = result["roi_scores"]
    crop = result["comparison_crop_energy"].clamp_min(1e-20)
    logits = 20.0 * scores / crop[:, None]
    classification = F.cross_entropy(logits.float(), labels)
    row_ids = torch.arange(len(labels), device=labels.device)
    correct = scores[row_ids, labels]
    incorrect = scores.clone()
    incorrect[row_ids, labels] = -float("inf")
    margin = F.relu(1.0 - (correct - incorrect.max(1).values) / crop).mean()
    leakage = (result["outside_roi_full_cmos_energy"] /
               result["detected_energy"].clamp_min(1e-20)).mean()
    eta = _eta_phase_to_cmos(result["detected_energy"], result["input_energy"].clamp_min(1e-20))
    shortfall = F.relu(float(target_efficiency) - eta)
    constraint = float(multiplier) * shortfall.mean() + 0.5 * float(rho) * shortfall.square().mean()
    phase_x = torch.sin((phase[..., 1:] - phase[..., :-1]) / 2).abs().mean()
    phase_y = torch.sin((phase[..., 1:, :] - phase[..., :-1, :]) / 2).abs().mean()
    weights = config["training"]
    loss = (classification + float(weights["classification_margin_weight"]) * margin
            + float(weights["roi_leakage_weight"]) * leakage + constraint
            + float(weights["phase_smoothness_weight"]) * (phase_x + phase_y))
    components = {"classification": classification.detach(), "margin": margin.detach(),
                  "leakage": leakage.detach(), "constraint": constraint.detach(),
                  "eta_mean": eta.detach().mean(), "eta_shortfall_mean": shortfall.detach().mean()}
    return loss, components


def _update_batch(variable: torch.nn.Parameter, optimizer: torch.optim.Optimizer,
                  context: dict[str, Any], rows: torch.Tensor, *, mode: str,
                  target_efficiency: float, multiplier: float, config: dict[str, Any],
                  noise_seed: int, edge_bias_nm: float, microbatch: int) -> dict[str, float]:
    total = int(len(rows))
    if total < 1:
        raise ValueError("training update requires at least one sample")
    device = context["device"]
    while True:
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {key: 0.0 for key in
                                  ("loss", "classification", "margin", "leakage", "constraint", "eta_mean", "eta_shortfall_mean")}
        try:
            for start in range(0, total, microbatch):
                selected_rows = rows[start:start + microbatch]
                phase = _effective_phase(variable, mode, config, noise_seed=noise_seed,
                                         edge_bias_nm=edge_bias_nm, training=True)
                fields = c34_runner._fields_for_rows(context["cache"], selected_rows)
                result = c34.model_b_forward(
                    fields, phase, float(config["geometry"]["aperture_diameter_mm"]),
                    lens_pitch_um=context["lens_pitch_um"],
                    plan_cache=context["propagation_cache"], return_camera=False,
                    screen_to_lens_um=context["screen_to_lens_um"],
                    lens_to_cmos_um=context["lens_to_cmos_um"])
                labels = context["labels"][selected_rows.detach().cpu()].to(device)
                loss, parts = _objective(result, labels, phase, target_efficiency,
                                         multiplier, float(config["energy"]["adaptive_lagrange_rho"]), config)
                weight = len(selected_rows) / total
                (loss * weight).backward()
                for key in sums:
                    source = loss.detach() if key == "loss" else parts[key]
                    sums[key] += float(source) * weight
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if microbatch <= 1:
                raise
            microbatch = max(1, microbatch // 2)
            print(f"C3.5 OOM; retrying unfinished logical batch with microbatch={microbatch}", flush=True)
            continue
        if variable.grad is None or not torch.isfinite(variable.grad).all():
            raise RuntimeError("C3.5 production update has absent or non-finite gradients")
        sums["gradient_norm"] = float(variable.grad.detach().norm())
        previous = variable.detach().clone()
        optimizer.step()
        if mode == "height":
            with torch.no_grad():
                variable.clamp_(0.0, float(config["fabrication_assumptions"]["maximum_height_um"]))
        if not torch.isfinite(variable).all():
            raise RuntimeError("C3.5 production update produced non-finite parameters")
        sums["parameter_delta_l2"] = float((variable.detach() - previous).norm())
        sums["microbatch_used"] = float(microbatch)
        return sums


def _selection_key(metrics: dict[str, Any], target: float) -> tuple[Any, ...]:
    mean_eta = float(metrics["eta_phase_to_cmos"]["mean"])
    p10_eta = float(metrics["eta_phase_to_cmos"]["p10"])
    accuracy = float(metrics["accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    min_recall = min(float(v) for v in metrics["recall"])
    if mean_eta >= target:
        return (0, -accuracy, -macro_f1, -p10_eta, -min_recall)
    return (1, target - mean_eta, -accuracy, -macro_f1, -p10_eta, -min_recall)


def _update_lagrange_multiplier(multiplier: float, target: float,
                                observed_efficiency: float, rho: float) -> float:
    values = (multiplier, target, observed_efficiency, rho)
    if not all(math.isfinite(float(value)) for value in values) or rho <= 0:
        raise ValueError("C3.5 Lagrange update requires finite values and positive rho")
    return min(1e6, max(0.0, float(multiplier) + float(rho) *
                        (float(target) - float(observed_efficiency))))


def _epoch_checkpoint(path: Path, *, mode: str, seed: int, target: float,
                      epoch: int, variable: torch.Tensor, optimizer,
                      multiplier: float, history: list[dict[str, Any]],
                      best_key: tuple[Any, ...] | None,
                      best_variable: torch.Tensor | None,
                      best_metrics: dict[str, Any] | None,
                      best_epoch: int | None,
                      train_rows: torch.Tensor, validation_rows: torch.Tensor,
                      config: dict[str, Any]) -> None:
    state = {
        "identity": CHECKPOINT_IDENTITY, "mode": mode, "seed": int(seed),
        "target_efficiency": float(target), "epoch": int(epoch),
        "variable": variable.detach().cpu(), "optimizer": optimizer.state_dict(),
        "multiplier": float(multiplier), "history": history,
        "best_key": best_key,
        "best_variable": None if best_variable is None else best_variable.detach().cpu(),
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "train_rows": train_rows.detach().cpu(), "validation_rows": validation_rows.detach().cpu(),
        "rng_cpu": torch.get_rng_state(), "config_identity": config["identity"],
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    _atomic_torch_save(path, state)


def _train_one(context: dict[str, Any], config: dict[str, Any], *, mode: str,
               seed: int, target: float, initial: torch.Tensor, output_dir: Path,
               resume_checkpoint: Path | None = None) -> dict[str, Any]:
    if mode not in {"phase", "height"}:
        raise ValueError("training mode must be phase or height")
    device = context["device"]
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    variable = torch.nn.Parameter(initial.to(device=device, dtype=torch.float32).detach().clone())
    optimizer = torch.optim.Adam([variable], lr=float(config["training"]["learning_rate"]))
    train_rows = context["split"]["train"].clone()
    validation_rows = context["split"]["validation"].clone()
    history: list[dict[str, Any]] = []
    multiplier = 0.0
    first_epoch = 0
    best_key: tuple[Any, ...] | None = None
    best_variable: torch.Tensor | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch: int | None = None
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if (state.get("identity") != CHECKPOINT_IDENTITY or state.get("mode") != mode
                or int(state.get("seed", -1)) != int(seed)
                or float(state.get("target_efficiency", -1)) != float(target)
                or state.get("config_identity") != config["identity"]
                or not torch.equal(state.get("train_rows"), train_rows)
                or not torch.equal(state.get("validation_rows"), validation_rows)):
            raise ValueError("explicit C3.5 resume checkpoint does not match this training contract")
        if (state.get("best_key") is None or state.get("best_variable") is None
                or state.get("best_metrics") is None or state.get("best_epoch") is None):
            raise ValueError("C3.5 resume checkpoint lacks the prior best-model selection state")
        variable.data.copy_(state["variable"].to(device))
        _restore_optimizer_state(optimizer, state["optimizer"])
        history = list(state["history"])
        multiplier = float(state["multiplier"])
        first_epoch = int(state["epoch"])
        if first_epoch < 0 or first_epoch > int(config["training"]["epochs"]):
            raise ValueError("C3.5 resume checkpoint epoch is outside the configured schedule")
        best_key = tuple(state["best_key"])
        best_variable = torch.as_tensor(state["best_variable"]).detach().cpu().clone()
        best_metrics = dict(state["best_metrics"])
        best_epoch = int(state["best_epoch"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and "rng_cuda" in state:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "epoch_state.pt"
    epochs = int(config["training"]["epochs"])
    batch_size = int(config["training"]["logical_batch_size"])
    microbatch = int(config["training"]["microbatch_size"])
    rho = float(config["energy"]["adaptive_lagrange_rho"])
    for epoch in range(first_epoch, epochs):
        epoch_start = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 1009 * (epoch + 1))
        shuffled = train_rows[torch.randperm(len(train_rows), generator=generator)]
        update_records = []
        eta_sum = 0.0
        row_sum = 0
        for batch_index, start in enumerate(range(0, len(shuffled), batch_size)):
            rows = shuffled[start:start + batch_size]
            edge_values = config["fabrication_assumptions"]["edge_height_bias_nm"]
            edge_bias = (float(edge_values[(int(seed) + epoch + batch_index) % len(edge_values)])
                         if mode == "height" else 0.0)
            update = _update_batch(
                variable, optimizer, context, rows, mode=mode, target_efficiency=target,
                multiplier=multiplier, config=config,
                noise_seed=(1_000_003 * seed + 9_176 * epoch + 97 * batch_index + 34_011) % (2**31 - 1),
                edge_bias_nm=edge_bias, microbatch=microbatch)
            update_records.append(update)
            eta_sum += update["eta_mean"] * len(rows)
            row_sum += len(rows)
        training_eta_proxy = eta_sum / max(row_sum, 1)
        multiplier = _update_lagrange_multiplier(multiplier, target, training_eta_proxy, rho)
        validation_phase = _effective_phase(
            variable, mode, config, noise_seed=int(config["fabrication_assumptions"]["height_noise_seeds"][0]),
            edge_bias_nm=0.0, training=False)
        metrics = _evaluate_phase(
            context, validation_phase.detach(), validation_rows, f"{mode}_seed{seed}_epoch{epoch + 1}",
            output_dir / f"epoch_{epoch + 1:02d}.npz", class_average=False, eval_batch=8,
            save_records=False)
        record = {
            "epoch": epoch + 1, "mode": mode, "seed": int(seed), "target_efficiency": float(target),
            "train_eta_proxy": float(training_eta_proxy), "lagrange_multiplier": float(multiplier),
            "validation": metrics,
            "mean_train_loss": float(np.mean([v["loss"] for v in update_records])),
            "mean_gradient_norm": float(np.mean([v["gradient_norm"] for v in update_records])),
            "elapsed_seconds": float(time.perf_counter() - epoch_start),
        }
        history.append(record)
        key = _selection_key(metrics, target)
        if best_key is None or key < best_key:
            best_key = key
            best_variable = variable.detach().cpu().clone()
            best_metrics = metrics
            best_epoch = epoch + 1
            _atomic_torch_save(output_dir / "best_model.pt", {
                "identity": "C35_BEST_PHASE_OR_HEIGHT", "mode": mode, "seed": int(seed),
                "target_efficiency": float(target), "epoch": epoch + 1,
                "variable": best_variable, "metrics": metrics, "selection_key": list(key),
            })
        _epoch_checkpoint(checkpoint_path, mode=mode, seed=seed, target=target,
                          epoch=epoch + 1, variable=variable, optimizer=optimizer,
                          multiplier=multiplier, history=history, train_rows=train_rows,
                          best_key=best_key, best_variable=best_variable,
                          best_metrics=best_metrics, best_epoch=best_epoch,
                          validation_rows=validation_rows, config=config)
        _write_json(output_dir / "training_history.json", {"identity": CHECKPOINT_IDENTITY,
                                                            "mode": mode, "seed": int(seed),
                                                            "records": history})
        print(json.dumps({"stage": mode, "seed": seed, "epoch": epoch + 1,
                          "target_eta": target, "val_accuracy": metrics["accuracy"],
                          "val_eta_mean": metrics["eta_phase_to_cmos"]["mean"],
                          "val_eta_p10": metrics["eta_phase_to_cmos"]["p10"],
                          "lambda": multiplier}, ensure_ascii=False), flush=True)
    if best_variable is None or best_metrics is None or best_epoch is None:
        raise RuntimeError("C3.5 training produced no validation checkpoint")
    if mode == "phase":
        best_phase = torch.remainder(best_variable, 2 * math.pi)
        np.save(output_dir / "phase_energy_constrained.npy", best_phase.numpy(), allow_pickle=False)
    else:
        np.save(output_dir / "height_map_um.npy", best_variable.numpy(), allow_pickle=False)
        best_phase = _effective_phase(best_variable.to(device), mode, config,
                                      noise_seed=int(config["fabrication_assumptions"]["height_noise_seeds"][0]),
                                      edge_bias_nm=0.0, training=False).detach().cpu()
        np.save(output_dir / "phase_after_nominal_process.npy",
                torch.remainder(best_phase, 2 * math.pi).numpy(), allow_pickle=False)
    return {"mode": mode, "seed": int(seed), "target_efficiency": float(target),
            "best_epoch": best_epoch,
            "best_metrics": best_metrics, "best_variable": best_variable,
            "best_phase": best_phase, "selection_key": list(best_key),
            "history_path": str(output_dir / "training_history.json")}


def _balanced_validation_rows(context: dict[str, Any], per_class: int) -> torch.Tensor:
    rows = []
    labels = context["labels"]
    validation = context["split"]["validation"]
    for cls in range(4):
        selected = validation[labels[validation] == cls]
        if len(selected) < per_class:
            raise ValueError(f"C3.5 validation has fewer than {per_class} samples of class {cls}")
        rows.append(selected[:per_class])
    return torch.cat(rows)


def run_p1(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    root = context["work_dir"] / "p1_energy_constrained"
    validation = context["split"]["validation"]
    phases = _load_control_phases(config)
    controls: dict[str, Any] = {}
    for name, phase in phases.items():
        controls[name] = _evaluate_phase(
            context, phase, validation, name, root / "controls" / f"{name}.npz",
            class_average=False, eval_batch=8, save_records=True)
    p3_initial = phases["c34_p3"]
    candidates = []
    targets = [float(v) for v in config["energy"]["pareto_targets"]]
    seeds = [int(v) for v in config["training"]["seeds"]]
    if not targets or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("C3.5 Pareto scan requires targets and unique predeclared seeds")
    for target in targets:
        for seed in seeds:
            output = root / f"target_{target:.6f}_seed_{seed}"
            result = _train_one(context, config, mode="phase", seed=seed, target=target,
                                initial=p3_initial, output_dir=output)
            metrics = _evaluate_phase(
                context, result["best_phase"], validation, f"p1_target_{target:.6f}_seed_{seed}",
                output / "validation_records.npz", class_average=False, eval_batch=8, save_records=True)
            candidate = {key: value for key, value in result.items() if key not in {"best_variable", "best_phase"}}
            candidate.update({"directory": str(output), "metrics": metrics,
                              "phase_path": str(output / "phase_energy_constrained.npy")})
            candidates.append(candidate)
    reference = float(config["energy"]["historical_reference_floor"])
    feasible = [c for c in candidates if c["metrics"]["eta_phase_to_cmos"]["mean"] >= reference]
    if feasible:
        selected = max(feasible, key=lambda c: (c["metrics"]["accuracy"], c["metrics"]["macro_f1"],
                                                c["metrics"]["eta_phase_to_cmos"]["p10"]))
        selection_status = "PROVISIONAL_REFERENCE_FEASIBLE"
    else:
        selected = min(candidates, key=lambda c: (
            reference - c["metrics"]["eta_phase_to_cmos"]["mean"],
            -c["metrics"]["accuracy"], -c["metrics"]["macro_f1"]))
        selection_status = "NO_CANDIDATE_REACHED_PROVISIONAL_REFERENCE"
    selected_phase = np.load(selected["phase_path"], allow_pickle=False).astype(np.float32)
    np.save(root / "phase_energy_constrained.npy", selected_phase, allow_pickle=False)
    selected_metrics = _evaluate_phase(
        context, torch.from_numpy(selected_phase), validation, "p1_selected_phase",
        root / "selected_validation_records.npz", class_average=True, eval_batch=4, save_records=True)
    selected_checkpoint = torch.load(Path(selected["directory"]) / "best_model.pt",
                                     map_location="cpu", weights_only=False)
    _atomic_torch_save(root / "phase_energy_constrained_weights.pt", selected_checkpoint)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 4))
        for candidate in candidates:
            m = candidate["metrics"]
            axis.scatter(100.0 * m["eta_phase_to_cmos"]["mean"], 100.0 * m["accuracy"],
                         label=f"target {candidate['target_efficiency']:.3g}; seed {candidate['seed']}")
        axis.scatter(100.0 * controls["no_phase"]["eta_phase_to_cmos"]["mean"],
                     100.0 * controls["no_phase"]["accuracy"], marker="x", color="black",
                     label="zero phase")
        axis.axvline(100.0 * reference, linestyle="--", color="gray", label="unverified reference")
        axis.set_xlabel("Mean eta_phase_to_cmos (%)")
        axis.set_ylabel("Fixed ROI validation accuracy (%)")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(root / "p1_pareto.png", dpi=160)
        plt.close(figure)
        phase_figure, phase_axis = plt.subplots(figsize=(5, 4))
        image = phase_axis.imshow(selected_phase, cmap="twilight", vmin=0, vmax=2 * math.pi)
        phase_figure.colorbar(image, ax=phase_axis, label="Wrapped phase (rad)")
        phase_axis.set_title("Selected C3.5 phase candidate")
        phase_figure.tight_layout()
        phase_figure.savefig(root / "phase_energy_constrained.png", dpi=160)
        plt.close(phase_figure)
    except ImportError as error:
        raise RuntimeError("matplotlib is required to deliver the C3.5 Pareto and phase figures") from error
    summary = {
        "identity": "C3.5_P1_ENERGY_CONSTRAINED_TRAINING",
        "status": selection_status,
        "selection_rule": "maximize validation accuracy, then Macro-F1 and P10 efficiency among candidates meeting the provisional mean-efficiency reference; if none, minimize reference shortfall then maximize accuracy and Macro-F1",
        "reference_threshold_status": "EFFICIENCY_THRESHOLD_UNVERIFIED",
        "controls": controls,
        "candidates": [{k: v for k, v in c.items() if k != "metrics"} | {"metrics": c["metrics"]}
                       for c in candidates],
        "selected": {"seed": selected["seed"], "target_efficiency": selected["target_efficiency"],
                     "best_epoch": selected["best_epoch"], "phase_path": str(root / "phase_energy_constrained.npy"),
                     "metrics": selected_metrics},
        "data_role": "C3.4 train/validation; previously used in C0 and C3.4 development, not independent",
        "digital_head_used": False,
    }
    _write_json(root / "p1_summary.json", summary)
    return summary


def run_p2(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    p1_root = context["work_dir"] / "p1_energy_constrained"
    phase_path = p1_root / "phase_energy_constrained.npy"
    if not phase_path.is_file():
        raise FileNotFoundError("C3.5 P2 requires the explicit selected P1 phase")
    phase = _resolve_phase(np.load(phase_path, allow_pickle=False))
    delta_n = float(config["fabrication_assumptions"]["delta_n_scenarios"][0])
    initial_height = c34.height_from_phase(phase, delta_n).float().clamp(
        0.0, float(config["fabrication_assumptions"]["maximum_height_um"]))
    target = float(config["energy"]["historical_reference_floor"])
    result = _train_one(
        context, config, mode="height", seed=int(config["training"]["p2_seed"]),
        target=target, initial=initial_height,
        output_dir=context["work_dir"] / "p2_process_aware")
    validation = context["split"]["validation"]
    metrics = _evaluate_phase(
        context, result["best_phase"], validation, "p2_nominal_processed_height",
        context["work_dir"] / "p2_process_aware" / "validation_records.npz",
        class_average=True, eval_batch=4, save_records=True)
    result["best_metrics"] = metrics
    summary = {
        "identity": "C3.5_P2_PROCESS_AWARE_HEIGHT_TRAINING",
        "status": "ASSUMPTION_ONLY_DIAGNOSTIC_COMPLETE",
        "selected_seed": result["seed"], "selected_epoch": result["best_epoch"],
        "target_efficiency": target, "target_status": "EFFICIENCY_THRESHOLD_UNVERIFIED_PROVISIONAL_REFERENCE",
        "process_assumptions": config["fabrication_assumptions"],
        "selected_height_path": "p2_process_aware/height_map_um.npy",
        "selected_phase_path": "p2_process_aware/phase_after_nominal_process.npy",
        "metrics": metrics,
        "data_role": "C3.4 train/validation; previously used in C0 and C3.4 development, not independent",
        "digital_head_used": False,
        "material_amplitude_model": "unit amplitude assumption; real complex transmission UNKNOWN",
    }
    _write_json(context["work_dir"] / "p2_process_aware" / "c35_process_aware_summary.json", summary)
    return summary


def _eta_for_phase(context: dict[str, Any], phase: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    fields = c34_runner._fields_for_rows(context["cache"], rows.to(context["device"]))
    result = c34.model_b_forward(
        fields, phase, context["aperture_mm"], lens_pitch_um=context["lens_pitch_um"],
        plan_cache=context["propagation_cache"], return_camera=False,
        screen_to_lens_um=context["screen_to_lens_um"],
        lens_to_cmos_um=context["lens_to_cmos_um"])
    return _eta_phase_to_cmos(result["detected_energy"], result["input_energy"].clamp_min(1e-20))


def run_preflight(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    preflight_started = time.perf_counter()
    device = context["device"]
    root = context["work_dir"] / "preflight"
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    phases = _load_control_phases(config)
    initial_phase = phases["c34_p3"].to(device)
    pilot_rows = _balanced_validation_rows(context, 2).to(device)
    one_row = pilot_rows[:1]

    probe = torch.nn.Parameter(initial_phase.clone())
    eta_value = _eta_for_phase(context, probe, one_row).sum()
    eta_gradient = torch.autograd.grad(eta_value, probe)[0]
    if not torch.isfinite(eta_gradient).all() or float(eta_gradient.abs().max()) == 0.0:
        raise RuntimeError("C3.5 throughput gradient is absent, non-finite, or identically zero")
    flat_index = int(eta_gradient.abs().argmax())
    y, x = divmod(flat_index, 100)
    epsilon = 0.02
    plus = initial_phase.clone()
    minus = initial_phase.clone()
    plus[y, x] += epsilon
    minus[y, x] -= epsilon
    with torch.no_grad():
        eta_plus = float(_eta_for_phase(context, plus, one_row).mean())
        eta_minus = float(_eta_for_phase(context, minus, one_row).mean())
    fd = (eta_plus - eta_minus) / (2 * epsilon)
    ad = float(eta_gradient[y, x])
    relative_error = abs(fd - ad) / max(abs(fd), abs(ad), 1e-8)
    if relative_error > 0.02:
        raise RuntimeError(f"C3.5 energy gradient finite-difference check failed: {relative_error:.4g}")

    target_high = max(float(config["energy"]["pareto_targets"][-1]), 0.4)
    constraint_probe = torch.nn.Parameter(initial_phase.clone())
    initial_eta = _eta_for_phase(context, constraint_probe, one_row).mean()
    initial_shortfall = F.relu(target_high - initial_eta)
    grad_shortfall = torch.autograd.grad(initial_shortfall, constraint_probe)[0]
    norm = grad_shortfall.norm().clamp_min(1e-12)
    with torch.no_grad():
        candidate_phase = constraint_probe - (0.05 / norm) * grad_shortfall
        after_shortfall = F.relu(target_high - _eta_for_phase(context, candidate_phase, one_row).mean())
    if not float(after_shortfall) < float(initial_shortfall):
        raise RuntimeError("C3.5 explicit energy constraint did not improve in a gradient-direction probe")
    rho = float(config["energy"]["adaptive_lagrange_rho"])
    lambda_increase = _update_lagrange_multiplier(0.0, target_high, target_high - 0.01, rho)
    lambda_decrease = _update_lagrange_multiplier(10.0, target_high, target_high + 0.01, rho)
    if not (lambda_increase > 0.0 and lambda_decrease < 10.0):
        raise RuntimeError("C3.5 adaptive Lagrange multiplier does not move with constraint residual")

    p1 = torch.nn.Parameter(initial_phase.clone())
    p1_optimizer = torch.optim.Adam([p1], lr=float(config["training"]["learning_rate"]))
    p1_updates = []
    for step in range(3):
        p1_updates.append(_update_batch(
            p1, p1_optimizer, context, pilot_rows, mode="phase", target_efficiency=target_high,
            multiplier=0.0, config=config, noise_seed=step + 11, edge_bias_nm=0.0,
            microbatch=2))
    if not all(v["parameter_delta_l2"] > 0 for v in p1_updates):
        raise RuntimeError("C3.5 P1 three-step CUDA preflight did not update phase parameters")

    initial_height = c34.height_from_phase(initial_phase, 0.5).float().clamp(0, 1.2)
    p2 = torch.nn.Parameter(initial_height.clone())
    p2_optimizer = torch.optim.Adam([p2], lr=float(config["training"]["learning_rate"]))
    p2_updates = []
    for step in range(3):
        p2_updates.append(_update_batch(
            p2, p2_optimizer, context, pilot_rows, mode="height",
            target_efficiency=float(config["energy"]["historical_reference_floor"]),
            multiplier=0.0, config=config, noise_seed=step + 23,
            edge_bias_nm=float(config["fabrication_assumptions"]["edge_height_bias_nm"][step % 2]),
            microbatch=2))
    if not all(v["parameter_delta_l2"] > 0 for v in p2_updates):
        raise RuntimeError("C3.5 P2 three-step CUDA preflight did not update height parameters")

    fields = c34_runner._fields_for_rows(context["cache"], pilot_rows[:2])
    with torch.no_grad():
        effective = _effective_phase(p2.detach(), "height", config, noise_seed=11,
                                     edge_bias_nm=0.0, training=False)
        no_camera = c34.model_b_forward(fields, effective, 2.0,
                                        lens_pitch_um=context["lens_pitch_um"],
                                        plan_cache=context["propagation_cache"], return_camera=False,
                                        screen_to_lens_um=context["screen_to_lens_um"],
                                        lens_to_cmos_um=context["lens_to_cmos_um"])
        with_camera = c34.model_b_forward(fields, effective, 2.0,
                                          lens_pitch_um=context["lens_pitch_um"],
                                          plan_cache=context["propagation_cache"], return_camera=True,
                                          screen_to_lens_um=context["screen_to_lens_um"],
                                          lens_to_cmos_um=context["lens_to_cmos_um"])
    parity_error = max(
        float((no_camera["roi_scores"] - with_camera["roi_scores"]).abs().max()),
        float((no_camera["detected_energy"] - with_camera["detected_energy"]).abs().max()),
        float((no_camera["comparison_crop_energy"] - with_camera["comparison_crop_energy"]).abs().max()))
    if parity_error > 1e-6:
        raise RuntimeError(f"C3.5 training and inference forward outputs differ: {parity_error}")

    negative_eval_rows = _balanced_validation_rows(context, 64)
    negative_source_path = root / "negative_control_source.npz"
    negative_source_metrics = _evaluate_phase(
        context, initial_phase.detach().cpu(), negative_eval_rows,
        "label_shuffle_negative_control_source", negative_source_path,
        class_average=False, eval_batch=8, save_records=True)
    with np.load(negative_source_path, allow_pickle=False) as source:
        negative_rows = source["original_rows"].copy()
        true_labels = source["labels"].copy()
        predictions = source["predictions"].copy()
        roi_scores = source["roi_scores"].copy()
    shuffled_labels, negative_metrics = _shuffled_label_negative_control(true_labels, predictions, 991)
    if negative_metrics["accuracy"] >= 0.40:
        raise RuntimeError("C3.5 shuffled-label negative control reached the preregistered 40% rejection limit")
    negative_path = root / "random_label_negative_control.npz"
    temporary_negative_path = negative_path.with_suffix(negative_path.suffix + ".tmp")
    with temporary_negative_path.open("wb") as handle:
        np.savez_compressed(handle, original_rows=negative_rows, true_labels=true_labels,
                            labels=shuffled_labels, predictions=predictions, roi_scores=roi_scores)
    temporary_negative_path.replace(negative_path)
    _write_json(root / "random_label_negative_control.json", {
        "identity": "C3.5_SHUFFLED_VALIDATION_LABEL_NEGATIVE_CONTROL",
        "status": "PASS", "seed": 991,
        "counterfactual": "fixed 2x2 ROI predictions on 256 balanced validation rows scored against a random permutation of the same labels",
        "source_model_metrics": negative_source_metrics,
        "negative_metrics": negative_metrics,
        "row_records": negative_path.name,
    })

    recovery_path = root / "recovery_checkpoint.pt"
    _atomic_torch_save(recovery_path, {
        "identity": CHECKPOINT_IDENTITY, "variable": p1.detach().cpu(),
        "optimizer": p1_optimizer.state_dict(), "seed": 701,
        "next_rows": pilot_rows.detach().cpu(),
    })
    saved = torch.load(recovery_path, map_location="cpu", weights_only=False)
    continuation = p1
    resumed = torch.nn.Parameter(saved["variable"].to(device).clone())
    resumed_optimizer = torch.optim.Adam([resumed], lr=float(config["training"]["learning_rate"]))
    _restore_optimizer_state(resumed_optimizer, saved["optimizer"])
    recovery_kwargs = {"mode": "phase", "target_efficiency": target_high, "multiplier": 0.0,
                       "config": config, "noise_seed": 701, "edge_bias_nm": 0.0, "microbatch": 2}
    _update_batch(continuation, p1_optimizer, context, saved["next_rows"].to(device), **recovery_kwargs)
    _update_batch(resumed, resumed_optimizer, context, saved["next_rows"].to(device), **recovery_kwargs)
    recovery_error = float((continuation.detach() - resumed.detach()).abs().max())
    if recovery_error != 0.0:
        raise RuntimeError(f"C3.5 checkpoint recovery did not reproduce next update: {recovery_error}")
    record = {
        "identity": "C3.5_PRODUCTION_PREFLIGHT",
        "status": "READY",
        "entrypoint": str(Path(__file__).resolve()),
        "source_modules": {
            "c34_runner": str(Path(c34_runner.__file__).resolve()),
            "c34_model": str(Path(c34.__file__).resolve()),
            "c33_phase": str(Path(c33.__file__).resolve()),
            "c34_process": str(Path(c34_process.__file__).resolve()),
        },
        "source_paths": dict(config["sources"]),
        "loaded_model": "C3.4 Model B scalar Fresnel forward; P3 phase initialization; fixed 100x100 screen",
        "data_roles": {"train": len(context["split"]["train"]), "validation": len(context["split"]["validation"]),
                       "previously_used_holdout_opened": False, "official_test_loaded": False},
        "p1_three_step_updates": p1_updates,
        "p2_three_step_updates": p2_updates,
        "energy_gradient": {"autodiff": ad, "finite_difference": fd, "relative_error": relative_error,
                             "pixel_yx": [y, x], "epsilon_rad": epsilon},
        "energy_constraint_direction": {"target": target_high,
                                         "shortfall_before": float(initial_shortfall),
                                         "shortfall_after": float(after_shortfall),
                                         "lambda_below_target": lambda_increase,
                                         "lambda_above_target": lambda_decrease},
        "training_inference_forward_max_abs_error": parity_error,
        "random_label_negative_control_accuracy": negative_metrics["accuracy"],
        "checkpoint_recovery_max_abs_error": recovery_error,
        "cache": context["cache_metadata"],
        "cuda": {"device": torch.cuda.get_device_name(device),
                 "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                 "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
        "elapsed_seconds": float(time.perf_counter() - preflight_started),
        "formal_command_authorized": "train P1 targets 7.7076% and 40.0% on C3.4 train/validation; then one P2 physical-height run at the provisional 7.7076% reference",
    }
    _write_json(root / "preflight.json", record)
    return record


def _processed_height(height: torch.Tensor, config: dict[str, Any], *, dn: float,
                     psf_sigma_px: float, minimum_feature_radius_px: int,
                     height_levels: int, noise_seed: int, noise_nm: float,
                     edge_bias_nm: float = 0.0) -> torch.Tensor:
    noise = c34_process._noise_field(height, noise_seed, noise_nm)
    printed = c34_process.process_height(
        height, psf_sigma_px=psf_sigma_px,
        minimum_feature_radius_px=minimum_feature_radius_px,
        height_levels=height_levels,
        maximum_height_um=float(config["fabrication_assumptions"]["maximum_height_um"]),
        noise_nm=noise_nm, noise_seed=noise_seed, edge_height_bias_nm=edge_bias_nm,
        noise_field=noise, straight_through_quantization=False)
    return c34.phase_from_height(printed, float(dn))


def run_sensitivity(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    root = context["work_dir"] / "p3_process_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    phases = _load_control_phases(config)
    process_cfg = config["fabrication_assumptions"]
    base_dn = float(process_cfg["delta_n_scenarios"][0])
    base_psf = float(process_cfg["psf_sigma_px"])
    base_feature = int(process_cfg["minimum_feature_radius_px"])
    base_levels = int(process_cfg["height_levels"])
    p1_path = context["work_dir"] / "p1_energy_constrained" / "phase_energy_constrained.npy"
    p2_height_path = context["work_dir"] / "p2_process_aware" / "height_map_um.npy"
    p2_nominal_phase_path = context["work_dir"] / "p2_process_aware" / "phase_after_nominal_process.npy"
    if not p1_path.is_file() or not p2_height_path.is_file() or not p2_nominal_phase_path.is_file():
        raise FileNotFoundError("C3.5 sensitivity scan requires completed P1 and P2 candidates")
    phases["c35_p1_energy_constrained"] = _resolve_phase(np.load(p1_path, allow_pickle=False))
    p2_height = torch.as_tensor(np.load(p2_height_path, allow_pickle=False), dtype=torch.float32)
    phases["c35_p2_process_aware"] = _resolve_phase(np.load(p2_nominal_phase_path, allow_pickle=False))
    candidate_heights = {
        name: (p2_height.clone() if name == "c35_p2_process_aware"
               else c34.height_from_phase(phase, base_dn).float().clamp(
                   0, float(process_cfg["maximum_height_um"])))
        for name, phase in phases.items()
    }
    rows = _balanced_validation_rows(context, int(config["sensitivity"]["validation_rows_per_class"]))
    baseline_predictions: dict[str, np.ndarray] = {}
    results_by_candidate: dict[str, list[dict[str, Any]]] = {name: [] for name in phases}
    scenario_specs: list[dict[str, Any]] = []
    for dn in config["sensitivity"]["delta_n"]:
        scenario_specs.append({"family": "delta_n", "name": f"delta_n_{dn:g}_nominal_process",
                              "dn": float(dn), "psf": base_psf,
                              "min_feature": base_feature, "levels": base_levels,
                              "noise": 0.0, "noise_seed": 0,
                              "edge": 0.0, "aperture": 2.0})
    for sigma in config["sensitivity"]["psf_sigma_px"]:
        scenario_specs.append({"family": "psf", "name": f"psf_{sigma:g}_px",
                              "dn": base_dn, "psf": float(sigma),
                              "min_feature": base_feature, "levels": base_levels,
                              "noise": 0.0, "noise_seed": 0,
                              "edge": 0.0, "aperture": 2.0})
    for radius in config["sensitivity"]["minimum_feature_radius_px"]:
        scenario_specs.append({"family": "minimum_feature", "name": f"minimum_feature_radius_{radius}_px",
                              "dn": base_dn, "psf": base_psf,
                              "min_feature": int(radius), "levels": base_levels,
                              "noise": 0.0, "noise_seed": 0,
                              "edge": 0.0, "aperture": 2.0})
    for seed in config["sensitivity"]["height_noise_seeds"]:
        scenario_specs.append({"family": "quantization_noise", "name": f"q64_noise20_seed_{seed}",
                              "dn": base_dn, "psf": base_psf,
                              "min_feature": base_feature, "levels": base_levels,
                              "noise": float(process_cfg["height_noise_nm"]),
                              "noise_seed": int(seed),
                              "edge": 0.0, "aperture": 2.0})
    for dn in config["sensitivity"]["delta_n"]:
        for seed in config["sensitivity"]["height_noise_seeds"]:
            for edge in config["fabrication_assumptions"]["edge_height_bias_nm"]:
                edge_name = "plus" if float(edge) >= 0 else "minus"
                scenario_specs.append({"family": "combined_adverse",
                                      "name": f"combined_worst_dn{dn:g}_seed{seed}_edge_{edge_name}20nm",
                                      "dn": float(dn), "psf": 1.2, "min_feature": 2,
                                      "levels": base_levels,
                                      "noise": float(process_cfg["height_noise_nm"]),
                                      "noise_seed": int(seed),
                                      "edge": float(edge), "aperture": 2.0})
    for aperture in config["sensitivity"]["apertures_mm"]:
        scenario_specs.append({"family": "aperture", "name": f"aperture_{aperture:g}_mm",
                              "dn": base_dn, "psf": base_psf,
                              "min_feature": base_feature, "levels": base_levels,
                              "noise": 0.0, "noise_seed": 0,
                              "edge": 0.0, "aperture": float(aperture)})

    for candidate_name, source_phase in phases.items():
        nominal_phase = _processed_height(
            candidate_heights[candidate_name], config, dn=base_dn,
            psf_sigma_px=base_psf, minimum_feature_radius_px=base_feature,
            height_levels=base_levels, noise_seed=0, noise_nm=0.0,
            edge_bias_nm=0.0)
        baseline_path = root / candidate_name / "baseline_nominal_process.npz"
        baseline_metrics = _evaluate_phase(
            context, nominal_phase, rows,
            f"{candidate_name}_baseline_nominal_process", baseline_path,
            class_average=False, eval_batch=8, save_records=True)
        with np.load(baseline_path, allow_pickle=False) as saved:
            baseline_predictions[candidate_name] = saved["predictions"].astype(np.int64)
        results = [{"scenario": "baseline_nominal_process", "family": "baseline",
                    "metrics": baseline_metrics, "flip_rate_from_ideal": 0.0}]
        for spec in scenario_specs:
            if spec["family"] == "aperture":
                evaluated_phase = source_phase
            else:
                evaluated_phase = _processed_height(
                    candidate_heights[candidate_name], config, dn=spec["dn"],
                    psf_sigma_px=spec["psf"], minimum_feature_radius_px=spec["min_feature"],
                    height_levels=spec["levels"], noise_seed=spec["noise_seed"],
                    noise_nm=spec["noise"], edge_bias_nm=spec["edge"])
            path = root / candidate_name / f"{spec['name']}.npz"
            metrics = _evaluate_phase(
                context, evaluated_phase, rows,
                f"{candidate_name}_{spec['name']}", path, aperture_mm=spec["aperture"],
                class_average=False, eval_batch=8, save_records=True)
            with np.load(path, allow_pickle=False) as saved:
                flips = float(np.mean(saved["predictions"].astype(np.int64)
                                      != baseline_predictions[candidate_name]))
            record = {"scenario": spec["name"], "family": spec["family"],
                      "delta_n_assumption": spec["dn"], "psf_sigma_px": spec["psf"],
                      "minimum_feature_radius_px": spec["min_feature"],
                      "height_levels": spec["levels"], "height_noise_nm": spec["noise"],
                      "noise_seed": spec["noise_seed"], "edge_bias_nm": spec["edge"],
                      "aperture_mm": spec["aperture"], "metrics": metrics,
                      "flip_rate_from_ideal": flips}
            results.append(record)
            print(f"sensitivity {candidate_name}/{spec['name']}: acc={metrics['accuracy']:.4f} eta={metrics['eta_phase_to_cmos']['mean']:.4f} flips={flips:.4f}", flush=True)
        results_by_candidate[candidate_name] = results

    candidate_summaries = {}
    for candidate_name, results in results_by_candidate.items():
        by_family: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            by_family.setdefault(item["family"], []).append(item)
        candidate_summaries[candidate_name] = {
            "scenario_count": len(results),
            "families": {
                family: {
                    "accuracy_mean": float(np.mean([r["metrics"]["accuracy"] for r in family_results])),
                    "accuracy_min": float(min(r["metrics"]["accuracy"] for r in family_results)),
                    "accuracy_max": float(max(r["metrics"]["accuracy"] for r in family_results)),
                    "eta_mean": float(np.mean([r["metrics"]["eta_phase_to_cmos"]["mean"] for r in family_results])),
                    "eta_p10_min": float(min(r["metrics"]["eta_phase_to_cmos"]["p10"] for r in family_results)),
                    "flip_rate_mean": float(np.mean([r["flip_rate_from_ideal"] for r in family_results])),
                    "worst_scenario": min(family_results, key=lambda r: r["metrics"]["accuracy"])["scenario"],
                } for family, family_results in by_family.items()
            },
            "best_scenario": max(results, key=lambda r: r["metrics"]["accuracy"])["scenario"],
            "worst_scenario": min(results, key=lambda r: r["metrics"]["accuracy"])["scenario"],
            "accuracy_mean_across_scenarios": float(np.mean([r["metrics"]["accuracy"] for r in results])),
            "eta_mean_across_scenarios": float(np.mean([r["metrics"]["eta_phase_to_cmos"]["mean"] for r in results])),
            "worst_accuracy": float(min(r["metrics"]["accuracy"] for r in results)),
            "mean_flip_rate": float(np.mean([r["flip_rate_from_ideal"] for r in results])),
        }
    summary = {
        "identity": "C3.5_PROCESS_SENSITIVITY",
        "status": "ASSUMPTION_SCENARIOS_COMPLETE",
        "population": {"role": "C3.4 validation development set; not independent",
                       "rows": int(len(rows)), "per_class": int(config["sensitivity"]["validation_rows_per_class"])},
        "nominal_process_reference": {
            "delta_n_assumption": base_dn,
            "psf_sigma_px": base_psf,
            "minimum_feature_radius_px": base_feature,
            "height_levels": base_levels,
            "height_noise_nm": 0.0,
            "edge_height_bias_nm": 0.0,
            "sensitivity_definition": "vary one listed parameter while holding the remaining nominal process terms fixed; combined_adverse replaces them with the listed joint adverse scenario",
        },
        "scenario_specs": scenario_specs,
        "candidates": results_by_candidate,
        "candidate_summaries": candidate_summaries,
        "probability_distributions_assumed": False,
        "hardware_material_parameters_measured": False,
    }
    _write_json(root / "c35_process_sensitivity.json", summary)
    return summary


def _save_stl_candidate(context: dict[str, Any], config: dict[str, Any], *,
                        name: str, design_phase: torch.Tensor,
                        design_height: torch.Tensor, delta_n: float) -> dict[str, Any]:
    root = context["work_dir"] / "p5_stl" / name / f"delta_n_{delta_n:.1f}"
    root.mkdir(parents=True, exist_ok=True)
    phase_rad = c34.phase_from_height(design_height, delta_n).detach().cpu().numpy().astype(np.float32)
    phase_wrapped = np.remainder(phase_rad, 2 * np.pi).astype(np.float32)
    height_np = design_height.detach().cpu().numpy().astype(np.float32)
    np.save(root / "phase_rad.npy", phase_rad, allow_pickle=False)
    np.save(root / "phase_wrapped.npy", phase_wrapped, allow_pickle=False)
    np.save(root / "height_map.npy", height_np, allow_pickle=False)
    np.savetxt(root / "height_map.csv", height_np, delimiter=",")
    triangles, mesh_metadata = c34.mesh_200um(
        height_np, substrate_um=float(config["fabrication_assumptions"]["substrate_thickness_um"]))
    stl_path = root / f"{name}_200x200um.stl"
    c34.write_binary_stl(stl_path, triangles, header="HW207 C3.5 PARAMETERIZED DIAGNOSTIC ONLY")
    metadata = {
        "identity": "C3.5_PARAMETERIZED_STL_DIAGNOSTIC",
        "status": "PARAMETERIZED_DIAGNOSTIC_ONLY",
        "unit": "um",
        "wavelength_nm": 532.0,
        "delta_n_assumption": float(delta_n),
        "delta_n_status": "UNKNOWN_FOR_ACTUAL_PRINTED_MATERIAL",
        "phase_height_mapping": "h=lambda*wrapped_phase/(2*pi*delta_n); P2 retains its trained physical-height branch",
        "phase_source": name,
        "amplitude_transmission": "UNKNOWN; unit amplitude assumed in numerical propagation",
        "substrate_thickness_um_assumption": float(config["fabrication_assumptions"]["substrate_thickness_um"]),
        "mesh": mesh_metadata,
        "self_intersection_independent_triangle_triangle_test": "NOT_RUN",
        "printer_process_status": "ASSUMPTION_ONLY_NOT_MEASURED",
    }
    _write_json(root / "stl_metadata.json", metadata)
    independent_audit = c34.audit_stl_independent(stl_path)
    recovered = torch.from_numpy(c34.recover_height_from_mesh_vertices(stl_path)).float()
    recovery_error = float((recovered - torch.from_numpy(height_np)).abs().max())
    if recovery_error > 1e-5:
        raise RuntimeError(f"C3.5 STL roundtrip height error exceeds tolerance: {recovery_error}")
    direct_process_phase = _processed_height(
        design_height.to(context["device"]), config, dn=delta_n,
        psf_sigma_px=float(config["fabrication_assumptions"]["psf_sigma_px"]),
        minimum_feature_radius_px=int(config["fabrication_assumptions"]["minimum_feature_radius_px"]),
        height_levels=int(config["fabrication_assumptions"]["height_levels"]),
        noise_seed=0, noise_nm=0.0, edge_bias_nm=0.0)
    stl_process_phase = _processed_height(
        recovered.to(context["device"]), config, dn=delta_n,
        psf_sigma_px=float(config["fabrication_assumptions"]["psf_sigma_px"]),
        minimum_feature_radius_px=int(config["fabrication_assumptions"]["minimum_feature_radius_px"]),
        height_levels=int(config["fabrication_assumptions"]["height_levels"]),
        noise_seed=0, noise_nm=0.0, edge_bias_nm=0.0)
    validation = context["split"]["validation"]
    ideal_metrics = _evaluate_phase(
        context, design_phase, validation, f"{name}_dn{delta_n:.1f}_ideal",
        root / "ideal_validation_records.npz", class_average=False, eval_batch=8, save_records=True)
    design_height_phase = c34.phase_from_height(design_height.to(context["device"]), delta_n)
    process_metrics = _evaluate_phase(
        context, direct_process_phase, validation, f"{name}_dn{delta_n:.1f}_assumed_process",
        root / "processed_validation_records.npz", class_average=False, eval_batch=8, save_records=True)
    stl_metrics = _evaluate_phase(
        context, stl_process_phase, validation, f"{name}_dn{delta_n:.1f}_stl_roundtrip_process",
        root / "stl_roundtrip_validation_records.npz", class_average=True, eval_batch=4, save_records=True)
    height_phase_metrics = _evaluate_phase(
        context, design_height_phase, validation, f"{name}_dn{delta_n:.1f}_height_only",
        root / "height_only_validation_records.npz", class_average=False, eval_batch=8, save_records=True)
    record = {
        "stl_path": str(stl_path), "metadata_path": str(root / "stl_metadata.json"),
        "phase_rad_path": str(root / "phase_rad.npy"), "phase_wrapped_path": str(root / "phase_wrapped.npy"),
        "height_map_path": str(root / "height_map.npy"), "height_csv_path": str(root / "height_map.csv"),
        "independent_audit": independent_audit,
        "height_recovery_max_abs_error_um": recovery_error,
        "height_only_metrics": height_phase_metrics,
        "ideal_phase_metrics": ideal_metrics,
        "assumed_process_metrics": process_metrics,
        "stl_roundtrip_process_metrics": stl_metrics,
        "stl_roundtrip_consistency": {
            "height_max_abs_error_um": recovery_error,
            "classification_accuracy_delta_vs_direct_process": float(stl_metrics["accuracy"] - process_metrics["accuracy"]),
            "mean_eta_delta_vs_direct_process": float(stl_metrics["eta_phase_to_cmos"]["mean"] - process_metrics["eta_phase_to_cmos"]["mean"]),
            "same_recovered_height_used_for_propagation": True,
        },
    }
    return record


def run_stl(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    p1_path = context["work_dir"] / "p1_energy_constrained" / "phase_energy_constrained.npy"
    p2_height_path = context["work_dir"] / "p2_process_aware" / "height_map_um.npy"
    if not p1_path.is_file() or not p2_height_path.is_file():
        raise FileNotFoundError("C3.5 STL export requires the selected P1 and P2 candidates")
    p1_phase = _resolve_phase(np.load(p1_path, allow_pickle=False))
    p2_height = torch.as_tensor(np.load(p2_height_path, allow_pickle=False), dtype=torch.float32)
    dn_values = [float(v) for v in config["fabrication_assumptions"]["delta_n_scenarios"]]
    records = []
    for dn in dn_values:
        p1_height = c34.height_from_phase(p1_phase, dn).float()
        records.append(_save_stl_candidate(context, config, name="p1_energy_constrained",
                                           design_phase=p1_phase, design_height=p1_height, delta_n=dn))
        p2_design_phase = c34.phase_from_height(p2_height, dn).float()
        records.append(_save_stl_candidate(context, config, name="p2_process_aware_height",
                                           design_phase=p2_design_phase, design_height=p2_height, delta_n=dn))
    audit = {
        "identity": "C3.5_STL_VALIDATION",
        "status": "PARTIAL_PARAMETERIZED_DIAGNOSTIC_ONLY",
        "designs": records,
        "all_stls_have_200x200um_mesh_bounds": all(
            record["independent_audit"]["bbox_numeric_without_intrinsic_unit"][0] == [0.0, 200.0]
            and record["independent_audit"]["bbox_numeric_without_intrinsic_unit"][1] == [0.0, 200.0]
            for record in records),
        "all_meshes_closed_and_oriented": all(
            record["independent_audit"]["closed_two_manifold_by_edge_incidence"]
            and record["independent_audit"]["opposite_directed_edge_pairing"]
            and record["independent_audit"]["degenerate_triangles"] == 0
            for record in records),
        "general_triangle_triangle_self_intersection_check": "NOT_RUN",
        "material_parameters_known": False,
        "printer_parameters_measured": False,
        "fabrication_ready": "BLOCKED",
    }
    _write_json(context["work_dir"] / "p5_stl" / "c35_stl_audit.json", audit)
    return audit


def build_summary(work_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    def read_if_exists(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    preflight = read_if_exists(work_dir / "preflight" / "preflight.json")
    p1 = read_if_exists(work_dir / "p1_energy_constrained" / "p1_summary.json")
    p2 = read_if_exists(work_dir / "p2_process_aware" / "c35_process_aware_summary.json")
    p3 = read_if_exists(work_dir / "p3_process_sensitivity" / "c35_process_sensitivity.json")
    p4 = read_if_exists(work_dir / "c35_independent_eval.json")
    p5 = read_if_exists(work_dir / "p5_stl" / "c35_stl_audit.json")
    energy_audit = read_if_exists(work_dir / "c35_energy_audit.json")
    states = {
        "C3.5_P0_ENERGY_AUDIT_PASS": "PASS_WITH_HISTORICAL_ROW_RECALCULATION_LIMITATION" if energy_audit else "NOT_RUN",
        "C3.5_P1_ENERGY_CONSTRAINED_PASS": "DIAGNOSTIC_ONLY" if p1 else "NOT_RUN",
        "C3.5_P2_PROCESS_AWARE_PASS": "ASSUMPTION_ONLY_DIAGNOSTIC_COMPLETE" if p2 else "NOT_RUN",
        "C3.5_P3_PROCESS_ROBUSTNESS_PASS": "ASSUMPTION_SCENARIOS_COMPLETE" if p3 else "NOT_RUN",
        "C3.5_P4_INDEPENDENT_EVAL_PASS": "INDEPENDENT_EVAL_BLOCKED" if p4 else "NOT_RUN",
        "C3.5_P5_STL_VALIDATION_PASS": "PARTIAL_PARAMETERIZED_DIAGNOSTIC_ONLY" if p5 else "NOT_RUN",
        "C3.5_FABRICATION_READY": "BLOCKED",
        "HW207_CURRENT_MAPPING": "BLOCKED",
    }
    complete = all(value in {"PASS", "PASS_WITH_HISTORICAL_ROW_RECALCULATION_LIMITATION",
                             "DIAGNOSTIC_ONLY", "ASSUMPTION_SCENARIOS_COMPLETE"}
                   for key, value in states.items() if key not in {"C3.5_P4_INDEPENDENT_EVAL_PASS", "C3.5_FABRICATION_READY", "HW207_CURRENT_MAPPING"})
    summary = {
        "identity": "HW207_MNIST4_C35_CLOSEOUT",
        "status": "C3.5_PARTIAL_PASS" if not complete or states["C3.5_P4_INDEPENDENT_EVAL_PASS"] != "PASS" else "C3.5_COMPLETE",
        "states": states,
        "preflight": preflight,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "energy_audit": energy_audit,
        "scientific_boundary": {
            "all_results_are_numerical_assumption_only": True,
            "no_digital_classifier_in_optical_inference": True,
            "no_official_test_images_evaluated": True,
            "no_actual_dmd_cmos_or_printer_operation": True,
            "material_and_printer_parameters": "UNKNOWN",
        },
    }
    _write_json(work_dir / "c35_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "preflight", "p1", "p2", "sensitivity", "stl", "summary"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_config(config_path)
    default_work_dir = Path(config["run"]["autodl_system_work_dir"])
    work_dir = (args.work_dir or default_work_dir).expanduser().resolve()
    if args.mode == "audit":
        local_run = ROOT / config["run"]["local_run_dir"]
        local_run.mkdir(parents=True, exist_ok=True)
        report = audit_saved_evidence(config, local_run)
        build_summary(local_run, config)
        print(json.dumps({"p0_status": report["status"], "threshold_status": report["threshold"]["status"],
                          "independent_eval": report["population_audit"]["independent_eval_status"]}, ensure_ascii=False))
        return
    device = torch.device(args.device)
    context = _load_context(config, work_dir, device)
    if args.mode == "preflight":
        result = run_preflight(context, config)
    elif args.mode == "p1":
        result = run_p1(context, config)
    elif args.mode == "p2":
        result = run_p2(context, config)
    elif args.mode == "sensitivity":
        result = run_sensitivity(context, config)
    elif args.mode == "stl":
        result = run_stl(context, config)
    elif args.mode == "summary":
        result = build_summary(work_dir, config)
    else:
        raise ValueError(f"unsupported C3.5 mode {args.mode}")
    if args.mode != "summary":
        build_summary(work_dir, config)
    print(json.dumps({"mode": args.mode, "status": result.get("status", "COMPLETED")}, ensure_ascii=False))


if __name__ == "__main__":
    main()





