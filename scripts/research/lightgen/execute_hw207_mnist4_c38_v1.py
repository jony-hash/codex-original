"""C3.8 fixed-model single-phase MNIST4 numerical study.

All optical stages call the frozen C3.7 Model-B producer through ``forward_field``.
The only trainable variable is one shared 100x100 phase map; classification is the
argmax of the four frozen CMOS ROI integrals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as c34_runner
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import hw207_mnist4_c37_v1 as c37


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/hw207_mnist4_c38_v1.json"
C37_RUN_DIR = ROOT / "outputs/20261105_hw207_mnist4_c37_v1/runs/attempt_01"
IDENTITY = "HW207_MNIST4_C38_JOINT_PHASE_SCREEN"
LENS_TO_CMOS_CHIRP_DTYPE = torch.complex128
PIXEL_AREA_UM2 = c37.CMOS_PITCH_UM ** 2
EPS = 1e-12
SOURCE_INPUT_NAMES = {
    "C35_P1": "C35_P1_phase.npy",
    "C37_E2_seed11": "C37_E2_seed11_phase.npy",
    "C37_E3_seed11": "C37_E3_seed11_phase.npy",
    "C37_E3_seed23": "C37_E3_seed23_phase.npy",
    "C37_E7_seed23": "C37_E7_seed23_phase.npy",
}
HISTORY_INPUT_NAMES = {key: f"history_{key}.npz" for key in SOURCE_INPUT_NAMES}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("identity") != IDENTITY:
        raise ValueError("unexpected C3.8 configuration identity")
    if config.get("numerics", {}).get("lens_to_cmos_fresnel_chirp_compute_dtype") != "complex128":
        raise ValueError("C3.8 lens-to-CMOS Fresnel chirps must use complex128 computation")
    geometry = c37.load_config()["geometry"]
    frozen = {
        "wavelength_nm": 532.0,
        "phase_grid": [100, 100],
        "phase_pitch_um": 2.0,
        "screen_to_lens_mm": 80.0,
        "lens_focal_length_mm": 30.0,
        "lens_to_cmos_mm": 100.0,
        "aperture_diameter_mm": 2.0,
        "camera_shape": [2048, 2048],
        "camera_pixel_pitch_um": 6.5,
    }
    if c34.WAVELENGTH_UM != 0.532 or geometry["phase_grid"] != frozen["phase_grid"]:
        raise ValueError("C3.7 physical-model source changed from the frozen C3.8 scenario")
    for name in ("screen_to_lens_mm", "lens_focal_length_mm", "lens_to_cmos_mm",
                 "aperture_diameter_mm", "camera_shape", "camera_pixel_pitch_um"):
        if geometry[name] != frozen[name]:
            raise ValueError(f"C3.7 Model-B field {name} changed")
    if config["core"]["core_side_px"] != c37.core_side_px(config["core"]["fraction_of_roi_side"]):
        raise ValueError("configured core side is inconsistent with frozen C3.7 ROI")
    if config["core"]["core_side_um"] != config["core"]["core_side_px"] * c37.CMOS_PITCH_UM:
        raise ValueError("configured physical core size is inconsistent with CMOS pixel pitch")
    if tuple(config["training"]["loss_weights"]) != tuple(config["training"]["loss_order"]):
        raise ValueError("C3.8 must activate all five frozen losses in the declared order")
    if len(config["source_candidates"]) < 5:
        raise ValueError("historical comparison must include C35 P1, E2, both E3 and E7")
    return config


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def phase_transmission(phase: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(phase), phase)


def forward_field(fields: torch.Tensor, phase: torch.Tensor, context: dict[str, Any], *,
                  lens_pitch_um: float | None = None,
                  return_camera: bool = True) -> dict[str, Any]:
    """One production optical forward for audit, training and frozen inference."""
    return c37.forward_batch(fields, phase, context, return_camera=return_camera,
                             lens_pitch_um=lens_pitch_um,
                             lens_to_cmos_chirp_dtype=LENS_TO_CMOS_CHIRP_DTYPE)


def predict_from_roi_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("four ROI scores are required for the optical argmax")
    return scores.argmax(dim=1)


def pareto_front(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated candidates over declared C3.8 objectives."""
    maximize = ("accuracy", "macro_f1", "min_class_recall", "peak_roi", "peak_core", "core_input_mean",
                "eta_mean", "eta_p10")
    minimize = ("spread_mean_um",)
    objectives = tuple(key for key in maximize + minimize
                       if records and all(key in row and math.isfinite(float(row[key]))
                                          for row in records))
    result = []
    for item in records:
        dominated = False
        for other in records:
            if other is item:
                continue
            no_worse = all(float(other[key]) >= float(item[key]) for key in objectives if key in maximize)
            no_worse &= all(float(other[key]) <= float(item[key]) for key in objectives if key in minimize)
            strictly = any(float(other[key]) > float(item[key]) for key in objectives if key in maximize)
            strictly |= any(float(other[key]) < float(item[key]) for key in objectives if key in minimize)
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            result.append(item)
    return result


def _resolved_artifact(relative: str, key: str, *, run_dir: Path,
                       history: bool = False) -> Path:
    candidate = ROOT / relative
    if candidate.is_file():
        return candidate
    fallback_name = HISTORY_INPUT_NAMES[key] if history else SOURCE_INPUT_NAMES[key]
    fallback = run_dir / "inputs" / fallback_name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"missing explicit C3.8 input for {key}: {candidate}; {fallback}")


def resolve_phase_sources(config: dict[str, Any] | None = None,
                          run_dir: Path | None = None) -> dict[str, torch.Tensor]:
    config = config or load_config()
    run_dir = run_dir or (ROOT / config["run"]["version_directory"] / "runs" /
                          config["run"]["attempt"])
    result = {}
    for key, relative in config["source_candidates"].items():
        path = _resolved_artifact(relative, key, run_dir=run_dir)
        result[key] = c37.load_frozen_phase(path)
    return result


def _resolve_history_sources(config: dict[str, Any], run_dir: Path) -> dict[str, Path]:
    return {key: _resolved_artifact(relative, key, run_dir=run_dir, history=True)
            for key, relative in config["historical_reference_records"].items()}


def _context(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C3.8 full optical work requires the authorized CUDA host")
    context = c37.build_context(c37.load_config(), device)
    if tuple(c37.roi_bounds()) != (
            (921, 1024, 921, 1024), (921, 1024, 1024, 1127),
            (1024, 1127, 921, 1024), (1024, 1127, 1024, 1127)):
        raise ValueError("C3.8 four fixed ROI coordinates changed")
    context["lens_pitch_um"] = 4.0
    return context


def _balanced_rows(rows: torch.Tensor, labels: torch.Tensor, per_class: int) -> torch.Tensor:
    selected = []
    for digit in range(4):
        part = rows[labels[rows] == digit]
        if len(part) < per_class:
            raise ValueError(f"C3.8 role has fewer than {per_class} rows for digit {digit}")
        selected.append(part[:per_class])
    return torch.cat(selected)


def _formal_training_rows(context: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    return _balanced_rows(context["split"]["train"], context["labels"],
                          int(config["data"]["train_rows_per_class_formal"]))


def _core_energies(intensity: torch.Tensor, fraction: float) -> torch.Tensor:
    values = []
    for r0, r1, c0, c1 in c37.core_bounds(fraction):
        values.append(intensity[:, r0:r1, c0:c1].sum((1, 2)) * PIXEL_AREA_UM2)
    return torch.stack(values, dim=1)


def loss_components(result: dict[str, torch.Tensor], labels: torch.Tensor,
                    phase: torch.Tensor, context: dict[str, Any], config: dict[str, Any], *,
                    eta_reference: torch.Tensor,
                    active_losses: Sequence[str] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Five-term physical-only objective; no label-dependent operation enters propagation."""
    intensity = result["camera_intensity"]
    scores = result["roi_scores"]
    if intensity is None or scores.shape != (len(labels), 4):
        raise ValueError("loss requires CMOS intensity and four ROI integrals")
    labels = labels.to(device=scores.device, dtype=torch.long)
    total = result["detected_energy"].clamp_min(EPS)
    share = scores / total[:, None]
    rows = torch.arange(len(labels), device=labels.device)
    correct = share[rows, labels]
    masked = share.clone()
    masked[rows, labels] = -torch.inf
    wrong = masked.max(1).values
    class_cfg = config["training"]["classification"]
    logits = share * float(class_cfg["temperature"])
    ce = F.cross_entropy(logits, labels)
    margin = F.relu(0.5 - correct + wrong).mean()
    roi_outside = (result["outside_roi_full_cmos_energy"].clamp_min(0.0) / total).mean()
    classification = ce + float(class_cfg["margin_weight"]) * margin + \
        float(class_cfg["roi_leakage_weight"]) * roi_outside

    cores = _core_energies(intensity, float(config["core"]["fraction_of_roi_side"]))
    core = cores[rows, labels] / total
    core_loss = -torch.log(core.clamp_min(1e-12)).mean()

    # The normalized power sum is a differentiable peak-competition surrogate.
    scaled = intensity / intensity.flatten(1).amax(1).clamp_min(EPS)[:, None, None]
    powered = scaled.clamp_min(0.0).pow(int(config["training"]["peak_power"]))
    peak_core = _core_energies(powered, float(config["core"]["fraction_of_roi_side"]))
    peak_prob = peak_core[rows, labels] / (
        powered.sum((1, 2)) * PIXEL_AREA_UM2).clamp_min(EPS)
    peak_loss = -torch.log(peak_prob.clamp_min(1e-12)).mean()

    eta = result["detected_energy"] / result["input_energy"].clamp_min(EPS)
    eta_reference = eta_reference.to(device=eta.device, dtype=eta.dtype).clamp_min(EPS)
    if eta_reference.shape != eta.shape:
        raise ValueError("per-row C3.5 P1 efficiency reference must align with the batch")
    retention = float(config["training"]["energy_reference_retention_target"])
    normalized_shortfall = (retention * eta_reference - eta) / eta_reference
    energy_loss = F.softplus(normalized_shortfall * 10.0).mean() / 10.0

    spot = {
        "core_fraction": core,
        "correct_roi_fraction": correct,
        "wrong_roi_fraction": (1.0 - correct -
                               result["outside_roi_full_cmos_energy"].clamp_min(0.0) / total).clamp_min(0.0),
        "outside_roi_fraction": result["outside_roi_full_cmos_energy"].clamp_min(0.0) / total,
        "own_roi_outside_core_fraction": (correct - core).clamp_min(0.0),
    }
    radius = c37.spread_radius_um(intensity, labels)
    spread_scale = float(config["training"]["spread_normalisation_um"])
    spread_loss = (radius / spread_scale).square().mean() + \
        0.5 * spot["own_roi_outside_core_fraction"].mean() + \
        spot["wrong_roi_fraction"].mean() + spot["outside_roi_fraction"].mean()

    components = {
        "classification": classification,
        "core": core_loss,
        "peak": peak_loss,
        "energy": energy_loss,
        "spread": spread_loss,
    }
    weights = dict(config["training"]["loss_weights"])
    if active_losses is not None:
        unknown = set(active_losses) - set(weights)
        if unknown:
            raise ValueError(f"unknown active C3.8 losses: {sorted(unknown)}")
        weights = {name: value if name in active_losses else 0.0
                   for name, value in weights.items()}
    loss = sum(float(weights[name]) * value for name, value in components.items())
    return loss, components


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, guess in zip(labels, predictions):
        confusion[int(truth), int(guess)] += 1
    precision, recall, f1 = [], [], []
    for digit in range(4):
        tp = int(confusion[digit, digit])
        precision.append(tp / max(int(confusion[:, digit].sum()), 1))
        recall.append(tp / max(int(confusion[digit, :].sum()), 1))
        f1.append(2 * precision[-1] * recall[-1] / max(precision[-1] + recall[-1], EPS))
    return {
        "accuracy": float((labels == predictions).mean()),
        "macro_f1": float(np.mean(f1)),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion.tolist(),
        "misclassified_count": int((labels != predictions).sum()),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {"mean": float(data.mean()), "median": float(np.median(data)),
            "p10": float(np.percentile(data, 10)), "minimum": float(data.min()),
            "maximum": float(data.max()), "std": float(data.std())}


def evaluate_frozen(context: dict[str, Any], config: dict[str, Any], phase: torch.Tensor,
                    rows: torch.Tensor, *, batch_size: int = 4,
                    lens_pitch_um: float | None = None,
                    save_map_rows: Sequence[int] = (),
                    aggregate_group_means: bool = False) -> dict[str, Any]:
    """Inference contains no trainable head and reads labels only after ROI scores."""
    device = context["device"]
    rows_cpu = torch.as_tensor(rows, dtype=torch.long).cpu()
    phase_fixed = phase.detach().to(device=device, dtype=torch.float32)
    row_parts: dict[str, list[np.ndarray]] = {key: [] for key in (
        "roi_scores", "input_energy_um2", "cmos_energy_um2", "core_energies_um2",
        "global_peak_row", "global_peak_col", "global_peak_intensity", "centroid_row",
        "centroid_col", "spread_radius_um", "outside_roi_energy_um2", "wrong_peak_intensity")}
    maps: dict[str, np.ndarray] = {}
    save_set = {int(value) for value in save_map_rows}
    group_all_sums = (np.zeros((4, c37.CMOS, c37.CMOS), dtype=np.float64)
                      if aggregate_group_means else None)
    group_correct_sums = (np.zeros((4, c37.CMOS, c37.CMOS), dtype=np.float64)
                          if aggregate_group_means else None)
    group_all_counts = np.zeros(4, dtype=np.int64) if aggregate_group_means else None
    group_correct_counts = np.zeros(4, dtype=np.int64) if aggregate_group_means else None
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows_cpu), batch_size):
            chunk = rows_cpu[start:start + batch_size]
            field = c37.input_fields(context, chunk)
            result = forward_field(field, phase_fixed, context, lens_pitch_um=lens_pitch_um)
            image = result["camera_intensity"]
            if image is None or not torch.isfinite(image).all() or (image < 0).any():
                raise RuntimeError("frozen optical inference produced invalid CMOS intensity")
            scores = result["roi_scores"]
            energies = _core_energies(image, float(config["core"]["fraction_of_roi_side"]))
            # Label use starts here, after the forward field and four ROI sums are fixed.
            chunk_labels = context["labels"][chunk.to(context["labels"].device)].to(device)
            chunk_predictions = scores.argmax(1)
            image_cpu = image.detach().cpu().numpy() if aggregate_group_means else None
            if aggregate_group_means:
                assert group_all_sums is not None and group_correct_sums is not None
                assert group_all_counts is not None and group_correct_counts is not None
                for local, label in enumerate(chunk_labels.tolist()):
                    label = int(label)
                    image_np = image_cpu[local]
                    group_all_sums[label] += image_np
                    group_all_counts[label] += 1
                    if int(chunk_predictions[local]) == label:
                        group_correct_sums[label] += image_np
                        group_correct_counts[label] += 1
            flat = image.flatten(1)
            peak_index = flat.argmax(1)
            peak_row = torch.div(peak_index, c37.CMOS, rounding_mode="floor")
            peak_col = peak_index.remainder(c37.CMOS)
            profile_y = image.sum(2)
            profile_x = image.sum(1)
            axis = torch.arange(c37.CMOS, device=device, dtype=image.dtype) - c37.FRAME_CENTER_PX
            detected = result["detected_energy"].clamp_min(EPS)
            center = c37.FRAME_CENTER_PX
            centroid_row = (profile_y * axis).sum(1) / image.sum((1, 2)).clamp_min(EPS) + center
            centroid_col = (profile_x * axis).sum(1) / image.sum((1, 2)).clamp_min(EPS) + center
            spread = c37.spread_radius_um(image, chunk_labels)
            wrong_peak = []
            for local, label in enumerate(chunk_labels.tolist()):
                r0, r1, c0, c1 = c37.roi_bounds()[int(label)]
                outside = image[local].clone()
                outside[r0:r1, c0:c1] = 0.0
                wrong_peak.append(outside.max())
            values = {
                "roi_scores": scores,
                "input_energy_um2": result["input_energy"],
                "cmos_energy_um2": result["detected_energy"],
                "core_energies_um2": energies,
                "global_peak_row": peak_row,
                "global_peak_col": peak_col,
                "global_peak_intensity": flat.max(1).values,
                "centroid_row": centroid_row,
                "centroid_col": centroid_col,
                "spread_radius_um": spread,
                "outside_roi_energy_um2": result["outside_roi_full_cmos_energy"],
                "wrong_peak_intensity": torch.stack(wrong_peak),
            }
            for key, value in values.items():
                row_parts[key].append(value.detach().cpu().numpy())
            for local, row in enumerate(chunk.tolist()):
                if row in save_set:
                    image_np = (image_cpu[local] if image_cpu is not None
                                else image[local].detach().cpu().numpy())
                    maps[str(row)] = image_np.astype(np.float32)
    arrays = {key: np.concatenate(parts) for key, parts in row_parts.items()}
    scores = arrays["roi_scores"]
    predictions = scores.argmax(1).astype(np.int64)
    labels = context["labels"][rows_cpu].cpu().numpy().astype(np.int64)
    core = arrays["core_energies_um2"]
    index = np.arange(len(labels))
    target_roi = scores[index, labels]
    target_core = core[index, labels]
    four_roi_energy = scores.sum(1)
    eta = arrays["cmos_energy_um2"] / np.maximum(arrays["input_energy_um2"], EPS)
    core_cmos_fraction = target_core / np.maximum(arrays["cmos_energy_um2"], EPS)
    core_input_fraction = target_core / np.maximum(arrays["input_energy_um2"], EPS)
    roi_input_fraction = target_roi / np.maximum(arrays["input_energy_um2"], EPS)
    own_roi_outside_core = np.maximum(target_roi - target_core, 0.0)
    wrong_roi_fraction = np.maximum(four_roi_energy - target_roi, 0.0) / np.maximum(
        arrays["cmos_energy_um2"], EPS)
    peak_in_roi, peak_in_core = [], []
    center_dist = []
    for label, row, col in zip(labels, arrays["global_peak_row"], arrays["global_peak_col"]):
        roi = c37.roi_bounds()[int(label)]
        core_box = c37.core_bounds(float(config["core"]["fraction_of_roi_side"]))[int(label)]
        peak_in_roi.append(roi[0] <= row < roi[1] and roi[2] <= col < roi[3])
        peak_in_core.append(core_box[0] <= row < core_box[1] and core_box[2] <= col < core_box[3])
        center = c37.roi_center_px(c37.roi_bounds())[int(label)]
        center_dist.append(math.hypot(float(row - center[0]), float(col - center[1]))
                           * c37.CMOS_PITCH_UM)
    metrics = _classification_metrics(labels, predictions)
    metrics.update({
        "population": int(len(labels)),
        "class_counts": {str(d): int((labels == d).sum()) for d in range(4)},
        "peak_in_correct_roi_rate": float(np.mean(peak_in_roi)),
        "peak_in_correct_core_rate": float(np.mean(peak_in_core)),
        "peak_distance_to_correct_roi_centre_um": _distribution(np.asarray(center_dist)),
        "correct_roi_input_fraction": _distribution(roi_input_fraction),
        "correct_core_cmos_fraction": _distribution(core_cmos_fraction),
        "correct_core_input_fraction": _distribution(core_input_fraction),
        "correct_roi_outside_core_energy_um2": _distribution(own_roi_outside_core),
        "correct_roi_outside_core_cmos_fraction": _distribution(
            own_roi_outside_core / np.maximum(arrays["cmos_energy_um2"], EPS)),
        "wrong_roi_cmos_fraction": _distribution(wrong_roi_fraction),
        "global_peak_intensity": _distribution(arrays["global_peak_intensity"]),
        "strongest_peak_outside_correct_roi_intensity": _distribution(arrays["wrong_peak_intensity"]),
        "eta_phase_to_cmos": _distribution(eta),
        "outside_all_four_rois_energy_um2": _distribution(arrays["outside_roi_energy_um2"]),
        "spread_radius_um": _distribution(arrays["spread_radius_um"]),
        "full_cmos_energy_um2": _distribution(arrays["cmos_energy_um2"]),
        "input_energy_um2": _distribution(arrays["input_energy_um2"]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "inference_rule": "frozen shared phase; four fixed ROI integrals; electronic argmax; labels used only for metrics",
        "lens_plane_pitch_um": float(lens_pitch_um or context["lens_pitch_um"]),
    })
    arrays.update({"rows": rows_cpu.numpy(), "labels": labels, "predictions": predictions,
                   "eta_phase_to_cmos": eta, "correct_core_input_fraction": core_input_fraction,
                   "correct_roi_input_fraction": roi_input_fraction,
                   "correct_roi_outside_core_energy_um2": own_roi_outside_core,
                   "wrong_roi_cmos_fraction": wrong_roi_fraction,
                   "peak_in_correct_roi": np.asarray(peak_in_roi, dtype=np.uint8),
                   "peak_in_correct_core": np.asarray(peak_in_core, dtype=np.uint8),
                   "peak_distance_to_core_centre_um": np.asarray(center_dist)})
    evaluation = {"metrics": metrics, "arrays": arrays, "maps": maps}
    if aggregate_group_means:
        assert group_all_sums is not None and group_correct_sums is not None
        assert group_all_counts is not None and group_correct_counts is not None
        all_means = np.divide(
            group_all_sums,
            group_all_counts[:, None, None],
            out=np.zeros_like(group_all_sums),
            where=group_all_counts[:, None, None] > 0,
        )
        correct_means = np.divide(
            group_correct_sums,
            group_correct_counts[:, None, None],
            out=np.zeros_like(group_correct_sums),
            where=group_correct_counts[:, None, None] > 0,
        )
        evaluation["group_mean_maps"] = {
            "all_class_means": all_means.astype(np.float32),
            "correct_class_means": correct_means.astype(np.float32),
            "overall_mean": group_all_sums.sum(axis=0).astype(np.float32) / float(len(labels)),
            "all_class_counts": group_all_counts.copy(),
            "correct_class_counts": group_correct_counts.copy(),
            "rows": rows_cpu.numpy().copy(),
            "labels": labels.copy(),
            "predictions": predictions.copy(),
        }
    return evaluation


def _save_eval(path: Path, evaluation: dict[str, Any]) -> None:
    atomic_savez(path, **evaluation["arrays"])


def _prepare_cache(context: dict[str, Any], rows: torch.Tensor, config: dict[str, Any],
                   chunk_size: int = 256) -> dict[str, Any]:
    cache_path = Path(config["run"]["system_cache_path"])
    return c37.prepare_field_cache(context, rows, cache_path=cache_path, chunk_size=chunk_size)


def _source_audit(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    phases = resolve_phase_sources(config, run_dir)
    histories = _resolve_history_sources(config, run_dir)
    records = {}
    for key, relative in config["source_candidates"].items():
        phase_path = _resolved_artifact(relative, key, run_dir=run_dir)
        records[key] = {
            "phase_path": str(phase_path),
            "phase_sha256": sha256_file(phase_path),
            "phase_shape": list(phases[key].shape),
            "phase_finite": bool(torch.isfinite(phases[key]).all()),
            "history_path": str(histories[key]),
            "history_sha256": sha256_file(histories[key]),
        }
    return records


def run_p0(config: dict[str, Any], context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Replay valid historical phases and audit C3.7 exceptions without relabeling them."""
    phase_sources = resolve_phase_sources(config, run_dir)
    history_sources = _resolve_history_sources(config, run_dir)
    rows = context["split"]["validation"].detach().cpu()
    if len(rows) != 2000:
        raise ValueError("C3.8 must reuse the exact C3.7 2,000-row development validation order")
    train_rows = _formal_training_rows(context, config)
    cache_rows = torch.cat((train_rows, rows)).unique(sorted=True)
    cache_metadata = _prepare_cache(context, cache_rows, config)
    results, comparisons = {}, {}
    for key, phase in phase_sources.items():
        evaluation = evaluate_frozen(context, config, phase, rows, batch_size=4)
        result_path = run_dir / "p0" / f"{key}_replay.npz"
        _save_eval(result_path, evaluation)
        results[key] = evaluation["metrics"]
        with np.load(history_sources[key], allow_pickle=False) as saved:
            historical = {name: saved[name] for name in saved.files}
        actual = evaluation["arrays"]
        required = ("rows", "labels", "predictions", "roi_scores", "core_energies_um2",
                    "input_energy_um2", "cmos_energy_um2", "eta_phase_to_cmos")
        # C3.7 uses one legacy key spelling for input energy; normalize only for comparison.
        if "core_energies_um2" not in historical and "core_energies" in historical:
            historical["core_energies_um2"] = historical["core_energies"]
        diffs = {}
        for field in required:
            if field not in historical or field not in actual:
                diffs[field] = {"available": False}
                continue
            left = np.asarray(actual[field])
            right = np.asarray(historical[field])
            same_shape = left.shape == right.shape
            diffs[field] = {
                "available": True, "same_shape": bool(same_shape),
                "exact_match": bool(same_shape and np.array_equal(left, right)),
                "max_abs_difference": float(np.max(np.abs(left - right))) if same_shape else None,
                "relative_l2_difference": float(np.linalg.norm(left - right) /
                                                  max(float(np.linalg.norm(right)), EPS))
                if same_shape and left.dtype.kind in "fci" else 0.0 if same_shape else None,
            }
        comparisons[key] = {
            "history_source": str(history_sources[key]),
            "same_rows_and_labels": bool(
                np.array_equal(actual["rows"], historical.get("rows", [])) and
                np.array_equal(actual["labels"], historical.get("labels", []))),
            "prediction_match": bool(np.array_equal(actual["predictions"],
                                                     historical.get("predictions", []))),
            "field_comparisons": diffs,
        }
    # Current status is read as context; the corrected record below is the C3.8 source.
    status_path = ROOT / "outputs/20261105_hw207_mnist4_c37_v1/runs/closeout_codex_01a18b9f_01/corrected_acceptance.json"
    if not status_path.is_file():
        status_path = run_dir / "inputs" / "c37_corrected_acceptance.json"
    corrected_c37 = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    epoch_audit = {}
    summary_roots = [C37_RUN_DIR / "train", run_dir / "inputs" / "c37_training_summaries"]
    summary_paths = sorted({path for root in summary_roots
                            for path in root.glob("*/training_summary.json")})
    for summary_path in summary_paths:
        record = json.loads(summary_path.read_text(encoding="utf-8"))
        epochs = [int(item["epoch"]) for item in record.get("history", []) if "epoch" in item]
        epoch_audit[summary_path.parent.name] = {
            "history_rows": len(epochs), "unique_epoch_count": len(set(epochs)),
            "unique_epochs": sorted(set(epochs)),
            "duplicate_epoch_rows": len(epochs) - len(set(epochs)),
        }
    payload = {
        "identity": "C38_P0_HISTORICAL_REPLAY",
        "population": {"role": "development_validation_reused_for_selection",
                       "rows": int(len(rows)), "source": "C37 exact validation split order",
                       "class_counts": {str(d): int((context["labels"][rows] == d).sum())
                                         for d in range(4)}},
        "replays": results,
        "input_field_cache": cache_metadata,
        "historical_comparisons": comparisons,
        "source_audit": _source_audit(config, run_dir),
        "historical_corrections": {
            "E6": "INVALID_SIGN_DEFECT_EXCLUDED_FROM_VALID_CANDIDATES",
            "E4": "FINAL_PROCESSING_BINDING_INVALID; NOT A FABRICATION_VALIDATION",
            "E5": "RAW_DESIGN_AND_PROCESSED_HEIGHT_STAGE_MISMATCH; DO NOT MIX",
            "oracle": "CLASS_CONDITIONED_REACHABILITY_CONTROL_ONLY; NOT A SINGLE_PHASE CAPACITY UPPER BOUND",
            "E7": "ONE_WEIGHT_CONTROL; NOT A BROAD ARCHITECTURE SEARCH",
            "training_history": "actual updates are counted from unique epoch identifiers, not duplicated history rows",
            "efficiency_7_708_percent": "historical reference only; not a validated physical threshold",
            "outside_energy": "target ROI outside, correct core outside, and all four ROIs outside are distinct quantities",
        },
        "unique_epoch_audit": epoch_audit,
        "c37_corrected_acceptance_available": corrected_c37 is not None,
        "input_model_lineage": {
            "input": "C34 input_field: C0 binary >=128, C2.1 geometry map, zero-defocus coherent pupil cutoff 0.1 cyc/um",
            "c33_context_note": "C37 constructs a C33 configuration with 52 mm metadata, but C34 input_field does not propagate that defocus; it applies the zero-defocus pupil mapping",
            "output": "C34 Model B: P80 -> 2 mm finite pupil and 30 mm thin lens -> P100 -> 2048 CMOS",
            "scope": "fixed numerical proxy, not measured DMD/relay/LightGen/CMOS behavior",
        },
        "all_historical_prediction_replays_match": all(
            item["same_rows_and_labels"] and item["prediction_match"] and
            all(item["field_comparisons"].get(name, {}).get("available", False)
                and item["field_comparisons"][name].get("same_shape", False)
                and item["field_comparisons"][name].get("relative_l2_difference", math.inf) <= 1e-5
                for name in ("input_energy_um2", "roi_scores", "core_energies_um2",
                             "cmos_energy_um2", "eta_phase_to_cmos"))
            for item in comparisons.values()),
    }
    write_json(run_dir / "p0" / "c38_p0_history_audit.json", payload)
    write_json(run_dir / "p0" / "c38_p0_replay_metrics.json", results)
    return payload


def _direct_fresnel_error(device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(3801)
    real = torch.randn((3, 4), generator=generator, dtype=torch.float64)
    imag = torch.randn((3, 4), generator=generator, dtype=torch.float64)
    field = torch.complex(real, imag).to(device)[None]
    actual = c34.scaled_fresnel(field, 2.0, (4, 5), 3.5, 1000.0)
    y = (torch.arange(4, dtype=torch.float64) - 1.5) * 3.5
    x = (torch.arange(5, dtype=torch.float64) - 2.0) * 3.5
    coords = [(float(xx), float(yy)) for yy in y for xx in x]
    expected = c34_runner._direct_fresnel(field[0], 2.0, coords, 1000.0).reshape(4, 5).to(device)
    relative = float((actual[0] - expected).norm() / expected.norm().clamp_min(1e-30))
    return {"relative_l2": relative, "tolerance": 1e-8,
            "passed": bool(relative <= 1e-8), "reference": "independent direct Fresnel quadrature",
            "probe_shape": [3, 4], "output_shape": [4, 5]}


def _single_frequency_check(device: torch.device) -> dict[str, Any]:
    count, pitch, z = 128, 2.0, 1000.0
    freq = 2.0 / (count * pitch)
    x = (torch.arange(count, device=device, dtype=torch.float64) - (count - 1) / 2) * pitch
    wave = torch.exp(2j * math.pi * freq * x)[None, None, :].expand(1, count, count).to(torch.complex128)
    propagated = c34.scaled_fresnel(wave, pitch, (count, count), pitch, z)
    probe_points = [(64, 64), (64, 48), (64, 80), (48, 64), (80, 64),
                    (48, 48), (48, 80), (80, 48), (80, 80)]
    axis_um = [(index - (count - 1) / 2) * pitch for index in range(count)]
    output_coords = [(axis_um[col], axis_um[row]) for row, col in probe_points]
    expected = c34_runner._direct_fresnel(wave[0], pitch, output_coords, z).to(device)
    reversed_reference = c34_runner._direct_fresnel(
        wave[0], pitch, output_coords, -z).to(device)
    actual = torch.stack([propagated[0, row, col] for row, col in probe_points])
    tolerance = 1e-8
    reverse_rejection_threshold = 1e-3
    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-30))
    reversed_relative = float((actual - reversed_reference).norm()
                              / expected.norm().clamp_min(1e-30))
    reverse_rejected = reversed_relative > reverse_rejection_threshold
    return {
        "relative_l2_vs_independent_finite_aperture_reference": relative,
        "tolerance": tolerance,
        "reverse_propagation_relative_l2": reversed_relative,
        "reverse_propagation_rejection_threshold": reverse_rejection_threshold,
        "reverse_propagation_rejected": reverse_rejected,
        "passed": bool(relative <= tolerance and reverse_rejected),
        "reference": "independent direct Fresnel quadrature on the same finite aperture",
        "spatial_frequency_cycles_per_um": freq,
        "analytic_infinite_plane_wave_phase_rad_for_reference": (
            -math.pi * c34.WAVELENGTH_UM * z * freq**2),
        "probe_points_row_col": [list(point) for point in probe_points],
    }


def _phase_modulation_check() -> dict[str, Any]:
    g = torch.Generator().manual_seed(3811)
    field = torch.complex(torch.randn((4, 100, 100), generator=g),
                          torch.randn((4, 100, 100), generator=g))
    phase = torch.rand((100, 100), generator=g) * (2 * math.pi)
    before = field.abs().square().sum((-2, -1))
    after = (field * phase_transmission(phase)).abs().square().sum((-2, -1))
    error = float(((after - before).abs() / before).max())
    periodic = float((phase_transmission(phase) - phase_transmission(phase + 2 * math.pi)).abs().max())
    return {"zero_distance_screen": "E_after = E_before * exp(i*phase); no free-space z=0 operator exists in Model B",
            "unit_amplitude_energy_relative_error": error,
            "2pi_periodicity_max_complex_error": periodic,
            "passed": bool(error <= 1e-6 and periodic <= 2e-6)}


def _full_model_precision(context: dict[str, Any], phase: torch.Tensor,
                          row: int) -> dict[str, Any]:
    device = context["device"]
    field32 = c37.input_fields(context, torch.tensor([row]))
    with torch.no_grad():
        result32 = forward_field(field32, phase.to(device), context)
        result64 = forward_field(field32.to(torch.complex128), phase.to(device=device, dtype=torch.float64), context)
    intensity32 = result32["camera_intensity"].double()
    intensity64 = result64["camera_intensity"]
    relative_l2 = float((intensity32 - intensity64).norm() / intensity64.norm().clamp_min(1e-30))
    e32 = float(result32["detected_energy"][0])
    e64 = float(result64["detected_energy"][0])
    relative_energy = abs(e32 - e64) / max(abs(e64), EPS)
    full_account = float(result64["roi_scores"].sum() + result64["outside_roi_full_cmos_energy"][0])
    account_relative = abs(full_account - e64) / max(abs(e64), EPS)
    return {"intensity_relative_l2_fp32_vs_fp64": relative_l2,
            "detected_energy_relative_error_fp32_vs_fp64": relative_energy,
            "roi_plus_outside_equals_cmos_relative_error": account_relative,
            "fp32_vs_fp64_tolerance": 1e-5,
            "passed": bool(relative_l2 <= 1e-5 and relative_energy <= 1e-5 and account_relative <= 1e-10),
            "input_energy_um2": float(result64["input_energy"][0]),
            "after_finite_pupil_energy_um2": float(result64["lens_energy"][0]),
            "full_cmos_energy_um2": e64,
            "finite_pupil_retention": float(result64["lens_energy"][0] /
                                             result64["input_energy"][0].clamp_min(EPS))}


def _sampling_sensitivity(context: dict[str, Any], config: dict[str, Any], phase: torch.Tensor,
                          rows: torch.Tensor) -> dict[str, Any]:
    results = {}
    for pitch in config["numerics"]["sampling_sensitivity_lens_plane_pitch_um"]:
        results[str(pitch)] = evaluate_frozen(context, config, phase, rows, batch_size=1,
                                               lens_pitch_um=float(pitch))
    reference = results[str(config["numerics"]["sampling_reference_pitch_um"])]["arrays"]
    comparison = {}
    for pitch, result in results.items():
        current = result["arrays"]
        n = len(rows)
        comparison[pitch] = {
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "peak_in_correct_roi_rate": result["metrics"]["peak_in_correct_roi_rate"],
            "correct_core_input_fraction_mean": result["metrics"]["correct_core_input_fraction"]["mean"],
            "eta_mean": result["metrics"]["eta_phase_to_cmos"]["mean"],
            "prediction_agreement_to_1um": float((current["predictions"] == reference["predictions"]).mean()),
            "roi_score_l1_over_input_max": float(np.max(
                np.abs(current["roi_scores"] - reference["roi_scores"]).sum(1) /
                np.maximum(current["input_energy_um2"], EPS))),
            "core_energy_l1_over_input_max": float(np.max(
                np.abs(current["core_energies_um2"] - reference["core_energies_um2"]).max(1) /
                np.maximum(current["input_energy_um2"], EPS))),
            "peak_displacement_vs_1um_max_px": float(np.max(np.hypot(
                current["global_peak_row"] - reference["global_peak_row"],
                current["global_peak_col"] - reference["global_peak_col"]))),
            "rows": n,
        }
    return {"reference_pitch_um": float(config["numerics"]["sampling_reference_pitch_um"]),
            "probe_rows": rows.tolist(), "comparison": comparison,
            "interpretation": "sampling sensitivity is reported, not used to change frozen Model B geometry"}


def run_p1(config: dict[str, Any], context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    phases = resolve_phase_sources(config, run_dir)
    phase = phases["C35_P1"]
    train_rows = _formal_training_rows(context, config)
    validation = context["split"]["validation"].detach().cpu()
    cache_rows = torch.cat((train_rows, validation)).unique(sorted=True)
    cache = _prepare_cache(context, cache_rows, config)
    direct = _direct_fresnel_error(context["device"])
    frequency = _single_frequency_check(context["device"])
    modulation = _phase_modulation_check()
    rows_per_class = int(config["numerics"]["sampling_probe_rows_per_class"])
    sample_rows = _balanced_rows(validation, context["labels"], rows_per_class)
    sampling = _sampling_sensitivity(context, config, phase, sample_rows)
    precision = _full_model_precision(context, phase, int(validation[0]))
    source_eta = evaluate_frozen(context, config, phase, train_rows, batch_size=4)
    eta_reference = source_eta["arrays"]["eta_phase_to_cmos"].astype(np.float32)
    atomic_savez(run_dir / "p1" / "c38_c35_p1_train_efficiency_reference.npz",
                 rows=train_rows.numpy(), eta_phase_to_cmos=eta_reference)
    numeric_pass = bool(direct["passed"] and frequency["passed"] and modulation["passed"]
                        and precision["passed"])
    payload = {
        "identity": "C38_P1_SHARED_MODEL_NUMERICAL_VALIDATION",
        "physical_model": {
            "status": "TAPE_ESTIMATE_AND_NUMERICAL_ASSUMPTION_ONLY",
            "wavelength_nm": 532.0, "phase_grid": [100, 100], "phase_pitch_um": 2.0,
            "phase_screen_size_um": [200.0, 200.0],
            "input": "C34 binary DMD/relay proxy; threshold 128, geometry map, zero-defocus coherent pupil cutoff 0.1 cycles/um",
            "output": "C37 forward_batch -> C34 Model B scalar Fresnel, screen-lens 80 mm, 2 mm pupil, f=30 mm, lens-CMOS 100 mm",
            "lens_sampling_nominal_um": 4.0,
            "cmos": [2048, 2048], "cmos_pixel_pitch_um": 6.5,
            "roi_bounds": c37.roi_bounds(),
            "core_bounds": c37.core_bounds(float(config["core"]["fraction_of_roi_side"])),
            "input_relay_and_hardware": "proxy only; current hardware mapping remains unverified",
        },
        "zero_distance_screen_check": modulation,
        "single_frequency_transfer_check": frequency,
        "independent_direct_fresnel_check": direct,
        "full_model_precision_and_energy": precision,
        "lens_sampling_sensitivity": sampling,
        "train_rows_for_c35_p1_eta_reference": int(len(train_rows)),
        "eta_reference_summary": _distribution(eta_reference),
        "input_cache": cache,
        "numerical_core_pass": numeric_pass,
        "sampling_note": "4/2/1 um differences are measured and retained; nominal pitch remains the frozen C3.7 4 um value",
    }
    write_json(run_dir / "p1" / "c38_p1_numerical_validation.json", payload)
    return payload


def _eta_reference_from_p1(run_dir: Path) -> dict[int, float]:
    path = run_dir / "p1" / "c38_c35_p1_train_efficiency_reference.npz"
    if not path.is_file():
        raise FileNotFoundError("P1 C3.5 P1 per-row efficiency reference is missing")
    with np.load(path, allow_pickle=False) as data:
        rows = data["rows"].astype(np.int64)
        eta = data["eta_phase_to_cmos"].astype(np.float64)
    if len(rows) != len(eta) or not np.isfinite(eta).all() or (eta <= 0).any():
        raise ValueError("invalid C3.5 P1 efficiency reference")
    return {int(row): float(value) for row, value in zip(rows, eta)}


def _eta_for_rows(rows: torch.Tensor, reference: dict[int, float], device: torch.device) -> torch.Tensor:
    try:
        return torch.tensor([reference[int(row)] for row in rows.detach().cpu().tolist()],
                            dtype=torch.float32, device=device)
    except KeyError as exc:
        raise ValueError(f"training row {exc.args[0]} lacks paired C3.5 P1 energy reference") from exc


def _production_update(context: dict[str, Any], config: dict[str, Any], phase: torch.nn.Parameter,
                       optimizer: torch.optim.Optimizer, rows: torch.Tensor,
                       eta_reference: torch.Tensor,
                       active_losses: Sequence[str] | None = None) -> dict[str, Any]:
    device = context["device"]
    micro = int(config["training"]["microbatch_size"])
    logical_count = len(rows)
    optimizer.zero_grad(set_to_none=True)
    component_sums = {name: 0.0 for name in config["training"]["loss_order"]}
    for start in range(0, logical_count, micro):
        batch_rows = rows[start:start + micro]
        labels = context["labels"][batch_rows].to(device)
        output = forward_field(c37.input_fields(context, batch_rows), phase, context)
        loss, components = loss_components(
            output, labels, phase, context, config,
            eta_reference=eta_reference[start:start + len(batch_rows)],
            active_losses=active_losses)
        scale = len(batch_rows) / logical_count
        (loss * scale).backward()
        for name, value in components.items():
            component_sums[name] += float(value.detach()) * scale
    if phase.grad is None or not torch.isfinite(phase.grad).all():
        raise RuntimeError("C3.8 production path has absent or non-finite phase gradient")
    gradient_norm = float(phase.grad.detach().norm())
    before = phase.detach().clone()
    optimizer.step()
    delta = float((phase.detach() - before).norm())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0 or not math.isfinite(delta) or delta <= 0:
        raise RuntimeError("C3.8 production optimizer failed to update the shared phase")
    enabled = set(active_losses) if active_losses is not None else set(component_sums)
    total_loss = sum(float(config["training"]["loss_weights"][name]) * component_sums[name]
                     for name in enabled)
    return {"loss": total_loss, "loss_components": component_sums,
            "gradient_norm": gradient_norm, "phase_update_l2": delta,
            "microbatches": int(math.ceil(logical_count / micro)),
            "logical_batch_size": logical_count}


def _production_update_with_labels(context: dict[str, Any], config: dict[str, Any],
                                   phase: torch.nn.Parameter, optimizer: torch.optim.Optimizer,
                                   rows: torch.Tensor, eta_reference: torch.Tensor,
                                   labels_override: torch.Tensor) -> dict[str, Any]:
    device = context["device"]
    micro = int(config["training"]["microbatch_size"])
    optimizer.zero_grad(set_to_none=True)
    sums = {name: 0.0 for name in config["training"]["loss_order"]}
    before = phase.detach().clone()
    for start in range(0, len(rows), micro):
        part = rows[start:start + micro]
        labels = labels_override[start:start + len(part)].to(device)
        output = forward_field(c37.input_fields(context, part), phase, context)
        loss, terms = loss_components(output, labels, phase, context, config,
                                      eta_reference=eta_reference[start:start + len(part)])
        scale = len(part) / len(rows)
        (loss * scale).backward()
        for name, value in terms.items():
            sums[name] += float(value.detach()) * scale
    if phase.grad is None or not torch.isfinite(phase.grad).all() or float(phase.grad.norm()) <= 0:
        raise RuntimeError("random-label negative control has invalid gradient")
    grad_norm = float(phase.grad.norm())
    optimizer.step()
    delta = float((phase.detach() - before).norm())
    return {"loss_components": sums, "gradient_norm": grad_norm, "phase_update_l2": delta}


def _finite_difference_checks(config: dict[str, Any], context: dict[str, Any],
                              sources: dict[str, torch.Tensor], train_rows: torch.Tensor,
                              eta_reference: dict[int, float]) -> dict[str, Any]:
    device = context["device"]
    chosen_rows = _balanced_rows(train_rows, context["labels"], 1)
    source_names = ["C35_P1", "C37_E3_seed23", "C35_P1", "C37_E3_seed23"]
    entries, class_details = [], []
    step = float(config["numerics"]["gradient_finite_difference_step_rad"])
    for digit, row_tensor in enumerate(chosen_rows):
        row = int(row_tensor)
        labels = context["labels"][row:row + 1].to(device)
        if int(labels[0]) != digit:
            raise ValueError("C3.8 gradient probe class ordering changed")
        field = c37.input_fields(context, torch.tensor([row])).to(torch.complex128)
        init_name = source_names[digit]
        initial = sources[init_name].to(device=device, dtype=torch.float64).clone()
        if digit == 3:
            generator = torch.Generator(device="cpu").manual_seed(3837)
            initial = (torch.rand((100, 100), generator=generator, dtype=torch.float64)
                       * (2 * math.pi)).to(device)
            init_name = "RANDOM_UNIFORM_PHASE"
        initial.requires_grad_(True)
        reference = torch.tensor([eta_reference[row]], dtype=torch.float64, device=device)
        optical = forward_field(field, initial, context)
        total_loss, parts = loss_components(optical, labels, initial, context, config,
                                            eta_reference=reference)
        gradient = torch.autograd.grad(total_loss, initial, retain_graph=True)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"C3.8 gradient check has non-finite values for class {digit}")
        flat = gradient.detach().abs().flatten()
        strong_index = int(flat.argmax())
        small_index = int(torch.argsort(flat)[min(len(flat) // 10, len(flat) - 1)])
        detail = {"class": digit, "row": row, "initialization": init_name,
                  "loss": float(total_loss.detach()), "term_gradient_norms": {}}
        for name, component in parts.items():
            grad = torch.autograd.grad(component, initial, retain_graph=True)[0]
            detail["term_gradient_norms"][name] = float(grad.norm())
        periodic_error = float((phase_transmission(initial.detach()) -
                                phase_transmission(initial.detach() + 2 * math.pi)).abs().max())
        for kind, flat_index in (("strong", strong_index), ("small", small_index),
                                 ("periodic_boundary", 0)):
            rr, cc = divmod(flat_index, 100)
            plus, minus = initial.detach().clone(), initial.detach().clone()
            plus[rr, cc] += step
            minus[rr, cc] -= step
            losses = []
            with torch.no_grad():
                for candidate in (plus, minus):
                    produced = forward_field(field, candidate, context)
                    value, _ = loss_components(produced, labels, candidate, context, config,
                                               eta_reference=reference)
                    losses.append(float(value))
            finite = (losses[0] - losses[1]) / (2 * step)
            analytic = float(gradient[rr, cc])
            entries.append({"class": digit, "row": row, "initialization": init_name,
                            "kind": kind, "pixel_yx": [rr, cc], "autograd": analytic,
                            "central_difference": finite,
                            "relative_error": abs(finite - analytic) /
                            max(abs(finite), abs(analytic), 1e-12),
                            "absolute_error": abs(finite - analytic),
                            "phase_periodicity_max_error": periodic_error})
        class_details.append(detail)
    strong = [item for item in entries if item["kind"] == "strong"]
    small = [item for item in entries if item["kind"] == "small"]
    passed = bool(
        all(item["relative_error"] <= float(config["numerics"]["gradient_relative_error_tolerance"])
            for item in strong)
        and all(item["absolute_error"] <= float(config["numerics"]["gradient_absolute_error_tolerance"])
                for item in small)
        and all(item["phase_periodicity_max_error"] <= 2e-6 for item in entries))
    return {"step_rad": step, "entries": entries, "class_details": class_details,
            "strong_checks": len(strong), "small_gradient_checks": len(small), "passed": passed}


def run_p2(config: dict[str, Any], context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    p0_path = run_dir / "p0" / "c38_p0_history_audit.json"
    p1_path = run_dir / "p1" / "c38_p1_numerical_validation.json"
    if not p0_path.is_file() or not p1_path.is_file():
        raise FileNotFoundError("P2 depends on complete P0 and P1 evidence")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    if not p0["all_historical_prediction_replays_match"] or not p1["numerical_core_pass"]:
        raise RuntimeError("P0 or P1 failed; downstream training is fail-closed")
    sources = resolve_phase_sources(config, run_dir)
    train_rows = _formal_training_rows(context, config)
    eta_map = _eta_reference_from_p1(run_dir)
    gradient_checks = _finite_difference_checks(config, context, sources, train_rows, eta_map)
    device = context["device"]
    phase = torch.nn.Parameter(sources["C35_P1"].to(device))
    sample_rows = _balanced_rows(train_rows, context["labels"], 1)
    labels = context["labels"][sample_rows].to(device)
    eta_ref = _eta_for_rows(sample_rows, eta_map, device)
    output = forward_field(c37.input_fields(context, sample_rows), phase, context)
    joint, parts = loss_components(output, labels, phase, context, config, eta_reference=eta_ref)
    term_gradients = {}
    for name, value in parts.items():
        grad = torch.autograd.grad(value, phase, retain_graph=True)[0]
        term_gradients[name] = {"value": float(value.detach()),
                                "gradient_norm": float(grad.norm()),
                                "gradient_finite": bool(torch.isfinite(grad).all()),
                                "gradient_nonzero": bool(float(grad.norm()) > 0)}
    joint_grad = torch.autograd.grad(joint, phase)[0]
    term_pass = all(record["gradient_finite"] and record["gradient_nonzero"]
                    for record in term_gradients.values())
    phase_before_updates = phase.detach().clone()
    optimizer = torch.optim.Adam([phase], lr=float(config["training"]["learning_rate"]))
    update_rows = _balanced_rows(train_rows, context["labels"], 16)[:64]
    eta_update = _eta_for_rows(update_rows, eta_map, device)
    real_updates = [_production_update(context, config, phase, optimizer, update_rows, eta_update)
                    for _ in range(3)]
    random_phase = torch.nn.Parameter(phase_before_updates.clone())
    random_phase_before = random_phase.detach().clone()
    random_optimizer = torch.optim.Adam([random_phase], lr=float(config["training"]["learning_rate"]))
    random_labels = context["labels"][update_rows].clone()
    label_rng = torch.Generator(device="cpu").manual_seed(3882)
    random_labels = random_labels[torch.randperm(len(random_labels), generator=label_rng)]
    labels_are_permuted = not torch.equal(random_labels, context["labels"][update_rows])
    negative_updates = [_production_update_with_labels(
        context, config, random_phase, random_optimizer, update_rows, eta_update, random_labels)
        for _ in range(3)]
    random_validation = _balanced_rows(context["split"]["validation"], context["labels"], 250)
    before_negative = evaluate_frozen(context, config, random_phase_before, random_validation, batch_size=4)
    after_negative = evaluate_frozen(context, config, random_phase.detach(), random_validation, batch_size=4)
    negative_update_l2_vs_correct = float(torch.linalg.vector_norm(phase.detach() - random_phase.detach()))
    negative_control_pass = bool(labels_are_permuted and negative_update_l2_vs_correct > 1e-6)
    updates_pass = all(item["gradient_norm"] > 0 and item["phase_update_l2"] > 0
                       and all(math.isfinite(value) for value in item["loss_components"].values())
                       for item in real_updates)

    # Exercise the exact epoch checkpoint and --resume path before P3. The
    # interrupted arm stops after epoch 1, preserves that checkpoint, and
    # resumes epoch 2. Its result must match an uninterrupted two-epoch run.
    recovery_root = run_dir / "p3" / "preflight_recovery"
    continuous_dir = recovery_root / "continuous"
    resumed_dir = recovery_root / "interrupted_then_resumed"
    if recovery_root.exists():
        raise FileExistsError(f"C3.8 P2 recovery evidence already exists: {recovery_root}")
    recovery_train_rows = update_rows.detach().cpu()
    recovery_validation_rows = _balanced_rows(context["split"]["validation"], context["labels"], 16)
    recovery_losses = tuple(config["training"]["loss_order"])
    recovery_seed = 3883
    continuous_summary = _train_one(
        config, context, continuous_dir, eta_map, sources,
        arm="P2_RECOVERY_CONTROL", initialization="C35_P1", seed=recovery_seed,
        train_rows=recovery_train_rows, validation_rows=recovery_validation_rows,
        active_losses=recovery_losses, max_epochs=2)
    _train_one(
        config, context, resumed_dir, eta_map, sources,
        arm="P2_RECOVERY_CONTROL", initialization="C35_P1", seed=recovery_seed,
        train_rows=recovery_train_rows, validation_rows=recovery_validation_rows,
        active_losses=recovery_losses, max_epochs=1)
    interrupted_receipt = resumed_dir / "training_summary.json"
    interrupted_receipt.rename(resumed_dir / "interrupted_epoch1_summary.json")
    resumed_summary = _train_one(
        config, context, resumed_dir, eta_map, sources,
        arm="P2_RECOVERY_CONTROL", initialization="C35_P1", seed=recovery_seed,
        train_rows=recovery_train_rows, validation_rows=recovery_validation_rows,
        active_losses=recovery_losses, max_epochs=2, resume=True)
    continuous_checkpoint = torch.load(continuous_dir / "checkpoint.pt", map_location="cpu",
                                      weights_only=False)
    resumed_checkpoint = torch.load(resumed_dir / "checkpoint.pt", map_location="cpu",
                                    weights_only=False)
    continuous_opt = continuous_checkpoint["optimizer"]
    resumed_opt = resumed_checkpoint["optimizer"]
    optimizer_state_match = continuous_opt["param_groups"] == resumed_opt["param_groups"]
    if optimizer_state_match:
        optimizer_state_match = continuous_opt["state"].keys() == resumed_opt["state"].keys()
    if optimizer_state_match:
        for state_id in continuous_opt["state"]:
            for key, expected in continuous_opt["state"][state_id].items():
                actual = resumed_opt["state"][state_id][key]
                if isinstance(expected, torch.Tensor):
                    if not torch.equal(expected, actual):
                        optimizer_state_match = False
                        break
                elif expected != actual:
                    optimizer_state_match = False
                    break
            if not optimizer_state_match:
                break
    history_keys = ("epoch", "updates", "train_loss_mean", "train_loss_components_sum_over_updates",
                    "gradient_norm_mean", "validation", "validation_selection_score",
                    "improved", "sample_order_rows")
    continuous_history, resumed_history = [], []
    for history, comparable in ((continuous_summary["history"], continuous_history),
                                (resumed_summary["history"], resumed_history)):
        for row in history:
            record = {}
            for key in history_keys:
                value = row[key]
                if key == "validation":
                    value = {name: metric for name, metric in value.items()
                             if name != "elapsed_seconds"}
                record[key] = value
            comparable.append(record)
    phase_continuous = np.load(continuous_dir / "phase_rad.npy", allow_pickle=False)
    phase_resumed = np.load(resumed_dir / "phase_rad.npy", allow_pickle=False)
    phase_recovery_relative_l2 = float(np.linalg.norm(phase_continuous - phase_resumed) /
                                       max(float(np.linalg.norm(phase_continuous)), EPS))
    roles_continuous = np.load(continuous_dir / "sample_roles.npz", allow_pickle=False)
    roles_resumed = np.load(resumed_dir / "sample_roles.npz", allow_pickle=False)
    sample_roles_match = (np.array_equal(roles_continuous["train_rows"], roles_resumed["train_rows"])
                          and np.array_equal(roles_continuous["validation_rows"],
                                             roles_resumed["validation_rows"]))
    recovery_checks = {
        "same_epoch_count_and_update_count": (
            continuous_summary["epochs_completed"] == resumed_summary["epochs_completed"] == 2
            and continuous_summary["training_updates"] == resumed_summary["training_updates"] == 2),
        "same_next_epoch_sample_order_and_scientific_history_excluding_elapsed_seconds": (
            continuous_history == resumed_history),
        "same_train_and_validation_roles": sample_roles_match,
        "same_active_losses_and_optimizer_schedule": (
            continuous_checkpoint["active_losses"] == resumed_checkpoint["active_losses"]
            and continuous_checkpoint["learning_rate"] == resumed_checkpoint["learning_rate"]
            and continuous_checkpoint["logical_batch_size"] == resumed_checkpoint["logical_batch_size"]
            and continuous_checkpoint["microbatch_size"] == resumed_checkpoint["microbatch_size"]),
        "phase_relative_l2_within_1e_6": phase_recovery_relative_l2 <= 1e-6,
        "optimizer_state_identical": bool(optimizer_state_match),
        "rng_state_identical": (
            torch.equal(continuous_checkpoint["torch_rng_state"], resumed_checkpoint["torch_rng_state"])
            and torch.equal(continuous_checkpoint["cuda_rng_state"], resumed_checkpoint["cuda_rng_state"])),
        "checkpoint_identity_and_epoch_match": (
            continuous_checkpoint["identity"] == resumed_checkpoint["identity"]
            and continuous_checkpoint["completed_epoch"] == resumed_checkpoint["completed_epoch"] == 2),
    }
    recovery_pass = all(recovery_checks.values())
    payload = {
        "identity": "C38_P2_JOINT_LOSS_PREFLIGHT",
        "loss_order_and_weights": config["training"]["loss_weights"],
        "component_gradients": term_gradients,
        "joint_gradient_norm": float(joint_grad.norm()),
        "joint_gradient_finite": bool(torch.isfinite(joint_grad).all()),
        "finite_difference": gradient_checks,
        "three_real_production_updates": real_updates,
        "random_label_negative_control": {
            "updates": negative_updates,
            "before_update_true_label_accuracy": before_negative["metrics"]["accuracy"],
            "after_three_updates_true_label_accuracy": after_negative["metrics"]["accuracy"],
            "expected": "with the same initialization, optimizer, and sample rows, permuted labels produce a different phase update than true labels",
            "labels_are_permuted": labels_are_permuted,
            "phase_update_l2_vs_true_label_branch": negative_update_l2_vs_correct,
            "passed": negative_control_pass,
            "interpretation": "label-sensitivity counterfactual; validation accuracy is diagnostic only and is not an independent accuracy gate",
            "train_label_permutation": random_labels.tolist(),
        },
        "checkpoint_recovery": {
            "mode": "engineering_preflight_only; one shared phase; no formal candidate",
            "continuous_summary": str(continuous_dir / "training_summary.json"),
            "interrupted_epoch1_receipt": str(resumed_dir / "interrupted_epoch1_summary.json"),
            "resumed_summary": str(resumed_dir / "training_summary.json"),
            "updates_per_branch": 2,
            "phase_relative_l2": phase_recovery_relative_l2,
            "checks": recovery_checks,
            "passed": recovery_pass,
        },
        "trainable_groups": ["one shared 100x100 phase map"],
        "frozen_inference_has_no_digital_head": True,
        "checks": {
            "all_five_components_have_finite_nonzero_gradient": bool(term_pass),
            "joint_gradient_finite_nonzero": bool(torch.isfinite(joint_grad).all() and joint_grad.norm() > 0),
            "finite_difference_matches": bool(gradient_checks["passed"]),
            "three_optimizer_updates_real": bool(updates_pass),
            "random_label_negative_control_rejects_label_independence": negative_control_pass,
            "explicit_checkpoint_resume_matches_continuous": recovery_pass,
        },
        "passed": bool(term_pass and torch.isfinite(joint_grad).all() and joint_grad.norm() > 0
                         and gradient_checks["passed"] and updates_pass
                         and negative_control_pass and recovery_pass),
    }
    write_json(run_dir / "p2" / "c38_preflight.json", payload)
    return payload


def _candidate_score(metrics: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> float:
    targets = config["evaluation"]
    eta0 = baseline["eta_phase_to_cmos"]
    core0 = baseline["correct_core_input_fraction"]
    current_eta = metrics["eta_phase_to_cmos"]
    current_core = metrics["correct_core_input_fraction"]
    ratios = [
        metrics["accuracy"] / float(targets["classification_accuracy_target"]),
        metrics["macro_f1"] / float(targets["macro_f1_target"]),
        metrics["peak_in_correct_roi_rate"] / float(targets["peak_in_correct_roi_target"]),
        metrics["peak_in_correct_core_rate"] / float(targets["peak_in_correct_core_target"]),
        current_eta["mean"] / max(float(eta0["mean"]) * float(targets["efficiency_reference_retention_target"]), EPS),
        current_eta["p10"] / max(float(eta0["p10"]) * float(targets["efficiency_reference_retention_target"]), EPS),
        current_core["mean"] / max(float(core0["mean"]) * 1.25, EPS),
    ]
    return float(np.mean(np.clip(ratios, 0.0, 1.0)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _initial_phase(name: str, sources: dict[str, torch.Tensor], seed: int,
                   device: torch.device) -> torch.Tensor:
    if name in sources:
        return sources[name].to(device=device, dtype=torch.float32).clone()
    if name != "RANDOM_UNIFORM_PHASE":
        raise ValueError(f"unknown C3.8 phase initialization: {name}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.rand((100, 100), generator=generator) * (2 * math.pi)).to(device)


def _train_one(config: dict[str, Any], context: dict[str, Any], run_dir: Path,
               eta_map: dict[int, float], sources: dict[str, torch.Tensor], *,
               arm: str, initialization: str, seed: int,
               train_rows: torch.Tensor, validation_rows: torch.Tensor,
               active_losses: Sequence[str], max_epochs: int,
               resume: bool = False) -> dict[str, Any]:
    _seed_everything(seed)
    device = context["device"]
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "sample_roles.npz"
    checkpoint = run_dir / "checkpoint.pt"
    summary_path = run_dir / "training_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = {
            "identity": "C38_SHARED_SINGLE_PHASE_TRAINING_RUN", "arm": arm,
            "seed": int(seed), "initialization": initialization,
            "active_losses": list(active_losses), "training_rows": train_rows.tolist(),
            "validation_rows": validation_rows.tolist(),
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise ValueError("completed C3.8 run does not match requested lineage")
        if int(summary.get("requested_max_epochs", -1)) != int(max_epochs):
            raise ValueError("completed C3.8 run has a different epoch budget")
        if not checkpoint.is_file() or not Path(summary["phase_path"]).is_file() or not rows_path.is_file():
            raise FileNotFoundError("completed C3.8 run is missing its roles, checkpoint, or selected phase")
        with np.load(rows_path, allow_pickle=False) as roles:
            if (not np.array_equal(roles["train_rows"], train_rows.numpy())
                    or not np.array_equal(roles["validation_rows"], validation_rows.numpy())):
                raise ValueError("completed C3.8 run sample roles differ from the saved run")
        return summary
    if checkpoint.is_file():
        if not resume:
            raise FileExistsError(f"incomplete C3.8 run exists; explicit --resume required: {run_dir}")
        if not rows_path.is_file():
            raise FileNotFoundError("C3.8 resume is missing saved sample roles")
        with np.load(rows_path, allow_pickle=False) as roles:
            if (not np.array_equal(roles["train_rows"], train_rows.numpy())
                    or not np.array_equal(roles["validation_rows"], validation_rows.numpy())):
                raise ValueError("C3.8 resume sample roles differ from the saved run")
    else:
        if rows_path.exists() or list(run_dir.glob("phase_epoch*.npy")):
            raise FileExistsError(f"partial C3.8 run lacks a recoverable checkpoint: {run_dir}")
        atomic_savez(rows_path, train_rows=train_rows.numpy(), validation_rows=validation_rows.numpy())
    phase = torch.nn.Parameter(_initial_phase(initialization, sources, seed, device))
    optimizer = torch.optim.Adam([phase], lr=float(config["training"]["learning_rate"]))
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_score = -1.0
    best_epoch = -1
    best_phase = None
    no_improvement = 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("arm") != arm or state.get("seed") != seed or state.get("initialization") != initialization:
            raise ValueError("explicit C3.8 resume state does not match requested run")
        if (state.get("active_losses") != list(active_losses)
                or not torch.equal(state["training_rows"].detach().cpu(), train_rows.detach().cpu())
                or not torch.equal(state["validation_rows"].detach().cpu(), validation_rows.detach().cpu())
                or float(state["learning_rate"]) != float(config["training"]["learning_rate"])
                or int(state["logical_batch_size"]) != int(config["training"]["logical_batch_size"])
                or int(state["microbatch_size"]) != int(config["training"]["microbatch_size"])):
            raise ValueError("explicit C3.8 resume configuration, losses, or sample roles differ")
        phase.data.copy_(state["phase"].to(device))
        optimizer.load_state_dict(state["optimizer"])
        history = list(state["history"])
        start_epoch = int(state["completed_epoch"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        best_phase = state["best_phase"].clone()
        no_improvement = int(state.get("no_improvement", 0))
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), device)
    else:
        best_phase = phase.detach().clone()
    baseline = json.loads((run_dir.parents[2] / "p0" / "c38_p0_replay_metrics.json").read_text(
        encoding="utf-8"))["C35_P1"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    patience = int(config["training"]["formal_early_stop_patience"])
    min_epochs = int(config["training"]["formal_min_epochs"])
    stop_reason = "MAX_EPOCHS"
    for epoch in range(start_epoch + 1, max_epochs + 1):
        epoch_started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(train_rows), generator=generator)
        shuffled = train_rows[order]
        component_totals = {name: 0.0 for name in config["training"]["loss_order"]}
        losses, gradient_norms = [], []
        updates = 0
        for offset in range(0, len(shuffled), int(config["training"]["logical_batch_size"])):
            rows = shuffled[offset:offset + int(config["training"]["logical_batch_size"])]
            refs = _eta_for_rows(rows, eta_map, device)
            record = _production_update(context, config, phase, optimizer, rows, refs,
                                        active_losses=active_losses)
            losses.append(record["loss"])
            gradient_norms.append(record["gradient_norm"])
            for name, value in record["loss_components"].items():
                component_totals[name] += value
            updates += 1
        validation = evaluate_frozen(context, config, phase.detach(), validation_rows, batch_size=4)
        score = _candidate_score(validation["metrics"], baseline, config)
        improved = score > best_score + 1e-8
        if improved:
            best_score, best_epoch = score, epoch
            best_phase = phase.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        record = {
            "epoch": epoch, "updates": updates, "train_loss_mean": float(np.mean(losses)),
            "train_loss_components_sum_over_updates": component_totals,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "validation": validation["metrics"], "validation_selection_score": score,
            "improved": improved, "sample_order_rows": shuffled.tolist(),
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        # The dictionary keyed by epoch prevents historical duplicate-append accounting.
        history = [item for item in history if int(item["epoch"]) != epoch] + [record]
        np.save(run_dir / f"phase_epoch{epoch}.npy",
                torch.remainder(phase.detach(), 2 * math.pi).cpu().numpy().astype(np.float32),
                allow_pickle=False)
        state = {
            "identity": "C38_SINGLE_PHASE_CHECKPOINT", "arm": arm, "seed": int(seed),
            "initialization": initialization, "phase": phase.detach().cpu(),
            "active_losses": list(active_losses), "training_rows": train_rows.detach().cpu(),
            "validation_rows": validation_rows.detach().cpu(),
            "learning_rate": float(config["training"]["learning_rate"]),
            "logical_batch_size": int(config["training"]["logical_batch_size"]),
            "microbatch_size": int(config["training"]["microbatch_size"]),
            "best_phase": best_phase.detach().cpu(), "best_epoch": best_epoch,
            "best_score": best_score, "optimizer": optimizer.state_dict(),
            "completed_epoch": epoch, "history": history,
            "no_improvement": no_improvement,
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device),
        }
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        readback = torch.load(temporary, map_location="cpu", weights_only=False)
        if (readback.get("identity") != "C38_SINGLE_PHASE_CHECKPOINT"
                or int(readback.get("completed_epoch", -1)) != epoch
                or not torch.equal(readback["phase"], phase.detach().cpu())
                or not torch.equal(readback["best_phase"], best_phase.detach().cpu())
                or not torch.equal(readback["training_rows"], train_rows.detach().cpu())
                or not torch.equal(readback["validation_rows"], validation_rows.detach().cpu())):
            raise RuntimeError("C3.8 epoch checkpoint failed its write/readback verification")
        temporary.replace(checkpoint)
        write_json(run_dir / "history.json", {"epochs": history})
        if epoch >= min_epochs and no_improvement >= patience:
            stop_reason = "EARLY_STOP_PATIENCE"
            break
    if best_phase is None or best_epoch < 1:
        raise RuntimeError("C3.8 training did not produce a selected phase")
    phase_out = torch.remainder(best_phase, 2 * math.pi).cpu().numpy().astype(np.float32)
    np.save(run_dir / "phase_rad.npy", phase_out, allow_pickle=False)
    best_validation = history[best_epoch - start_epoch - 1]["validation"] if start_epoch == 0 else None
    if best_validation is None:
        epoch_record = next(item for item in history if int(item["epoch"]) == best_epoch)
        best_validation = epoch_record["validation"]
    summary = {
        "identity": "C38_SHARED_SINGLE_PHASE_TRAINING_RUN", "arm": arm, "seed": int(seed),
        "initialization": initialization, "active_losses": list(active_losses),
        "loss_weights": {name: float(config["training"]["loss_weights"][name])
                          for name in active_losses},
        "phase_path": str(run_dir / "phase_rad.npy"),
        "phase_sha256": sha256_file(run_dir / "phase_rad.npy"),
        "best_epoch": int(best_epoch), "epochs_completed": len(history),
        "requested_max_epochs": int(max_epochs),
        "stop_reason": stop_reason,
        "training_updates": int(sum(int(item["updates"]) for item in history)),
        "training_rows": train_rows.tolist(), "validation_rows": validation_rows.tolist(),
        "best_validation": best_validation,
        "best_validation_selection_score": float(best_score),
        "elapsed_seconds": float(time.perf_counter() - started),
        "training_updates_per_second": float(sum(int(item["updates"]) for item in history) /
                                               max(time.perf_counter() - started, EPS)),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "history": history,
        "checkpoint_path": str(checkpoint),
    }
    write_json(run_dir / "training_summary.json", summary)
    return summary


def run_p3(config: dict[str, Any], context: dict[str, Any], run_dir: Path,
           *, resume: bool = False) -> dict[str, Any]:
    preflight_path = run_dir / "p2" / "c38_preflight.json"
    if not preflight_path.is_file() or not json.loads(preflight_path.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("C3.8 P2 preflight is missing or failed; training is not admitted")
    sources = resolve_phase_sources(config, run_dir)
    eta_map = _eta_reference_from_p1(run_dir)
    design = config["training"]["pilot_design"]
    all_train = _formal_training_rows(context, config)
    formal_val = context["split"]["validation"].detach().cpu()
    input_cache_rows = torch.cat((all_train, formal_val)).unique(sorted=True)
    input_cache_metadata = _prepare_cache(context, input_cache_rows, config)
    pilot_train = _balanced_rows(all_train, context["labels"],
                                 int(config["data"]["train_rows_per_class_pilot"]))
    pilot_val = _balanced_rows(context["split"]["validation"], context["labels"],
                               int(config["data"]["pilot_validation_rows_per_class"]))
    output_dir = run_dir / "p3"
    output_dir.mkdir(parents=True, exist_ok=True)
    pilot_specs = [("T4", source) for source in design["initialisation_sources"]]
    t4_results = []
    for _, source in pilot_specs:
        summary = _train_one(config, context, output_dir / "pilots" / f"T4_{source}", eta_map, sources,
                             arm="T4", initialization=source, seed=int(config["training"]["pilot_seed"]),
                             train_rows=pilot_train, validation_rows=pilot_val,
                             active_losses=design["ablation_arms"]["T4"],
                             max_epochs=int(config["training"]["pilot_epochs_max"]), resume=resume)
        t4_results.append(summary)
    selected_t4 = max(t4_results, key=lambda item: item["best_validation_selection_score"])
    selected_init = selected_t4["initialization"]
    arm_summaries = {"T4": selected_t4}
    for arm in ("T0", "T1", "T2", "T3"):
        arm_summaries[arm] = _train_one(
            config, context, output_dir / "pilots" / f"{arm}_{selected_init}", eta_map, sources,
            arm=arm, initialization=selected_init, seed=int(config["training"]["pilot_seed"]),
            train_rows=pilot_train, validation_rows=pilot_val,
            active_losses=design["ablation_arms"][arm],
            max_epochs=int(config["training"]["pilot_epochs_max"]), resume=resume)
    baseline = json.loads((run_dir / "p0" / "c38_p0_replay_metrics.json").read_text(
        encoding="utf-8"))["C35_P1"]
    pilot_records = []
    for arm, summary in arm_summaries.items():
        metric = summary["best_validation"]
        record = {
            "name": arm, "arm": arm, "initialization": summary["initialization"],
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
            "min_class_recall": min(metric["recall"]),
            "peak_roi": metric["peak_in_correct_roi_rate"],
            "peak_core": metric["peak_in_correct_core_rate"],
            "core_input_mean": metric["correct_core_input_fraction"]["mean"],
            "eta_mean": metric["eta_phase_to_cmos"]["mean"],
            "eta_p10": metric["eta_phase_to_cmos"]["p10"],
            "spread_mean_um": metric["spread_radius_um"]["mean"],
            "selection_score": _candidate_score(metric, baseline, config),
            "summary_path": summary["phase_path"],
        }
        pilot_records.append(record)
    front = pareto_front(pilot_records)
    front = sorted(front, key=lambda item: item["selection_score"], reverse=True)
    if len(front) < int(design["formal_shortlist_size"]):
        remaining = sorted((item for item in pilot_records if item not in front),
                           key=lambda item: item["selection_score"], reverse=True)
        front.extend(remaining[:int(design["formal_shortlist_size"]) - len(front)])
    shortlist = front[:int(design["formal_shortlist_size"])]
    init_by_seed = {11: "C35_P1", 23: "C37_E3_seed23", 37: "RANDOM_UNIFORM_PHASE"}
    formal_train = all_train
    formal = []
    for candidate in shortlist:
        arm = candidate["arm"]
        active = design["ablation_arms"][arm]
        for seed in config["training"]["seeds"]:
            seed = int(seed)
            initialization = init_by_seed[seed]
            summary = _train_one(
                config, context, output_dir / "formal" / f"{arm}_seed{seed}", eta_map, sources,
                arm=arm, initialization=initialization, seed=seed,
                train_rows=formal_train, validation_rows=formal_val,
                active_losses=active, max_epochs=int(config["training"]["formal_epochs_max"]),
                resume=resume)
            metric = summary["best_validation"]
            formal.append({
                "name": f"{arm}_seed{seed}", "arm": arm, "seed": seed,
                "initialization": initialization,
                "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
                "min_class_recall": min(metric["recall"]),
                "peak_roi": metric["peak_in_correct_roi_rate"],
                "peak_core": metric["peak_in_correct_core_rate"],
                "core_input_mean": metric["correct_core_input_fraction"]["mean"],
                "eta_mean": metric["eta_phase_to_cmos"]["mean"],
                "eta_p10": metric["eta_phase_to_cmos"]["p10"],
                "spread_mean_um": metric["spread_radius_um"]["mean"],
                "selection_score": _candidate_score(metric, baseline, config),
                "phase_path": summary["phase_path"],
                "summary_path": str(output_dir / "formal" / f"{arm}_seed{seed}" / "training_summary.json"),
            })
    payload = {
        "identity": "C38_P3_SINGLE_SHARED_PHASE_TRAINING",
        "pilot_rows_per_class": len(pilot_train) // 4,
        "pilot_validation_rows_per_class": len(pilot_val) // 4,
        "pilot_count": len(t4_results) + 4,
        "pilot_summaries": [*t4_results, *[arm_summaries[key] for key in ("T0", "T1", "T2", "T3")]],
        "pilot_records": pilot_records,
        "pilot_pareto_front": front,
        "selected_initialization_for_ablation": selected_init,
        "formal_shortlist": shortlist,
        "formal_runs": formal,
        "formal_run_count": len(formal),
        "formal_update_cap_per_run": int(config["training"]["formal_update_budget"]["maximum_updates_per_run"]),
        "maximum_total_updates": int(config["training"]["formal_update_budget"]["maximum_total_updates"]),
        "input_field_cache": input_cache_metadata,
        "actual_total_updates": int(sum(json.loads(Path(item["summary_path"]).read_text(
            encoding="utf-8"))["training_updates"] for item in formal)),
        "unique_epoch_counts": {item["name"]: json.loads(Path(item["summary_path"]).read_text(
            encoding="utf-8"))["epochs_completed"] for item in formal},
        "seeds": config["training"]["seeds"],
        "test_role": "official MNIST test remains closed; no test rows used",
    }
    write_json(run_dir / "c38_training_summary.json", payload)
    return payload


def run_p4(config: dict[str, Any], context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    training_path = run_dir / "c38_training_summary.json"
    if not training_path.is_file():
        raise FileNotFoundError("P4 requires completed C3.8 pilot and formal training")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if int(training["formal_run_count"]) != int(config["training"]["pilot_design"]["formal_shortlist_size"]) * 3:
        raise ValueError("P4 formal candidate set is incomplete")
    rows = context["split"]["validation"].detach().cpu()
    if len(rows) != 2000:
        raise ValueError("P4 must use the frozen C3.7 development validation population")
    p4_dir = run_dir / "p4"
    p4_dir.mkdir(parents=True, exist_ok=True)
    p0_metrics = json.loads((run_dir / "p0" / "c38_p0_replay_metrics.json").read_text(
        encoding="utf-8"))
    baseline = p0_metrics["C35_P1"]
    baseline_eta = baseline["eta_phase_to_cmos"]
    baseline_core = baseline["correct_core_input_fraction"]
    evaluations, records = {}, []
    best_group_means = None
    best_group_name = None
    best_group_score = -math.inf
    for run in training["formal_runs"]:
        name = run["name"]
        phase_path = Path(run["phase_path"])
        phase = c37.load_frozen_phase(phase_path)
        if phase.requires_grad:
            raise RuntimeError("frozen C3.8 phase unexpectedly requires gradients")
        evaluation = evaluate_frozen(context, config, phase, rows, batch_size=4,
                                     aggregate_group_means=True)
        _save_eval(p4_dir / f"{name}_validation_rows.npz", evaluation)
        metric = evaluation["metrics"]
        evaluations[name] = metric
        record = {
            "name": name, "arm": run["arm"], "seed": int(run["seed"]),
            "initialization": run["initialization"],
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
            "min_class_recall": min(metric["recall"]),
            "peak_roi": metric["peak_in_correct_roi_rate"],
            "peak_core": metric["peak_in_correct_core_rate"],
            "core_input_mean": metric["correct_core_input_fraction"]["mean"],
            "eta_mean": metric["eta_phase_to_cmos"]["mean"],
            "eta_p10": metric["eta_phase_to_cmos"]["p10"],
            "spread_mean_um": metric["spread_radius_um"]["mean"],
            "phase_path": str(phase_path), "phase_sha256": sha256_file(phase_path),
            "validation_rows_path": str(p4_dir / f"{name}_validation_rows.npz"),
            "metrics": metric,
        }
        record["mean_efficiency_retention_vs_C35_P1"] = \
            record["eta_mean"] / max(baseline_eta["mean"], EPS)
        record["p10_efficiency_retention_vs_C35_P1"] = \
            record["eta_p10"] / max(baseline_eta["p10"], EPS)
        record["correct_core_input_mean_gain_vs_C35_P1"] = \
            record["core_input_mean"] - baseline_core["mean"]
        record["selection_score"] = _candidate_score(metric, baseline, config)
        records.append(record)
        if record["selection_score"] > best_group_score:
            best_group_score = record["selection_score"]
            best_group_name = name
            best_group_means = evaluation["group_mean_maps"]
    front = pareto_front(records)
    selected = max(records, key=lambda item: item["selection_score"])
    if selected["name"] != best_group_name or best_group_means is None:
        raise RuntimeError("selected candidate and P4 grouped CMOS means diverged")
    group_means_path = p4_dir / "c38_selected_cmos_group_means.npz"
    atomic_savez(group_means_path, **best_group_means)
    target = config["evaluation"]
    checks = {
        "accuracy_target": bool(selected["accuracy"] >= target["classification_accuracy_target"]),
        "macro_f1_target": bool(selected["macro_f1"] >= target["macro_f1_target"]),
        "peak_correct_roi_target": bool(selected["peak_roi"] >= target["peak_in_correct_roi_target"]),
        "peak_correct_core_target": bool(selected["peak_core"] >= target["peak_in_correct_core_target"]),
        "mean_eta_reference_retention": bool(selected["mean_efficiency_retention_vs_C35_P1"] >=
                                               target["efficiency_reference_retention_target"]),
        "p10_eta_reference_retention": bool(selected["p10_efficiency_retention_vs_C35_P1"] >=
                                             target["efficiency_reference_retention_target"]),
        "correct_core_energy_improved": bool(selected["correct_core_input_mean_gain_vs_C35_P1"] > 0),
    }
    passed = all(checks.values())
    payload = {
        "identity": "C38_P4_FROZEN_PHASE_OPTICAL_INFERENCE",
        "population": {"role": "reused development validation; not independent",
                       "rows": int(len(rows)), "row_indices": rows.tolist(),
                       "labels": context["labels"][rows].tolist(),
                       "class_counts": {str(d): int((context["labels"][rows] == d).sum())
                                         for d in range(4)}},
        "decision": "argmax of four fixed ROI integrals; no learned digital classifier or per-image normalization",
        "frozen_phase_candidates": records,
        "pareto_front": [item["name"] for item in front],
        "selected_for_visualization": selected["name"],
        "group_mean_maps_candidate": best_group_name,
        "group_mean_maps_path": str(group_means_path.relative_to(run_dir)),
        "group_mean_maps_source": "selected candidate's same full validation forward pass used for P4 metrics; no extra optical replay",
        "group_mean_class_counts": best_group_means["all_class_counts"].tolist(),
        "group_mean_correct_counts": best_group_means["correct_class_counts"].tolist(),
        "selected_metrics": evaluations[selected["name"]],
        "checks": checks,
        "all_research_targets_pass": bool(passed),
        "independent_evaluation": "INDEPENDENT_EVAL_BLOCKED; official MNIST test remains closed due historical exposure",
        "hardware_mapping": "HW207_CURRENT_MAPPING=BLOCKED",
        "classification_scope": "numerical fixed Model-B candidate only; not hardware-validated optical performance",
    }
    write_json(run_dir / "c38_evaluation.json", payload)
    return payload


def _save_figure(fig: Any, output_stem: Path, dpi: int) -> list[dict[str, Any]]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
        outputs.append({"path": str(path), "size_bytes": path.stat().st_size})
    return outputs


def _physical_box(box: Sequence[int]) -> tuple[float, float, float, float]:
    r0, r1, c0, c1 = [int(x) for x in box]
    center = c37.FRAME_CENTER_PX
    pitch = c37.CMOS_PITCH_UM
    return ((c0 - center - 0.5) * pitch, (r0 - center - 0.5) * pitch,
            (c1 - c0) * pitch, (r1 - r0) * pitch)


def _draw_cmos(ax: Any, image: np.ndarray, label: int, prediction: int,
               peak_row: float, peak_col: float, centroid_row: float, centroid_col: float,
               vmax: float, *, title: str = "") -> Any:
    import matplotlib.patches as patches

    extent_um = c37.CMOS * c37.CMOS_PITCH_UM / 2.0
    artist = ax.imshow(image, origin="lower", extent=(-extent_um, extent_um, -extent_um, extent_um),
                       cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    for digit, box in enumerate(c37.roi_bounds()):
        x, y, width, height = _physical_box(box)
        ax.add_patch(patches.Rectangle((x, y), width, height, fill=False,
                                       edgecolor="#31d158" if digit == label else "#e7edf4",
                                       linewidth=1.5 if digit == label else 0.7, alpha=0.95))
        if digit == label:
            cx, cy, cw, ch = _physical_box(c37.core_bounds(0.4)[digit])
            ax.add_patch(patches.Rectangle((cx, cy), cw, ch, fill=False,
                                           edgecolor="#31d158", linewidth=1.2,
                                           linestyle="--", alpha=0.95))
    pitch = c37.CMOS_PITCH_UM
    center = c37.FRAME_CENTER_PX
    ax.scatter([(peak_col - center) * pitch], [(peak_row - center) * pitch],
               marker="x", s=45, c="#00e5ff", linewidths=1.6, label="peak")
    ax.scatter([(centroid_col - center) * pitch], [(centroid_row - center) * pitch],
               marker="+", s=48, c="#ff44d5", linewidths=1.7, label="centroid")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_xlim(-extent_um, extent_um)
    ax.set_ylim(-extent_um, extent_um)
    ax.set_aspect("equal")
    return artist


def _map_peak_centroid(image: np.ndarray) -> tuple[float, float, float, float] | None:
    values = np.asarray(image, dtype=np.float64)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return None
    peak_row, peak_col = np.unravel_index(int(np.argmax(values)), values.shape)
    center = c37.FRAME_CENTER_PX
    axis = np.arange(values.shape[0], dtype=np.float64) - center
    centroid_row = float(np.dot(values.sum(axis=1), axis) / total + center)
    centroid_col = float(np.dot(values.sum(axis=0), axis) / total + center)
    return float(peak_row), float(peak_col), centroid_row, centroid_col


def _draw_cmos_summary(ax: Any, image: np.ndarray, vmax: float, *, title: str,
                       target_label: int | None,
                       core_fraction: float) -> Any:
    import matplotlib.patches as patches

    extent_um = c37.CMOS * c37.CMOS_PITCH_UM / 2.0
    artist = ax.imshow(image, origin="lower", extent=(-extent_um, extent_um, -extent_um, extent_um),
                       cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    for digit, box in enumerate(c37.roi_bounds()):
        x, y, width, height = _physical_box(box)
        is_target = digit == target_label
        ax.add_patch(patches.Rectangle(
            (x, y), width, height, fill=False,
            edgecolor="#31d158" if is_target else "#e7edf4",
            linewidth=1.5 if is_target else 0.7, alpha=0.95,
        ))
        core_box = c37.core_bounds(core_fraction)[digit]
        cx, cy, cw, ch = _physical_box(core_box)
        ax.add_patch(patches.Rectangle(
            (cx, cy), cw, ch, fill=False,
            edgecolor="#31d158" if is_target else "#b8c88a",
            linewidth=1.2 if is_target else 0.65, linestyle="--", alpha=0.95,
        ))
    peak_centroid = _map_peak_centroid(image)
    if peak_centroid is not None:
        peak_row, peak_col, centroid_row, centroid_col = peak_centroid
        pitch = c37.CMOS_PITCH_UM
        center = c37.FRAME_CENTER_PX
        ax.scatter([(peak_col - center) * pitch], [(peak_row - center) * pitch],
                   marker="x", s=38, c="#00e5ff", linewidths=1.5, label="peak")
        ax.scatter([(centroid_col - center) * pitch], [(centroid_row - center) * pitch],
                   marker="+", s=42, c="#ff44d5", linewidths=1.6, label="centroid")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_xlim(-extent_um, extent_um)
    ax.set_ylim(-extent_um, extent_um)
    ax.set_aspect("equal")
    return artist


def _sample_selection(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rows, labels, predictions, scores = (arrays[key] for key in
                                         ("rows", "labels", "predictions", "roi_scores"))
    margin = scores[np.arange(len(labels)), labels] - np.max(
        np.where(np.eye(4, dtype=bool)[labels], -np.inf, scores), axis=1)
    selected: dict[str, Any] = {"typical_correct": {}, "difficult_correct": {}, "representative_errors": {}}
    for digit in range(4):
        class_idx = np.flatnonzero(labels == digit)
        correct = class_idx[predictions[class_idx] == digit]
        incorrect = class_idx[predictions[class_idx] != digit]
        ordered = correct[np.argsort(margin[correct])]
        if len(ordered):
            quantile_positions = sorted(set(int(round(q * (len(ordered) - 1)))
                                            for q in (0.25, 0.5, 0.75)))
            selected["typical_correct"][str(digit)] = [int(rows[ordered[pos]])
                                                        for pos in quantile_positions]
            selected["difficult_correct"][str(digit)] = [int(rows[index]) for index in ordered[:3]]
        else:
            selected["typical_correct"][str(digit)] = []
            selected["difficult_correct"][str(digit)] = []
        if len(incorrect):
            destinations, counts = np.unique(predictions[incorrect], return_counts=True)
            destination = int(destinations[np.argmax(counts)])
            representative = incorrect[predictions[incorrect] == destination]
            selected["representative_errors"][str(digit)] = [int(rows[index])
                                                              for index in representative[:3]]
            selected.setdefault("most_common_error_destination", {})[str(digit)] = destination
        else:
            selected["representative_errors"][str(digit)] = []
    selected["rules"] = "per class: correct-margin q25/q50/q75, three lowest-margin correct, and first three errors to most frequent wrong destination"
    return selected


def run_p5(config: dict[str, Any], context: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation_path = run_dir / "c38_evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("P5 needs the frozen P4 evaluation and selected candidate")
    evaluation_record = json.loads(evaluation_path.read_text(encoding="utf-8"))
    selected_name = evaluation_record["selected_for_visualization"]
    selected = next(item for item in evaluation_record["frozen_phase_candidates"]
                    if item["name"] == selected_name)
    phase_path = Path(selected["phase_path"])
    phase = c37.load_frozen_phase(phase_path)
    with np.load(selected["validation_rows_path"], allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    group_path = Path(evaluation_record["group_mean_maps_path"])
    if not group_path.is_absolute():
        group_path = run_dir / group_path
    if not group_path.is_file():
        raise FileNotFoundError("P5 requires class and population mean maps from the P4 forward pass")
    with np.load(group_path, allow_pickle=False) as saved:
        group_maps = {name: saved[name] for name in saved.files}
    if evaluation_record.get("group_mean_maps_candidate") != selected_name:
        raise ValueError("P4 grouped CMOS maps are not bound to the selected frozen phase")
    if not np.array_equal(group_maps["rows"], arrays["rows"]):
        raise ValueError("P4 grouped CMOS maps use a different validation row order")
    if not np.array_equal(group_maps["labels"], arrays["labels"]):
        raise ValueError("P4 grouped CMOS maps use different validation labels")
    if not np.array_equal(group_maps["predictions"], arrays["predictions"]):
        raise ValueError("P4 grouped CMOS maps use different frozen ROI predictions")
    if not np.array_equal(group_maps["all_class_counts"], np.bincount(arrays["labels"], minlength=4)):
        raise ValueError("P4 grouped CMOS means do not cover the full selected population")
    if (group_maps["all_class_means"].shape != (4, c37.CMOS, c37.CMOS)
            or group_maps["correct_class_means"].shape != (4, c37.CMOS, c37.CMOS)
            or group_maps["overall_mean"].shape != (c37.CMOS, c37.CMOS)):
        raise ValueError("P4 grouped CMOS mean maps have an unexpected shape")
    selection = _sample_selection(arrays)
    main_rows = [selection["typical_correct"][str(d)][1]
                 if len(selection["typical_correct"][str(d)]) > 1
                 else (selection["typical_correct"][str(d)][0]
                       if selection["typical_correct"][str(d)] else
                       int(arrays["rows"][np.flatnonzero(arrays["labels"] == d)[0]]))
                 for d in range(4)]
    figure_rows = []
    for digit in range(4):
        figure_rows.extend(selection["typical_correct"][str(digit)])
        figure_rows.extend(selection["difficult_correct"][str(digit)])
        figure_rows.extend(selection["representative_errors"][str(digit)])
    figure_rows = list(dict.fromkeys(main_rows + figure_rows))
    maps_evaluation = evaluate_frozen(context, config, phase, torch.tensor(figure_rows),
                                      batch_size=1, save_map_rows=figure_rows)
    maps = maps_evaluation["maps"]
    if set(maps) != {str(row) for row in figure_rows}:
        raise RuntimeError("C3.8 P5 could not retrieve all deterministic selected CMOS maps")
    replay_arrays = maps_evaluation["arrays"]
    p4_position = {int(row): idx for idx, row in enumerate(arrays["rows"])}
    replay_indices = np.asarray([p4_position[int(row)] for row in figure_rows], dtype=np.int64)
    p4_expected_scores = arrays["roi_scores"][replay_indices]
    replayed_scores = replay_arrays["roi_scores"]
    score_relative_error = np.abs(replayed_scores - p4_expected_scores) / np.maximum(
        np.abs(p4_expected_scores), EPS)
    p4_expected_cmos_energy = arrays["cmos_energy_um2"][replay_indices]
    replay_cmos_energy = replay_arrays["cmos_energy_um2"]
    cmos_energy_relative_error = np.abs(replay_cmos_energy - p4_expected_cmos_energy) / np.maximum(
        np.abs(p4_expected_cmos_energy), EPS)
    p4_replay_matches = {
        "rows_match": bool(np.array_equal(replay_arrays["rows"], np.asarray(figure_rows))),
        "labels_match": bool(np.array_equal(replay_arrays["labels"], arrays["labels"][replay_indices])),
        "roi_predictions_match": bool(np.array_equal(
            replay_arrays["predictions"], arrays["predictions"][replay_indices])),
        "roi_scores_close": bool(np.allclose(replayed_scores, p4_expected_scores,
                                              rtol=1e-4, atol=1e-8)),
        "cmos_energy_close": bool(np.allclose(replay_cmos_energy, p4_expected_cmos_energy,
                                               rtol=1e-4, atol=1e-8)),
        "global_peak_positions_match": bool(
            np.array_equal(replay_arrays["global_peak_row"],
                           arrays["global_peak_row"][replay_indices])
            and np.array_equal(replay_arrays["global_peak_col"],
                               arrays["global_peak_col"][replay_indices])),
        "centroids_close": bool(
            np.allclose(replay_arrays["centroid_row"], arrays["centroid_row"][replay_indices],
                        rtol=1e-5, atol=1e-3)
            and np.allclose(replay_arrays["centroid_col"], arrays["centroid_col"][replay_indices],
                            rtol=1e-5, atol=1e-3)),
        "max_relative_roi_score_error": float(score_relative_error.max(initial=0.0)),
        "max_relative_cmos_energy_error": float(cmos_energy_relative_error.max(initial=0.0)),
    }
    all_display_maps = [*maps.values(), *group_maps["all_class_means"],
                        *group_maps["correct_class_means"], group_maps["overall_mean"]]
    vmax = max(float(np.max(image)) for image in all_display_maps)
    if not math.isfinite(vmax) or vmax <= 0:
        raise RuntimeError("C3.8 P5 CMOS maps have no positive linear intensity")
    if not all(np.isfinite(image).all() and np.min(image) >= 0 for image in all_display_maps):
        raise RuntimeError("C3.8 P5 CMOS sample or grouped mean map is invalid")
    out = run_dir / "p5"
    out.mkdir(parents=True, exist_ok=True)
    phase_np = np.remainder(phase.cpu().numpy(), 2 * math.pi).astype(np.float32)
    data_rows = context["images"]
    labels = context["labels"].cpu().numpy()
    p4_arrays = arrays
    position = {int(row): idx for idx, row in enumerate(p4_arrays["rows"])}
    replay_position = {int(row): idx for idx, row in enumerate(replay_arrays["rows"])}

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(4, 4, figsize=(14.4, 12.0), constrained_layout=True)
    titles = ["Binary DMD command", "Shared phase screen φ (rad)",
              "CMOS intensity (linear, shared scale)", "Four fixed ROI integrals"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11, weight="bold")
    image_artist = None
    for digit, row in enumerate(main_rows):
        idx = position[row]
        label = int(labels[row])
        prediction = int(p4_arrays["predictions"][idx])
        axes[digit, 0].imshow((data_rows[row].cpu().numpy() >= 128).astype(np.uint8),
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[digit, 0].set_title(f"row {row}; label {label}", fontsize=8)
        axes[digit, 0].axis("off")
        axes[digit, 1].imshow(phase_np, cmap="twilight", vmin=0, vmax=2 * math.pi,
                              extent=(-100, 100, -100, 100), origin="lower")
        axes[digit, 1].set_title("Same frozen phase", fontsize=8)
        axes[digit, 1].set_xlabel("x (µm)")
        axes[digit, 1].set_ylabel("y (µm)")
        image_artist = _draw_cmos(
            axes[digit, 2], maps[str(row)], label, prediction,
            replay_arrays["global_peak_row"][replay_position[row]],
            replay_arrays["global_peak_col"][replay_position[row]],
            replay_arrays["centroid_row"][replay_position[row]],
            replay_arrays["centroid_col"][replay_position[row]], vmax,
            title=f"row {row}: true {label}, pred {prediction}")
        if digit == 0:
            axes[digit, 2].legend(loc="upper right", fontsize=6, frameon=False)
        vals = p4_arrays["roi_scores"][idx]
        colors = ["#31a354" if i == label else "#8a96a3" for i in range(4)]
        bars = axes[digit, 3].bar(np.arange(4), vals, color=colors)
        for bar, value in zip(bars, vals):
            axes[digit, 3].text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height(), f"{value:.2g}", ha="center", va="bottom", fontsize=7)
        axes[digit, 3].set_xticks(range(4), ["TL 0", "TR 1", "BL 2", "BR 3"], rotation=15)
        axes[digit, 3].set_ylabel("∫I dA (a.u. µm²)")
        axes[digit, 3].set_title(f"true {label}; ROI argmax {prediction}", fontsize=8)
    fig.colorbar(image_artist, ax=axes[:, 2], shrink=0.68, label="Model intensity (a.u.; linear)")
    main_files = _save_figure(fig, out / "c38_main_4x4_cmos_routing", int(config["visualization"]["png_dpi"]))
    plt.close(fig)
    atomic_savez(out / "c38_selected_cmos_maps_raw_float32.npz",
                 rows=np.asarray(figure_rows, dtype=np.int64),
                 intensity=np.stack([maps[str(row)] for row in figure_rows]).astype(np.float32),
                 main_rows=np.asarray(main_rows, dtype=np.int64),
                 shared_linear_vmax=np.asarray(vmax, dtype=np.float64))

    # Compact atlas still includes typical, difficult and representative failed classes.
    atlas_columns = [
        ("typical_correct", slot) for slot in range(3)
    ] + [("difficult_correct", slot) for slot in range(3)] + [
        ("representative_errors", slot) for slot in range(3)]
    fig2, axes2 = plt.subplots(4, len(atlas_columns), figsize=(18, 8), constrained_layout=True)
    for digit in range(4):
        for col, (role, slot) in enumerate(atlas_columns):
            selected_rows = selection[role][str(digit)]
            ax = axes2[digit, col]
            if slot >= len(selected_rows):
                ax.axis("off")
                continue
            row = selected_rows[slot]
            idx = position[row]
            label = int(labels[row])
            prediction = int(p4_arrays["predictions"][idx])
            _draw_cmos(ax, maps[str(row)], label, prediction,
                       replay_arrays["global_peak_row"][replay_position[row]],
                       replay_arrays["global_peak_col"][replay_position[row]],
                       replay_arrays["centroid_row"][replay_position[row]],
                       replay_arrays["centroid_col"][replay_position[row]], vmax,
                       title=f"row {row}; {label}→{prediction}")
            ax.set_xticks([])
            ax.set_yticks([])
        axes2[digit, 0].set_ylabel(f"digit {digit}\nCMOS linear")
    role_headers = ["typical q25", "typical q50", "typical q75", "hard 1", "hard 2", "hard 3",
                    "error 1", "error 2", "error 3"]
    for ax, title in zip(axes2[0], role_headers):
        ax.set_xlabel(title, fontsize=7)
    atlas_files = _save_figure(fig2, out / "c38_cmos_examples_and_errors", int(config["visualization"]["png_dpi"]))
    plt.close(fig2)

    # Grouped mean maps come from the same selected-candidate P4 validation pass.
    fig_mean, axes_mean = plt.subplots(2, 4, figsize=(14.0, 7.2), constrained_layout=True)
    mean_artist = None
    mean_groups = (
        ("all_class_means", "all validation", group_maps["all_class_counts"]),
        ("correct_class_means", "correct only", group_maps["correct_class_counts"]),
    )
    for row_idx, (map_key, role, counts) in enumerate(mean_groups):
        for digit in range(4):
            mean_artist = _draw_cmos_summary(
                axes_mean[row_idx, digit], group_maps[map_key][digit], vmax,
                title=f"digit {digit} — {role} (n={int(counts[digit])})",
                target_label=digit,
                core_fraction=float(config["core"]["fraction_of_roi_side"]),
            )
    fig_mean.colorbar(mean_artist, ax=axes_mean.ravel().tolist(), shrink=0.78,
                      label="Model intensity (a.u.; shared absolute linear scale)")
    class_mean_files = _save_figure(
        fig_mean, out / "c38_class_mean_cmos_all_vs_correct",
        int(config["visualization"]["png_dpi"]))
    plt.close(fig_mean)

    fig_overall, ax_overall = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    overall_artist = _draw_cmos_summary(
        ax_overall, group_maps["overall_mean"], vmax,
        title=f"Overall mean CMOS intensity — n={len(arrays['rows'])}",
        target_label=None,
        core_fraction=float(config["core"]["fraction_of_roi_side"]),
    )
    fig_overall.colorbar(overall_artist, ax=ax_overall, shrink=0.78,
                         label="Model intensity (a.u.; shared absolute linear scale)")
    overall_mean_files = _save_figure(
        fig_overall, out / "c38_overall_mean_cmos",
        int(config["visualization"]["png_dpi"]))
    plt.close(fig_overall)

    roi_scores = np.asarray(arrays["roi_scores"], dtype=np.float64)
    class_labels = np.asarray(arrays["labels"], dtype=np.int64)
    roi_means = np.stack([roi_scores[class_labels == digit].mean(axis=0) for digit in range(4)])
    roi_stds = np.stack([roi_scores[class_labels == digit].std(axis=0, ddof=1)
                         for digit in range(4)])
    fig_roi, ax_roi = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    roi_x = np.arange(4, dtype=np.float64)
    roi_width = 0.18
    roi_labels = ["TL 0", "TR 1", "BL 2", "BR 3"]
    roi_colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]
    for roi in range(4):
        offset = (roi - 1.5) * roi_width
        ax_roi.bar(roi_x + offset, roi_means[:, roi], width=roi_width,
                   yerr=roi_stds[:, roi], capsize=2, label=roi_labels[roi],
                   color=roi_colors[roi], alpha=0.9)
    ax_roi.set_xticks(roi_x, [f"true {digit}" for digit in range(4)])
    ax_roi.set_ylabel("Mean ROI integral ± SD (model a.u. µm²)")
    ax_roi.set_title("Frozen P4 ROI energy by true class (no per-image normalization)")
    ax_roi.legend(ncol=4, fontsize=7, frameon=False)
    roi_energy_files = _save_figure(
        fig_roi, out / "c38_roi_energy_by_true_class",
        int(config["visualization"]["png_dpi"]))
    plt.close(fig_roi)

    peak_roi_rates = np.asarray([
        arrays["peak_in_correct_roi"][class_labels == digit].mean() for digit in range(4)
    ])
    peak_core_rates = np.asarray([
        arrays["peak_in_correct_core"][class_labels == digit].mean() for digit in range(4)
    ])
    fig_peak, ax_peak = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    peak_width = 0.34
    ax_peak.bar(roi_x - peak_width / 2, peak_roi_rates, peak_width,
                label="global peak in correct ROI", color="#31a354")
    ax_peak.bar(roi_x + peak_width / 2, peak_core_rates, peak_width,
                label="global peak in correct core", color="#756bb1")
    ax_peak.set_xticks(roi_x, [f"digit {digit}" for digit in range(4)])
    ax_peak.set_ylim(0.0, 1.0)
    ax_peak.set_ylabel("Fraction of validation samples")
    ax_peak.set_title("Peak localization under the fixed ROI/core geometry")
    ax_peak.legend(fontsize=7, frameon=False)
    peak_localization_files = _save_figure(
        fig_peak, out / "c38_peak_localization_by_class",
        int(config["visualization"]["png_dpi"]))
    plt.close(fig_peak)

    eta_by_class = [arrays["eta_phase_to_cmos"][class_labels == digit] for digit in range(4)]
    core_input_by_class = [arrays["correct_core_input_fraction"][class_labels == digit]
                           for digit in range(4)]
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    axes_dist[0].boxplot(eta_by_class, labels=[str(d) for d in range(4)], showfliers=False)
    axes_dist[0].set_title("CMOS efficiency η")
    axes_dist[0].set_ylabel("E_CMOS / E_input")
    axes_dist[1].boxplot(core_input_by_class, labels=[str(d) for d in range(4)], showfliers=False)
    axes_dist[1].set_title("Correct core energy / input energy")
    axes_dist[1].set_ylabel("E_core,y / E_input")
    for ax in axes_dist:
        ax.set_xlabel("True MNIST class")
    distribution_files = _save_figure(
        fig_dist, out / "c38_efficiency_core_input_distributions",
        int(config["visualization"]["png_dpi"]))
    plt.close(fig_dist)

    cm = np.asarray(evaluation_record["selected_metrics"]["confusion_matrix"], dtype=np.int64)
    fig3, ax3 = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax3.imshow(cm, cmap="Blues")
    for r in range(4):
        for c in range(4):
            ax3.text(c, r, str(cm[r, c]), ha="center", va="center",
                     color="white" if cm[r, c] > cm.max() * 0.55 else "black")
    ax3.set(xticks=range(4), yticks=range(4), xlabel="ROI argmax", ylabel="True MNIST digit",
            title=f"Frozen optical ROI decision — {selected_name}")
    confusion_files = _save_figure(fig3, out / "c38_confusion_matrix", int(config["visualization"]["png_dpi"]))
    plt.close(fig3)

    front = evaluation_record["pareto_front"]
    all_candidates = evaluation_record["frozen_phase_candidates"]
    fig4, ax4 = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    for candidate in all_candidates:
        ax4.scatter(candidate["eta_mean"], candidate["accuracy"],
                    s=50 + 800 * candidate["core_input_mean"],
                    facecolors="#2b8cbe" if candidate["name"] in front else "none",
                    edgecolors="#045a8d", alpha=0.85)
        ax4.annotate(candidate["name"], (candidate["eta_mean"], candidate["accuracy"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax4.axhline(config["evaluation"]["classification_accuracy_target"],
                color="#b2182b", linestyle="--", linewidth=0.8)
    ax4.set(xlabel="Mean full-CMOS efficiency (model a.u.)", ylabel="Accuracy",
            title="Formal candidates: accuracy–efficiency; marker size ∝ core/input")
    pareto_files = _save_figure(fig4, out / "c38_formal_candidate_tradeoff", int(config["visualization"]["png_dpi"]))
    plt.close(fig4)

    all_files = (main_files + atlas_files + class_mean_files + overall_mean_files
                 + roi_energy_files + peak_localization_files + distribution_files
                 + confusion_files + pareto_files)
    p4_replay_ok = all(p4_replay_matches[key] for key in (
        "rows_match", "labels_match", "roi_predictions_match", "roi_scores_close",
        "cmos_energy_close"))
    group_means_ok = bool(
        np.array_equal(group_maps["all_class_counts"], np.full(4, 500))
        and np.all(group_maps["correct_class_counts"] > 0)
        and np.isfinite(group_maps["all_class_means"]).all()
        and np.isfinite(group_maps["correct_class_means"]).all()
        and np.isfinite(group_maps["overall_mean"]).all()
        and np.min(group_maps["all_class_means"]) >= 0
        and np.min(group_maps["correct_class_means"]) >= 0
        and np.min(group_maps["overall_mean"]) >= 0
    )
    manifest = {
        "identity": "C38_P5_NUMERICAL_CMOS_VISUALIZATION",
        "selected_candidate": selected_name,
        "phase_sha256": selected["phase_sha256"],
        "sample_selection": selection,
        "main_rows": main_rows,
        "selected_population_role": "development validation reused for model selection, not independent",
        "raw_maps_saved": str(out / "c38_selected_cmos_maps_raw_float32.npz"),
        "raw_map_shape": [len(figure_rows), c37.CMOS, c37.CMOS],
        "p4_group_mean_maps_source": str(group_path),
        "p4_group_mean_maps_candidate": evaluation_record["group_mean_maps_candidate"],
        "p4_group_mean_counts": {
            "all_by_class": group_maps["all_class_counts"].tolist(),
            "correct_by_class": group_maps["correct_class_counts"].tolist(),
        },
        "p4_selected_row_replay": p4_replay_matches,
        "figure_groups": {
            "main_routing": [str(item["path"]) for item in main_files],
            "examples_and_errors": [str(item["path"]) for item in atlas_files],
            "class_means_all_vs_correct": [str(item["path"]) for item in class_mean_files],
            "overall_mean_cmos": [str(item["path"]) for item in overall_mean_files],
            "roi_energy_by_true_class": [str(item["path"]) for item in roi_energy_files],
            "peak_localization_by_class": [str(item["path"]) for item in peak_localization_files],
            "efficiency_and_core_input_distributions": [str(item["path"]) for item in distribution_files],
            "confusion_matrix": [str(item["path"]) for item in confusion_files],
            "candidate_tradeoff": [str(item["path"]) for item in pareto_files],
        },
        "cmos_display": {"vmin": 0.0, "vmax": vmax, "scale": "shared absolute linear intensity; no per-image normalization",
                         "intensity_units": "model intensity a.u., not calibrated camera counts"},
        "caption_scope": "532 nm fixed Model-B numerical simulation; 2,000 reused development-validation rows; ROI/core coordinates fixed before training; averages are unnormalized linear intensity in model units",
        "coordinate_mapping": "row=y, column=x; origin lower; CMOS axes centered at optical axis",
        "files": all_files,
        "programmatic_qa": {
            "all_formats_present": all(Path(item["path"]).is_file() and item["size_bytes"] > 0
                                        for item in all_files),
            "single_shared_cmos_scale": True,
            "all_sample_maps_positive_and_finite": all(
                np.isfinite(image).all() and image.min() >= 0 for image in maps.values()),
            "all_group_mean_maps_positive_and_finite": group_means_ok,
            "p4_group_means_cover_all_validation_rows": group_means_ok,
            "p4_selected_rows_replay_saved_roi_statistics": p4_replay_ok,
            "p4_replay_peak_positions_match": p4_replay_matches["global_peak_positions_match"],
            "p4_replay_centroids_close": p4_replay_matches["centroids_close"],
            "same_phase_repeated_in_main_rows": True,
            "roi_bounds_frozen": c37.roi_bounds(),
            "core_bounds_frozen": c37.core_bounds(float(config["core"]["fraction_of_roi_side"])),
        },
    }
    write_json(out / "c38_visualization_manifest.json", manifest)
    return manifest


def run_completion_audit(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    paths = {
        "p0": run_dir / "p0" / "c38_p0_history_audit.json",
        "p1": run_dir / "p1" / "c38_p1_numerical_validation.json",
        "p2": run_dir / "p2" / "c38_preflight.json",
        "p3": run_dir / "c38_training_summary.json",
        "p4": run_dir / "c38_evaluation.json",
        "p5": run_dir / "p5" / "c38_visualization_manifest.json",
    }
    available = {name: path.is_file() for name, path in paths.items()}
    records = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in paths.items() if path.is_file()}
    p0_ok = bool(records.get("p0", {}).get("all_historical_prediction_replays_match", False))
    p1_ok = bool(records.get("p1", {}).get("numerical_core_pass", False))
    p2_ok = bool(records.get("p2", {}).get("passed", False))
    p3_record = records.get("p3", {})
    formal_runs = p3_record.get("formal_runs", [])
    formal_run_summaries = []
    for run in formal_runs:
        summary_path = Path(str(run.get("summary_path", "")))
        if not summary_path.is_absolute():
            project_path = ROOT / summary_path
            run_path = run_dir / summary_path
            summary_path = project_path if project_path.is_file() else run_path
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        formal_run_summaries.append({
            "name": str(run.get("name", "")),
            "summary_path": str(summary_path),
            "summary_available": summary_path.is_file(),
            "active_losses": list(summary.get("active_losses", [])),
            "epochs_completed": int(summary.get("epochs_completed", 0)),
            "training_updates": int(summary.get("training_updates", 0)),
        })
    p3_ok = bool(p3_record.get("formal_run_count", 0) == 6
                 and len(formal_runs) == 6
                 and all(item["summary_available"] and item["epochs_completed"] > 0
                         and item["training_updates"] > 0 for item in formal_run_summaries))
    formal_losses = {loss for item in formal_run_summaries for loss in item["active_losses"]}
    full_joint_loss_in_formal_training = set(config["training"]["loss_order"]).issubset(formal_losses)
    p4_record = records.get("p4", {})
    p4_group_path = run_dir / str(p4_record.get("group_mean_maps_path", ""))
    p4_ok = bool(p4_record.get("selected_for_visualization")
                 and p4_record.get("group_mean_maps_candidate")
                 == p4_record.get("selected_for_visualization")
                 and p4_group_path.is_file())
    qa = records.get("p5", {}).get("programmatic_qa", {})
    p5_ok = bool(records.get("p5") and qa.get("all_formats_present")
                  and qa.get("single_shared_cmos_scale")
                  and qa.get("all_sample_maps_positive_and_finite")
                  and qa.get("all_group_mean_maps_positive_and_finite")
                  and qa.get("p4_group_means_cover_all_validation_rows")
                  and qa.get("p4_selected_rows_replay_saved_roi_statistics"))
    experiment_valid = all((p0_ok, p1_ok, p2_ok, p3_ok, p4_ok, p5_ok))
    p4_target_pass = bool(records.get("p4", {}).get("all_research_targets_pass", False))
    target_pass = bool(p4_target_pass and full_joint_loss_in_formal_training)
    payload = {
        "identity": "HW207_C38_COMPLETION_AUDIT",
        "status": "C3.8_PASS" if experiment_valid and target_pass else
                  "C3.8_PARTIAL_PASS" if experiment_valid else "C3.8_BLOCKED",
        "stage_artifacts_available": available,
        "stage_checks": {"P0_historical_replay": p0_ok, "P1_numerical_core": p1_ok,
                         "P2_preflight": p2_ok, "P3_formal_runs": p3_ok,
                         "P3_full_joint_loss_in_formal_training": full_joint_loss_in_formal_training,
                         "P4_frozen_inference": p4_ok, "P5_visualization": p5_ok},
        "formal_loss_activation_audit": {
            "required_losses": list(config["training"]["loss_order"]),
            "activated_losses": sorted(formal_losses),
            "runs": formal_run_summaries,
            "all_required_losses_reached_formal_training": full_joint_loss_in_formal_training,
        },
        "P4_research_targets_pass": p4_target_pass,
        "research_targets_all_pass": target_pass,
        "scientific_boundary": "all outcomes are numerical Model-B diagnostics; no hardware calibration or independent evaluation",
        "independent_evaluation": config["data"]["independent_evaluation_status"],
        "hardware_mapping": "HW207_CURRENT_MAPPING=BLOCKED",
        "fabrication": "not evaluated; no STL or printer/material claim",
        "next_action": "NONE_AUTOMATIC; successor work requires a separate decision",
    }
    write_json(run_dir / "c38_completion_audit.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["p0", "p1", "p2", "p3", "p4", "p5", "all", "audit"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing incomplete epoch checkpoint explicitly")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_dir = args.run_dir or (ROOT / config["run"]["version_directory"] / "runs" /
                               config["run"]["attempt"])
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_config = run_dir / "c38_config.json"
    if not copied_config.exists():
        copied_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.stage == "audit":
        result = run_completion_audit(config, run_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.device != "cuda":
        raise ValueError("C3.8 production execution is CUDA-only")
    context = _context(config, torch.device(args.device))
    try:
        if args.stage in ("p0", "all"):
            run_p0(config, context, run_dir)
        if args.stage in ("p1", "all"):
            run_p1(config, context, run_dir)
        if args.stage in ("p2", "all"):
            preflight = run_p2(config, context, run_dir)
            if not preflight["passed"]:
                print(json.dumps({"stage": "P2", "passed": False}, indent=2))
                return 2
        if args.stage in ("p3", "all"):
            run_p3(config, context, run_dir, resume=args.resume)
        if args.stage in ("p4", "all"):
            run_p4(config, context, run_dir)
        if args.stage in ("p5", "all"):
            run_p5(config, context, run_dir)
        result = run_completion_audit(config, run_dir) if args.stage == "all" else {"stage": args.stage}
    except Exception as exc:
        failure = {"identity": "C38_STAGE_FAILURE", "stage": args.stage,
                   "exception_type": type(exc).__name__, "message": str(exc),
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(run_dir / "c38_stage_failure.json", failure)
        raise
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
