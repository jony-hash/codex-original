"""Reproduce frozen C3.5 CMOS maps and publish diagnostic-only figures."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import torch

from scripts.research.lightgen import execute_hw207_mnist4_c34_v1 as c34_runner
from scripts.research.lightgen import execute_hw207_mnist4_c35_v1 as c35_runner
from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import hw207_mnist4_roi_v1 as roi_readout


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "outputs/20261105_hw207_mnist4_c35_v1/runs/autodl_backup_attempt_03"
DEFAULT_P1_DIR = SOURCE_ROOT / "p1_energy_constrained/target_0.400000_seed_11"
DEFAULT_P2_TRAIN_DIR = SOURCE_ROOT / "p2_process_aware"
DEFAULT_P2_STL_DIR = SOURCE_ROOT / "p5_stl/p2_process_aware_height/delta_n_0.5"
DEFAULT_OUTPUT_DIR = ROOT / (
    "outputs/20261105_hw207_mnist4_cmos_demo_v1/runs/"
    "visualization_03_01a1a9b1-5c54-7d81-9999-0cb3b09557a2"
)
PIXEL_AREA_UM2 = c34.CMOS_PITCH_UM**2
ROI_POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
CLASS_COLORS = ("#2878B5", "#E17C05", "#2A9D55", "#8A63B8")
CMOS_ZOOM_BOUNDS = (824, 1224, 824, 1224)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_alignment_checker(path: Path):
    spec = importlib.util.spec_from_file_location("hw207_c36_panel_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Matplotlib panel-alignment audit tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_matplotlib_panel_alignment


def save_figure(
    fig: plt.Figure,
    path: Path,
    require_matplotlib_panel_alignment,
    *,
    row_groups: list[dict[str, Any]] | None = None,
    column_groups: list[dict[str, Any]] | None = None,
    alignment_axes: list[plt.Axes] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment_groups = {}
    if row_groups is not None:
        alignment_groups["row_groups"] = row_groups
    if column_groups is not None:
        alignment_groups["column_groups"] = column_groups
    require_matplotlib_panel_alignment(
        fig,
        axes=alignment_axes,
        json_out=path.with_suffix(".alignment.json"),
        overlay_svg=path.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        **alignment_groups,
    )
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def assert_roi_contract(config_root: Path) -> tuple[tuple[int, int, int, int], ...]:
    c0 = read_json(config_root / "hw207_mnist4_c0_v1.json")
    c31 = read_json(config_root / "hw207_mnist4_c31_c32_cmos_v1.json")
    c34_config = read_json(config_root / "hw207_mnist4_c34_v1.json")
    if c0["roi"]["class_to_roi"] != list(ROI_POSITIONS):
        raise ValueError("C0 ROI mapping conflicts with the frozen C3.3 mapping")
    if c31["cmos"]["roi_order"] != list(ROI_POSITIONS):
        raise ValueError("C3.1/C3.2 ROI order conflicts with C3.3")
    if c31["cmos"]["roi_to_label"] != dict(zip(ROI_POSITIONS, range(4), strict=True)):
        raise ValueError("C3.1/C3.2 ROI class mapping conflicts with C3.3")
    if c34_config["roi"]["class_to_roi"] != [0, 1, 2, 3]:
        raise ValueError("C3.4 class-to-ROI mapping conflicts with C3.3")

    bounds = c34.fixed_roi_bounds("cpu")
    occupancy = np.zeros(c34.CMOS_SHAPE, dtype=np.uint8)
    areas = []
    centers = []
    for r0, r1, c0_, c1 in bounds:
        if not (0 <= r0 < r1 <= c34.CMOS_SHAPE[0] and 0 <= c0_ < c1 <= c34.CMOS_SHAPE[1]):
            raise ValueError(f"fixed ROI outside CMOS frame: {(r0, r1, c0_, c1)}")
        if occupancy[r0:r1, c0_:c1].any():
            raise ValueError("fixed CMOS ROIs overlap")
        occupancy[r0:r1, c0_:c1] = 1
        areas.append((r1 - r0) * (c1 - c0_))
        centers.append(((r0 + r1) / 2, (c0_ + c1) / 2))
    if len(bounds) != 4 or len(set(areas)) != 1:
        raise ValueError("fixed CMOS ROIs are not four equal-area rectangles")
    if not (
        centers[0][0] == centers[1][0] < centers[2][0] == centers[3][0]
        and centers[0][1] == centers[2][1] < centers[1][1] == centers[3][1]
    ):
        raise ValueError("fixed CMOS ROIs are not ordered as a 2x2 grid")
    return bounds


def load_candidates(
    config: dict[str, Any],
    p1_dir: Path,
    p2_train_dir: Path,
    p2_stl_dir: Path,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    p1_checkpoint_path = p1_dir / "best_model.pt"
    p1_phase_path = p1_dir / "phase_energy_constrained.npy"
    p1_records_path = p1_dir / "validation_records.npz"
    p1_checkpoint = torch.load(p1_checkpoint_path, map_location="cpu", weights_only=False)
    p1_phase_np = np.load(p1_phase_path, allow_pickle=False)
    if (
        p1_checkpoint.get("identity") != "C35_BEST_PHASE_OR_HEIGHT"
        or p1_checkpoint.get("mode") != "phase"
        or int(p1_checkpoint.get("seed", -1)) != 11
        or float(p1_checkpoint.get("target_efficiency", -1)) != 0.4
        or p1_phase_np.shape != (100, 100)
        or not np.isfinite(p1_phase_np).all()
    ):
        raise ValueError("P1 checkpoint or phase does not match target 0.4, seed 11")
    p1_checkpoint_phase = p1_checkpoint["variable"].detach().cpu().numpy()
    p1_wrapped_difference = float(
        np.max(np.abs(np.angle(np.exp(1j * (p1_checkpoint_phase - p1_phase_np)))))
    )
    if p1_wrapped_difference > 1e-5:
        raise ValueError("P1 checkpoint and saved phase are not modulo-2pi equivalent")
    p1_record = read_json(p1_dir / "validation_records.json")
    if p1_record.get("tag") != "p1_target_0.400000_seed_11":
        raise ValueError("P1 validation record is not for the requested candidate")
    with np.load(p1_records_path, allow_pickle=False) as saved:
        p1_records = {name: saved[name] for name in saved.files}

    p2_checkpoint_path = p2_train_dir / "best_model.pt"
    p2_height_path = p2_stl_dir / "height_map.npy"
    p2_stl_path = p2_stl_dir / "p2_process_aware_height_200x200um.stl"
    p2_records_path = p2_stl_dir / "stl_roundtrip_validation_records.npz"
    p2_checkpoint = torch.load(p2_checkpoint_path, map_location="cpu", weights_only=False)
    p2_height = np.load(p2_height_path, allow_pickle=False)
    if (
        p2_checkpoint.get("identity") != "C35_BEST_PHASE_OR_HEIGHT"
        or p2_checkpoint.get("mode") != "height"
        or int(p2_checkpoint.get("seed", -1)) != int(config["training"]["p2_seed"])
        or p2_height.shape != (100, 100)
        or not np.isfinite(p2_height).all()
        or not np.array_equal(p2_checkpoint["variable"].detach().cpu().numpy(), p2_height)
    ):
        raise ValueError("P2 saved height does not match its frozen checkpoint")
    p2_summary = read_json(p2_train_dir / "c35_process_aware_summary.json")
    p2_meta = read_json(p2_stl_dir / "stl_metadata.json")
    p2_record = read_json(p2_stl_dir / "stl_roundtrip_validation_records.json")
    if (
        p2_summary.get("selected_seed") != p2_checkpoint["seed"]
        or p2_summary.get("selected_epoch") != p2_checkpoint["epoch"]
        or p2_meta.get("delta_n_assumption") != 0.5
        or p2_meta.get("phase_source") != "p2_process_aware_height"
        or p2_record.get("tag") != "p2_process_aware_height_dn0.5_stl_roundtrip_process"
    ):
        raise ValueError("P2 STL artifacts do not match the requested delta_n=0.5 candidate")
    recovered_height = c34.recover_height_from_mesh_vertices(p2_stl_path)
    height_error = float(np.max(np.abs(recovered_height - p2_height)))
    if height_error > 1e-5:
        raise ValueError(f"P2 STL read-back height error exceeds 1e-5 um: {height_error}")
    process = config["fabrication_assumptions"]
    p2_phase = c35_runner._processed_height(
        torch.as_tensor(recovered_height, dtype=torch.float32, device=device),
        config,
        dn=0.5,
        psf_sigma_px=float(process["psf_sigma_px"]),
        minimum_feature_radius_px=int(process["minimum_feature_radius_px"]),
        height_levels=int(process["height_levels"]),
        noise_seed=0,
        noise_nm=0.0,
        edge_bias_nm=0.0,
    ).detach()
    with np.load(p2_records_path, allow_pickle=False) as saved:
        p2_records = {name: saved[name] for name in saved.files}

    return {
        "p1": {
            "name": "P1 high-throughput phase, target 0.40, seed 11",
            "phase": torch.as_tensor(p1_phase_np, dtype=torch.float32, device=device),
            "records": p1_records,
            "records_path": p1_records_path,
            "checkpoint_path": p1_checkpoint_path,
            "model_path": p1_phase_path,
            "seed": int(p1_checkpoint["seed"]),
            "epoch": int(p1_checkpoint["epoch"]),
            "selection": {
                "target_efficiency": float(p1_checkpoint["target_efficiency"]),
                "checkpoint_phase_wrapped_max_difference_rad": p1_wrapped_difference,
            },
            "process": "unprocessed learned phase candidate; no P2 height or print operator",
        },
        "p2": {
            "name": "P2 process-aware height, delta_n 0.5 STL round-trip",
            "phase": p2_phase,
            "records": p2_records,
            "records_path": p2_records_path,
            "checkpoint_path": p2_checkpoint_path,
            "model_path": p2_stl_path,
            "height_path": p2_height_path,
            "seed": int(p2_checkpoint["seed"]),
            "epoch": int(p2_checkpoint["epoch"]),
            "selection": {
                "target_efficiency": float(p2_checkpoint["target_efficiency"]),
                "delta_n": 0.5,
                "stl_height_recovery_max_abs_error_um": height_error,
                "process": {
                    "psf_sigma_px": float(process["psf_sigma_px"]),
                    "minimum_feature_radius_px": int(process["minimum_feature_radius_px"]),
                    "height_levels": int(process["height_levels"]),
                    "noise_nm": 0.0,
                    "edge_bias_nm": 0.0,
                    "amplitude_transmission": "unit amplitude assumption",
                },
            },
            "process": "STL read-back height plus the frozen nominal assumed height operator",
        },
    }


def forward_batch(
    fields: torch.Tensor,
    phase: torch.Tensor,
    config: dict[str, Any],
    plan_cache: dict[tuple, Any],
) -> dict[str, torch.Tensor]:
    result = c34.model_b_forward(
        fields,
        phase,
        float(config["geometry"]["aperture_diameter_mm"]),
        lens_pitch_um=float(config["geometry"]["lens_plane_pitch_um"]),
        plan_cache=plan_cache,
        return_camera=True,
        screen_to_lens_um=float(config["geometry"]["screen_to_lens_mm"]) * 1000.0,
        lens_to_cmos_um=float(config["geometry"]["lens_to_cmos_mm"]) * 1000.0,
    )
    if result["camera_intensity"] is None or result["camera_intensity"].shape[-2:] != c34.CMOS_SHAPE:
        raise RuntimeError("forward did not return the complete CMOS intensity frame")
    return result


def preflight(
    candidates: dict[str, dict[str, Any]],
    context: dict[str, Any],
    validation_rows: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    row = int(validation_rows[0])
    row_tensor = torch.tensor([row], dtype=torch.long, device=context["device"])
    fields = c34_runner._fields_for_rows(context["cache"], row_tensor)
    checks = {}
    for key, candidate in candidates.items():
        with torch.inference_mode():
            first = forward_batch(fields, candidate["phase"], context["config"], context["plans"][key])
            second = forward_batch(fields, candidate["phase"], context["config"], context["plans"][key])
        actual = first["roi_scores"][0].detach().cpu().double().numpy()
        expected = candidate["records"]["roi_scores"][0].astype(np.float64)
        if int(np.argmax(actual)) != int(candidate["records"]["predictions"][0]) or not np.allclose(
            actual, expected, rtol=1e-4, atol=2e-4
        ):
            raise RuntimeError(f"{key} no-update preflight disagrees with saved C3.5 scores")
        camera = first["camera_intensity"][0].detach().cpu().numpy()
        repeated = second["camera_intensity"][0].detach().cpu().numpy()
        relative_l2 = float(
            np.linalg.norm((camera - repeated).ravel())
            / max(np.linalg.norm(camera.ravel()), np.finfo(np.float64).tiny)
        )
        if not np.allclose(camera, repeated, rtol=2e-6, atol=1e-8):
            raise RuntimeError(f"{key} repeated same-input forward is inconsistent")
        checks[key] = {
            "original_row": row,
            "label_after_forward": int(labels[row]),
            "prediction": int(np.argmax(actual)),
            "score_max_abs_difference": float(np.max(np.abs(actual - expected))),
            "repeat_map_relative_l2_difference": relative_l2,
            "batch_size": 1,
            "gradients": "disabled",
            "optimizer_updates": 0,
        }
    return {"status": "PASS", "candidate_checks": checks}


def run_population(
    key: str,
    candidate: dict[str, Any],
    context: dict[str, Any],
    rows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    count = len(rows)
    scores = np.empty((count, 4), dtype=np.float64)
    input_energy = np.empty(count, dtype=np.float64)
    cmos_energy = np.empty(count, dtype=np.float64)
    predictions = np.empty(count, dtype=np.int64)
    class_sums = np.zeros((4, 2, *c34.CMOS_SHAPE), dtype=np.float64)
    class_counts = np.zeros((4, 2), dtype=np.int64)
    vmax = 0.0
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        batch_rows = rows[start:end]
        row_tensor = torch.as_tensor(batch_rows, dtype=torch.long, device=context["device"])
        fields = c34_runner._fields_for_rows(context["cache"], row_tensor)
        with torch.inference_mode():
            output = forward_batch(fields, candidate["phase"], context["config"], context["plans"][key])
        batch_scores = output["roi_scores"].detach().cpu().double().numpy()
        batch_predictions = batch_scores.argmax(1).astype(np.int64)
        intensity = output["camera_intensity"].detach().cpu()
        scores[start:end] = batch_scores
        predictions[start:end] = batch_predictions
        input_energy[start:end] = output["input_energy"].detach().cpu().double().numpy()
        cmos_energy[start:end] = output["detected_energy"].detach().cpu().double().numpy()
        batch_labels = labels[batch_rows]
        vmax = max(vmax, float(intensity.max()))
        for cls in range(4):
            true_mask = batch_labels == cls
            if true_mask.any():
                class_sums[cls, 0] += intensity[true_mask].double().sum(0).numpy()
                class_counts[cls, 0] += int(true_mask.sum())
                correct_mask = true_mask & (batch_predictions == cls)
                if correct_mask.any():
                    class_sums[cls, 1] += intensity[correct_mask].double().sum(0).numpy()
                    class_counts[cls, 1] += int(correct_mask.sum())
        if end == count or end % 256 == 0:
            print(f"{key}: frozen forward {end}/{count}", flush=True)
    if (class_counts == 0).any():
        raise RuntimeError(f"{key} lacks all-sample or correct-sample CMOS maps for a class")
    means = (class_sums / class_counts[:, :, None, None]).astype(np.float32)
    saved = candidate["records"]
    checks = {
        "saved_row_order_matches": bool(np.array_equal(saved["original_rows"], rows)),
        "saved_labels_match": bool(np.array_equal(saved["labels"], labels[rows])),
        "saved_predictions_match": bool(np.array_equal(saved["predictions"], predictions)),
        "saved_roi_scores_allclose": bool(np.allclose(saved["roi_scores"], scores, rtol=1e-4, atol=2e-4)),
        "saved_cmos_energy_allclose": bool(
            np.allclose(saved["cmos_energy_um2"], cmos_energy, rtol=1e-4, atol=2e-4)
        ),
        "roi_score_max_abs_difference": float(np.max(np.abs(saved["roi_scores"] - scores))),
        "cmos_energy_max_abs_difference_um2": float(np.max(np.abs(saved["cmos_energy_um2"] - cmos_energy))),
    }
    if not all(checks[name] for name in (
        "saved_row_order_matches", "saved_labels_match", "saved_predictions_match",
        "saved_roi_scores_allclose", "saved_cmos_energy_allclose",
    )):
        raise RuntimeError(f"{key} population forward does not reproduce saved C3.5 results")
    if key == "p2":
        old_path = candidate["records_path"].with_name(
            "stl_roundtrip_validation_records_class_average_cmos.npy"
        )
        old_mean = np.load(old_path, allow_pickle=False)
        denominator = max(float(np.linalg.norm(old_mean.ravel())), np.finfo(np.float64).tiny)
        mean_l2 = float(np.linalg.norm((means[:, 0] - old_mean).ravel()) / denominator)
        checks["saved_all_sample_class_mean_relative_l2"] = mean_l2
        checks["saved_all_sample_class_mean_max_abs_difference"] = float(
            np.max(np.abs(means[:, 0] - old_mean))
        )
        if mean_l2 > 1e-3:
            raise RuntimeError("P2 recomputed all-sample means disagree with saved class-average maps")
    return {
        "rows": rows.copy(),
        "labels": labels[rows].copy(),
        "predictions": predictions,
        "roi_scores": scores,
        "input_energy_um2": input_energy,
        "cmos_energy_um2": cmos_energy,
        "roi_outside_energy_um2": np.maximum(cmos_energy - scores.sum(1), 0.0),
        "class_means": means,
        "class_counts": class_counts,
        "vmax": vmax,
        "stored_comparison": checks,
    }


def classification_stats(result: dict[str, Any]) -> dict[str, Any]:
    labels = result["labels"]
    predictions = result["predictions"]
    scores = result["roi_scores"]
    confusion = np.zeros((4, 4), dtype=np.int64)
    for true, predicted in zip(labels, predictions, strict=True):
        confusion[int(true), int(predicted)] += 1
    class_accuracy = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    precision = np.diag(confusion) / np.maximum(confusion.sum(0), 1)
    f1 = 2 * precision * class_accuracy / np.maximum(precision + class_accuracy, 1e-30)
    total = np.maximum(result["cmos_energy_um2"], 1e-30)
    correct = scores[np.arange(len(labels)), labels]
    wrong = scores.copy()
    wrong[np.arange(len(labels)), labels] = -np.inf
    strongest_wrong = wrong.max(1)
    eta = result["cmos_energy_um2"] / np.maximum(result["input_energy_um2"], 1e-30)
    class_energy = np.stack([scores[labels == cls].mean(0) for cls in range(4)])
    return {
        "sample_count": int(len(labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "class_accuracy": class_accuracy.tolist(),
        "mean_efficiency": float(eta.mean()),
        "p10_efficiency": float(np.percentile(eta, 10)),
        "mean_cmos_energy_um2": float(result["cmos_energy_um2"].mean()),
        "mean_roi_energy_um2": scores.mean(0).tolist(),
        "class_mean_roi_energy_um2": class_energy.tolist(),
        "correct_roi_energy_fraction_mean": float(np.mean(correct / total)),
        "strongest_wrong_roi_energy_fraction_mean": float(np.mean(strongest_wrong / total)),
        "correct_to_strongest_wrong_roi_energy_ratio_mean": float(
            np.mean(correct / np.maximum(strongest_wrong, 1e-30))
        ),
        "correct_minus_strongest_wrong_energy_um2_mean": float(np.mean(correct - strongest_wrong)),
        "roi_outside_energy_um2_mean": float(result["roi_outside_energy_um2"].mean()),
        "roi_outside_fraction_mean": float(np.mean(result["roi_outside_energy_um2"] / total)),
        "roi_winner_counts": np.bincount(predictions, minlength=4).tolist(),
        "eta_per_sample": eta,
    }


def choose_rows(results: dict[str, dict[str, Any]], random_seed: int) -> tuple[dict[str, Any], np.ndarray]:
    correct: dict[str, dict[str, list[int]]] = {}
    failures: dict[str, dict[str, int | None]] = {}
    for key, result in results.items():
        correct[key] = {}
        failures[key] = {}
        for cls in range(4):
            class_positions = np.flatnonzero(result["labels"] == cls)
            order = np.argsort(result["rows"][class_positions])
            class_positions = class_positions[order]
            class_rows = result["rows"][class_positions]
            good = class_rows[result["predictions"][class_positions] == cls]
            bad = class_rows[result["predictions"][class_positions] != cls]
            if len(good) < 3:
                raise RuntimeError(f"{key} class {cls} has fewer than three correct samples")
            correct[key][str(cls)] = [int(value) for value in good[:3]]
            failures[key][str(cls)] = int(bad[0]) if len(bad) else None
    p2 = results["p2"]
    class1_rows = p2["rows"][p2["labels"] == 1]
    demo_row = int(np.min(class1_rows))
    rng = np.random.default_rng(random_seed)
    random_rows = np.sort(rng.choice(p2["rows"], size=8, replace=False)).astype(np.int64)
    selected = {}
    for key in results:
        chosen = set(random_rows.tolist())
        chosen.add(demo_row)
        for values in correct[key].values():
            chosen.update(values)
        chosen.update(row for row in failures[key].values() if row is not None)
        selected[key] = sorted(chosen)
    return {
        "correct_examples_first_by_original_index": correct,
        "first_error_by_class": failures,
        "random_independent_roi_check_rows": [int(value) for value in random_rows],
        "input_1_demo_original_row": demo_row,
        "selected_raw_cmos_rows_by_candidate": selected,
        "random_seed": random_seed,
    }, random_rows


def collect_maps(
    key: str,
    candidate: dict[str, Any],
    full_result: dict[str, Any],
    selected_rows: list[int],
    context: dict[str, Any],
) -> dict[int, np.ndarray]:
    positions = {int(row): i for i, row in enumerate(full_result["rows"])}
    maps = {}
    for row in selected_rows:
        row_tensor = torch.tensor([row], dtype=torch.long, device=context["device"])
        fields = c34_runner._fields_for_rows(context["cache"], row_tensor)
        with torch.inference_mode():
            first = forward_batch(fields, candidate["phase"], context["config"], context["plans"][key])
            repeated = forward_batch(fields, candidate["phase"], context["config"], context["plans"][key])
        image = first["camera_intensity"][0].detach().cpu().numpy().astype(np.float32, copy=True)
        second = repeated["camera_intensity"][0].detach().cpu().numpy()
        if not np.isfinite(image).all() or (image < 0).any():
            raise RuntimeError(f"{key} produced invalid CMOS intensity")
        if not np.allclose(image, second, rtol=2e-6, atol=1e-8):
            raise RuntimeError(f"{key} selected-sample repeat forward is inconsistent")
        if not np.allclose(
            first["roi_scores"][0].detach().cpu().double().numpy(),
            full_result["roi_scores"][positions[row]],
            rtol=1e-4, atol=2e-4,
        ):
            raise RuntimeError(f"{key} selected-sample score differs from full-population forward")
        maps[row] = image
    return maps


def collect_dmd_patterns(rows: list[int], images: torch.Tensor, configuration: Any) -> dict[int, np.ndarray]:
    unique_rows = sorted(set(int(row) for row in rows))
    binary = (images[unique_rows].to(configuration.device) >= 128).float()
    mapped = c33.geo.map_binary_images(
        binary,
        configuration.base.source_matrix,
        configuration.base.target_matrix,
        configuration.base.affine,
    )
    command_shape = tuple(int(value) for value in configuration.base.lattice_shape)
    expected_elements = len(unique_rows) * int(np.prod(command_shape))
    if mapped["commands"].numel() != expected_elements:
        raise RuntimeError(
            f"C3.3 command tensor has {mapped['commands'].numel()} values; "
            f"expected {expected_elements} for {command_shape} mirrors"
        )
    commands = mapped["commands"].reshape(len(unique_rows), *command_shape).detach().cpu().numpy().astype(np.uint8)
    return {row: commands[index] for index, row in enumerate(unique_rows)}


def independent_roi_checks(
    results: dict[str, dict[str, Any]],
    maps: dict[str, dict[int, np.ndarray]],
    random_rows: np.ndarray,
    rectangles: tuple[tuple[int, int, int, int], ...],
) -> dict[str, Any]:
    output = {}
    for key, result in results.items():
        checks = []
        for row in random_rows:
            position = int(np.flatnonzero(result["rows"] == row)[0])
            label = int(result["labels"][position])
            independent = roi_readout.readout(maps[key][int(row)], rectangles, label)
            scores = np.asarray(independent["scores"], dtype=np.float64) * PIXEL_AREA_UM2
            saved_scores = result["roi_scores"][position]
            if (
                independent["prediction"] != int(result["predictions"][position])
                or not np.allclose(scores, saved_scores, rtol=1e-5, atol=2e-4)
            ):
                raise RuntimeError(f"{key} independent ROI sum failed for row {int(row)}")
            checks.append({
                "original_row": int(row),
                "true_class_after_forward": label,
                "independent_scores_um2": scores.tolist(),
                "saved_scores_um2": saved_scores.tolist(),
                "prediction": int(independent["prediction"]),
                "max_abs_difference_um2": float(np.max(np.abs(scores - saved_scores))),
            })
        output[key] = {"sample_count": len(checks), "checks": checks}
    return output


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def draw_roi_boxes(
    axis: plt.Axes,
    rectangles: tuple[tuple[int, int, int, int], ...],
    true_class: int,
    prediction: int,
    labels: bool,
) -> None:
    for index, (r0, r1, c0_, c1) in enumerate(rectangles):
        axis.add_patch(Rectangle(
            (c0_ - 0.5, r0 - 0.5), c1 - c0_, r1 - r0,
            fill=False, edgecolor="white", linewidth=0.7, zorder=4,
        ))
        if index == true_class:
            axis.add_patch(Rectangle(
                (c0_ - 0.5, r0 - 0.5), c1 - c0_, r1 - r0,
                fill=False, edgecolor="#12D66B", linewidth=1.6, zorder=5,
            ))
        if prediction != true_class and index == prediction:
            axis.add_patch(Rectangle(
                (c0_ - 0.5, r0 - 0.5), c1 - c0_, r1 - r0,
                fill=False, edgecolor="#FF463A", linewidth=1.6, linestyle="--", zorder=6,
            ))
        if labels:
            label_y = r0 - 16
            axis.text(
                (c0_ + c1 - 1) / 2, label_y, f"ROI_{index}→{index}",
                color="white", fontsize=7, ha="center", va="bottom",
                bbox={"facecolor": "black", "alpha": 0.76, "pad": 1.0, "edgecolor": "none"},
                zorder=7,
            )


def show_cmos(
    axis: plt.Axes,
    image: np.ndarray,
    rectangles: tuple[tuple[int, int, int, int], ...],
    true_class: int,
    prediction: int,
    norm: Normalize,
    labels: bool,
    zoom: bool = False,
) -> Any:
    if zoom:
        r0, r1, c0, c1 = CMOS_ZOOM_BOUNDS
        plot_image = image[r0:r1, c0:c1]
        if plot_image.shape != (400, 400):
            raise ValueError(f"CMOS zoom window is not 400x400: {plot_image.shape}")
        extent = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
    else:
        plot_image = image
        extent = None
    artist = axis.imshow(
        plot_image, cmap="magma", norm=norm, origin="upper", extent=extent,
        interpolation="nearest", aspect="equal",
    )
    if not np.array_equal(np.asarray(artist.get_array()), plot_image):
        raise RuntimeError("plot image differs from the raw saved CMOS intensity array")
    draw_roi_boxes(axis, rectangles, true_class, prediction, labels)
    if zoom:
        r0, r1, c0, c1 = CMOS_ZOOM_BOUNDS
        axis.set_xlim(c0 - 0.5, c1 - 0.5)
        axis.set_ylim(r1 - 0.5, r0 - 0.5)
    else:
        axis.set_xlim(-0.5, c34.CMOS_SHAPE[1] - 0.5)
        axis.set_ylim(c34.CMOS_SHAPE[0] - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    return artist


def show_cmos_locator(
    axis: plt.Axes,
    image: np.ndarray,
    rectangles: tuple[tuple[int, int, int, int], ...],
    true_class: int,
    prediction: int,
    norm: Normalize,
) -> None:
    axis.imshow(image, cmap="magma", norm=norm, origin="upper", interpolation="nearest")
    draw_roi_boxes(axis, rectangles, true_class, prediction, False)
    r0, r1, c0, c1 = CMOS_ZOOM_BOUNDS
    axis.add_patch(Rectangle(
        (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
        fill=False, edgecolor="#F4B942", linewidth=1.1, linestyle="--", zorder=8,
    ))
    axis.set_xlim(-0.5, c34.CMOS_SHAPE[1] - 0.5)
    axis.set_ylim(c34.CMOS_SHAPE[0] - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Full CMOS\norange outline = zoom", fontsize=6.5)


def draw_input(axis: plt.Axes, images: torch.Tensor, row: int, compact: bool = False) -> None:
    binary = (images[row].cpu().numpy() >= 128).astype(np.uint8)
    axis.imshow(binary, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"Binary input\nindex {row}" if not compact else f"{row}", fontsize=8 if not compact else 7)


def draw_scheme(axis: plt.Axes, candidate: dict[str, Any]) -> None:
    axis.set_axis_off()
    axis.set_box_aspect(1)
    labels = ("Binary MNIST input", "Frozen 100×100\nphase / height", "P80 → lens\nØ2 mm, f=30 mm", "P100 → CMOS\n2048×2048")
    ys = np.linspace(0.86, 0.24, len(labels))
    for i, (label, y) in enumerate(zip(labels, ys, strict=True)):
        axis.text(
            0.5, y, label, ha="center", va="center", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.38", "facecolor": "#F6F7F8", "edgecolor": "#6B7280", "linewidth": 0.7},
            transform=axis.transAxes,
        )
        if i < len(labels) - 1:
            axis.add_patch(FancyArrowPatch(
                (0.5, y - 0.07), (0.5, ys[i + 1] + 0.07),
                arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color="#4B5563",
                transform=axis.transAxes,
            ))
    axis.text(
        0.5, 0.05, f"532 nm · {candidate['name'].split(',')[0]}",
        ha="center", va="center", fontsize=6.5, transform=axis.transAxes,
    )


def draw_frozen_layer(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.add_patch(FancyArrowPatch(
        (0.5, 0.84), (0.5, 0.68), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.7, color="#4B5563", transform=axis.transAxes,
    ))
    axis.text(
        0.5, 0.5, "Frozen\n100×100\nphase / height", ha="center", va="center",
        fontsize=6.2, bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F6F7F8",
                           "edgecolor": "#6B7280", "linewidth": 0.6},
        transform=axis.transAxes,
    )
    axis.add_patch(FancyArrowPatch(
        (0.5, 0.32), (0.5, 0.16), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.7, color="#4B5563", transform=axis.transAxes,
    ))


def energy_bars(axis: plt.Axes, scores: np.ndarray, maximum: float) -> None:
    axis.barh(np.arange(4), np.asarray(scores) / 1000.0, color=CLASS_COLORS, height=0.66)
    axis.set_yticks(np.arange(4), [f"ROI_{i}→C{i}" for i in range(4)])
    axis.invert_yaxis()
    axis.set_xlabel("ROI integral (×10³ model a.u.·µm²)")
    axis.set_xlim(0, maximum / 1000.0)
    axis.grid(axis="x", color="#D9DDE1", linewidth=0.45)
    axis.set_axisbelow(True)


def figure_input_one(
    candidate: dict[str, Any],
    result: dict[str, Any],
    maps: dict[int, np.ndarray],
    dmd_patterns: dict[int, np.ndarray],
    images: torch.Tensor,
    rectangles: tuple[tuple[int, int, int, int], ...],
    vmax: float,
    row: int,
    path: Path,
    title: str,
    require_matplotlib_panel_alignment,
) -> None:
    position = int(np.flatnonzero(result["rows"] == row)[0])
    true_class = int(result["labels"][position])
    prediction = int(result["predictions"][position])
    score = result["roi_scores"][position]
    fig = plt.figure(figsize=(17, 8), constrained_layout=True)
    grid = fig.add_gridspec(
        3, 5, height_ratios=(5.5, 0.48, 1.5),
        width_ratios=(0.72, 0.72, 0.58, 2.8, 0.85),
    )
    raw_axis = fig.add_subplot(grid[0, 0])
    binary_axis = fig.add_subplot(grid[0, 1])
    raw_axis.set_gid("raw-mnist")
    binary_axis.set_gid("dmd-command")
    raw_axis.imshow(images[row].cpu().numpy(), cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
    raw_axis.set_title(f"Raw MNIST\nindex {row}")
    raw_axis.set_xticks([])
    raw_axis.set_yticks([])
    binary_axis.imshow(dmd_patterns[row], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    binary_axis.set_title(
        f"Local DMD command window\n{dmd_patterns[row].shape[0]}×{dmd_patterns[row].shape[1]} mirrors"
    )
    binary_axis.set_xticks([])
    binary_axis.set_yticks([])
    draw_scheme(fig.add_subplot(grid[0, 2]), candidate)
    norm = Normalize(0.0, vmax)
    map_axis = fig.add_subplot(grid[0, 3])
    artist = show_cmos(map_axis, maps[row], rectangles, true_class, prediction, norm, True, zoom=True)
    locator_axis = fig.add_subplot(grid[0, 4])
    map_axis.set_gid("cmos-zoom")
    locator_axis.set_gid("cmos-full-locator")
    show_cmos_locator(locator_axis, maps[row], rectangles, true_class, prediction, norm)
    map_axis.set_title(
        f"CMOS intensity · True class {true_class} · Predicted class {prediction} · "
        f"{'correct' if true_class == prediction else 'error'}"
    )
    colorbar_axis = fig.add_subplot(grid[1, 3:])
    cbar = fig.colorbar(artist, cax=colorbar_axis, orientation="horizontal")
    cbar.set_label("Intensity (model a.u.)")
    cbar.ax.tick_params(labelsize=7, length=2)
    bar_axis = fig.add_subplot(grid[2, :])
    energy_bars(bar_axis, score, max(float(np.max(score)) * 1.12, 1e-9))
    bar_axis.set_title("Frozen optical forward → CMOS intensity → fixed ROI integration → electronic argmax", fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="medium")
    save_figure(
        fig, path, require_matplotlib_panel_alignment,
        row_groups=[{"id": "input-panel-row", "panels": ["raw-mnist", "dmd-command"]}],
        alignment_axes=[raw_axis, binary_axis, map_axis, locator_axis],
    )


def figure_four_class(
    key: str,
    candidate: dict[str, Any],
    result: dict[str, Any],
    maps: dict[int, np.ndarray],
    selection: dict[str, Any],
    images: torch.Tensor,
    rectangles: tuple[tuple[int, int, int, int], ...],
    vmax: float,
    path: Path,
    require_matplotlib_panel_alignment,
) -> None:
    correct = selection["correct_examples_first_by_original_index"][key]
    fig = plt.figure(figsize=(27, 16), constrained_layout=True)
    grid = fig.add_gridspec(
        4, 5, width_ratios=(1.0, 1.0, 1.0, 0.72, 0.045), hspace=0.02, wspace=0.09,
    )
    norm = Normalize(0.0, vmax)
    last_image = None
    camera_axes: list[plt.Axes] = []
    locator_axes: list[plt.Axes] = []
    for cls in range(4):
        for sample_number, row in enumerate(correct[str(cls)]):
            position = int(np.flatnonzero(result["rows"] == row)[0])
            prediction = int(result["predictions"][position])
            camera_axis = fig.add_subplot(grid[cls, sample_number])
            camera_gid = f"class-{cls}-sample-{sample_number}-cmos"
            camera_axis.set_gid(camera_gid)
            camera_axes.append(camera_axis)
            last_image = show_cmos(camera_axis, maps[row], rectangles, cls, prediction, norm, True, zoom=True)
            camera_axis.set_title(
                f"Sample {sample_number + 1} · index {row} · true {cls} / pred {prediction}",
                fontsize=7.5, pad=2,
            )
        locator_axis = fig.add_subplot(grid[cls, 3])
        locator_gid = f"class-{cls}-cmos-locator"
        locator_axis.set_gid(locator_gid)
        locator_axes.append(locator_axis)
        show_cmos_locator(locator_axis, result["class_means"][cls, 0], rectangles, cls, cls, norm)
        locator_axis.set_title(f"Class {cls} overview", fontsize=8)
    colorbar_axis = fig.add_subplot(grid[:, 4])
    colorbar_axis.set_gid("shared-linear-colorbar")
    fig.suptitle(
        f"{candidate['name']} · three correct examples per class\n"
        "Shared flow: MNIST input → frozen optical model → CMOS window [824:1224, 824:1224] → ROI argmax",
        fontsize=12,
    )
    cbar = fig.colorbar(last_image, cax=colorbar_axis)
    cbar.set_label("Intensity (model a.u.)")
    row_groups = [
        {
            "id": f"class-{cls}-cmos-zoom-row",
            "panels": [f"class-{cls}-sample-{sample_number}-cmos" for sample_number in range(3)],
        }
        for cls in range(4)
    ]
    column_groups = [
        {
            "id": f"sample-{sample_number}-cmos-zoom-column",
            "panels": [f"class-{cls}-sample-{sample_number}-cmos" for cls in range(4)],
        }
        for sample_number in range(3)
    ] + [{"id": "class-mean-locator-column", "panels": [f"class-{cls}-cmos-locator" for cls in range(4)]}]
    save_figure(
        fig, path, require_matplotlib_panel_alignment,
        row_groups=row_groups, column_groups=column_groups,
        alignment_axes=camera_axes + locator_axes,
    )


def figure_failures(
    key: str,
    candidate: dict[str, Any],
    result: dict[str, Any],
    maps: dict[int, np.ndarray],
    selection: dict[str, Any],
    images: torch.Tensor,
    rectangles: tuple[tuple[int, int, int, int], ...],
    vmax: float,
    path: Path,
    require_matplotlib_panel_alignment,
) -> None:
    failures = selection["first_error_by_class"][key]
    rows = [failures[str(cls)] for cls in range(4) if failures[str(cls)] is not None]
    if not rows:
        return
    fig = plt.figure(figsize=(21, 15), constrained_layout=True)
    grid = fig.add_gridspec(len(rows), 4, width_ratios=(0.65, 3.4, 1.8, 1.05), hspace=0.18)
    last_image = None
    maximum = max(float(np.max(result["roi_scores"])) * 1.04, 1e-9)
    camera_axes: list[plt.Axes] = []
    camera_ids: list[str] = []
    for row_number, row in enumerate(rows):
        position = int(np.flatnonzero(result["rows"] == row)[0])
        true_class = int(result["labels"][position])
        prediction = int(result["predictions"][position])
        input_axis = fig.add_subplot(grid[row_number, 0])
        input_axis.set_gid(f"error-{row_number}-input")
        draw_input(input_axis, images, row, compact=True)
        camera_axis = fig.add_subplot(grid[row_number, 1])
        camera_gid = f"error-{row_number}-cmos"
        camera_axis.set_gid(camera_gid)
        camera_axes.append(camera_axis)
        camera_ids.append(camera_gid)
        last_image = show_cmos(
            camera_axis, maps[row], rectangles, true_class, prediction, Normalize(0.0, vmax), True, zoom=True,
        )
        camera_axis.set_title(
            f"First error for true class {true_class} · index {row} · predicted {prediction}",
            fontsize=8,
        )
        locator_axis = fig.add_subplot(grid[row_number, 3])
        locator_axis.set_gid(f"error-{row_number}-locator")
        show_cmos_locator(locator_axis, maps[row], rectangles, true_class, prediction, Normalize(0.0, vmax))
        bar_axis = fig.add_subplot(grid[row_number, 2])
        bar_axis.set_gid(f"error-{row_number}-roi")
        energy_bars(bar_axis, result["roi_scores"][position], maximum)
        bar_axis.tick_params(axis="both", labelsize=7)
    fig.suptitle(f"{candidate['name']} · preserved first error per class", fontsize=11)
    cbar = fig.colorbar(last_image, ax=camera_axes, fraction=0.035, pad=0.025)
    cbar.set_label("Intensity (model a.u.)")
    row_groups = [
        {
            "id": f"error-{row_number}-{role}",
            "panels": [f"error-{row_number}-{first}", f"error-{row_number}-{second}"],
        }
        for row_number in range(len(rows))
        for role, first, second in (("input-cmos", "input", "cmos"), ("cmos-roi", "cmos", "roi"))
    ]
    save_figure(
        fig, path, require_matplotlib_panel_alignment,
        column_groups=[{"id": "first-error-cmos-zoom-column", "panels": camera_ids}],
        alignment_axes=camera_axes,
    )


def figure_class_means(
    candidate: dict[str, Any],
    result: dict[str, Any],
    rectangles: tuple[tuple[int, int, int, int], ...],
    vmax: float,
    path: Path,
    require_matplotlib_panel_alignment,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(18, 17), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.8, 1.8, 0.72)})
    norm = Normalize(0.0, vmax)
    last_image = None
    alignment_axes: list[plt.Axes] = []
    row_groups = []
    column_groups = []
    for cls in range(4):
        for column, group in enumerate((0, 1)):
            last_image = show_cmos(
                axes[cls, column], result["class_means"][cls, group],
                rectangles, cls, cls, norm, True, zoom=True,
            )
            axes[cls, column].set_gid(f"class-{cls}-mean-{column}")
            alignment_axes.append(axes[cls, column])
            count = int(result["class_counts"][cls, group])
            label = "All true-class validation rows" if group == 0 else "Correct rows only"
            axes[cls, column].set_title(f"Class {cls} · {label} · n={count} · 400×400 px")
        show_cmos_locator(
            axes[cls, 2], result["class_means"][cls, 0], rectangles, cls, cls, norm,
        )
        axes[cls, 2].set_gid(f"class-{cls}-mean-locator")
        alignment_axes.append(axes[cls, 2])
        axes[cls, 2].set_title(f"Class {cls} mean locator", fontsize=7)
        row_groups.append({"id": f"class-{cls}-mean-zoom-row", "panels": [f"class-{cls}-mean-0", f"class-{cls}-mean-1"]})
    column_groups.extend(
        {"id": f"class-mean-column-{column}", "panels": [f"class-{cls}-mean-{column}" for cls in range(4)]}
        for column in range(2)
    )
    column_groups.append({"id": "class-mean-locator-column", "panels": [f"class-{cls}-mean-locator" for cls in range(4)]})
    fig.suptitle(f"{candidate['name']} · linear intensity means · shared scale 0–{vmax:.4g}", fontsize=12)
    cbar = fig.colorbar(last_image, ax=axes[:, :2].ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Mean intensity (model a.u.)")
    save_figure(fig, path, require_matplotlib_panel_alignment,
                row_groups=row_groups, column_groups=column_groups, alignment_axes=alignment_axes)


def figure_energy(candidate: dict[str, Any], stats: dict[str, Any], path: Path, require_matplotlib_panel_alignment) -> None:
    class_energy = np.asarray(stats["class_mean_roi_energy_um2"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    image = axes[0].imshow(class_energy / 1000.0, cmap="Blues", vmin=0)
    axes[0].set_xticks(range(4), [f"ROI_{i}" for i in range(4)])
    axes[0].set_yticks(range(4), [f"True {i}" for i in range(4)])
    axes[0].set_title("Mean ROI energy by true class")
    for row in range(4):
        for col in range(4):
            axes[0].text(col, row, f"{class_energy[row, col] / 1000:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="Mean integral (×10³ model a.u.·µm²)")
    x = np.arange(4)
    width = 0.18
    for roi in range(4):
        axes[1].bar(
            x + (roi - 1.5) * width, class_energy[:, roi] / 1000.0,
            width=width, color=CLASS_COLORS[roi], label=f"ROI_{roi}",
        )
    axes[1].set_xticks(x, [f"True {i}" for i in range(4)])
    axes[1].set_ylabel("Mean ROI integral (×10³ model a.u.·µm²)")
    axes[1].set_title("Four-region route by input class")
    axes[1].grid(axis="y", color="#D9DDE1", linewidth=0.45)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle(candidate["name"], fontsize=11)
    save_figure(fig, path, require_matplotlib_panel_alignment)


def figure_confusion(candidate: dict[str, Any], stats: dict[str, Any], path: Path, require_matplotlib_panel_alignment) -> None:
    matrix = np.asarray(stats["confusion_matrix"], dtype=np.int64)
    percent = matrix / np.maximum(matrix.sum(1, keepdims=True), 1) * 100
    fig, axis = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = axis.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    axis.set_xticks(range(4), [f"Pred {i}" for i in range(4)])
    axis.set_yticks(range(4), [str(i) for i in range(4)])
    axis.set_title(f"{candidate['name']} · n={matrix.sum()}\nRows: true class · columns: predicted class")
    for row in range(4):
        for col in range(4):
            axis.text(
                col, row, f"{matrix[row, col]}\n{percent[row, col]:.1f}%",
                ha="center", va="center", fontsize=8,
                color="white" if percent[row, col] > 48 else "#1F2937",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row-normalized count (%)")
    save_figure(fig, path, require_matplotlib_panel_alignment)


def figure_comparison(
    candidates: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    maps: dict[str, dict[int, np.ndarray]],
    demo_row: int,
    rectangles: tuple[tuple[int, int, int, int], ...],
    vmax: float,
    path: Path,
    require_matplotlib_panel_alignment,
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True,
                             gridspec_kw={"height_ratios": (4.0, 1.0)})
    norm = Normalize(0.0, vmax)
    alignment_axes: list[plt.Axes] = []
    for column, key in enumerate(("p1", "p2")):
        result = results[key]
        position = int(np.flatnonzero(result["rows"] == demo_row)[0])
        true_class = int(result["labels"][position])
        prediction = int(result["predictions"][position])
        zoom_axis = axes[0, column]
        zoom_axis.set_gid(f"candidate-{key}-cmos-zoom")
        locator_axis = axes[1, column]
        locator_axis.set_gid(f"candidate-{key}-cmos-locator")
        alignment_axes.append(zoom_axis)
        artist = show_cmos(
            zoom_axis, maps[key][demo_row], rectangles, true_class, prediction, norm, True, zoom=True,
        )
        zoom_axis.set_title(
            f"{candidates[key]['name']} · 400×400 px\nindex {demo_row} · true 1 / predicted {prediction}",
            fontsize=8,
        )
        show_cmos_locator(
            locator_axis, maps[key][demo_row], rectangles, true_class, prediction, norm,
        )
    row_groups = [
        {"id": "candidate-zoom-row", "panels": ["candidate-p1-cmos-zoom", "candidate-p2-cmos-zoom"]},
    ]
    fig.colorbar(artist, ax=axes[0, :], fraction=0.035, pad=0.02,
                 label="Intensity (model a.u.) · shared linear scale")
    fig.suptitle(
        f"Same input and ROIs · P1 accuracy {stats['p1']['accuracy']:.2%}, mean η {stats['p1']['mean_efficiency']:.2%} · "
        f"P2 STL accuracy {stats['p2']['accuracy']:.2%}, mean η {stats['p2']['mean_efficiency']:.2%}",
        fontsize=10,
    )
    save_figure(fig, path, require_matplotlib_panel_alignment,
                row_groups=row_groups, alignment_axes=alignment_axes)
    first = maps["p1"][demo_row].astype(np.float64)
    second = maps["p2"][demo_row].astype(np.float64)
    relative_l2 = float(np.linalg.norm((first - second).ravel()) / max(np.linalg.norm(first.ravel()), 1e-30))
    return {
        "original_row": demo_row,
        "p1_max_intensity": float(first.max()),
        "p2_max_intensity": float(second.max()),
        "p1_to_p2_mean_intensity_ratio": float(first.mean() / max(second.mean(), 1e-30)),
        "map_relative_l2_difference": relative_l2,
        "shared_color_scale_vmax": vmax,
    }


def report_text(
    stats: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    preflight: dict[str, Any],
    checks: dict[str, Any],
    bounds: tuple[tuple[int, int, int, int], ...],
    output_dir: Path,
    map_comparison: dict[str, Any],
    shared_vmax: float,
) -> str:
    demo_winners = {
        key: int(np.argmax(stats[key]["demo_roi_scores_um2"]))
        for key in ("p1", "p2")
    }
    lines = [
        "# HW207 MNIST4 C3.6 CMOS 2×2 optical classification visualization",
        "",
        "## 科学认识",
        "",
        "本轮只对冻结候选进行数值 forward、CMOS 强度与固定 ROI 读出；标签仅在 forward 返回后用于评价和样本选择。",
        "",
        f"1. **输入数字 1：** P2 主演示原始 MNIST 索引为 {selection['input_1_demo_original_row']}；同一图像由 P1、P2 分别 forward。True/Predicted、四区积分与原始强度图见 01_input_1_cmos_demo、01_input_1_cmos_demo_p1 和 06_c35_candidate_comparison。",
        f"   - P2：True=1、Predicted={stats['p2']['demo_prediction']}、最强 ROI_{demo_winners['p2']}；ROI_1 是否最强：{'是' if demo_winners['p2'] == 1 else '否'}；ROI 积分（ROI_0…ROI_3）={np.round(stats['p2']['demo_roi_scores_um2'], 3).tolist()}。",
        f"   - P1：True=1、Predicted={stats['p1']['demo_prediction']}、最强 ROI_{demo_winners['p1']}；ROI_1 是否最强：{'是' if demo_winners['p1'] == 1 else '否'}；ROI 积分（ROI_0…ROI_3）={np.round(stats['p1']['demo_roi_scores_um2'], 3).tolist()}。",
        "2. **类别路由：** 下表赢家来自每类全部验证输入的四个 ROI 平均积分；全部样本均值和正确样本均值另行绘制。",
        "",
        "| 候选 | 类别 | 类均值最强 ROI | 类准确率 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("p2", "p1"):
        for cls in range(4):
            winner = int(np.argmax(stats[key]["class_mean_roi_energy_um2"][cls]))
            lines.append(f"| {key.upper()} | {cls} | ROI_{winner} | {stats[key]['class_accuracy'][cls]:.2%} |")
    for key in ("p2", "p1"):
        mean_winner = int(np.argmax(stats[key]["class_mean_roi_energy_um2"][1]))
        same = demo_winners[key] == mean_winner
        lines.append(
            f"- {key.upper()}：数字 1 单样本赢家 ROI_{demo_winners[key]}；真实类别 1 全部验证样本均值赢家 ROI_{mean_winner}；两者{'一致' if same else '不一致'}。"
        )
    lines.extend([
        "",
        "3. **其他 ROI 相对强度：** ROI 面积相等，积分排名也等于区域平均强度排名；逐样本正确 ROI 占 CMOS 能量比例、最强错误 ROI 比例和两者差值如下。",
        "",
        "| 候选 | 正确 ROI / CMOS | 最强错误 ROI / CMOS | 正确/最强错误积分比 | 正确−最强错误平均能量差 |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in ("p2", "p1"):
        s = stats[key]
        lines.append(
            f"| {key.upper()} | {s['correct_roi_energy_fraction_mean']:.2%} | "
            f"{s['strongest_wrong_roi_energy_fraction_mean']:.2%} | "
            f"{s['correct_to_strongest_wrong_roi_energy_ratio_mean']:.3g} | "
            f"{s['correct_minus_strongest_wrong_energy_um2_mean']:.4g} model a.u.·µm² |"
        )
    lines.extend([
        "",
        "4. **ROI 外能量：**",
        "",
        "| 候选 | CMOS 外四区平均比例 | 外四区平均积分 |",
        "|---|---:|---:|",
    ])
    for key in ("p2", "p1"):
        s = stats[key]
        lines.append(f"| {key.upper()} | {s['roi_outside_fraction_mean']:.2%} | {s['roi_outside_energy_um2_mean']:.4g} model a.u.·µm² |")
    lines.extend([
        "",
        f"5. **单样本与类均值：** 数字 1 主图取 C3.4 validation 中原始索引最早的真实类别 1 样本，不按预测结果筛选；Figure 02 每类显示按原始索引最早的 3 个正确样本，首错样本图单独保存每类第一个错误样本；Figure 03 并列显示全部 500 个真实类别样本的均值和正确样本均值，避免只用正确样本导致选择偏差。",
        "6. **错误路由：**",
        "",
        "| 候选 | 最低准确类别 | 类准确率 | 最大错误去向 |",
        "|---|---:|---:|---:|",
    ])
    for key in ("p2", "p1"):
        s = stats[key]
        cls = int(np.argmin(s["class_accuracy"]))
        row = np.asarray(s["confusion_matrix"])[cls].copy()
        row[cls] = -1
        route = int(np.argmax(row))
        lines.append(f"| {key.upper()} | {cls} | {s['class_accuracy'][cls]:.2%} | ROI_{route} / class {route} |")
    lines.extend([
        "",
        f"7. **P1/P2 差异：** P1 为目标效率 40%、seed 11 的冻结相位；P2 为 seed 37 冻结高度，经 Δn=0.5 STL 回读与名义假设工艺算子。数字 1 同一输入（row {map_comparison['original_row']}）的 P1/P2 光强均值比={map_comparison['p1_to_p2_mean_intensity_ratio']:.4g}，全图相对 L2 差异={map_comparison['map_relative_l2_difference']:.4g}；图使用共同色标 0–{map_comparison['shared_color_scale_vmax']:.7g}。二者分开计算，样本来自相同开发验证人口，不是独立测试。",
        "",
        "| 候选 | Accuracy | Macro-F1 | 平均效率 | P10 效率 | 平均 CMOS 积分 | CMOS 外四区 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in ("p2", "p1"):
        s = stats[key]
        lines.append(
            f"| {key.upper()} | {s['accuracy']:.2%} | {s['macro_f1']:.2%} | {s['mean_efficiency']:.2%} | "
            f"{s['p10_efficiency']:.2%} | {s['mean_cmos_energy_um2']:.4g} model a.u.·µm² | "
            f"{s['roi_outside_fraction_mean']:.2%} |"
        )
    lines.extend([
        "",
        "8. **Science 2018 展示逻辑：** 是，图件采用输入数字与实际二值 DMD 命令→固定衍射层→输出面强度→固定探测区→最大积分判决的分类展示逻辑，仅参考探测区能量路由表达，不复制论文图片。Lin et al. 报告了按类别平均的输出面强度及对应 detector 区域（[DOI 10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084)）。",
        "9. **结果类型：** 全部为 C3.4 Model B 标量 Fresnel 数值传播；P2 还经过假设高度工艺和 STL 网格回读。不是实物 CMOS 采集或已标定的 HW207 硬件结果。",
        "",
        "## 目标缺口",
        "",
        "C3.5 报告将 2,000 行人口标为已复用开发验证集；C0/C3.4 已使用相关 MNIST 0–3 人口，故不能称为独立测试。P2 的 Δn=0.5、PSF σ=0.8 px、最小特征半径 1 px、64 级高度、单位振幅和无噪声/无边缘偏差均为数值假设。材料复透射、打印机响应、真实光路和 CMOS 坐标没有实测闭环。",
        "",
        "## 工程检查",
        "",
        f"- 人口与行序：C3.4 validation n={checks['p2']['row_count']}，每类 500；P1/P2 输入行和标签与已保存 C3.5 记录相同。",
        f"- 固定 ROI 半开坐标（row start,end,column start,end）：{list(bounds)}；顺序 ROI_0→0、ROI_1→1、ROI_2→2、ROI_3→3；像素面积 {PIXEL_AREA_UM2:g} µm²。",
        f"- 无更新单样本预检：{preflight['status']}；同输入重复 forward 通过；梯度关闭、优化器更新为 0。",
        "- 输入图：左侧保留原始 MNIST 索引和灰度图，中间显示由实际 C3.3 编码生成的 104×104 本地二值 DMD 命令窗口；面积积分后映射到 100×100 目标面，再进入 C3.4 输入 relay/pupil crop；未人工绘制光场或探测亮斑。",
        "- 全人口重放：原始行、标签、预测、四区积分与 CMOS 能量均复现 C3.5 保存记录；P2 全类均值另与已存 class-average 数组比对。",
        "- 图件布局：各多面板图在 Matplotlib 最终版式上通过 1.5 pt 对齐 Gate；布局 manifest 和 QA overlay 随图保存。",
        f"- 独立 ROI 复算：每候选随机 8 个验证行，以 NumPy 对原始强度分区求和并乘 {PIXEL_AREA_UM2:g} µm²；积分和 argmax 均与 forward 结果一致。",
        "- 图像方向：轴 0 为 CMOS 行（row 0 在上），轴 1 为列；绘图无翻转、转置、插值或裁剪。绘制的图像数组与保存强度数组逐像素相同。",
        f"- 光强使用全人口统一线性色标，vmin=0、vmax={shared_vmax:.8g} model a.u.；没有逐图归一化、对数变换或背景压暗。",
        "",
        "## 决策影响与下一步",
        "",
        "这些图可解释当前假设传播模型的四区路由。P1/P2 可按数值 throughput 与 assumed-process 场景并列展示；不可据此声称独立泛化、材料/打印鲁棒性、实物相机测量或硬件就绪。任何独立人口验证、正式加工、DMD/CMOS 操作或部署均需单独规划和授权；本报告不授予这些许可。",
        "",
        "## 本地状态",
        "",
        "- 输出目录：与本报告同级的独立 C3.6 可视化 run。",
        "- 图件：01 输入 1；02 四类正确样本与首错样本；03 类均值；04 ROI 能量矩阵；05 混淆矩阵；06 P1/P2 候选对照。对应候选分别保存。",
        "- 每候选保存 24 个选定样本的原始 CMOS 强度图及类均值；全人口 2,000 行的 ROI 分数、CMOS/输入/ROI 外能量、标签与预测保存在 roi_energy_scores.npz；样本索引、8 行独立复算和来源信息保存在 arrays/ 与 reports/。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/lightgen/hw207_mnist4_c35_v1.json")
    parser.add_argument("--p1-dir", type=Path, default=DEFAULT_P1_DIR)
    parser.add_argument("--p2-train-dir", type=Path, default=DEFAULT_P2_TRAIN_DIR)
    parser.add_argument("--p2-stl-dir", type=Path, default=DEFAULT_P2_STL_DIR)
    parser.add_argument(
        "--source-p1-dir",
        default="outputs/20261105_hw207_mnist4_c35_v1/runs/autodl_backup_attempt_03/p1_energy_constrained/target_0.400000_seed_11",
    )
    parser.add_argument(
        "--source-p2-train-dir",
        default="outputs/20261105_hw207_mnist4_c35_v1/runs/autodl_backup_attempt_03/p2_process_aware",
    )
    parser.add_argument(
        "--source-p2-stl-dir",
        default="outputs/20261105_hw207_mnist4_c35_v1/runs/autodl_backup_attempt_03/p5_stl/p2_process_aware_height/delta_n_0.5",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20261105)
    parser.add_argument("--alignment-tool", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.device.split(":")[0] != "cuda":
        raise ValueError("use an explicit CUDA device and a positive small batch size")
    for source_path in (args.source_p1_dir, args.source_p2_train_dir, args.source_p2_stl_dir):
        parsed = Path(source_path)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("source provenance paths must be repository-relative")
    device = torch.device(args.device)
    require_matplotlib_panel_alignment = load_alignment_checker(args.alignment_tool.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this frozen production optical forward")
    config = c35_runner._load_config(args.config.resolve())
    rectangles = assert_roi_contract(ROOT / "configs/research/lightgen")
    candidates = load_candidates(
        config, args.p1_dir.resolve(), args.p2_train_dir.resolve(), args.p2_stl_dir.resolve(), device,
    )
    candidates["p1"]["source_artifacts"] = {
        "checkpoint": f"{args.source_p1_dir}/best_model.pt",
        "phase": f"{args.source_p1_dir}/phase_energy_constrained.npy",
        "validation_records": f"{args.source_p1_dir}/validation_records.npz",
    }
    candidates["p2"]["source_artifacts"] = {
        "training_checkpoint": f"{args.source_p2_train_dir}/best_model.pt",
        "training_summary": f"{args.source_p2_train_dir}/c35_process_aware_summary.json",
        "stl": f"{args.source_p2_stl_dir}/p2_process_aware_height_200x200um.stl",
        "height_map": f"{args.source_p2_stl_dir}/height_map.npy",
        "validation_records": f"{args.source_p2_stl_dir}/stl_roundtrip_validation_records.npz",
    }
    images, labels, split = c34_runner._split_and_data()
    rows = split["validation"].detach().cpu().numpy().astype(np.int64)
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    if (
        len(rows) != int(config["data"]["validation_count"])
        or len(split["train"]) != int(config["data"]["train_count"])
        or len(split["holdout"]) != int(config["data"]["previously_used_holdout_count"])
        or not np.array_equal(np.bincount(labels_np[rows], minlength=4), np.full(4, 500))
    ):
        raise ValueError("frozen C3.5 validation population or class counts changed")
    for candidate in candidates.values():
        if (
            not np.array_equal(candidate["records"]["original_rows"], rows)
            or not np.array_equal(candidate["records"]["labels"], labels_np[rows])
            or candidate["records"]["roi_scores"].shape != (len(rows), 4)
        ):
            raise ValueError("saved candidate rows, labels or ROI scores differ from frozen validation")

    input_config = c33.build_configuration(52_000.0, str(device))
    input_config.base.screen.requires_grad_(False)
    cache_rows, fields, cache_metadata = c34_runner._prepare_field_cache(
        images, split["validation"], input_config, device, chunk_size=args.batch_size,
    )
    context = {
        "device": device, "cache": (cache_rows, fields), "config": config,
        "plans": {"p1": {}, "p2": {}},
    }
    preflight_result = preflight(candidates, context, rows, labels_np)
    print(json.dumps({"stage": "production_preflight", **preflight_result}, ensure_ascii=False), flush=True)
    results = {
        key: run_population(key, candidate, context, rows, labels_np, args.batch_size)
        for key, candidate in candidates.items()
    }
    stats = {key: classification_stats(value) for key, value in results.items()}
    selection, random_rows = choose_rows(results, args.random_seed)
    maps = {
        key: collect_maps(
            key, candidates[key], results[key],
            selection["selected_raw_cmos_rows_by_candidate"][key], context,
        )
        for key in candidates
    }
    roi_checks = independent_roi_checks(results, maps, random_rows, rectangles)
    dmd_rows = sorted({
        row for candidate_rows in selection["selected_raw_cmos_rows_by_candidate"].values()
        for row in candidate_rows
    })
    dmd_patterns = collect_dmd_patterns(dmd_rows, images, input_config)
    demo_row = int(selection["input_1_demo_original_row"])
    shared_vmax = max(float(results["p1"]["vmax"]), float(results["p2"]["vmax"]))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    figures_dir = output_dir / "figures"
    arrays_dir = output_dir / "arrays"
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir()
    arrays_dir.mkdir()
    reports_dir.mkdir()
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 8, "axes.linewidth": 0.7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    map_comparison = figure_comparison(
        candidates, results, stats, maps, demo_row, rectangles, shared_vmax,
        figures_dir / "06_c35_candidate_comparison", require_matplotlib_panel_alignment,
    )
    for key in ("p1", "p2"):
        pos = int(np.flatnonzero(results[key]["rows"] == demo_row)[0])
        stats[key]["demo_roi_scores_um2"] = results[key]["roi_scores"][pos].tolist()
        stats[key]["demo_prediction"] = int(results[key]["predictions"][pos])

    for key in ("p2", "p1"):
        candidate = candidates[key]
        result = results[key]
        figure_input_one(
            candidate, result, maps[key], dmd_patterns, images, rectangles, shared_vmax, demo_row,
            figures_dir / ("01_input_1_cmos_demo" if key == "p2" else "01_input_1_cmos_demo_p1"),
            "HW207 MNIST4 · frozen single-layer optical classification",
            require_matplotlib_panel_alignment,
        )
        figure_four_class(
            key, candidate, result, maps[key], selection, images, rectangles,
            shared_vmax, figures_dir / (
                "02_four_class_optical_routing" if key == "p2"
                else "02_four_class_optical_routing_p1"
            ),
            require_matplotlib_panel_alignment,
        )
        figure_failures(
            key, candidate, result, maps[key], selection, images, rectangles,
            shared_vmax, figures_dir / f"02_first_error_cases_{key}",
            require_matplotlib_panel_alignment,
        )
        figure_class_means(
            candidate, result, rectangles, shared_vmax,
            figures_dir / ("03_class_average_cmos" if key == "p2" else "03_class_average_cmos_p1"),
            require_matplotlib_panel_alignment,
        )
        figure_energy(
            candidate, stats[key],
            figures_dir / ("04_roi_energy_matrix" if key == "p2" else "04_roi_energy_matrix_p1"),
            require_matplotlib_panel_alignment,
        )
        figure_confusion(
            candidate, stats[key],
            figures_dir / ("05_confusion_matrix" if key == "p2" else "05_confusion_matrix_p1"),
            require_matplotlib_panel_alignment,
        )
    alignment_paths = sorted(figures_dir.glob("*.alignment.json"))
    if not alignment_paths:
        raise RuntimeError("no render-time Matplotlib panel-alignment reports were produced")
    alignment_results = {}
    for alignment_path in alignment_paths:
        alignment_report = read_json(alignment_path)
        verdict = alignment_report.get("verdict")
        if verdict not in ("PASS", "NOT APPLICABLE"):
            raise RuntimeError(f"panel-alignment report did not pass: {alignment_path.name} ({verdict})")
        alignment_results[alignment_path.name] = {
            "verdict": verdict,
            "summary": alignment_report.get("summary", {}),
        }
    map_payload: dict[str, np.ndarray] = {}
    score_payload: dict[str, np.ndarray] = {}
    for key in ("p1", "p2"):
        selected_rows = np.asarray(sorted(maps[key]), dtype=np.int64)
        positions = {int(row): i for i, row in enumerate(results[key]["rows"])}
        selected_positions = np.asarray([positions[int(row)] for row in selected_rows], dtype=np.int64)
        map_payload[f"{key}_sample_rows"] = selected_rows
        map_payload[f"{key}_sample_labels"] = results[key]["labels"][selected_positions]
        map_payload[f"{key}_sample_predictions"] = results[key]["predictions"][selected_positions]
        map_payload[f"{key}_sample_intensity"] = np.stack([maps[key][int(row)] for row in selected_rows])
        map_payload[f"{key}_class_mean_all"] = results[key]["class_means"][:, 0]
        map_payload[f"{key}_class_mean_correct"] = results[key]["class_means"][:, 1]
        map_payload[f"{key}_class_mean_counts"] = results[key]["class_counts"]
        score_payload[f"{key}_original_rows"] = results[key]["rows"]
        score_payload[f"{key}_labels"] = results[key]["labels"]
        score_payload[f"{key}_predictions"] = results[key]["predictions"]
        score_payload[f"{key}_roi_scores_um2"] = results[key]["roi_scores"]
        score_payload[f"{key}_input_energy_um2"] = results[key]["input_energy_um2"]
        score_payload[f"{key}_cmos_energy_um2"] = results[key]["cmos_energy_um2"]
        score_payload[f"{key}_roi_outside_energy_um2"] = results[key]["roi_outside_energy_um2"]
    save_npz(arrays_dir / "cmos_intensity_maps.npz", **map_payload)
    save_npz(
        arrays_dir / "input_dmd_patterns.npz",
        original_rows=np.asarray(dmd_rows, dtype=np.int64),
        binary_commands=np.stack([dmd_patterns[row] for row in dmd_rows]),
    )
    save_npz(arrays_dir / "roi_energy_scores.npz", **score_payload)
    write_json(arrays_dir / "sample_indices.json", selection)

    clean_stats = {
        key: {name: value for name, value in stats[key].items() if name != "eta_per_sample"}
        for key in ("p1", "p2")
    }
    manifest = {
        "identity": "HW207_C3.6_CMOS_2X2_OPTICAL_CLASSIFICATION_VISUALIZATION",
        "status": "NUMERICAL_DIAGNOSTIC_ONLY",
        "data": {
            "image_source": "MNIST train-images loaded by the frozen C3.4 loader",
            "population": "C3.4 validation rows, reused in C0/C3.4 development; not independent",
            "count": int(len(rows)),
            "class_counts": np.bincount(labels_np[rows], minlength=4).tolist(),
            "original_row_indices_preserved": True,
            "official_test_images_loaded": False,
        },
        "models": {
            key: {
                "name": candidates[key]["name"],
                "checkpoint_path": str(candidates[key]["checkpoint_path"]),
                "frozen_model_path": str(candidates[key]["model_path"]),
                "source_artifacts": candidates[key]["source_artifacts"],
                "seed": candidates[key]["seed"],
                "checkpoint_epoch": candidates[key]["epoch"],
                "process": candidates[key]["process"],
                "selection_and_process": candidates[key]["selection"],
                "saved_validation_records": str(candidates[key]["records_path"]),
                "metrics": clean_stats[key],
                "full_population_forward_check": results[key]["stored_comparison"],
            }
            for key in ("p1", "p2")
        },
        "optics": {
            "forward": "C3.4 Model B scalar Fresnel propagation from the C3.5 production path",
            "wavelength_nm": float(config["wavelength_nm"]),
            "screen_grid": config["geometry"]["phase_grid"],
            "screen_pitch_um": float(config["geometry"]["phase_pitch_um"]),
            "screen_to_lens_mm": float(config["geometry"]["screen_to_lens_mm"]),
            "lens_focal_length_mm": float(config["geometry"]["lens_focal_length_mm"]),
            "lens_to_cmos_mm": float(config["geometry"]["lens_to_cmos_mm"]),
            "aperture_diameter_mm": float(config["geometry"]["aperture_diameter_mm"]),
            "lens_plane_pitch_um": float(config["geometry"]["lens_plane_pitch_um"]),
            "cmos_shape": list(c34.CMOS_SHAPE),
            "cmos_pixel_pitch_um": float(c34.CMOS_PITCH_UM),
            "intensity": "abs(camera_complex_field)**2 in model arbitrary units",
            "absolute_photometry": "not calibrated or measured",
            "input_encoding": "MNIST uint8 threshold >=128, then C3.3 nearest-map to the binary local DMD command window; its area-integrated target-plane map is 100x100 before C3.4 input relay/pupil crop",
            "local_dmd_command_window_shape": list(input_config.base.lattice_shape),
            "target_plane_map_shape": [100, 100],
        },
        "roi": {
            "coordinate_frame": "full CMOS row/column, half-open",
            "bounds": [list(rect) for rect in rectangles],
            "areas_pixels": [
                int((r1 - r0) * (c1_ - c0))
                for r0, r1, c0, c1_ in rectangles
            ],
            "mapping": {
                f"ROI_{i}": {"class": i, "position": ROI_POSITIONS[i]}
                for i in range(4)
            },
            "pixel_area_um2": PIXEL_AREA_UM2,
            "decision": "argmax of four fixed ROI integrals after optical forward",
        },
        "checks": {
            "preflight": preflight_result,
            "figure_panel_alignment": alignment_results,
            "independent_roi_recalculation": roi_checks,
            "roi_mapping": "PASS",
            "gradient_or_optimizer_update": "NONE",
            "physical_dmd_or_cmos_operation": "NONE",
            "orientation": "axis 0 rows top-to-bottom; axis 1 columns left-to-right; no flip/transpose",
            "plot_array_identity": "raw array passed directly to imshow; shared linear scale; no per-image normalization",
            "shared_color_scale": {"normalization": "linear", "vmin": 0.0, "vmax": shared_vmax},
            "input_field_cache": cache_metadata,
            "same_input_comparison": map_comparison,
        },
        "sample_selection": selection,
        "outputs": {
            "figures": sorted(path.name for path in figures_dir.iterdir()),
            "arrays": [
                "arrays/cmos_intensity_maps.npz",
                "arrays/input_dmd_patterns.npz",
                "arrays/roi_energy_scores.npz",
                "arrays/sample_indices.json",
            ],
            "report": "reports/visualization_report.md",
        },
    }
    write_json(reports_dir / "visualization_manifest.json", manifest)
    report_checks = {
        "p1": {"row_count": len(results["p1"]["rows"])},
        "p2": {"row_count": len(results["p2"]["rows"])},
    }
    report = report_text(
        stats, selection, preflight_result, report_checks, rectangles, output_dir,
        map_comparison, shared_vmax,
    )
    (reports_dir / "visualization_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "output_dir": str(output_dir),
        "input_1_original_row": demo_row,
        "candidate_metrics": clean_stats,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
