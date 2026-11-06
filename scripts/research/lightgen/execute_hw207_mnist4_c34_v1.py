"""Execute C3.4 audit, numerical preflight, frozen inference and phase training.

This entrypoint is diagnostic-only.  The model-B geometry is a tape-estimate
scenario; it does not establish HW207 hardware calibration or print readiness.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as c33
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34

DEFAULT_RUN = c34.ROOT / "outputs/20261104_hw207_mnist4_c34_v1/runs/attempt_01"
DEFAULT_C33 = c34.ROOT / "outputs/20261104_hw207_mnist4_c33_v1/runs/attempt_01"


def _select_phase(c33_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any], torch.Tensor, dict[str, Any]]:
    candidates = []
    for run in sorted(c33_root.glob("train_*")):
        summary_path = run / "training_summary.json"
        checkpoint_path = run / "phase.pt"
        if not summary_path.exists() or not checkpoint_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation_row = next((r for r in summary.get("history", [])
                               if int(r["epoch"]) == int(summary["best_epoch"])), None)
        if validation_row is None:
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        phase = checkpoint["phase"].detach().float().cpu()
        fingerprint = c33.tensor_sha256(phase)
        if fingerprint != summary["phase"]["fingerprint"]:
            raise ValueError(f"checkpoint and summary phase mismatch: {run.name}")
        candidates.append((float(summary["validation"]["accuracy"]),
                           float(validation_row["val_loss"]), run.name,
                           checkpoint_path, summary, checkpoint, phase))
    if not candidates:
        raise FileNotFoundError(f"no complete C3.3 phase runs under {c33_root}")
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    best = candidates[0]
    ties = [row[2] for row in candidates if row[0] == best[0]]
    if len(ties) != 1:
        raise ValueError(f"validation selection is not unique: {ties}")
    selection = {"run": best[2], "checkpoint": str(best[3]),
                 "validation_accuracy": best[0], "validation_loss_at_best_epoch": best[1],
                 "best_epoch": int(best[4]["best_epoch"]), "seed": int(best[4]["seed"]),
                 "phase_tensor_sha256": c33.tensor_sha256(best[6]),
                 "selection_rule": "maximum saved validation accuracy, then validation loss; holdout unused",
                 "validation_accuracy_ties": ties}
    return best[3], best[4], best[5], best[6], selection


def _select_c33_fabrication_arm(c33_root: Path, smooth_sigma: float) -> tuple[torch.Tensor, dict[str, Any]]:
    candidates = []
    for run in sorted(c33_root.glob("train_*")):
        summary_path, checkpoint_path = run / "training_summary.json", run / "phase.pt"
        if not summary_path.exists() or not checkpoint_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if float(summary.get("defocus_um", 0.0)) != 52_000.0:
            continue
        if not math.isclose(float(summary.get("smooth_sigma", 0.0)), smooth_sigma, abs_tol=1e-9):
            continue
        entry = next((r for r in summary.get("history", [])
                      if int(r["epoch"]) == int(summary["best_epoch"])), None)
        if entry is None:
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        phase = checkpoint.get("effective", checkpoint["phase"]).detach().float().cpu()
        if phase.shape not in {(100, 100), (1, 1, 100, 100)}:
            raise ValueError(f"C3.3 phase must be a 100x100 screen, got {tuple(phase.shape)}")
        phase = phase.reshape(100, 100)
        candidates.append((float(summary["validation"]["accuracy"]),
                           float(entry["val_loss"]), run.name, phase, summary))
    if not candidates:
        raise FileNotFoundError(f"no C3.3 phase for smooth_sigma={smooth_sigma}")
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    best = candidates[0]
    ties = [r[2] for r in candidates if r[0] == best[0]]
    return best[3], {"run": best[2], "smooth_sigma_px": smooth_sigma,
                     "validation_accuracy": best[0], "validation_loss_at_best_epoch": best[1],
                     "validation_accuracy_ties": ties,
                     "selection_rule": "C33 52mm arm only; validation accuracy then validation loss; no holdout selection",
                     "phase_tensor_sha256": c33.tensor_sha256(best[3])}


def _split_and_data():
    images, labels = c33.load_mnist()
    split = c34.stratified_c34_split(c33.c0_train_indices(), labels)
    for role in ("train", "validation", "holdout"):
        counts = [int((labels[split[role]] == cls).sum()) for cls in range(4)]
        if role != "train" and counts != [500, 500, 500, 500]:
            raise ValueError(f"C3.4 {role} class counts changed: {counts}")
    return images, labels, split


def _balanced_rows(rows: torch.Tensor, labels: torch.Tensor, per_class: int) -> torch.Tensor:
    selected = []
    for cls in range(4):
        part = rows[labels[rows] == cls]
        if len(part) < per_class:
            raise ValueError("not enough samples in one C3.4 class")
        selected.append(part[:per_class])
    return torch.cat(selected)


def _direct_fresnel(field: torch.Tensor, input_pitch_um: float, output_coords_um: list[tuple[float, float]],
                    z_um: float, wavelength_um: float = c34.WAVELENGTH_UM) -> torch.Tensor:
    """Independent direct quadrature used to verify the Bluestein producer."""
    field = field.detach().cpu().to(torch.complex128)
    ny, nx = field.shape
    axis_x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * input_pitch_um
    axis_y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * input_pitch_um
    xx, yy = torch.meshgrid(axis_x, axis_y, indexing="xy")
    results = []
    for xout, yout in output_coords_um:
        phase = math.pi / (wavelength_um * z_um) * ((xout - xx).square() + (yout - yy).square())
        results.append((-1j * input_pitch_um**2 / (wavelength_um * z_um)
                        * (field * torch.exp(1j * phase)).sum()))
    return torch.stack(results)


def _rayleigh_sommerfeld(field: torch.Tensor, pitch_um: float,
                         output_coords_um: list[tuple[float, float]],
                         z_um: float, wavelength_um: float = c34.WAVELENGTH_UM) -> torch.Tensor:
    """Scalar Rayleigh-Sommerfeld samples for independent paraxial checks."""
    field = field.detach().cpu().to(torch.complex128)
    ny, nx = field.shape
    axis_x = (torch.arange(nx, dtype=torch.float64) - (nx - 1) / 2) * pitch_um
    axis_y = (torch.arange(ny, dtype=torch.float64) - (ny - 1) / 2) * pitch_um
    xx, yy = torch.meshgrid(axis_x, axis_y, indexing="xy")
    k = 2 * math.pi / wavelength_um
    results = []
    for xout, yout in output_coords_um:
        rho2 = (xout - xx).square() + (yout - yy).square()
        radius = torch.sqrt(z_um**2 + rho2)
        kernel = (z_um / (1j * wavelength_um * radius.square())) * torch.exp(1j * k * (radius - z_um))
        results.append((field * kernel).sum() * pitch_um**2)
    return torch.stack(results)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def run_p0_p1(args) -> dict[str, Any]:
    run_dir = args.run_dir
    p0_dir = run_dir / "p0_p2"
    p0_dir.mkdir(parents=True, exist_ok=True)
    c33_root = args.c33_root
    audit = c34.audit_c33_artifacts(c33_root)
    c34.write_json(p0_dir / "c33_artifact_audit.json", audit)
    phase_path, phase_summary, checkpoint, phase, selection = _select_phase(c33_root)
    images, labels, split = _split_and_data()
    probe_rows = _balanced_rows(split["validation"], labels, 16)
    probe_images = images[probe_rows]
    device = torch.device(args.device)
    configuration = c33.build_configuration(52_000.0, args.device)
    configuration.base.screen.requires_grad_(False)
    phase = phase.to(device=device, dtype=torch.float32)
    cache: dict[tuple, c34.CZTPlan] = {}

    with torch.no_grad():
        c33_result = c33.forward_field(probe_images[:8].to(device), configuration, phase)
        c34_field = c34.input_field(probe_images[:8].to(device), configuration)
    input_equivalence_abs = float((c33_result["lightgen_input_field"] - c34_field).abs().max())

    tiny = torch.randn((1, 3, 4), dtype=torch.complex128)
    tiny_output = c34.scaled_fresnel(tiny, 2.0, (4, 5), 3.5, 1000.0)
    tiny_points = [((col - 2) * 3.5, (row - 1.5) * 3.5)
                   for row in range(4) for col in range(5)]
    tiny_ref = _direct_fresnel(tiny[0], 2.0, tiny_points, 1000.0).reshape(4, 5)
    czt_direct_error = _relative_l2(tiny_output[0].cpu(), tiny_ref)

    example_field = c34.input_field(probe_images[:1].to(device), configuration)
    zero_phase = torch.zeros((GRID := 100, GRID), device=device)
    random_generator = torch.Generator(device="cpu").manual_seed(207034)
    random_phase = (torch.rand((GRID, GRID), generator=random_generator) * (2 * math.pi)).to(device)
    with torch.no_grad():
        baseline = c34.model_b_forward(example_field, zero_phase, 0.5, lens_pitch_um=4.0,
                                       plan_cache=cache)
        const_shift = c34.model_b_forward(example_field, zero_phase + 1.234, 0.5,
                                          lens_pitch_um=4.0, plan_cache=cache)
        random_result = c34.model_b_forward(example_field, random_phase, 0.5,
                                            lens_pitch_um=4.0, plan_cache=cache)
    base_i = baseline["camera_intensity"]
    shifted_i = const_shift["camera_intensity"]
    random_i = random_result["camera_intensity"]
    assert base_i is not None and shifted_i is not None and random_i is not None
    global_phase_error = float((base_i - shifted_i).abs().sum() /
                               base_i.abs().sum().clamp_min(1e-30))
    random_phase_effect = float((base_i - random_i).abs().sum() /
                                base_i.abs().sum().clamp_min(1e-30))

    aperture_checks = {}
    with torch.no_grad():
        for diameter in (0.5, 1.0, 2.0):
            result = c34.model_b_forward(example_field, phase, diameter, lens_pitch_um=4.0,
                                         plan_cache=cache, return_camera=False)
            aperture_checks[str(diameter)] = {
                "input_energy_um2": float(result["input_energy"].mean()),
                "energy_after_pupil_um2": float(result["lens_energy"].mean()),
                "full_sensor_energy_um2": float(result["detected_energy"].mean()),
                "pupil_throughput": float((result["lens_energy"] / result["input_energy"].clamp_min(1e-30)).mean()),
            }

    sampling_rows = _balanced_rows(split["validation"], labels, 4)
    roi_l1 = []
    crop_intensity_l1 = []
    with torch.no_grad():
        for row in sampling_rows:
            sample = c34.input_field(images[row:row + 1].to(device), configuration)
            fine = c34.model_b_forward(sample, phase, 2.0, lens_pitch_um=2.0,
                                       plan_cache=cache, return_camera=True)
            coarse = c34.model_b_forward(sample, phase, 2.0, lens_pitch_um=4.0,
                                         plan_cache=cache, return_camera=True)
            fine_i, coarse_i = fine["camera_intensity"], coarse["camera_intensity"]
            if fine_i is None or coarse_i is None:
                raise RuntimeError("sampling check did not return CMOS intensities")
            input_e = fine["input_energy"].clamp_min(1e-30)
            roi_l1.append(float((fine["roi_scores"] - coarse["roi_scores"]).abs().sum()
                                / input_e))
            f_crop = fine_i[:, 640:1408, 640:1408]
            c_crop = coarse_i[:, 640:1408, 640:1408]
            crop_intensity_l1.append(float((f_crop - c_crop).abs().sum() * c34.CMOS_PITCH_UM**2
                                           / input_e))
    max_roi_l1_4_vs_2 = max(roi_l1, default=float("inf"))
    max_crop_l1_4_vs_2 = max(crop_intensity_l1, default=float("inf"))
    sampling_pass = max(max_roi_l1_4_vs_2, max_crop_l1_4_vs_2) <= 1e-3
    selected_pitch = 4.0 if sampling_pass else 2.0
    sampling_2_vs_1 = None
    if not sampling_pass:
        sampling_rows_2_vs_1 = _balanced_rows(split["validation"], labels, 1)
        roi_diffs, crop_diffs = [], []
        with torch.no_grad():
            for row in sampling_rows_2_vs_1:
                sample = c34.input_field(images[row:row + 1].to(device), configuration)
                fine = c34.model_b_forward(sample, phase, 2.0, lens_pitch_um=1.0,
                                           plan_cache=cache, return_camera=True)
                coarse = c34.model_b_forward(sample, phase, 2.0, lens_pitch_um=2.0,
                                             plan_cache=cache, return_camera=True)
                fine_i, coarse_i = fine["camera_intensity"], coarse["camera_intensity"]
                if fine_i is None or coarse_i is None:
                    raise RuntimeError("fine sampling reference did not return CMOS intensity")
                input_e = fine["input_energy"].clamp_min(1e-30)
                roi_diffs.append(float((fine["roi_scores"] - coarse["roi_scores"]).abs().sum() / input_e))
                fine_crop = fine_i[:, 640:1408, 640:1408]
                coarse_crop = coarse_i[:, 640:1408, 640:1408]
                crop_diffs.append(float((fine_crop - coarse_crop).abs().sum() * c34.CMOS_PITCH_UM**2 / input_e))
        sampling_2_vs_1 = {"rows": sampling_rows_2_vs_1.tolist(),
                           "max_roi_l1_over_input": max(roi_diffs, default=float("inf")),
                           "max_crop_intensity_l1_over_input": max(crop_diffs, default=float("inf"))}
        sampling_pass = max(sampling_2_vs_1["max_roi_l1_over_input"],
                            sampling_2_vs_1["max_crop_intensity_l1_over_input"]) <= 1e-3
        selected_pitch = 2.0 if sampling_pass else 1.0

    one_cpu = c34.to_complex128(example_field[:1].detach().cpu())
    phase_cpu = phase.detach().cpu().double().reshape(100, 100)
    screen_phase = torch.polar(torch.ones_like(phase_cpu), phase_cpu)
    d_um, lens_pitch = 500.0, 4.0
    pupil_n = int(d_um / lens_pitch)
    p80 = c34.scaled_fresnel(one_cpu * screen_phase,
                             c34.PITCH_UM, (pupil_n, pupil_n), lens_pitch,
                             c34.MODEL_B_D1_UM)[0]
    lens_coord = (torch.arange(pupil_n, dtype=torch.float64) - (pupil_n - 1) / 2) * lens_pitch
    ly, lx = torch.meshgrid(lens_coord, lens_coord, indexing="ij")
    pupil_mask = ((lx.square() + ly.square()) <= (d_um / 2) ** 2).double()
    at_lens = p80 * pupil_mask * torch.exp((-1j * math.pi / (c34.WAVELENGTH_UM * c34.MODEL_B_F_UM))
                                          * (lx.square() + ly.square()))
    p100 = c34.scaled_fresnel(at_lens[None], lens_pitch, c34.CMOS_SHAPE,
                              c34.CMOS_PITCH_UM, c34.MODEL_B_D2_UM)[0]
    p80_positions = [(-248.0, -248.0), (0.0, 0.0), (248.0, 248.0)]
    fresnel80 = c34.scaled_fresnel(
        one_cpu * screen_phase, c34.PITCH_UM,
        (pupil_n, pupil_n), lens_pitch, c34.MODEL_B_D1_UM
    )[0]
    def sample_grid(field_value, pitch, coords):
        out = []
        for x, y in coords:
            col = int(round(x / pitch + (field_value.shape[-1] - 1) / 2))
            row = int(round(y / pitch + (field_value.shape[-2] - 1) / 2))
            out.append(field_value[row, col])
        return torch.stack(out)
    exact80 = _rayleigh_sommerfeld(one_cpu[0] * screen_phase,
                                   c34.PITCH_UM, p80_positions, c34.MODEL_B_D1_UM)
    fresnel80_samples = sample_grid(fresnel80, lens_pitch, p80_positions)
    rs80_error = _relative_l2(fresnel80_samples, exact80)
    roi = c34.fixed_roi_bounds()
    p100_positions = []
    for r0, r1, c0, c1 in roi:
        r, col = (r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2
        x = (col - (c34.CMOS_SHAPE[1] - 1) / 2) * c34.CMOS_PITCH_UM
        y = (r - (c34.CMOS_SHAPE[0] - 1) / 2) * c34.CMOS_PITCH_UM
        p100_positions.append((x, y))
    exact100 = _rayleigh_sommerfeld(at_lens, lens_pitch, p100_positions, c34.MODEL_B_D2_UM)
    fresnel100_samples = sample_grid(p100, c34.CMOS_PITCH_UM, p100_positions)
    rs100_error = _relative_l2(fresnel100_samples, exact100)

    precision_rows = probe_rows[:4]
    precision_records = []
    with torch.no_grad():
        for row in precision_rows.tolist():
            sample = c34.input_field(images[row:row + 1].to(device), configuration)
            f32_result = c34.model_b_forward(sample, phase, 0.5, lens_pitch_um=4.0,
                                             plan_cache=cache)
            f64_result = c34.model_b_forward(c34.to_complex128(sample), phase.double(), 0.5,
                                             lens_pitch_um=4.0)
            f32_field = f32_result["camera_complex_field"].detach().cpu().to(torch.complex128)
            f64_field = f64_result["camera_complex_field"].detach().cpu()
            f32_intensity = f32_field.abs().square()
            f64_intensity = f64_field.abs().square()
            f64_energy = f64_intensity.sum().clamp_min(1e-30)
            f32_score = f32_result["roi_scores"].detach().cpu().double()
            f64_score = f64_result["roi_scores"].detach().cpu().double()
            f32_detected = f32_result["detected_energy"].detach().cpu().double()
            f64_detected = f64_result["detected_energy"].detach().cpu().double()
            f32_prediction = c34.predictions_from_scores(f32_score, f32_detected)
            f64_prediction = c34.predictions_from_scores(f64_score, f64_detected)
            precision_records.append({
                "original_index": int(row),
                "field_relative_l2": _relative_l2(f32_field, f64_field),
                "full_cmos_intensity_relative_l1": float((f32_intensity - f64_intensity).abs().sum() / f64_energy),
                "roi_score_l1_over_fp64_detected_energy": float((f32_score - f64_score).abs().sum() / f64_detected.clamp_min(1e-30)),
                "fp32_roi_winner": int(f32_prediction.item()),
                "fp64_roi_winner": int(f64_prediction.item()),
                "roi_winner_agreement": bool(torch.equal(f32_prediction, f64_prediction)),
            })
    fp32_fp64 = max((row["field_relative_l2"] for row in precision_records), default=float("inf"))
    precision_pass = bool(precision_records and all(
        math.isfinite(row["field_relative_l2"])
        and math.isfinite(row["full_cmos_intensity_relative_l1"])
        and math.isfinite(row["roi_score_l1_over_fp64_detected_energy"])
        and row["roi_winner_agreement"] for row in precision_records
    ))

    distance_sensitivity = {}
    with torch.no_grad():
        for z1 in (70_000.0, 80_000.0, 90_000.0):
            r = c34.model_b_forward(example_field, phase, 0.5, lens_pitch_um=4.0,
                                    plan_cache=cache, return_camera=False,
                                    screen_to_lens_um=z1)
            distance_sensitivity[f"P{int(z1/1000)}L30P100"] = r["roi_scores"][0].cpu().tolist()
        for z2 in (90_000.0, 100_000.0, 110_000.0):
            r = c34.model_b_forward(example_field, phase, 0.5, lens_pitch_um=4.0,
                                    plan_cache=cache, return_camera=False,
                                    lens_to_cmos_um=z2)
            distance_sensitivity[f"P80L30P{int(z2/1000)}"] = r["roi_scores"][0].cpu().tolist()

    report = {
        "identity": "C3.4_OPTICAL_MODEL_AUDIT",
        "session_source_id": c34.SESSION_SOURCE_ID,
        "status": "NUMERICAL_CANDIDATE_MODEL_NOT_HW207_CALIBRATION",
        "phase_source": selection,
        "input_model": "C33/C2.1 binary threshold 128, polygon-area DMD mapping, coherent pupil cutoff 0.1 cycles/um, scalar_binary_approximation",
        "input_field_max_abs_difference_from_C33_forward": input_equivalence_abs,
        "wavelength_nm": 532.0,
        "phase_screen": {"samples": [100, 100], "sample_pitch_um": 2.0,
                         "cell_boundary_extent_um": [200.0, 200.0],
                         "operation": "field times exp(i phase), fixed during P2"},
        "model_a": {"identity": "C33 ideal 4f relay plus 52 mm camera defocus",
                    "magnification": -6.6666666667,
                    "status": "HISTORICAL_NUMERICAL_REFERENCE_NOT_MODEL_B"},
        "model_b": {"sequence": "P(80 mm) L(30 mm) P(100 mm)",
                    "distance_status": "TAPE_ESTIMATE_CENTIMETER_ROUNDED_NOT_CALIBRATED",
                    "lens_clear_aperture_status": "UNKNOWN",
                    "aperture_scenarios_mm": [0.5, 1.0, 2.0],
                    "aperture_status": "ASSUMPTION_ONLY",
                    "abcd": c34.abcd_one_lens(80_000, 30_000, 100_000),
                    "lens_plane_pitch_um": {"tested": [4.0, 2.0] + ([1.0] if sampling_2_vs_1 else []),
                                            "4_vs_2": {"rows": sampling_rows.tolist(),
                                                       "max_roi_l1_over_input": max_roi_l1_4_vs_2,
                                                       "max_crop_intensity_l1_over_input": max_crop_l1_4_vs_2},
                                            "2_vs_1": sampling_2_vs_1,
                                            "tolerance": 1e-3,
                                            "selected_for_next_stage_um": selected_pitch,
                                            "passed": sampling_pass},
                    "finite_pupil_energy_scenarios": aperture_checks,
                    "distance_sensitivity_roi_scores": distance_sensitivity,
                    "fresnel_vs_exact_scalar_RS_reference": {"P80_relative_field_l2": rs80_error,
                                                              "P100_within_fixed_ROI_centers_relative_field_l2": rs100_error,
                                                              "interpretation": "independent scalar Rayleigh-Sommerfeld point checks; paraxial-model approximation, not a hardware measurement"}},
        "numerical_checks": {"Bluestein_vs_direct_fresnel_relative_l2": czt_direct_error,
                             "global_phase_intensity_relative_l1": global_phase_error,
                             "random_phase_intensity_relative_l1_change": random_phase_effect,
                             "FP32_vs_FP64_camera_field_relative_l2_max_over_one_validation_sample_per_class": fp32_fp64,
                             "FP32_vs_FP64_validation_precision_samples": precision_records,
                             "FP32_vs_FP64_roi_decision_stability_pass": precision_pass,
                             "input_mapping_exactly_matches_C33": input_equivalence_abs == 0.0,
                             "sampling_convergence_pass": sampling_pass},
        "coordinate_transform": {"origin": "center between middle CMOS pixels; input and pupil sample centers centered on axis",
                                 "x_axis": "increasing columns to the right",
                                 "y_axis": "increasing rows downward in image arrays; physical coordinate sign tied to row index",
                                 "screen_pitch_um": 2.0, "pupil_pitch_selected_um": selected_pitch,
                                 "CMOS_pitch_um": 6.5, "CMOS_shape": [2048, 2048],
                                 "fixed_crop_offset_px": [640, 640],
                                 "fixed_roi_bounds_full_cmos": [list(x) for x in roi],
                                 "coordinate_mapping_status": "C33_NUMERICAL_MAPPING_ONLY_NOT_HARDWARE_CALIBRATED"},
        "unknown_hardware": json.loads(c34.CONFIG_PATH.read_text(encoding="utf-8"))["unknown_hardware_parameters"],
    }
    report["numerical_pass"] = bool(input_equivalence_abs == 0.0 and czt_direct_error < 1e-10
                                    and global_phase_error < 1e-4 and precision_pass
                                    and sampling_pass and rs80_error < 0.02 and rs100_error < 0.02)
    c34.write_json(p0_dir / "optical_model_audit.json", report)
    c34.write_json(p0_dir / "split.json", {"seed": 20261105,
                                           "roles": {k: {"rows": len(v), "original_indices": v.tolist(),
                                                         "class_counts": [int((labels[v] == cls).sum()) for cls in range(4)]}
                                                     for k, v in split.items()}})
    return report


def _infer_model_a(images: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor,
                   phase: torch.Tensor, device: torch.device, run_dir: Path,
                   batch_size: int = 8) -> dict[str, Any]:
    configuration = c33.build_configuration(52_000.0, str(device))
    configuration.base.screen.requires_grad_(False)
    phase = phase.to(device).detach()
    scores, totals, predictions, class_sum, class_count = [], [], [], np.zeros((4, 768, 768), np.float64), np.zeros(4, np.int64)
    labels_part = labels[rows].long()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk_rows = rows[start:start + batch_size]
            result = c33.forward_field(images[chunk_rows].to(device), configuration, phase)
            sc = result["roi_scores"].detach().cpu().double()
            total = result["detected_energy"].detach().cpu().double()
            scores.append(sc)
            totals.append(total)
            predictions.append(c33.predictions_from_scores(sc, total))
            pixels = result["cmos_pixel_integral"].detach().cpu().numpy()
            for offset, true_label in enumerate(labels_part[start:start + len(chunk_rows)].tolist()):
                class_sum[true_label] += pixels[offset]
                class_count[true_label] += 1
    scores_t, totals_t, preds_t = torch.cat(scores), torch.cat(totals), torch.cat(predictions)
    metrics = c34.classification_metrics(labels_part, preds_t, scores_t, totals_t)
    class_avg = class_sum / np.maximum(class_count[:, None, None], 1)
    return {"scores": scores_t, "detected": totals_t, "crop_energy": totals_t,
            "predictions": preds_t, "metrics": metrics, "class_average": class_avg,
            "class_count": class_count.tolist(), "labels": labels_part,
            "rows": rows}


def _infer_model_b(images: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor,
                   phase: torch.Tensor, aperture_mm: float, lens_pitch_um: float,
                   device: torch.device, batch_size: int, *, save_class_average: bool,
                   cache: dict[tuple, c34.CZTPlan],
                   field_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
    scores, detected, crop_energy, inputs, lens_energy, predictions = [], [], [], [], [], []
    labels_part = labels[rows].long()
    class_sum = np.zeros((4, 768, 768), np.float64) if save_class_average else None
    class_count = np.zeros(4, np.int64) if save_class_average else None
    configuration = c33.build_configuration(52_000.0, str(device))
    configuration.base.screen.requires_grad_(False)
    with torch.no_grad():
        start = 0
        current_batch = batch_size
        while start < len(rows):
            chunk_rows = rows[start:start + current_batch]
            fields = (_fields_for_rows(field_cache, chunk_rows) if field_cache is not None
                      else c34.input_field(images[chunk_rows].to(device), configuration))
            try:
                result = c34.model_b_forward(fields, phase.to(device).detach(), aperture_mm,
                                             lens_pitch_um=lens_pitch_um, plan_cache=cache,
                                             return_camera=save_class_average)
            except torch.cuda.OutOfMemoryError:
                if current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                torch.cuda.empty_cache()
                continue
            score = result["roi_scores"].detach().cpu().double()
            full = result["detected_energy"].detach().cpu().double()
            crop = result["comparison_crop_energy"].detach().cpu().double()
            scores.append(score)
            detected.append(full)
            crop_energy.append(crop)
            inputs.append(result["input_energy"].detach().cpu().double())
            lens_energy.append(result["lens_energy"].detach().cpu().double())
            predictions.append(c34.predictions_from_scores(score, crop))
            if save_class_average:
                pixels = result["camera_intensity"][:, 640:1408, 640:1408].detach().cpu().numpy()
                assert class_sum is not None and class_count is not None
                for offset, true_label in enumerate(labels_part[start:start + len(chunk_rows)].tolist()):
                    class_sum[true_label] += pixels[offset]
                    class_count[true_label] += 1
            start += len(chunk_rows)
            if start % max(current_batch * 50, 1) == 0 or start == len(rows):
                print(f"model-B aperture={aperture_mm:g}mm rows={start}/{len(rows)} batch={current_batch}", flush=True)
    score_t, full_t, crop_t = torch.cat(scores), torch.cat(detected), torch.cat(crop_energy)
    pred_t = torch.cat(predictions)
    metrics = c34.classification_metrics(labels_part, pred_t, score_t, full_t,
                                         fraction_denominator=crop_t)
    metrics["mean_cmos_crop_energy_um2"] = float(crop_t.mean())
    metrics["mean_roi_outside_crop_energy_um2"] = float((crop_t - score_t.sum(1)).mean())
    metrics["mean_full_sensor_efficiency_vs_input"] = float((full_t / torch.cat(inputs).clamp_min(1e-30)).mean())
    metrics["mean_pupil_efficiency_vs_input"] = float((torch.cat(lens_energy) / torch.cat(inputs).clamp_min(1e-30)).mean())
    return {"scores": score_t, "detected": full_t, "crop_energy": crop_t,
            "input_energy": torch.cat(inputs), "lens_energy": torch.cat(lens_energy),
            "predictions": pred_t, "metrics": metrics,
            "class_average": (class_sum / np.maximum(class_count[:, None, None], 1)
                              if class_sum is not None else None),
            "class_count": class_count.tolist() if class_count is not None else None,
            "labels": labels_part, "rows": rows}


def _infer_fields_b(fields: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor,
                    phase: torch.Tensor, aperture_mm: float, lens_pitch_um: float,
                    device: torch.device, batch_size: int,
                    cache: dict[tuple, c34.CZTPlan] | None = None) -> dict[str, Any]:
    cache = {} if cache is None else cache
    scores, detected, crop_energy, inputs, lens_energy, predictions = [], [], [], [], [], []
    labels_part = labels[rows].long()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            field_chunk = fields[start:start + batch_size].to(device)
            result = c34.model_b_forward(field_chunk, phase.to(device).detach(), aperture_mm,
                                         lens_pitch_um=lens_pitch_um, plan_cache=cache,
                                         return_camera=False)
            score = result["roi_scores"].detach().cpu().double()
            full = result["detected_energy"].detach().cpu().double()
            crop = result["comparison_crop_energy"].detach().cpu().double()
            scores.append(score)
            detected.append(full)
            crop_energy.append(crop)
            inputs.append(result["input_energy"].detach().cpu().double())
            lens_energy.append(result["lens_energy"].detach().cpu().double())
            predictions.append(c34.predictions_from_scores(score, crop))
    score_t, full_t, crop_t = torch.cat(scores), torch.cat(detected), torch.cat(crop_energy)
    metrics = c34.classification_metrics(labels_part, torch.cat(predictions), score_t, full_t,
                                         fraction_denominator=crop_t)
    metrics["mean_cmos_crop_energy_um2"] = float(crop_t.mean())
    metrics["mean_roi_outside_crop_energy_um2"] = float((crop_t - score_t.sum(1)).mean())
    metrics["mean_full_sensor_efficiency_vs_input"] = float((full_t / torch.cat(inputs).clamp_min(1e-30)).mean())
    metrics["mean_pupil_efficiency_vs_input"] = float((torch.cat(lens_energy) / torch.cat(inputs).clamp_min(1e-30)).mean())
    return {"scores": score_t, "detected": full_t, "crop_energy": crop_t,
            "input_energy": torch.cat(inputs), "lens_energy": torch.cat(lens_energy),
            "predictions": torch.cat(predictions), "metrics": metrics, "labels": labels_part,
            "rows": rows}


def _prepare_field_cache(images: torch.Tensor, rows: torch.Tensor, configuration,
                         device: torch.device, *, chunk_size: int = 16
                         ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Cache deterministic input-plane fields once for the fixed C3.4 population."""
    if chunk_size < 1:
        raise ValueError("field-cache chunk size must be positive")
    cache_rows = torch.sort(torch.unique(rows.detach().cpu().long())).values.to(device)
    fields = torch.empty((len(cache_rows), 100, 100), dtype=torch.complex64, device=device)
    start_time = time.perf_counter()
    used_chunk = chunk_size
    start = 0
    while start < len(cache_rows):
        end = min(len(cache_rows), start + used_chunk)
        try:
            with torch.no_grad():
                fields[start:end] = c34.input_field(images[cache_rows[start:end].cpu()].to(device),
                                                    configuration)
        except torch.cuda.OutOfMemoryError:
            if used_chunk <= 1:
                raise
            used_chunk = max(1, used_chunk // 2)
            torch.cuda.empty_cache()
            continue
        start = end
        if start % max(used_chunk * 100, 1) == 0 or start == len(cache_rows):
            print(f"input-field cache rows={start}/{len(cache_rows)} chunk={used_chunk}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metadata = {"rows": int(len(cache_rows)), "shape": [int(v) for v in fields.shape],
                "dtype": str(fields.dtype), "bytes": int(fields.numel() * fields.element_size()),
                "chunk_size_used": int(used_chunk),
                "elapsed_seconds": float(time.perf_counter() - start_time),
                "device": str(device),
                "cache_semantics": "fixed C3.4 binary input encoding and relay proxy; no labels or phase"}
    return cache_rows, fields, metadata


def _fields_for_rows(field_cache: tuple[torch.Tensor, torch.Tensor] | None,
                     rows: torch.Tensor) -> torch.Tensor:
    if field_cache is None:
        raise ValueError("field cache is required for indexed lookup")
    cache_rows, fields = field_cache
    positions = torch.searchsorted(cache_rows, rows.to(cache_rows.device, dtype=torch.long))
    if len(positions) and (int(positions.max()) >= len(cache_rows)
                           or not torch.equal(cache_rows[positions], rows.to(cache_rows.device, dtype=torch.long))):
        raise ValueError("requested input rows are absent from the prepared field cache")
    return fields[positions]


def _load_or_prepare_formal_field_cache(images: torch.Tensor, split: dict[str, torch.Tensor],
                                        configuration, device: torch.device,
                                        cache_path: Path | None) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    expected_rows = torch.sort(torch.cat([split["train"], split["validation"], split["holdout"]])).values
    if cache_path is None:
        cache_rows, cache_fields, metadata = _prepare_field_cache(images, expected_rows,
                                                                   configuration, device, chunk_size=16)
        return cache_rows, cache_fields, metadata
    if not cache_path.is_file():
        raise FileNotFoundError(f"explicit C3.4 field cache does not exist: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("identity") != "C34_FIXED_BINARY_INPUT_FIELD_CACHE":
        raise ValueError("field-cache identity is not the fixed C3.4 input preprocessing")
    saved_rows = torch.as_tensor(payload.get("rows"), dtype=torch.long)
    saved_fields = payload.get("fields")
    if (not torch.equal(saved_rows, expected_rows) or not isinstance(saved_fields, torch.Tensor)
            or saved_fields.shape != (len(expected_rows), 100, 100)
            or saved_fields.dtype != torch.complex64 or not torch.isfinite(saved_fields).all()):
        raise ValueError("field cache rows, shape, dtype or values do not match the C3.4 population")
    cache_rows = saved_rows.to(device)
    cache_fields = saved_fields.to(device)
    probe_rows = expected_rows[:2]
    with torch.no_grad():
        direct = c34.input_field(images[probe_rows].to(device), configuration)
        cached = _fields_for_rows((cache_rows, cache_fields), probe_rows)
    if not torch.equal(direct, cached):
        raise ValueError("loaded system-disk field cache differs from the production preprocessing path")
    metadata = {**payload.get("metadata", {}),
                "source_path": str(cache_path.resolve()),
                "loaded_from_system_disk": True,
                "exact_direct_preprocessing_probe": True}
    return cache_rows, cache_fields, metadata


def _shuffle_generator_after_completed_epochs(seed: int, completed_epochs: int,
                                              train_count: int) -> torch.Generator:
    if completed_epochs < 0 or train_count < 1:
        raise ValueError("resume epoch and training population must be nonnegative/positive")
    generator = torch.Generator().manual_seed(seed)
    torch.rand((100, 100), generator=generator)
    for _ in range(completed_epochs):
        torch.randperm(train_count, generator=generator)
    return generator


def _mcnemar_worse_pvalue(correct_a: torch.Tensor, correct_b: torch.Tensor) -> dict[str, Any]:
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    # Exact one-sided Binomial(n, 0.5) upper tail; use integers so the result
    # does not depend on an optional SciPy installation or an approximation.
    p = (sum(math.comb(n, k) for k in range(a_only, n + 1)) / (1 << n)) if n else 1.0
    return {"A_correct_B_wrong": a_only, "A_wrong_B_correct": b_only,
            "discordant_pairs": n, "one_sided_exact_binomial_pvalue_B_worse": p,
            "adaptation_trigger_rule": "Model B at D=2 mm is lower than Model A by paired exact McNemar direction test p<0.05",
            "trigger": bool(p < 0.05)}


def run_p2(args) -> dict[str, Any]:
    p0_dir = args.run_dir / "p0_p2"
    model_audit = json.loads((p0_dir / "optical_model_audit.json").read_text(encoding="utf-8"))
    if not model_audit.get("numerical_pass"):
        raise RuntimeError("C3.4 P1 numerical audit did not pass; P2 is stopped")
    _, _, checkpoint, phase, selection = _select_phase(args.c33_root)
    images, labels, split = _split_and_data()
    c33_holdout = checkpoint["holdout_rows"].long()
    if len(c33_holdout) != 2000:
        raise ValueError("C3.3 historical holdout must contain 2,000 rows")
    device = torch.device(args.device)
    lens_pitch = float(model_audit["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    phase = phase.to(device).detach()
    output = args.run_dir / "p2_frozen_inference"
    output.mkdir(parents=True, exist_ok=False)
    model_a = _infer_model_a(images, labels, c33_holdout, phase, device, output, args.batch_size)
    with torch.no_grad():
        no_phase = _infer_model_b(images, labels, c33_holdout, torch.zeros_like(phase), 2.0,
                                  lens_pitch, device, args.batch_size, save_class_average=False,
                                  cache={})
    b_results = {}
    cache: dict[tuple, c34.CZTPlan] = {}
    for aperture in (0.5, 1.0, 2.0):
        result = _infer_model_b(images, labels, c33_holdout, phase, aperture,
                                lens_pitch, device, args.batch_size,
                                save_class_average=(aperture == 2.0), cache=cache)
        b_results[str(aperture)] = result
        c34.write_json(output / f"model_b_aperture_{aperture:g}mm_metrics.json", result["metrics"])
        np.savez_compressed(output / f"model_b_aperture_{aperture:g}mm_scores.npz",
                            original_indices=c33_holdout.numpy(), labels=labels[c33_holdout].numpy(),
                            predictions=result["predictions"].numpy(), roi_scores=result["scores"].numpy(),
                            full_sensor_energy=result["detected"].numpy(), crop_energy=result["crop_energy"].numpy(),
                            input_energy=result["input_energy"].numpy(), lens_energy=result["lens_energy"].numpy())
    model_a_metrics = model_a["metrics"]
    b_main = b_results["2.0"]
    np.savez_compressed(output / "model_a_fixed_roi_scores.npz",
                        original_indices=c33_holdout.numpy(), labels=model_a["labels"].numpy(),
                        predictions=model_a["predictions"].numpy(), roi_scores=model_a["scores"].numpy(),
                        detected_energy=model_a["detected"].numpy())
    adaptation = _mcnemar_worse_pvalue(
        model_a["predictions"].eq(model_a["labels"]), b_main["predictions"].eq(b_main["labels"])
    )
    example_rows = _balanced_rows(c33_holdout, labels, 1)
    model_a_examples = _infer_model_a(images, labels, example_rows, phase, device, output, batch_size=1)
    model_b_examples = _infer_model_b(images, labels, example_rows, phase, 2.0, lens_pitch,
                                      device, 1, save_class_average=True, cache={})
    np.savez_compressed(output / "model_a_fixed_examples.npz",
                        original_indices=example_rows.numpy(), labels=labels[example_rows].numpy(),
                        class_order=np.arange(4), cmos_crop_by_class=model_a_examples["class_average"].astype(np.float32))
    np.savez_compressed(output / "model_b_2mm_fixed_examples.npz",
                        original_indices=example_rows.numpy(), labels=labels[example_rows].numpy(),
                        class_order=np.arange(4), cmos_crop_by_class=model_b_examples["class_average"].astype(np.float32))
    comparison = {
        "identity": "C3.4_FROZEN_PHASE_PAIRED_INFERENCE",
        "session_source_id": c34.SESSION_SOURCE_ID,
        "phase_source": selection,
        "population": {"role": "C3.3 historical phase_holdout; consumed by C3.3 and used here only for frozen replay/paired diagnosis",
                       "rows": int(len(c33_holdout)),
                       "original_indices": c33_holdout.tolist(),
                       "class_counts": [int((labels[c33_holdout] == cls).sum()) for cls in range(4)]},
        "fixed_example_selection": {"rule": "first row by original holdout order for each class, selected before viewing predictions",
                                    "original_indices": example_rows.tolist(),
                                    "labels": labels[example_rows].tolist()},
        "decision_chain": "frozen binary input -> fixed phase exp(i phi) -> scalar propagation -> CMOS intensity -> fixed C3.3 2x2 ROI -> argmax",
        "model_a_c33": {"metrics": model_a_metrics,
                        "no_phase_accuracy": None,
                        "status": "HISTORICAL_NUMERICAL_MODEL"},
        "model_b_estimated_geometry": {"aperture_assumptions_mm": {key: value["metrics"] for key, value in b_results.items()},
                                       "no_phase_d2mm_metrics": no_phase["metrics"],
                                       "main_assumption_aperture_mm": 2.0,
                                       "lens_plane_pitch_um": lens_pitch,
                                       "status": "SCALAR_FRESNEL_ASSUMPTION_MODEL"},
        "paired_adaptation_decision": adaptation,
        "guardrails": {"phase_frozen": True, "optimizer_constructed": False,
                       "backpropagation_used": False, "digital_head_used": False,
                       "labels_used_in_forward_or_decision": False, "roi_reselected": False,
                       "official_mnist_test_loaded": False},
    }
    c34.write_json(output / "frozen_phase_comparison.json", comparison)
    if model_a["class_average"] is not None:
        np.save(output / "model_a_class_average_cmos_crop.npy", model_a["class_average"].astype(np.float32))
    if b_main["class_average"] is not None:
        np.save(output / "model_b_2mm_class_average_cmos_crop.npy", b_main["class_average"].astype(np.float32))
    torch.save({"phase": phase.cpu(), "run": selection, "phase_requires_grad": phase.requires_grad},
               output / "frozen_phase_provenance.pt")
    return comparison


def _c34_loss(result: dict[str, Any], labels: torch.Tensor, phase: torch.Tensor,
              throughput_floor: float) -> torch.Tensor:
    scores = result["roi_scores"]
    crop = result["comparison_crop_energy"].clamp_min(1e-20)
    logits = 20.0 * scores / crop[:, None]
    ce = F.cross_entropy(logits.float(), labels)
    rows = torch.arange(len(labels), device=labels.device)
    correct = scores[rows, labels]
    wrong = scores.clone()
    wrong[rows, labels] = -float("inf")
    margin = F.relu(1.0 - (correct - wrong.max(1).values) / crop).mean()
    leakage = (result["outside_roi_energy"] / crop).mean()
    throughput = result["detected_energy"] / result["input_energy"].clamp_min(1e-20)
    throughput_loss = F.relu(throughput_floor - throughput).square().mean()
    dx = torch.sin((phase[..., 1:, :] - phase[..., :-1, :]) / 2).abs().mean()
    dy = torch.sin((phase[..., :, 1:] - phase[..., :, :-1]) / 2).abs().mean()
    return ce + 0.5 * margin + 0.05 * leakage + 0.1 * throughput_loss + 1e-3 * (dx + dy)


def _update_rows(phase: torch.nn.Parameter, optimizer, raw_images: torch.Tensor,
                 labels: torch.Tensor, rows: torch.Tensor, configuration,
                 aperture: float, lens_pitch: float, microbatch: int,
                 device: torch.device, cache: dict[tuple, c34.CZTPlan],
                 throughput_floor: float,
                 field_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[float, float]:
    """One logical-batch update with sequential optical microbatches."""
    optimizer.zero_grad(set_to_none=True)
    total = len(rows)
    if total == 0:
        raise ValueError("cannot update from an empty logical batch")
    loss_values = []
    for start in range(0, total, microbatch):
        batch_rows = rows[start:start + microbatch]
        fields = (_fields_for_rows(field_cache, batch_rows) if field_cache is not None
                  else c34.input_field(raw_images[batch_rows].to(device), configuration))
        result = c34.model_b_forward(fields, phase, aperture, lens_pitch_um=lens_pitch,
                                     plan_cache=cache, return_camera=False)
        loss = _c34_loss(result, labels[batch_rows].to(device), phase, throughput_floor)
        (loss * (len(batch_rows) / total)).backward()
        loss_values.append(float(loss.detach()))
    if phase.grad is None or not torch.isfinite(phase.grad).all():
        raise RuntimeError("C3.4 logical-batch gradient is absent or non-finite")
    gradient_norm = float(phase.grad.detach().norm())
    optimizer.step()
    return float(np.mean(loss_values)), gradient_norm


def _evaluate_phase(images: torch.Tensor, labels: torch.Tensor, rows: torch.Tensor,
                    phase: torch.Tensor, aperture: float, lens_pitch: float,
                    device: torch.device, batch_size: int,
                    field_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
    result = _infer_model_b(images, labels, rows, phase.detach(), aperture,
                            lens_pitch, device, batch_size, save_class_average=False, cache={},
                            field_cache=field_cache)
    return result


def run_resource_tune(args) -> dict[str, Any]:
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("C3.4 resource tuning requires the authorized CUDA execution host")
    if args.resume_checkpoint is None or not args.resume_checkpoint.is_file():
        raise FileNotFoundError("resource tuning requires an explicit completed-epoch checkpoint")
    if args.resource_output_dir is None:
        raise ValueError("resource tuning requires an explicit result directory on the system disk")
    output = args.resource_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache_path = args.field_cache_path.resolve() if args.field_cache_path is not None else output / "input_field_cache.pt"
    p1_path = args.run_dir / "p0_p2/optical_model_audit.json"
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    if not p1.get("numerical_pass"):
        raise RuntimeError("resource tuning stopped because the C3.4 numerical preflight did not pass")
    images, labels, split = _split_and_data()
    preflight_path = args.run_dir / "p3_geometry_adapted_training/training_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("resource tuning requires the existing passing C3.4 training preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("preflight_pass") is not True:
        raise RuntimeError("resource tuning stopped because the original training preflight did not pass")
    throughput_floor = float(preflight["throughput_floor"]["efficiency_vs_input"])
    fast_path = args.run_dir / "p3_geometry_adapted_training/fast_train_1000/fast_training_summary.json"
    if not fast_path.is_file() or json.loads(fast_path.read_text(encoding="utf-8")).get("fast_training_pass") is not True:
        raise RuntimeError("resource tuning requires the existing passing 1,000-row fast-training gate")
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("identity") != "C3.4_PHASE_TRAINING_CHECKPOINT":
        raise ValueError("resource checkpoint identity is not a C3.4 epoch checkpoint")
    seed = int(checkpoint["seed"])
    epoch = int(checkpoint["epoch"])
    if (seed not in (1, 2, 3) or not 1 <= epoch <= args.epochs
            or checkpoint.get("session_source_id") != c34.SESSION_SOURCE_ID):
        raise ValueError("resource checkpoint seed, epoch or session identity is invalid")
    train_rows = checkpoint["train_rows"].long()
    validation_rows = checkpoint["validation_rows"].long()
    holdout_rows = checkpoint["holdout_rows"].long()
    if (not torch.equal(train_rows, split["train"]) or not torch.equal(validation_rows, split["validation"])
            or not torch.equal(holdout_rows, split["holdout"])):
        raise ValueError("checkpoint train/validation/holdout rows differ from the fixed C3.4 split")
    train_rows = _balanced_rows(train_rows, labels, 48)
    per_class = [train_rows[labels[train_rows] == cls] for cls in range(4)]
    update_rows = [torch.cat([per_class[cls][step * 16:(step + 1) * 16] for cls in range(4)])
                   for step in range(3)]
    val_rows = _balanced_rows(validation_rows, labels, 16)
    device = torch.device(args.device)
    input_configuration = c33.build_configuration(52_000.0, str(device))
    input_configuration.base.screen.requires_grad_(False)
    if args.field_cache_path is not None:
        cache_rows, cache_fields, cache_metadata = _load_or_prepare_formal_field_cache(
            images, split, input_configuration, device, cache_path)
        cache_file_bytes = int(cache_fields.numel() * cache_fields.element_size())
        available_bytes = int(shutil.disk_usage(cache_path.parent).free)
        cache_metadata["reused_existing_cache_for_resource_preflight"] = True
    else:
        cache_rows_needed = torch.cat([split["train"], split["validation"], split["holdout"]])
        cache_rows, cache_fields, cache_metadata = _prepare_field_cache(
            images, cache_rows_needed, input_configuration, device, chunk_size=16)
        cache_file_bytes = int(cache_fields.numel() * cache_fields.element_size())
        available_bytes = int(shutil.disk_usage(output).free)
        required_bytes = cache_file_bytes + 512 * 1024 * 1024
        if available_bytes < required_bytes:
            raise OSError(f"system-disk space is insufficient for the explicit field cache: "
                          f"need {required_bytes} bytes, have {available_bytes}")
        cache_temporary = cache_path.with_suffix(".pt.tmp")
        torch.save({"identity": "C34_FIXED_BINARY_INPUT_FIELD_CACHE",
                    "rows": cache_rows.detach().cpu(),
                    "fields": cache_fields.detach().cpu(),
                    "metadata": cache_metadata}, cache_temporary)
        cache_temporary.replace(cache_path)
    field_cache = (cache_rows, cache_fields)
    direct_check_rows = update_rows[0][:2]
    with torch.no_grad():
        direct_fields = c34.input_field(images[direct_check_rows].to(device), input_configuration)
        cached_fields = _fields_for_rows(field_cache, direct_check_rows)
    cache_exact = bool(torch.equal(direct_fields, cached_fields))
    if not cache_exact:
        raise RuntimeError("prepared input-field cache differs from the production preprocessing path")
    lens_pitch = float(p1["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    start_phase = torch.nn.Parameter(torch.as_tensor(checkpoint["phase"], dtype=torch.float32,
                                                     device=device).clone())
    saved_optimizer = checkpoint["optimizer"]

    def run_three_updates(microbatch: int, *, use_cache: bool):
        parameter = torch.nn.Parameter(start_phase.detach().clone())
        optimizer = torch.optim.Adam([parameter], lr=0.05)
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        rows_cache = field_cache if use_cache else None
        losses, gradient_norms = [], []
        first_gradient = None
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step_rows in update_rows:
            loss, grad_norm = _update_rows(parameter, optimizer, images, labels, step_rows,
                                           input_configuration, 2.0, lens_pitch, microbatch,
                                           device, {}, throughput_floor,
                                           rows_cache)
            losses.append(loss)
            gradient_norms.append(grad_norm)
            if first_gradient is None:
                first_gradient = parameter.grad.detach().flatten().clone()
            if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError("resource preflight produced an invalid real training update")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        evaluated = _infer_fields_b(_fields_for_rows(field_cache, val_rows), labels, val_rows,
                                    parameter.detach(), 2.0, lens_pitch, device, 1, {})
        return {"parameter": parameter, "optimizer": optimizer,
                "first_gradient": first_gradient, "losses": losses,
                "gradient_norms": gradient_norms, "elapsed_seconds": elapsed,
                "validation": evaluated}

    torch.cuda.reset_peak_memory_stats(device)
    direct_reference = run_three_updates(2, use_cache=False)
    direct_peak_bytes = int(torch.cuda.max_memory_allocated(device))
    reference_parameter = direct_reference["parameter"].detach().clone()
    reference_gradient = direct_reference["first_gradient"].detach().clone()
    reference_validation = direct_reference["validation"]
    direct_summary = {"three_logical_updates_seconds": direct_reference["elapsed_seconds"],
                      "train_losses": direct_reference["losses"],
                      "gradient_norms": direct_reference["gradient_norms"],
                      "peak_allocated_mib": direct_peak_bytes / (1024 ** 2)}
    del direct_reference
    torch.cuda.empty_cache()
    gc.collect()
    candidates = []
    selected_candidate = None
    for microbatch in (2, 4, 8, 16, 32, 64):
        torch.cuda.reset_peak_memory_stats(device)
        candidate = None
        try:
            candidate = run_three_updates(microbatch, use_cache=True)
        except torch.cuda.OutOfMemoryError:
            candidates.append({"microbatch": microbatch, "status": "CUDA_OOM",
                               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))})
            if candidate is not None:
                del candidate
            torch.cuda.empty_cache()
            gc.collect()
            break
        gradient = candidate["first_gradient"]
        grad_cos = float(F.cosine_similarity(reference_gradient, gradient, dim=0))
        grad_rel = float((gradient - reference_gradient).norm()
                         / reference_gradient.norm().clamp_min(1e-30))
        param_rel = float((candidate["parameter"].detach() - reference_parameter).norm()
                          / (reference_parameter - start_phase.detach()).norm().clamp_min(1e-30))
        validation = candidate["validation"]
        score_rel = float((validation["scores"] - reference_validation["scores"]).abs().sum()
                          / reference_validation["scores"].abs().sum().clamp_min(1e-30))
        loss_rel = max(abs(float(left) - float(right)) / max(abs(float(right)), 1e-12)
                       for left, right in zip(candidate["losses"], direct_summary["train_losses"], strict=True))
        prediction_agreement = float((validation["predictions"] == reference_validation["predictions"])
                                      .float().mean())
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        record = {"microbatch": microbatch, "status": "COMPLETED",
                  "three_logical_updates_seconds": candidate["elapsed_seconds"],
                  "logical_updates_per_second": 3.0 / max(candidate["elapsed_seconds"], 1e-12),
                  "train_losses": candidate["losses"], "gradient_norms": candidate["gradient_norms"],
                  "first_update_gradient_cosine_vs_direct_microbatch2": grad_cos,
                  "first_update_gradient_relative_l2_vs_direct_microbatch2": grad_rel,
                  "three_update_parameter_relative_l2_vs_direct_microbatch2": param_rel,
                  "three_update_train_loss_max_relative_error_vs_direct_microbatch2": loss_rel,
                  "validation_roi_score_relative_l1_vs_direct_microbatch2": score_rel,
                  "validation_prediction_agreement_vs_direct_microbatch2": prediction_agreement,
                  "peak_allocated_bytes_including_field_cache": peak_bytes,
                  "peak_allocated_mib_including_field_cache": peak_bytes / (1024 ** 2),
                  "numerical_parity_candidate": bool(grad_cos >= 0.9999 and grad_rel <= 1e-3
                                                      and param_rel <= 1e-2 and loss_rel <= 1e-3
                                                      and score_rel <= 1e-4 and prediction_agreement == 1.0)}
        candidates.append(record)
        if record["numerical_parity_candidate"]:
            if selected_candidate is None or (record["logical_updates_per_second"]
                                               > selected_candidate["logical_updates_per_second"]):
                selected_candidate = record
        del candidate
        torch.cuda.empty_cache()
        gc.collect()
    if selected_candidate is None:
        selected_candidate = next(row for row in candidates
                                  if row.get("microbatch") == 2 and row.get("status") == "COMPLETED")

    eval_candidates = []
    eval_reference = _infer_fields_b(_fields_for_rows(field_cache, val_rows), labels, val_rows,
                                     start_phase.detach(), 2.0, lens_pitch, device, 1, {})
    eval_selected = None
    for batch_size in (1, 2, 4, 8, 16, 32, 64):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            result = _infer_fields_b(_fields_for_rows(field_cache, val_rows), labels, val_rows,
                                     start_phase.detach(), 2.0, lens_pitch, device, batch_size, {})
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            eval_candidates.append({"batch_size": batch_size, "status": "CUDA_OOM",
                                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))})
            torch.cuda.empty_cache()
            gc.collect()
            break
        elapsed = time.perf_counter() - started
        score_rel = float((result["scores"] - eval_reference["scores"]).abs().sum()
                          / eval_reference["scores"].abs().sum().clamp_min(1e-30))
        prediction_agreement = float((result["predictions"] == eval_reference["predictions"])
                                      .float().mean())
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        record = {"batch_size": batch_size, "status": "COMPLETED", "elapsed_seconds": elapsed,
                  "images_per_second": len(val_rows) / max(elapsed, 1e-12),
                  "roi_score_relative_l1_vs_batch1": score_rel,
                  "prediction_agreement_vs_batch1": prediction_agreement,
                  "peak_allocated_mib_including_field_cache": peak_bytes / (1024 ** 2),
                  "numerical_parity_candidate": bool(score_rel <= 1e-4 and prediction_agreement == 1.0)}
        eval_candidates.append(record)
        if record["numerical_parity_candidate"] and (eval_selected is None
                or record["images_per_second"] > eval_selected["images_per_second"]):
            eval_selected = record
        del result
        torch.cuda.empty_cache()
        gc.collect()
    if eval_selected is None:
        eval_selected = next(row for row in eval_candidates
                             if row.get("batch_size") == 1 and row.get("status") == "COMPLETED")
    report = {"identity": "C3.4_CUDA_RESOURCE_PREFLIGHT",
              "status": "ENGINEERING_PREFLIGHT_COMPLETE",
              "session_source_id": c34.SESSION_SOURCE_ID,
              "explicit_checkpoint": str(args.resume_checkpoint.resolve()),
              "checkpoint_seed": seed, "checkpoint_completed_epoch": epoch,
              "logical_update_batch": 64,
              "population": {"role": "fixed C3.4 train and validation only",
                             "three_balanced_train_update_rows": [r.tolist() for r in update_rows],
                             "validation_rows_for_numerical_parity": val_rows.tolist(),
                             "official_mnist_test_loaded": False},
              "input_field_cache": {**cache_metadata,
                                    "exact_equal_to_direct_preprocessing_for_probe": cache_exact,
                                    "system_disk_cache_path": str(cache_path),
                                    "system_disk_cache_bytes": cache_file_bytes,
                                    "system_disk_available_at_preflight_bytes": available_bytes},
              "direct_reference_microbatch2": direct_summary,
              "training_candidates": candidates,
              "selected_microbatch": int(selected_candidate["microbatch"]),
              "selected_train_images_per_second": float(selected_candidate["logical_updates_per_second"]
                                                         * 64),
              "measured_speedup_vs_direct_microbatch2": float(
                  direct_summary["three_logical_updates_seconds"]
                  / max(float(selected_candidate["three_logical_updates_seconds"]), 1e-12)),
              "evaluation_candidates": eval_candidates,
              "selected_evaluation_batch_size": int(eval_selected["batch_size"]),
              "selected_evaluation_images_per_second": float(eval_selected["images_per_second"]),
              "selection_rule": "highest measured throughput among completed candidates that pass the declared gradient/score/prediction parity checks; fallback to microbatch 2 or evaluation batch 1",
              "parity_tolerances": {"first_update_gradient_cosine_minimum": 0.9999,
                                    "first_update_gradient_relative_l2_maximum": 0.001,
                                    "three_update_parameter_relative_l2_maximum": 0.01,
                                    "three_update_train_loss_max_relative_error": 0.001,
                                    "validation_roi_score_relative_l1_maximum": 0.0001,
                                    "validation_prediction_agreement_minimum": 1.0},
              "resume_contract": ["same seed sequence, train/validation/holdout rows, logical update batch, Adam state, epoch limit, patience and loss",
                                  "cache changes deterministic input-field reuse only",
                                  "new microbatch changes only floating-point accumulation grouping; report its parity evidence",
                                  "completed epochs are restored from the explicit epoch_state.pt checkpoint"]}
    c34.write_json(output / "resource_preflight.json", report)
    return report


def _training_preflight_pass(gradient_norm: float, update_records: list[tuple[float, float]],
                             update_l2: float, negative_control_accuracy: float,
                             baseline_efficiency: float) -> bool:
    return bool(math.isfinite(gradient_norm) and gradient_norm > 0
                and math.isfinite(update_l2) and update_l2 > 0
                and math.isfinite(baseline_efficiency) and baseline_efficiency > 0
                and all(math.isfinite(loss) and math.isfinite(norm) and norm > 0
                        for loss, norm in update_records)
                and negative_control_accuracy < 0.40)


def _fast_training_pass(history: list[dict[str, Any]], gradient_norms: list[float]) -> dict[str, bool]:
    if not history:
        return {"loss_decreased": False, "energy_retained": False, "gradients_valid": False,
                "fast_training_pass": False}
    loss_decreased = (math.isfinite(history[0]["train_loss"])
                      and math.isfinite(history[-1]["train_loss"])
                      and history[-1]["train_loss"] < history[0]["train_loss"])
    energy_retained = (math.isfinite(history[-1]["efficiency_fraction_of_zero_phase_validation"])
                       and history[-1]["efficiency_fraction_of_zero_phase_validation"] >= 0.1)
    gradients_valid = bool(gradient_norms and all(math.isfinite(value) and value > 0
                                                   for value in gradient_norms))
    return {"loss_decreased": loss_decreased, "energy_retained": energy_retained,
            "gradients_valid": gradients_valid,
            "fast_training_pass": bool(loss_decreased and energy_retained and gradients_valid)}


def _p2_training_triggered(comparison: dict[str, Any]) -> bool:
    decision = comparison.get("paired_adaptation_decision", {})
    p_value = decision.get("one_sided_exact_binomial_pvalue_B_worse")
    return bool(decision.get("trigger") is True and isinstance(p_value, (int, float))
                and math.isfinite(p_value) and p_value < 0.05)


def run_training(args) -> dict[str, Any]:
    p2_path = args.run_dir / "p0_p2/optical_model_audit.json"
    model_audit = json.loads(p2_path.read_text(encoding="utf-8"))
    if not model_audit.get("numerical_pass"):
        raise RuntimeError("C3.4 P1 numerical audit did not pass; training is stopped")
    p2_path = args.run_dir / "p2_frozen_inference/frozen_phase_comparison.json"
    p2_audit_path = args.run_dir / "p2_frozen_inference/p2_independent_audit.json"
    if not p2_path.is_file() or not p2_audit_path.is_file():
        raise FileNotFoundError("C3.4 P3 requires saved P2 results and their independent audit")
    p2_comparison = json.loads(p2_path.read_text(encoding="utf-8"))
    p2_audit = json.loads(p2_audit_path.read_text(encoding="utf-8"))
    if not p2_audit.get("independent_audit_pass"):
        raise RuntimeError("C3.4 P2 independent decision audit did not pass; training is stopped")
    if not _p2_training_triggered(p2_comparison):
        raise RuntimeError("C3.4 P2 did not trigger geometry adaptation; training is not authorized by this plan")
    images, labels, split = _split_and_data()
    device = torch.device(args.device)
    lens_pitch = float(model_audit["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    train_rows = split["train"]
    val_rows = split["validation"]
    preflight_rows = _balanced_rows(train_rows, labels, 64)
    preflight_rows = preflight_rows[torch.randperm(
        len(preflight_rows), generator=torch.Generator().manual_seed(2070341))]
    cache: dict[tuple, c34.CZTPlan] = {}
    input_configuration = c33.build_configuration(52_000.0, str(device))
    input_configuration.base.screen.requires_grad_(False)
    preflight_dir = args.run_dir / "p3_geometry_adapted_training"
    preflight_path = preflight_dir / "training_preflight.json"

    if args.stage == "preflight":
        init_generator = torch.Generator().manual_seed(207034)
        initial = ((torch.rand((100, 100), generator=init_generator) - 0.5)
                   * (0.2 * math.pi)).to(device)
        zero_phase = torch.zeros_like(initial)
        baseline = _evaluate_phase(images, labels, preflight_rows, zero_phase, 2.0,
                                   lens_pitch, device, args.batch_size)
        zero_phase_efficiency = baseline["metrics"]["mean_full_sensor_efficiency_vs_input"]
        if not math.isfinite(zero_phase_efficiency) or zero_phase_efficiency <= 0:
            raise RuntimeError("training-only zero-phase throughput reference is invalid")
        throughput_floor = 0.1 * zero_phase_efficiency
        phase = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([phase], lr=0.05)
        optimizer.zero_grad(set_to_none=True)
        preflight_losses = []
        for start in range(0, len(preflight_rows), args.microbatch):
            rows = preflight_rows[start:start + args.microbatch]
            result = c34.model_b_forward(c34.input_field(images[rows].to(device), input_configuration),
                                         phase, 2.0, lens_pitch_um=lens_pitch,
                                         plan_cache=cache, return_camera=False)
            loss = _c34_loss(result, labels[rows].to(device), phase, throughput_floor)
            (loss * (len(rows) / len(preflight_rows))).backward()
            preflight_losses.append(float(loss.detach()))
        if phase.grad is None or not torch.isfinite(phase.grad).all():
            raise RuntimeError("256-sample C3.4 gradient preflight failed")
        preflight_gradient_norm = float(phase.grad.detach().norm())
        optimizer.zero_grad(set_to_none=True)
        before = phase.detach().clone()
        update_records = []
        for step in range(3):
            rows = preflight_rows[step * args.update_batch:(step + 1) * args.update_batch]
            update_records.append(_update_rows(phase, optimizer, images, labels, rows,
                                               input_configuration, 2.0, lens_pitch,
                                               args.microbatch, device, cache,
                                               throughput_floor))
        update_l2 = float((phase.detach() - before).norm())
        if not math.isfinite(update_l2) or update_l2 == 0:
            raise RuntimeError("C3.4 real-update preflight did not change phase")

        neg_phase = torch.nn.Parameter(initial.clone())
        neg_optimizer = torch.optim.Adam([neg_phase], lr=0.05)
        neg_rows = preflight_rows[:args.update_batch]
        neg_labels = labels[neg_rows][torch.randperm(
            len(neg_rows), generator=torch.Generator().manual_seed(80034))]
        neg_label_table = labels.clone()
        neg_label_table[neg_rows] = neg_labels
        neg_records = []
        for _ in range(3):
            neg_records.append(_update_rows(neg_phase, neg_optimizer, images, neg_label_table,
                                            neg_rows, input_configuration, 2.0, lens_pitch,
                                            args.microbatch, device, cache, throughput_floor))
        neg_eval_rows = _balanced_rows(val_rows, labels, 64)
        true_eval = _evaluate_phase(images, labels, neg_eval_rows, phase, 2.0,
                                    lens_pitch, device, args.batch_size)
        neg_eval = _evaluate_phase(images, labels, neg_eval_rows, neg_phase, 2.0,
                                   lens_pitch, device, args.batch_size)
        neg_accuracy = neg_eval["metrics"]["accuracy_all_samples"]
        negative_control_pass = neg_accuracy < 0.40
        preflight_pass = _training_preflight_pass(
            preflight_gradient_norm, update_records + neg_records, update_l2,
            neg_accuracy, zero_phase_efficiency
        )
        preflight = {
            "identity": "C3.4_PHASE_TRAINING_PREFLIGHT",
            "session_source_id": c34.SESSION_SOURCE_ID,
            "geometry": "P(80 mm)L(30 mm)P(100 mm), D=2 mm assumption",
            "gradient_population": {"role": "train", "rows": preflight_rows.tolist()},
            "zero_phase_reference": {"role": "train", "rows": preflight_rows.tolist(),
                                     "mean_full_sensor_efficiency_vs_input": zero_phase_efficiency},
            "throughput_floor": {"fraction_of_train_zero_phase_reference": 0.1,
                                  "efficiency_vs_input": throughput_floor,
                                  "purpose": "retain at least 10 percent of the training-set zero-phase full-sensor throughput"},
            "gradient_norm_256_rows": preflight_gradient_norm,
            "mean_loss_256_rows": float(np.mean(preflight_losses)),
            "three_real_updates": [{"loss": row[0], "gradient_norm": row[1]} for row in update_records],
            "phase_parameter_l2_update": update_l2,
            "random_label_negative_control": {"rows": neg_rows.tolist(), "permuted_labels": neg_labels.tolist(),
                                               "validation_rows": neg_eval_rows.tolist(),
                                               "three_updates": [{"loss": row[0], "gradient_norm": row[1]} for row in neg_records],
                                               "true_label_control_accuracy_after_three_updates": true_eval["metrics"]["accuracy_all_samples"],
                                               "shuffled_label_validation_accuracy": neg_accuracy,
                                               "acceptance": "shuffled-label validation accuracy < 0.40 on 64 balanced validation rows",
                                               "passed": negative_control_pass},
            "preflight_pass": preflight_pass,
            "outer_gate": "PRELIGHT_NUMERICAL_AND_OPTIMIZER_PREFLIGHT_PASS" if preflight_pass else "INVALID_HARNESS",
        }
        preflight_dir.mkdir(parents=True, exist_ok=True)
        c34.write_json(preflight_path, preflight)
        return preflight

    if not preflight_path.is_file():
        raise FileNotFoundError("C3.4 P3 requires the saved passing training preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("preflight_pass"):
        raise RuntimeError("C3.4 P3 training preflight did not pass; downstream training is stopped")
    throughput_floor = float(preflight["throughput_floor"]["efficiency_vs_input"])
    neg_eval_rows = torch.tensor(preflight["random_label_negative_control"]["validation_rows"], dtype=torch.long)

    if args.stage == "fast":
        fast_dir = preflight_dir / "fast_train_1000"
        fast_dir.mkdir(parents=True, exist_ok=False)
        init_generator = torch.Generator().manual_seed(207034)
        initial = ((torch.rand((100, 100), generator=init_generator) - 0.5)
                   * (0.2 * math.pi)).to(device)
        fast_phase = torch.nn.Parameter(initial.clone())
        fast_opt = torch.optim.Adam([fast_phase], lr=0.05)
        fast_rows = _balanced_rows(train_rows, labels, 250)
        zero_eval = _evaluate_phase(images, labels, neg_eval_rows, torch.zeros_like(initial),
                                    2.0, lens_pitch, device, args.batch_size)
        zero_efficiency = zero_eval["metrics"]["mean_full_sensor_efficiency_vs_input"]
        fast_history = []
        all_gradient_norms = []
        for epoch in range(3):
            order = torch.randperm(len(fast_rows), generator=torch.Generator().manual_seed(3450 + epoch))
            epoch_losses = []
            for start in range(0, len(order), args.update_batch):
                batch_rows = fast_rows[order[start:start + args.update_batch]]
                loss, grad_norm = _update_rows(fast_phase, fast_opt, images, labels, batch_rows,
                                               input_configuration, 2.0, lens_pitch,
                                               args.microbatch, device, cache, throughput_floor)
                epoch_losses.append(loss)
                all_gradient_norms.append(grad_norm)
            val_small = _evaluate_phase(images, labels, neg_eval_rows, fast_phase, 2.0,
                                        lens_pitch, device, args.batch_size)
            efficiency = val_small["metrics"]["mean_full_sensor_efficiency_vs_input"]
            entry = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)),
                     "validation_accuracy_64": val_small["metrics"]["accuracy_all_samples"],
                     "validation_macro_f1_64": val_small["metrics"]["macro_f1_classes_0_to_3"],
                     "mean_full_sensor_efficiency_vs_input": efficiency,
                     "efficiency_fraction_of_zero_phase_validation": efficiency / max(zero_efficiency, 1e-30),
                     "phase_grad_norm_last": grad_norm}
            fast_history.append(entry)
            print(json.dumps(entry), flush=True)
        fast_gate = _fast_training_pass(fast_history, all_gradient_norms)
        fast_pass = fast_gate["fast_training_pass"]
        fast_report = {"identity": "C3.4_FAST_TRAINING_GATE", "session_source_id": c34.SESSION_SOURCE_ID,
                       "rows": int(len(fast_rows)), "train_rows": fast_rows.tolist(),
                       "class_counts": [int((labels[fast_rows] == cls).sum()) for cls in range(4)],
                       "zero_phase_validation_efficiency": zero_efficiency,
                       "gate_thresholds": {"train_loss_final_below_initial": True,
                                           "minimum_efficiency_fraction_of_zero_phase": 0.1,
                                           "all_gradient_norms_finite_and_positive": True},
                       "history": fast_history, **fast_gate,
                       "fast_training_pass": fast_pass,
                       "outer_gate": "FAST_TRAINING_DIAGNOSTIC_PASS" if fast_pass else "VALID_FAILED"}
        c34.write_json(fast_dir / "fast_training_summary.json", fast_report)
        torch.save({"phase": fast_phase.detach().cpu(), "optimizer": fast_opt.state_dict(),
                    "session_source_id": c34.SESSION_SOURCE_ID}, fast_dir / "fast_phase.pt")
        return fast_report

    fast_summary_path = preflight_dir / "fast_train_1000/fast_training_summary.json"
    if not fast_summary_path.is_file():
        raise FileNotFoundError("C3.4 formal P3 training requires the saved fast-training gate")
    fast_report = json.loads(fast_summary_path.read_text(encoding="utf-8"))
    if not fast_report.get("fast_training_pass"):
        raise RuntimeError("C3.4 fast-training gate did not pass; formal training is stopped")

    formal_dir = preflight_dir / "formal"
    if args.resume_formal:
        if args.resume_checkpoint is None or not args.resume_checkpoint.is_file():
            raise FileNotFoundError("formal resume requires the explicit last completed epoch checkpoint")
        formal_dir.mkdir(parents=True, exist_ok=True)
    else:
        formal_dir.mkdir(parents=True, exist_ok=False)
    field_cache_rows, field_cache_values, field_cache_metadata = _load_or_prepare_formal_field_cache(
        images, split, input_configuration, device, args.field_cache_path)
    field_cache = (field_cache_rows, field_cache_values)
    if args.resume_formal:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        resume_seed = int(resume_checkpoint["seed"])
        resume_seed_dir = formal_dir / f"seed_{resume_seed}"
        if args.resume_checkpoint.resolve() != (resume_seed_dir / "epoch_state.pt").resolve():
            raise ValueError("explicit resume checkpoint does not match its formal seed directory")
        if resume_checkpoint.get("identity") != "C3.4_PHASE_TRAINING_CHECKPOINT":
            raise ValueError("formal resume checkpoint has an unexpected identity")
        if (int(resume_checkpoint["seed"]) not in (1, 2, 3)
                or not 1 <= int(resume_checkpoint["epoch"]) <= args.epochs
                or len(resume_checkpoint["history"]) != int(resume_checkpoint["epoch"])
                or int(resume_checkpoint["best_epoch"]) > int(resume_checkpoint["epoch"])
                or resume_checkpoint["best_phase"] is None
                or resume_checkpoint.get("session_source_id") != c34.SESSION_SOURCE_ID):
            raise ValueError("formal resume checkpoint state is incomplete or inconsistent")
        if (not torch.equal(resume_checkpoint["train_rows"].long(), train_rows)
                or not torch.equal(resume_checkpoint["validation_rows"].long(), val_rows)
                or not torch.equal(resume_checkpoint["holdout_rows"].long(), split["holdout"])):
            raise ValueError("formal resume checkpoint has a different data population or role assignment")
    c34.write_json(args.run_dir / "progress.json", {"status": "RUNNING", "stage": "formal",
                    "current_seed": None, "current_epoch": None,
                    "microbatch": args.microbatch, "evaluation_batch_size": args.batch_size,
                    "input_field_cache_bytes": int(field_cache_values.numel() * field_cache_values.element_size())})
    seed_results = []
    for seed in (1, 2, 3):
        seed_dir = formal_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=args.resume_formal)
        completed_summary = seed_dir / "training_summary.json"
        completed_phase = seed_dir / "phase.pt"
        if args.resume_formal and completed_summary.is_file() and completed_phase.is_file():
            summary = json.loads(completed_summary.read_text(encoding="utf-8"))
            checkpoint = torch.load(completed_phase, map_location="cpu", weights_only=False)
            if (int(summary["seed"]) != seed or int(checkpoint["seed"]) != seed
                    or not math.isfinite(float(summary["validation_accuracy"]))):
                raise ValueError(f"completed formal seed artifacts are inconsistent: seed_{seed}")
            seed_results.append({"seed": seed, "best_epoch": int(summary["best_epoch"]),
                                 "validation_accuracy": float(summary["validation_accuracy"]),
                                 "phase_path": str(completed_phase)})
            continue
        existing_epoch_state = seed_dir / "epoch_state.pt"
        if (args.resume_formal and existing_epoch_state.is_file()
                and seed != resume_seed):
            raise ValueError(f"unbound partial seed checkpoint requires an explicit resume path: seed_{seed}")
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        start_epoch = 0
        if args.resume_formal and seed == resume_seed:
            state = resume_checkpoint
            phase_param = torch.nn.Parameter(state["phase"].to(device=device, dtype=torch.float32).clone())
            opt = torch.optim.Adam([phase_param], lr=0.05)
            opt.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"])
            best_val = float(state["best_validation_accuracy"])
            best_epoch = int(state["best_epoch"])
            best_phase = state["best_phase"].detach().float().cpu().clone()
            history = list(state["history"])
            generator = _shuffle_generator_after_completed_epochs(seed, start_epoch, len(train_rows))
        else:
            phase_param = torch.nn.Parameter(((torch.rand((100, 100), generator=generator) - 0.5)
                                              * (0.2 * math.pi)).to(device))
            opt = torch.optim.Adam([phase_param], lr=0.05)
            best_val = -1.0
            best_epoch = 0
            best_phase = None
            history = []
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            c34.write_json(args.run_dir / "progress.json", {"status": "RUNNING", "stage": "formal",
                            "current_seed": seed, "current_epoch": epoch + 1,
                            "microbatch": args.microbatch, "evaluation_batch_size": args.batch_size,
                            "input_field_cache_bytes": int(field_cache_values.numel()
                                                            * field_cache_values.element_size())})
            order = torch.randperm(len(train_rows), generator=generator)
            losses = []
            for start in range(0, len(order), args.update_batch):
                row_slice = train_rows[order[start:start + args.update_batch]]
                loss, _ = _update_rows(phase_param, opt, images, labels, row_slice,
                                       input_configuration, 2.0, lens_pitch,
                                       args.microbatch, device, cache, throughput_floor,
                                       field_cache)
                losses.append(loss)
            evaluation = _evaluate_phase(images, labels, val_rows, phase_param.detach(), 2.0,
                                         lens_pitch, device, args.batch_size, field_cache)
            val_accuracy = evaluation["metrics"]["accuracy_all_samples"]
            record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                      "validation_accuracy": val_accuracy,
                      "validation_macro_f1": evaluation["metrics"]["macro_f1_classes_0_to_3"],
                      "mean_full_sensor_energy_um2": evaluation["metrics"]["mean_detected_energy_um2"],
                      "elapsed_s": time.time() - epoch_start}
            history.append(record)
            print(json.dumps({"seed": seed, **record}), flush=True)
            if val_accuracy > best_val:
                best_val, best_epoch = val_accuracy, epoch + 1
                best_phase = phase_param.detach().cpu().clone()
            state = {"identity": "C3.4_PHASE_TRAINING_CHECKPOINT", "seed": seed,
                     "epoch": epoch + 1, "phase": phase_param.detach().cpu(),
                     "optimizer": opt.state_dict(), "history": history,
                     "best_phase": best_phase, "best_epoch": best_epoch,
                     "best_validation_accuracy": best_val,
                     "train_rows": train_rows, "validation_rows": val_rows,
                     "holdout_rows": split["holdout"], "session_source_id": c34.SESSION_SOURCE_ID}
            temporary = seed_dir / "epoch_state.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(seed_dir / "epoch_state.pt")
            if epoch + 1 - best_epoch >= args.patience:
                break
        if best_phase is None:
            raise RuntimeError("C3.4 training produced no validation-selected phase")
        torch.save({"phase": best_phase, "seed": seed, "best_epoch": best_epoch,
                    "validation_accuracy": best_val, "train_rows": train_rows,
                    "validation_rows": val_rows, "holdout_rows": split["holdout"],
                    "session_source_id": c34.SESSION_SOURCE_ID}, seed_dir / "phase.pt")
        c34.write_json(seed_dir / "training_summary.json", {"seed": seed, "best_epoch": best_epoch,
                      "validation_accuracy": best_val, "history": history,
                      "train_count": len(train_rows), "validation_count": len(val_rows),
                      "holdout_count_not_opened": len(split["holdout"]), "device": str(device)})
        seed_results.append({"seed": seed, "best_epoch": best_epoch,
                             "validation_accuracy": best_val,
                             "phase_path": str(seed_dir / "phase.pt")})
    seed_results.sort(key=lambda row: (-row["validation_accuracy"], row["seed"]))
    selected = seed_results[0]
    selected_checkpoint = torch.load(selected["phase_path"], map_location="cpu", weights_only=False)
    selected_phase = selected_checkpoint["phase"].to(device)
    holdout_eval = _evaluate_phase(images, labels, split["holdout"], selected_phase, 2.0,
                                   lens_pitch, device, args.batch_size, field_cache)
    report = {"identity": "C3.4_GEOMETRY_ADAPTED_PHASE_TRAINING",
              "session_source_id": c34.SESSION_SOURCE_ID,
              "status": "NUMERICAL_MODEL_ONLY",
              "geometry": "P(80 mm)L(30 mm)P(100 mm); D=2 mm aperture assumption",
              "data_split": {key: {"count": len(value),
                                    "class_counts": [int((labels[value] == cls).sum()) for cls in range(4)]}
                             for key, value in split.items()},
              "seed_results_selected_by_validation_only": seed_results,
              "selected_seed": selected,
              "throughput_floor_from_train_zero_phase_reference": throughput_floor,
              "resource_configuration": {"logical_update_batch": args.update_batch,
                                         "training_microbatch": args.microbatch,
                                         "evaluation_batch_size": args.batch_size,
                                         "fixed_input_field_cache": field_cache_metadata,
                                         "resume_checkpoint": (str(args.resume_checkpoint)
                                                               if args.resume_formal else None)},
              "holdout_opened_once_after_selection": True,
              "holdout_metrics": holdout_eval["metrics"],
              "official_mnist_test_loaded": False,
              "trainable_only": "100x100 phase, no digital head or geometry/ROI/input updates"}
    c34.write_json(formal_dir / "training_summary.json", report)
    np.save(formal_dir / "phase_rad.npy", selected_phase.detach().cpu().numpy())
    np.save(formal_dir / "phase_wrapped.npy", torch.remainder(selected_phase.detach().cpu(), 2 * math.pi).numpy())
    torch.save({"phase": selected_phase.detach().cpu(), "selection": selected,
                "session_source_id": c34.SESSION_SOURCE_ID}, formal_dir / "selected_phase.pt")
    c34.write_json(formal_dir / "phase_metadata.json", {
        "identity": "C3.4_SELECTED_FROZEN_PHASE",
        "phase_shape": [100, 100], "phase_units": "radian",
        "selected_seed": selected["seed"], "best_epoch": selected["best_epoch"],
        "validation_accuracy": selected["validation_accuracy"],
        "selection_role": "C3.4 validation only",
        "geometry": "P(80 mm)L(30 mm)P(100 mm), aperture D=2 mm assumption",
        "physical_mapping": "not calibrated; phase-to-height remains delta_n-parameterized",
        "session_source_id": c34.SESSION_SOURCE_ID})
    c34.write_json(args.run_dir / "progress.json", {"status": "COMPLETED", "stage": "formal",
                    "current_seed": selected["seed"], "current_epoch": selected["best_epoch"],
                    "microbatch": args.microbatch, "evaluation_batch_size": args.batch_size,
                    "holdout_opened_once_after_selection": True})
    return report


def _next_attempt_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = parent / f"attempt_{number:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
            continue
        return candidate


def _load_geometry_adapted_phase(run_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    training_dir = run_dir / "p3_geometry_adapted_training" / "formal"
    summary_path = training_dir / "training_summary.json"
    phase_path = training_dir / "phase_rad.npy"
    checkpoint_path = training_dir / "selected_phase.pt"
    if not all(path.is_file() for path in (summary_path, phase_path, checkpoint_path)):
        raise FileNotFoundError("P4 requires completed P3 formal output and selected phase")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("identity") != "C3.4_GEOMETRY_ADAPTED_PHASE_TRAINING"
            or summary.get("official_mnist_test_loaded") is not False
            or summary.get("holdout_opened_once_after_selection") is not True):
        raise ValueError("P3 formal summary does not satisfy the frozen role contract")
    seed_rows = summary.get("seed_results_selected_by_validation_only", [])
    if len(seed_rows) != 3 or any(not math.isfinite(float(row["validation_accuracy"]))
                                  for row in seed_rows):
        raise ValueError("P4 requires three finite validation-selected P3 seed results")
    selected = summary.get("selected_seed", {})
    if not selected or selected not in seed_rows:
        raise ValueError("P3 selected seed is not one of the recorded validation candidates")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    phase = torch.as_tensor(checkpoint.get("phase"), dtype=torch.float32)
    saved_phase = torch.from_numpy(np.load(phase_path, allow_pickle=False)).float()
    if (phase.shape != (100, 100) or saved_phase.shape != phase.shape
            or not torch.isfinite(phase).all() or not torch.equal(phase, saved_phase)):
        raise ValueError("P3 selected phase tensor is missing, non-finite or inconsistent")
    if checkpoint.get("selection") != selected:
        raise ValueError("P3 selected phase checkpoint and summary selection disagree")
    return phase.detach(), {"selected_seed": selected,
                            "validation_candidates": seed_rows,
                            "source": str(phase_path)}


def _load_fabrication_arms(c33_root: Path, run_dir: Path):
    naive_phase, naive = _select_c33_fabrication_arm(c33_root, 0.0)
    smooth_phase, smooth = _select_c33_fabrication_arm(c33_root, 0.8)
    c34_phase, c34_source = _load_geometry_adapted_phase(run_dir)
    arms = {
        "c33_naive": (naive_phase.detach().float().cpu(), naive),
        "c33_bandwidth_constrained": (smooth_phase.detach().float().cpu(), smooth),
        "c34_geometry_adapted": (c34_phase.detach().float().cpu(), c34_source),
    }
    if any(phase.shape != (100, 100) or not torch.isfinite(phase).all()
           or phase.requires_grad for phase, _ in arms.values()):
        raise ValueError("fabrication comparison phases must be finite, frozen 100x100 arrays")
    return arms


def _fabrication_variants(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostic = configuration["fabrication_diagnostic"]
    variants: list[dict[str, Any]] = [{
        "scenario": "ideal_phase", "psf_sigma_px": 0.0,
        "minimum_feature_radius_px": 0, "height_levels": 0,
        "height_noise_nm": 0.0, "noise_seed": 0, "edge_height_bias_nm": 0.0,
    }]
    for sigma in diagnostic["psf_sensitivity_sigma_px"]:
        if float(sigma) == 0.0:
            continue
        variants.append({"scenario": f"psf_{float(sigma):g}px", "psf_sigma_px": float(sigma),
                         "minimum_feature_radius_px": 0, "height_levels": 0,
                         "height_noise_nm": 0.0, "noise_seed": 0, "edge_height_bias_nm": 0.0})
    for radius in (1, 2):
        variants.append({"scenario": f"minimum_feature_radius_{radius}px",
                         "psf_sigma_px": 0.0, "minimum_feature_radius_px": radius,
                         "height_levels": 0, "height_noise_nm": 0.0,
                         "noise_seed": 0, "edge_height_bias_nm": 0.0})
    seeds = [int(seed) for seed in diagnostic["height_noise_seeds"]]
    for seed in seeds:
        variants.append({"scenario": "height_quantization64_plus_noise20nm",
                         "psf_sigma_px": 0.0, "minimum_feature_radius_px": 0,
                         "height_levels": int(diagnostic["height_levels"]),
                         "height_noise_nm": float(diagnostic["height_noise_sigma_nm"]),
                         "noise_seed": seed, "edge_height_bias_nm": 0.0})
        variants.append({"scenario": "combined_psf0p8_minfeature1_q64_noise20nm",
                         "psf_sigma_px": float(diagnostic["historical_psf_sigma_px"]),
                         "minimum_feature_radius_px": int(diagnostic["minimum_feature_opening_radius_px"]),
                         "height_levels": int(diagnostic["height_levels"]),
                         "height_noise_nm": float(diagnostic["height_noise_sigma_nm"]),
                         "noise_seed": seed, "edge_height_bias_nm": 0.0})
        for edge_bias in diagnostic["edge_height_error_scenarios_nm"]:
            variants.append({"scenario": f"combined_edge_height_bias_{edge_bias:+g}nm",
                             "psf_sigma_px": float(diagnostic["historical_psf_sigma_px"]),
                             "minimum_feature_radius_px": int(diagnostic["minimum_feature_opening_radius_px"]),
                             "height_levels": int(diagnostic["height_levels"]),
                             "height_noise_nm": float(diagnostic["height_noise_sigma_nm"]),
                             "noise_seed": seed, "edge_height_bias_nm": float(edge_bias)})
    return variants


def _check_p4_p6_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    p1_path = run_dir / "p0_p2/optical_model_audit.json"
    p2_path = run_dir / "p2_frozen_inference/frozen_phase_comparison.json"
    p2_audit_path = run_dir / "p2_frozen_inference/p2_independent_audit.json"
    if not all(path.is_file() for path in (p1_path, p2_path, p2_audit_path)):
        raise FileNotFoundError("P4/P6 requires saved P1 and independently audited P2 evidence")
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    p2_audit = json.loads(p2_audit_path.read_text(encoding="utf-8"))
    if not p1.get("numerical_pass") or not p2_audit.get("independent_audit_pass"):
        raise RuntimeError("P4/P6 stopped because upstream numerical/audit evidence did not pass")
    if not _p2_training_triggered(p2):
        raise RuntimeError("P4/P6 P3-dependent arms require the registered P2 adaptation trigger")
    return p1, p2


def run_p4_preflight(args) -> dict[str, Any]:
    p1, _ = _check_p4_p6_inputs(args.run_dir)
    arms = _load_fabrication_arms(args.c33_root, args.run_dir)
    images, labels, split = _split_and_data()
    rows = _balanced_rows(split["validation"], labels, 2)
    device = torch.device(args.device)
    lens_pitch = float(p1["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    configuration = c33.build_configuration(52_000.0, str(device))
    configuration.base.screen.requires_grad_(False)
    with torch.no_grad():
        fields = c34.input_field(images[rows].to(device), configuration)
    baseline_checks = []
    cache: dict[tuple, c34.CZTPlan] = {}
    for arm_name, (phase, source) in arms.items():
        phase_before = phase.clone()
        ideal = _infer_fields_b(fields, labels, rows, phase, 2.0, lens_pitch,
                                device, 1, cache)
        if not torch.isfinite(ideal["scores"]).all():
            raise RuntimeError(f"non-finite preflight ROI scores for {arm_name}")
        for delta_n in (0.5, 0.6):
            height = c34.fabricate_height(phase, delta_n, psf_sigma_px=0.0,
                                          minimum_feature_radius_px=0, height_levels=0)
            phase_recovered = c34.phase_from_height(height, delta_n).float()
            reconstructed = _infer_fields_b(fields, labels, rows, phase_recovered,
                                            2.0, lens_pitch, device, 1, cache)
            if (not torch.isfinite(reconstructed["scores"]).all()
                    or not torch.equal(phase, phase_before)):
                raise RuntimeError(f"phase-freeze/finite-output preflight failed for {arm_name}")
            prediction_agreement = float((ideal["predictions"] == reconstructed["predictions"]).float().mean())
            score_rel_l1 = float((ideal["scores"] - reconstructed["scores"]).abs().sum()
                                 / ideal["scores"].abs().sum().clamp_min(1e-30))
            if prediction_agreement != 1.0 or score_rel_l1 > 2e-5:
                raise RuntimeError(f"phase-height ideal round-trip failed for {arm_name}, delta_n={delta_n}")
            baseline_checks.append({"arm": arm_name, "delta_n": delta_n,
                                    "sample_count": len(rows),
                                    "accuracy": ideal["metrics"]["accuracy_all_samples"],
                                    "roi_score_relative_l1_after_height_round_trip": score_rel_l1,
                                    "prediction_agreement": prediction_agreement,
                                    "phase_frozen": True, "source": source})
    primary_phase, _ = arms["c34_geometry_adapted"]
    primary_height = c34.fabricate_height(primary_phase, 0.5, psf_sigma_px=0.8,
                                          minimum_feature_radius_px=1, height_levels=64,
                                          height_noise_nm=20.0, noise_seed=11,
                                          edge_error_nm=20.0, maximum_height_um=1.2)
    primary_phase_after_process = c34.phase_from_height(primary_height, 0.5).float()
    processed = _infer_fields_b(fields, labels, rows, primary_phase_after_process,
                                2.0, lens_pitch, device, 1, cache)
    if not torch.isfinite(processed["scores"]).all():
        raise RuntimeError("combined assumed-process preflight generated non-finite outputs")
    preflight_parent = (args.p4_output_root / "preflight" if args.p4_output_root is not None
                        else args.run_dir / "p4_p6_preflight")
    output = _next_attempt_dir(preflight_parent)
    report = {"identity": "C3.4_P4_P6_PRODUCTION_PREFLIGHT",
              "session_source_id": c34.SESSION_SOURCE_ID,
              "status": "PASS",
              "population": {"role": "C3.4 validation only; P3 used validation for selection",
                             "rows": rows.tolist(), "labels": labels[rows].tolist(),
                             "class_counts": [int((labels[rows] == cls).sum()) for cls in range(4)]},
              "arms": {key: value[1] for key, value in arms.items()},
              "ideal_height_round_trip_checks": baseline_checks,
              "combined_process_smoke": {"arm": "c34_geometry_adapted", "delta_n": 0.5,
                                          "psf_sigma_px": 0.8, "minimum_feature_radius_px": 1,
                                          "height_levels": 64, "height_noise_nm_sigma": 20.0,
                                          "height_noise_seed": 11, "outer_border_height_bias_nm": 20.0,
                                          "accuracy": processed["metrics"]["accuracy_all_samples"],
                                          "mean_full_sensor_efficiency_vs_input": processed["metrics"]["mean_full_sensor_efficiency_vs_input"]},
              "guardrails": {"phase_frozen": True, "labels_used_only_for_metrics": True,
                             "roi_reselected": False, "official_mnist_test_loaded": False,
                             "all_process_parameters_are_assumption_scenarios": True,
                             "lateral_edge_displacement_resolved": False}}
    c34.write_json(output / "preflight.json", report)
    return report


def _save_phase_and_height_figures(output: Path, phase: torch.Tensor,
                                   height_by_delta: dict[float, tuple[np.ndarray, np.ndarray]]) -> None:
    import matplotlib.pyplot as plt

    phase_array = torch.remainder(phase.detach().cpu(), 2 * math.pi).numpy()
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = ax.imshow(phase_array, cmap="twilight", vmin=0, vmax=2 * math.pi)
    ax.set_title("Selected C3.4 phase (wrapped)")
    ax.set_xlabel("x sample")
    ax.set_ylabel("y sample")
    fig.colorbar(image, ax=ax, label="rad")
    fig.savefig(output / "phase_wrapped.png", dpi=150)
    plt.close(fig)
    for delta_n, (ideal, process) in height_by_delta.items():
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis, values, title in ((axes[0], ideal, "Ideal wrapped-height map"),
                                    (axes[1], process, "Nominal process-constrained height")):
            image = axis.imshow(values, cmap="viridis", vmin=0)
            axis.set_title(title)
            axis.set_xlabel("x sample")
            axis.set_ylabel("y sample")
            fig.colorbar(image, ax=axis, label="height (µm)")
        fig.suptitle(f"Parameterized Δn={delta_n:g} scenario; not material measurement")
        fig.savefig(output / f"height_maps_delta_n_{delta_n:g}.png", dpi=150)
        plt.close(fig)


def _plot_p4_accuracy(output: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    scenarios = ["ideal_phase", "psf_0.8px",
                 "combined_psf0p8_minfeature1_q64_noise20nm"]
    arms = ["c33_naive", "c33_bandwidth_constrained", "c34_geometry_adapted"]
    selected = [row for row in records if row["delta_n_assumption"] == 0.5
                and row["scenario"] in scenarios]
    grouped = {}
    for arm in arms:
        grouped[arm] = {}
        for scenario in scenarios:
            values = [row["metrics"]["accuracy_all_samples"] for row in selected
                      if row["arm"] == arm and row["scenario"] == scenario]
            if values:
                grouped[arm][scenario] = (float(np.mean(values)),
                                          float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(scenarios))
    width = 0.24
    for offset, arm in enumerate(arms):
        means = [grouped[arm].get(s, (np.nan, 0.0))[0] for s in scenarios]
        errors = [grouped[arm].get(s, (np.nan, 0.0))[1] for s in scenarios]
        axis.bar(x + (offset - 1) * width, means, width, yerr=errors, capsize=3, label=arm)
    axis.set_xticks(x, ["ideal", "PSF σ=0.8 px", "combined + 5 noise seeds"])
    axis.set_ylabel("validation accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Matched assumed-process comparison (Δn=0.5 scene)")
    axis.legend()
    fig.savefig(output / "p4_matched_fabrication_accuracy.png", dpi=150)
    plt.close(fig)


def run_p4_p6(args) -> dict[str, Any]:
    p1, _ = _check_p4_p6_inputs(args.run_dir)
    arms = _load_fabrication_arms(args.c33_root, args.run_dir)
    images, labels, split = _split_and_data()
    rows = _balanced_rows(split["validation"], labels, 16)
    device = torch.device(args.device)
    lens_pitch = float(p1["model_b"]["lens_plane_pitch_um"]["selected_for_next_stage_um"])
    process_config = c34.load_config()["fabrication_diagnostic"]
    max_height = float(process_config["maximum_height_um"])
    p4_parent = (args.p4_output_root / "runs" if args.p4_output_root is not None
                 else args.run_dir / "p4_p6")
    output = _next_attempt_dir(p4_parent)
    p4_dir = output / "p4_fabrication_sweep"
    p5_dir = output / "p5_parameterized_stl"
    p6_dir = output / "p6_stl_recovered_simulation"
    p4_dir.mkdir()
    p5_dir.mkdir()
    p6_dir.mkdir()
    configuration = c33.build_configuration(52_000.0, str(device))
    configuration.base.screen.requires_grad_(False)
    with torch.no_grad():
        fields = c34.input_field(images[rows].to(device), configuration)
    variants = _fabrication_variants(c34.load_config())
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    cache: dict[tuple, c34.CZTPlan] = {}
    for arm_name, (phase, source) in arms.items():
        baseline = _infer_fields_b(fields, labels, rows, phase, 2.0, lens_pitch,
                                   device, args.batch_size, cache)
        baselines[arm_name] = baseline
        for delta_n in process_config["delta_n_scenarios"]:
            for variant_number, variant in enumerate(variants):
                if variant["scenario"] == "ideal_phase":
                    height = c34.height_from_phase(phase, float(delta_n))
                    processed_phase = phase.clone()
                else:
                    height = c34.fabricate_height(
                        phase, float(delta_n), psf_sigma_px=variant["psf_sigma_px"],
                        minimum_feature_radius_px=variant["minimum_feature_radius_px"],
                        height_levels=variant["height_levels"],
                        height_noise_nm=variant["height_noise_nm"],
                        noise_seed=variant["noise_seed"],
                        edge_error_nm=variant["edge_height_bias_nm"],
                        maximum_height_um=max_height)
                    processed_phase = c34.phase_from_height(height, float(delta_n)).float()
                if not torch.isfinite(processed_phase).all():
                    raise RuntimeError(f"non-finite reconstructed phase in {arm_name}/{variant}")
                result = _infer_fields_b(fields, labels, rows, processed_phase,
                                         2.0, lens_pitch, device, args.batch_size, cache)
                phase_delta = torch.angle(torch.exp(1j * (processed_phase - phase)))
                phase_rms = float(phase_delta.double().square().mean().sqrt())
                if height.ndim != 2 or not torch.isfinite(height).all():
                    raise RuntimeError(f"invalid height map in {arm_name}/{variant}")
                safe_id = f"r{len(records):04d}"
                arrays[f"{safe_id}_scores"] = result["scores"].numpy()
                arrays[f"{safe_id}_predictions"] = result["predictions"].numpy()
                arrays[f"{safe_id}_detected_energy"] = result["detected"].numpy()
                arrays[f"{safe_id}_crop_energy"] = result["crop_energy"].numpy()
                baseline_prediction = baselines[arm_name]["predictions"]
                record = {"readout_id": safe_id, "arm": arm_name, "arm_source": source,
                          "delta_n_assumption": float(delta_n), **variant,
                          "sample_count": len(rows), "phase_circular_rms_error_rad": phase_rms,
                          "height_range_um": [float(height.min()), float(height.max())],
                          "height_max_neighbor_step_um": float(max(
                              (height[:, 1:] - height[:, :-1]).abs().max(),
                              (height[1:, :] - height[:-1, :]).abs().max())),
                          "height_at_lower_or_upper_clamp_fraction": float(
                              ((height <= 1e-7) | (height >= max_height - 1e-7)).float().mean()),
                          "paired_decision_change_count_vs_same_arm_ideal": int(
                              (result["predictions"] != baseline_prediction).sum()),
                          "paired_decision_agreement_vs_same_arm_ideal": float(
                              (result["predictions"] == baseline_prediction).float().mean()),
                          "metrics": result["metrics"],
                          "process_status": "ASSUMPTION_ONLY_NOT_PRINTER_MEASUREMENT",
                          "edge_error_interpretation": (
                              "vertical outer-border height bias only; lateral edge displacement unresolved at 2 um pitch"
                              if variant["edge_height_bias_nm"] != 0 else "not applied")}
                records.append(record)
    np.savez_compressed(p4_dir / "p4_paired_readouts.npz", **arrays)
    # Summarize repeated random perturbations without treating pixels/images as independent fabrication runs.
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["arm"], record["delta_n_assumption"], record["scenario"])
        grouped.setdefault(key, []).append(record)
    grouped_summary = []
    for (arm, delta_n, scenario), items in grouped.items():
        accuracy = [item["metrics"]["accuracy_all_samples"] for item in items]
        macro_f1 = [item["metrics"]["macro_f1_classes_0_to_3"] for item in items]
        grouped_summary.append({"arm": arm, "delta_n_assumption": delta_n,
                                "scenario": scenario, "perturbation_replicates": len(items),
                                "accuracy_mean": float(np.mean(accuracy)),
                                "accuracy_sd_across_perturbation_seeds": float(np.std(accuracy, ddof=1))
                                if len(accuracy) > 1 else 0.0,
                                "macro_f1_mean": float(np.mean(macro_f1)),
                                "macro_f1_sd_across_perturbation_seeds": float(np.std(macro_f1, ddof=1))
                                if len(macro_f1) > 1 else 0.0})
    p4_report = {"identity": "C3.4_MATCHED_FABRICATION_SENSITIVITY",
                 "status": "NUMERICAL_ASSUMPTION_SCENARIOS",
                 "session_source_id": c34.SESSION_SOURCE_ID,
                 "geometry": "P(80 mm)L(30 mm)P(100 mm), 2 mm aperture scenario; no hardware calibration",
                 "population": {"role": "C3.4 validation; used during P3 seed/epoch selection",
                                "count": len(rows), "class_counts": [int((labels[rows] == cls).sum()) for cls in range(4)],
                                "original_indices": rows.tolist()},
                 "arms": {key: value[1] for key, value in arms.items()},
                 "matched_operator_order": ["wrapped phase to height", "height-domain Gaussian PSF",
                                             "grey-scale minimum-feature opening", "64-level height quantization",
                                             "Gaussian height noise", "outer-border height bias", "height to phase"],
                 "parameter_scenarios": process_config,
                 "variant_count": len(records), "records": records,
                 "grouped_repeated_noise_summary": grouped_summary,
                 "readout_arrays": "p4_paired_readouts.npz; each record readout_id indexes score/prediction/energy arrays",
                 "limitations": ["All PSF, feature, quantization, noise and edge terms are assumed scenarios, not printer measurements.",
                                 "edge_height_error_scenarios_nm are vertical border-height biases; lateral edge displacement is not resolved.",
                                 "Validation rows were used for P3 selection and are not independent test evidence.",
                                 "The assumed material delta_n values do not identify the printed material."]}
    c34.write_json(p4_dir / "p4_fabrication_sweep.json", p4_report)

    selected_phase, selected_source = arms["c34_geometry_adapted"]
    height_by_delta: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    p5_records = []
    p6_records = []
    p6_arrays: dict[str, np.ndarray] = {}
    for delta_n in process_config["delta_n_scenarios"]:
        delta_n = float(delta_n)
        delta_dir = p5_dir / f"delta_n_{delta_n:g}"
        delta_dir.mkdir()
        ideal_height = c34.height_from_phase(selected_phase, delta_n).detach().cpu().numpy().astype(np.float64)
        nominal_height_t = c34.fabricate_height(
            selected_phase, delta_n,
            psf_sigma_px=float(process_config["historical_psf_sigma_px"]),
            minimum_feature_radius_px=int(process_config["minimum_feature_opening_radius_px"]),
            height_levels=int(process_config["height_levels"]),
            height_noise_nm=0.0, noise_seed=0, edge_error_nm=0.0,
            maximum_height_um=max_height)
        nominal_height = nominal_height_t.detach().cpu().numpy().astype(np.float64)
        height_by_delta[delta_n] = (ideal_height, nominal_height)
        np.save(delta_dir / "height_map_ideal_wrapped_phase.npy", ideal_height)
        np.savetxt(delta_dir / "height_map_ideal_wrapped_phase.csv", ideal_height,
                   delimiter=",", fmt="%.9f")
        np.save(delta_dir / "height_map_nominal_process.npy", nominal_height)
        np.savetxt(delta_dir / "height_map_nominal_process.csv", nominal_height,
                   delimiter=",", fmt="%.9f")
        ideal_mesh_dir = delta_dir / "stl_ideal_parameterized_diagnostic"
        nominal_mesh_dir = delta_dir / "stl_nominal_process_parameterized_diagnostic"
        mesh_records = []
        for mesh_dir, height, design_name in (
                (ideal_mesh_dir, ideal_height, "ideal_wrapped_phase"),
                (nominal_mesh_dir, nominal_height, "nominal_process_no_random_noise")):
            mesh_dir.mkdir()
            triangles, mesh_metadata = c34.mesh_200um(height, substrate_um=1.2)
            stl_path = mesh_dir / "phase_mask_parameterized_diagnostic.stl"
            c34.write_binary_stl(stl_path, triangles)
            c34.write_json(mesh_dir / "stl_metadata.json", {
                "identity": "C3.4_PARAMETERIZED_DIAGNOSTIC_ONLY",
                "unit": "um", "delta_n_assumption": delta_n,
                "wavelength_um": c34.WAVELENGTH_UM,
                "phase_source": selected_source,
                "design": design_name,
                "surface_mesh": mesh_metadata,
                "substrate_thickness_um": 1.2,
                "substrate_thickness_status": "geometric placeholder, not measured or printer specified",
                "material_identity": "UNKNOWN", "fabrication_ready": False})
            independent_audit = c34.audit_stl_independent(stl_path)
            recovered_height = c34.recover_height_from_mesh_vertices(stl_path)
            recovery_error = float(np.max(np.abs(recovered_height - height)))
            if (independent_audit["triangles"] != mesh_metadata["triangles"]
                    or not independent_audit["closed_two_manifold_by_edge_incidence"]
                    or not independent_audit["opposite_directed_edge_pairing"]
                    or independent_audit["declared_unit_status"] != "SIDECAR_DECLARES_UM"
                    or recovery_error > 6e-6):
                raise RuntimeError(f"P5 independent STL audit or height recovery failed: {stl_path.name}")
            mesh_record = {"design": design_name, "stl_path": str(stl_path),
                           "mesh_generation_metadata": mesh_metadata,
                           "independent_stl_audit": independent_audit,
                           "center_height_max_abs_recovery_error_um": recovery_error,
                           "status": "PARAMETERIZED_DIAGNOSTIC_ONLY"}
            c34.write_json(mesh_dir / "stl_independent_audit.json", mesh_record)
            mesh_records.append(mesh_record)
            recovered_phase = torch.as_tensor(c34.phase_from_height(recovered_height, delta_n),
                                              dtype=selected_phase.dtype, device=device)
            process_phase = torch.as_tensor(c34.phase_from_height(height, delta_n),
                                             dtype=selected_phase.dtype, device=device)
            if design_name.startswith("ideal"):
                printed_eval = _infer_fields_b(fields, labels, rows, recovered_phase,
                                               2.0, lens_pitch, device, args.batch_size, cache)
                phase_kind = "ideal_height_and_STL_round_trip"
            else:
                process_eval = _infer_fields_b(fields, labels, rows, process_phase,
                                               2.0, lens_pitch, device, args.batch_size, cache)
                printed_eval = _infer_fields_b(fields, labels, rows, recovered_phase,
                                               2.0, lens_pitch, device, args.batch_size, cache)
                phase_kind = "nominal_process_height_and_STL_round_trip"
                process_error = float(torch.angle(torch.exp(1j * (process_phase - recovered_phase)))
                                      .double().abs().max())
                p6_records.append({"delta_n_assumption": delta_n,
                                   "phase_source": phase_kind,
                                   "process_phase_metrics": process_eval["metrics"],
                                   "stl_recovered_phase_metrics": printed_eval["metrics"],
                                   "stl_recovered_minus_process_max_circular_phase_error_rad": process_error,
                                   "stl_float32_height_recovery_max_abs_error_um": recovery_error,
                                   "paired_prediction_agreement_process_vs_stl": float(
                                       (process_eval["predictions"] == printed_eval["predictions"]).float().mean()),
                                   "assumption_only": True})
            key = f"dn{delta_n:g}_{design_name}"
            p6_arrays[f"{key}_scores"] = printed_eval["scores"].numpy()
            p6_arrays[f"{key}_predictions"] = printed_eval["predictions"].numpy()
            p6_arrays[f"{key}_detected_energy"] = printed_eval["detected"].numpy()
            p6_records.append({"delta_n_assumption": delta_n, "design": design_name,
                               "phase_kind": phase_kind,
                               "metrics": printed_eval["metrics"],
                               "phase_source": selected_source,
                               "readout_key": key,
                               "stl_round_trip_height_error_um": recovery_error,
                               "assumption_only": True})
        p5_records.append({"delta_n_assumption": delta_n,
                           "h_2pi_um": c34.WAVELENGTH_UM / delta_n,
                           "height_branch": "wrapped minimum height; phase mod 2pi; k=0",
                           "footprint_um": [200.0, 200.0],
                           "sample_centers_um_first_last": [1.0, 199.0],
                           "physical_boundary_um": [0.0, 200.0],
                           "nominal_process_operator": {"psf_sigma_px": 0.8,
                                                         "minimum_feature_radius_px": 1,
                                                         "height_levels": 64,
                                                         "height_noise_applied_to_export": False,
                                                         "edge_height_bias_applied_to_export": False},
                           "material_identity": "UNKNOWN",
                           "status": "PARAMETERIZED_DIAGNOSTIC_ONLY",
                           "mesh_records": mesh_records})
    np.savez_compressed(p6_dir / "p6_stl_recovered_readouts.npz", **p6_arrays)
    c34.write_json(p5_dir / "p5_height_stl_report.json", {
        "identity": "C3.4_PHASE_TO_HEIGHT_STL_GEOMETRY",
        "status": "PARAMETERIZED_DIAGNOSTIC_ONLY",
        "phase_source": selected_source,
        "material_and_printer_parameters": "UNKNOWN",
        "fabrication_ready": False,
        "delta_n_scenarios": p5_records,
        "independent_triangle_triangle_self_intersection_test": "NOT_RUN",
        "geometry_statement": "200x200 um cell-boundary footprint; 100x100 centers span 198 um",
        "unit_statement": "STL has no intrinsic unit; matching JSON sidecars declare um"})
    c34.write_json(p6_dir / "p6_printed_structure_resimulation.json", {
        "identity": "C3.4_STL_RECOVERED_PARAMETERIZED_STRUCTURE_RESIMULATION",
        "status": "NUMERICAL_ASSUMPTION_SCENARIOS",
        "population": {"role": "C3.4 validation; used during P3 selection",
                       "count": len(rows), "class_counts": [int((labels[rows] == cls).sum()) for cls in range(4)],
                       "original_indices": rows.tolist()},
        "fixed_geometry": "P(80 mm)L(30 mm)P(100 mm), D=2 mm assumption; fixed C3.3 ROI",
        "records": p6_records,
        "limitations": ["The recovered structure uses assumed delta_n and does not represent a measured printed device.",
                        "CMOS noise/QE/gain, actual lens pupil and measured relay mapping are not modeled.",
                        "Validation data were used for phase selection; no independent generalization claim is made."]})
    _save_phase_and_height_figures(p5_dir, selected_phase, height_by_delta)
    _plot_p4_accuracy(p4_dir, records)
    report = {"identity": "C3.4_P4_P5_P6_CLOSEOUT",
              "status": "NUMERICAL_ASSUMPTION_SCENARIOS_COMPLETE",
              "session_source_id": c34.SESSION_SOURCE_ID,
              "run_dir": str(output),
              "p4_variant_count": len(records), "p4_report": str(p4_dir / "p4_fabrication_sweep.json"),
              "p5_report": str(p5_dir / "p5_height_stl_report.json"),
              "p6_report": str(p6_dir / "p6_printed_structure_resimulation.json"),
              "sample_count": len(rows), "all_phase_screens_frozen": True,
              "material_printer_identity": "UNKNOWN",
              "fabrication_ready": False,
              "p2_historical_population_use": "C3.3 holdout was used for frozen replay/paired diagnosis; P4-P6 use C3.4 validation, which was used for P3 selection."}
    c34.write_json(output / "p4_p6_closeout.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("p0p1", "p2", "preflight", "fast", "formal",
                                             "resource_tune", "p4_preflight", "p4_p6"), required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--c33-root", type=Path, default=DEFAULT_C33)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-formal", action="store_true")
    parser.add_argument("--resource-output-dir", type=Path)
    parser.add_argument("--field-cache-path", type=Path)
    parser.add_argument("--p4-output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="CMOS inference microbatch; reduce on CUDA allocation failure")
    parser.add_argument("--update-batch", type=int, default=64,
                        help="logical Adam batch for C3.4 phase optimization")
    parser.add_argument("--microbatch", type=int, default=2,
                        help="gradient microbatch within the fixed logical Adam batch")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.c33_root = args.c33_root.resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.resource_output_dir is not None:
        args.resource_output_dir = args.resource_output_dir.resolve()
    if args.field_cache_path is not None:
        args.field_cache_path = args.field_cache_path.resolve()
    if args.p4_output_root is not None:
        args.p4_output_root = args.p4_output_root.resolve()
    if args.stage == "p0p1":
        report = run_p0_p1(args)
    elif args.stage == "p2":
        report = run_p2(args)
    elif args.stage == "resource_tune":
        report = run_resource_tune(args)
    elif args.stage == "p4_preflight":
        report = run_p4_preflight(args)
    elif args.stage == "p4_p6":
        report = run_p4_p6(args)
    elif args.stage in {"preflight", "fast", "formal"}:
        report = run_training(args)
    else:
        raise ValueError(args.stage)
    print(json.dumps({"stage": args.stage, "status": report.get("status", report.get("numerical_pass")),
                      "run_dir": str(args.run_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
