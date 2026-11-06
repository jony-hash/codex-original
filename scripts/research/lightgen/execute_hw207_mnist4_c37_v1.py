"""HW207 MNIST4 C3.7: single layer phase focus + CMOS four-ROI joint optimisation.

Sub-commands implement the C3.7 work packages in one shared numerical path:

``bench``       forward/backward timing probe (no science record)
``corecheck``   freeze the four target cores and prove containment/consistency
``p0``          C3.6 spot breakpoint diagnosis, coordinate audit, plane sensitivity
``gradcheck``   autograd vs finite difference and the minimal gradient validation
``ablate``      E1..E3 small-population screening
``train``       formal fine-tune of one arm and one seed
``fabricate``   P4 common manufacturing comparison
``vizdata``     export the arrays consumed by the P5 visualisation entry point

Every stage writes machine readable JSON next to its arrays.  The frozen phase
of the selected candidate is always stored as a raw 100x100 ``.npy`` matrix.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34
from scripts.research.lightgen import hw207_mnist4_c34_process_aware_v1 as c34_process
from scripts.research.lightgen import hw207_mnist4_c37_v1 as c37

ROOT = c37.ROOT


# --------------------------------------------------------------------------- #
# argument handling
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["bench", "corecheck", "p0", "gradcheck", "ablate",
                                          "train", "finalize", "select", "spotlimit", "oracle", "fabricate", "vizdata",
                                          "accept"])
    parser.add_argument("--config", type=Path, default=c37.DEFAULT_CONFIG)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--init-phase", type=Path, default=None)
    parser.add_argument("--p2-height", type=Path, default=None)
    parser.add_argument("--reference-phase", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-class", type=int, default=None)
    parser.add_argument("--eval-per-class", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--focus-weight", type=float, default=None)
    parser.add_argument("--focus-mode", default=None,
                        choices=["mean", "worst_class"])
    parser.add_argument("--eta-retention", type=float, default=None)
    parser.add_argument("--core-fraction", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--arm", default=None, help="ablation arm name, e.g. E2")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "holdout"])
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--candidates", default=None,
                        help="comma separated candidate list for fabricate/vizdata")
    parser.add_argument("--tag", default=None)
    return parser


def resolve_init_phase(args, config: dict[str, Any]) -> Path:
    if args.init_phase is not None:
        return Path(args.init_phase)
    return ROOT / config["init"]["p1_phase"]


def resolve_p2_references(args, config: dict[str, Any]) -> tuple[Path, Path]:
    height = Path(args.p2_height) if args.p2_height is not None else ROOT / config["init"]["p2_height_reference"]
    phase = ROOT / config["init"]["p2_phase_reference"]
    return height, phase


def resolve_reference_phase(args, config: dict[str, Any]) -> Path:
    """Frozen C3.5 P1 phase for reference arms, independent of the fine-tune initialiser."""
    if args.reference_phase is not None:
        return Path(args.reference_phase)
    return ROOT / config["init"]["p1_phase"]


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def stage_bench(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    device = context["device"]
    phase = c37.load_frozen_phase(resolve_init_phase(args, config)).to(device)
    rows = c37.balanced_rows(context["split"]["train"], context["labels"], 4)[:32]
    records = []
    c37.prepare_field_cache(context, rows)
    for batch in (1, 2, 4, 8):
        for _ in range(2):
            with torch.no_grad():
                c37.forward_batch(c37.input_fields(context, rows[:batch]), phase, context)
        torch.cuda.synchronize()
        start = time.perf_counter()
        repeats = 5
        for _ in range(repeats):
            with torch.no_grad():
                c37.forward_batch(c37.input_fields(context, rows[:batch]), phase, context)
        torch.cuda.synchronize()
        forward_seconds = (time.perf_counter() - start) / repeats
        variable = torch.nn.Parameter(phase.clone())
        optimizer = torch.optim.Adam([variable], lr=0.0)
        masks = c37.core_masks(float(config["core"]["frozen_fraction"]), device)
        training_seconds = None
        try:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = c37.forward_batch(c37.input_fields(context, rows[:batch]), variable,
                                           context)
                labels = context["labels"][rows[:batch]].to(device)
                loss, _ = c37.objective(result, labels, variable, context, config,
                                        groups=["focus", "sidelobe", "throughput", "spread"],
                                        eta_target=0.0, multiplier=0.0, masks=masks)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                result = c37.forward_batch(c37.input_fields(context, rows[:batch]), variable,
                                           context)
                loss, _ = c37.objective(result, labels, variable, context, config,
                                        groups=["focus", "sidelobe", "throughput", "spread"],
                                        eta_target=0.0, multiplier=0.0, masks=masks)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            training_seconds = (time.perf_counter() - start) / 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        records.append({
            "batch": batch,
            "forward_seconds_per_batch": forward_seconds,
            "forward_seconds_per_sample": forward_seconds / batch,
            "train_seconds_per_batch": training_seconds,
            "train_seconds_per_sample": (training_seconds / batch) if training_seconds else None,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20),
        })
        del variable, optimizer
        torch.cuda.empty_cache()
    payload = {"stage": "bench", "device": str(device),
               "device_name": torch.cuda.get_device_name(0), "records": records}
    c37.write_json(args.work_dir / "c37_bench.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# corecheck
# --------------------------------------------------------------------------- #
def stage_corecheck(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    fraction = float(config["core"]["frozen_fraction"])
    spec = c37.core_spec(fraction)
    rois = np.array(c37.roi_bounds())
    cores = np.array([record["core_bounds_px"] for record in spec["records"]])
    checks: dict[str, Any] = {}
    checks["four_cores_inside_own_roi"] = bool(all(
        roi[0] <= core[0] and core[1] <= roi[1] and roi[2] <= core[2] and core[3] <= roi[3]
        for roi, core in zip(rois, cores)))
    checks["cores_do_not_overlap"] = bool(len({tuple(core) for core in cores}) == 4)
    checks["cores_have_identical_side_px"] = bool(len({int(c[1] - c[0]) for c in cores}) == 1)
    checks["cores_are_centred_on_roi_centres"] = bool(np.array_equal(
        np.stack([[(c[0] + c[1] - 1) / 2, (c[2] + c[3] - 1) / 2] for c in cores]),
        c37.roi_center_px(rois)))
    masks = c37.core_masks(fraction, "cpu")
    checks["core_mask_area_matches_spec"] = bool(all(
        int(masks[index].sum().item()) == (cores[index][1] - cores[index][0]) * (cores[index][3] - cores[index][2])
        for index in range(4)))
    checks["core_masks_pairwise_disjoint"] = bool(float((masks.sum(0) > 1).sum()) == 0)
    roi_stack = c37.roi_masks("cpu")
    checks["roi_masks_pairwise_disjoint"] = bool(float((roi_stack.sum(0) > 1).sum()) == 0)
    checks["core_masks_inside_roi_masks"] = bool(float(((masks.sum(0) > 0) & (roi_stack.sum(0) == 0)).sum()) == 0)
    checks["roi_block_centred_on_optical_axis"] = bool(np.allclose(
        c37.roi_center_px(rois).mean(0), [c37.FRAME_CENTER_PX, c37.FRAME_CENTER_PX]))

    # diffraction limited scale of this geometry for the declared core sizes
    wavelength_um = 532e-3
    na = (float(config["geometry"]["aperture_diameter_mm"]) * 1000.0 / 2.0) / (
        float(config["geometry"]["lens_to_cmos_mm"]) * 1000.0)
    fwhm_um = 0.51 * wavelength_um / na
    candidates = []
    for candidate in config["core"]["diagnostic_candidates_fraction"]:
        side = c37.core_side_px(float(candidate))
        candidates.append({
            "fraction": float(candidate),
            "side_px": int(side),
            "side_um": float(side * c37.CMOS_PITCH_UM),
            "side_in_diffraction_fwhm": float(side * c37.CMOS_PITCH_UM / fwhm_um),
            "frozen": bool(abs(float(candidate) - fraction) < 1e-12),
        })
    payload = {
        "stage": "corecheck",
        "identity": "C3.7_TARGET_CORE_DEFINED",
        "frozen_fraction": fraction,
        "spec": spec,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "diffraction_limit": {
            "numerical_na": float(na),
            "fwhm_um": float(fwhm_um),
            "fwhm_px": float(fwhm_um / c37.CMOS_PITCH_UM),
            "note": "geometric estimate from the frozen Model B aperture and lens-to-CMOS distance; "
                    "not a measured point spread function",
        },
        "candidates": candidates,
    }
    c37.write_json(args.work_dir / "c37_core_spec.json", payload)
    print(json.dumps({"all_checks_passed": payload["all_checks_passed"], "checks": checks,
                      "diffraction_limit": payload["diffraction_limit"]}, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# P0 spot diagnosis
# --------------------------------------------------------------------------- #
def _coordinate_audit(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Prove the row/column to y/x convention with asymmetric phase ramps.

    A uniform plane wave is used as the probe so that the deflection direction is
    not contaminated by the asymmetric MNIST input pattern.
    """
    device = context["device"]
    probe = torch.ones((1, c37.GRID, c37.GRID), dtype=torch.complex64, device=device)
    axis = torch.arange(c37.GRID, device=device, dtype=torch.float32)
    ramp_columns = (0.35 * axis)[None, :].expand(c37.GRID, c37.GRID).contiguous()
    ramp_rows = (0.35 * axis)[:, None].expand(c37.GRID, c37.GRID).contiguous()
    records = {}
    for name, phase in (("column_ramp", ramp_columns), ("row_ramp", ramp_rows),
                        ("zero_phase", torch.zeros_like(ramp_rows))):
        with torch.no_grad():
            result = c37.forward_batch(probe, phase, context)
        image = result["camera_intensity"][0].detach().to(torch.float64)
        index = int(torch.argmax(image))
        peak_row, peak_col = divmod(index, c37.CMOS)
        profile_row = image.sum(1)
        profile_column = image.sum(0)
        base = c37.FRAME_CENTER_PX
        axis_px = torch.arange(c37.CMOS, dtype=torch.float64, device=image.device) - base
        centroid_row = float((profile_row * axis_px).sum() / profile_row.sum()) + base
        centroid_col = float((profile_column * axis_px).sum() / profile_column.sum()) + base
        records[name] = {
            "peak_row": int(peak_row), "peak_col": int(peak_col),
            "centroid_row": centroid_row, "centroid_col": centroid_col,
            "centroid_shift_row": centroid_row - base,
            "centroid_shift_col": centroid_col - base,
        }
    column_shift = records["column_ramp"]
    row_shift = records["row_ramp"]
    checks = {
        "column_ramp_moves_columns": bool(abs(column_shift["centroid_shift_col"]) > 100.0),
        "column_ramp_does_not_move_rows": bool(
            abs(column_shift["centroid_shift_row"]) < 0.2 * abs(column_shift["centroid_shift_col"])),
        "row_ramp_moves_rows": bool(abs(row_shift["centroid_shift_row"]) > 100.0),
        "row_ramp_does_not_move_columns": bool(
            abs(row_shift["centroid_shift_col"]) < 0.2 * abs(row_shift["centroid_shift_row"])),
        "both_ramps_share_one_sign_convention": bool(
            np.sign(column_shift["centroid_shift_col"]) == np.sign(row_shift["centroid_shift_row"])),
        "zero_phase_probe_stays_on_axis": bool(
            abs(records["zero_phase"]["centroid_shift_row"]) < 5.0
            and abs(records["zero_phase"]["centroid_shift_col"]) < 5.0),
        "roi_block_is_centred_on_the_optical_axis": bool(np.allclose(
            c37.roi_center_px(c37.roi_bounds()).mean(0),
            [c37.FRAME_CENTER_PX, c37.FRAME_CENTER_PX])),
    }
    return {
        "records": records,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "convention": "array index (row, column) maps to physical (y, x); index 0 is the lowest y and x; "
                      "the optical axis is the array centre; figures must use origin='lower' with an extent "
                      "measured from that centre",
        "probe": "uniform complex plane wave on the frozen 100x100 phase screen (no MNIST pattern)",
    }


def stage_p0(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    phase = c37.load_frozen_phase(resolve_init_phase(args, config))
    rows = context["split"][args.split]
    full = c37.evaluate_phase(context, config, phase, rows, tag=f"E0_{args.split}",
                              batch_size=max(1, int(args.batch_size or 4)))
    c37.save_evaluation(args.work_dir / "p0" / "c37_p0_row_records.npz", full)
    (args.work_dir / "p0" / "maps").mkdir(parents=True, exist_ok=True)
    for row, image in full["maps"].items():
        np.save(args.work_dir / "p0" / "maps" / f"cmos_intensity_row_{row}.npy", image)
    c37.write_json(args.work_dir / "p0" / "c37_p0_e0_metrics.json", full["metrics"])

    audit = _coordinate_audit(args, config, context)
    c37.write_json(args.work_dir / "p0" / "c37_p0_coordinate_audit.json", audit)

    subset = c37.balanced_rows(rows, context["labels"], int(config["p0"]["subset_rows_per_class"]))
    sensitivity = []
    for plane in config["p0"]["defocus_planes_mm"]:
        for aperture in config["p0"]["aperture_scenarios_mm"]:
            for name, candidate in (("E0_P1_phase", phase), ("zero_phase", torch.zeros_like(phase))):
                evaluation = c37.evaluate_phase(
                    context, config, candidate, subset,
                    tag=f"plane{plane}_ap{aperture}_{name}", batch_size=int(args.batch_size or 4),
                    aperture_mm=float(aperture), lens_to_cmos_um=float(plane) * 1000.0,
                    verbose=False)
                sensitivity.append({
                    "plane_mm": float(plane), "aperture_mm": float(aperture), "phase": name,
                    "accuracy": evaluation["metrics"]["accuracy"],
                    "peak_localisation_rate": evaluation["metrics"]["peak_localisation_rate"],
                    "centroid_localisation_rate": evaluation["metrics"]["centroid_localisation_rate"],
                    "core_fraction_median": evaluation["metrics"]["core_fraction"]["p50"],
                    "target_roi_share_of_cmos_median": evaluation["metrics"]["target_roi_share_of_cmos"]["p50"],
                    "eta_mean": evaluation["metrics"]["eta_phase_to_cmos"]["mean"],
                })
    convergence = []
    for pitch in config["p0"]["sampling_convergence_lens_pitch_um"]:
        evaluation = c37.evaluate_phase(
            context, config, phase, subset[:100], tag=f"pitch{pitch}", batch_size=2,
            lens_pitch_um=float(pitch), verbose=False)
        convergence.append({
            "lens_plane_pitch_um": float(pitch),
            "accuracy": evaluation["metrics"]["accuracy"],
            "peak_localisation_rate": evaluation["metrics"]["peak_localisation_rate"],
            "core_fraction_median": evaluation["metrics"]["core_fraction"]["p50"],
            "eta_mean": evaluation["metrics"]["eta_phase_to_cmos"]["mean"],
            "mean_peak_distance_um": evaluation["metrics"]["peak_distance_to_target_centre_um"]["mean"],
        })
    # field level convergence: identical rows, only the lens-plane sampling changes
    field_difference = None
    pitches = list(config["p0"]["sampling_convergence_lens_pitch_um"])
    if len(pitches) >= 2:
        probe_rows = subset[:4]
        fields = c37.input_fields(context, probe_rows)
        with torch.no_grad():
            coarse = c37.forward_batch(fields, phase.to(context["device"]), context,
                                       lens_pitch_um=float(max(pitches)))["camera_intensity"]
            fine = c37.forward_batch(fields, phase.to(context["device"]), context,
                                     lens_pitch_um=float(min(pitches)))["camera_intensity"]
        denominator = coarse.abs().max().clamp_min(1e-30)
        field_difference = {
            "rows": int(len(probe_rows)),
            "coarse_lens_pitch_um": float(max(pitches)),
            "fine_lens_pitch_um": float(min(pitches)),
            "max_abs_intensity_difference_over_peak": float((coarse - fine).abs().max() / denominator),
            "mean_abs_intensity_difference_over_peak": float((coarse - fine).abs().mean() / denominator),
            "roi_score_max_relative_difference": None,
        }
        with torch.no_grad():
            coarse_roi = c37.forward_batch(fields, phase.to(context["device"]), context,
                                           lens_pitch_um=float(max(pitches)))["roi_scores"]
            fine_roi = c37.forward_batch(fields, phase.to(context["device"]), context,
                                         lens_pitch_um=float(min(pitches)))["roi_scores"]
        field_difference["roi_score_max_relative_difference"] = float(
            ((coarse_roi - fine_roi).abs() / coarse_roi.abs().clamp_min(1e-30)).max())
    payload = {
        "stage": "p0",
        "identity": "C3.7_P0_SPOT_DIAGNOSTIC",
        "population": {
            "rows": int(len(rows)), "split": args.split,
            "class_counts": full["metrics"]["class_counts"],
            "role": "development validation reused from C3.4/C3.5; not an independent test set",
            "note": "the ~54% peak localisation of the C3.6 figure was a visualisation sample and is "
                    "not a population estimate; this record recomputes the statistic on the whole split",
        },
        "phase_source": str(resolve_init_phase(args, config)),
        "metrics": full["metrics"],
        "three_success_definitions": {
            "CLASSIFICATION_SUCCESS": "argmax of the four fixed ROI integrals equals the label",
            "PEAK_LOCALIZATION_SUCCESS": "the global CMOS intensity maximum falls inside the correct ROI",
            "FOCUS_CONCENTRATION_SUCCESS": "the correct core fraction reaches the declared design target "
                                           "while the throughput constraint holds",
        },
        "classification_success_rate": full["metrics"]["accuracy"],
        "peak_localization_success_rate": full["metrics"]["peak_localisation_rate"],
        "centroid_localization_success_rate": full["metrics"]["centroid_localisation_rate"],
        "focus_concentration_success_rate": float(np.mean(
            (full["core_fraction"] >= float(config["acceptance"]["core_fraction_median_min"]))
            & (full["eta"] >= float(c37.derive_eta_target(config, {"metrics": full["metrics"]}))))),
        "coordinate_audit": audit,
        "defocus_and_aperture_sensitivity": sensitivity,
        "sampling_convergence": convergence,
        "sampling_field_convergence": field_difference,
    }
    c37.write_json(args.work_dir / "c37_p0_spot_diagnostic.json", payload)
    c37.write_json(args.work_dir / "c37_eta_target.json", {
        "eta_target": c37.derive_eta_target(config, {"metrics": full["metrics"]}),
        "retention_fraction": float(config["energy"]["eta_retention_fraction"]),
        "baseline_eta_mean": full["metrics"]["eta_phase_to_cmos"]["mean"],
        "baseline_eta_p10": full["metrics"]["eta_phase_to_cmos"]["p10"],
        "source": str(args.work_dir / "c37_p0_spot_diagnostic.json"),
    })
    print(json.dumps({key: payload[key] for key in
                      ("classification_success_rate", "peak_localization_success_rate",
                       "centroid_localization_success_rate", "focus_concentration_success_rate")},
                     indent=2))
    print(json.dumps({"coordinate_audit": audit["checks"],
                      "eta_mean": full["metrics"]["eta_phase_to_cmos"]["mean"],
                      "core_fraction_median": full["metrics"]["core_fraction"]["p50"],
                      "target_roi_share_of_cmos_median": full["metrics"]["target_roi_share_of_cmos"]["p50"]},
                     indent=2))
    return payload


# --------------------------------------------------------------------------- #
# gradient validation
# --------------------------------------------------------------------------- #
def _loss_for_variable(variable: torch.nn.Parameter, context, config, rows, groups,
                       masks, eta_target, multiplier) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fields = c37.input_fields(context, rows)
    phase = variable if variable.ndim == 2 else variable.reshape(c37.GRID, c37.GRID)
    result = c37.forward_batch(fields, phase, context)
    labels = context["labels"][rows].to(context["device"])
    return c37.objective(result, labels, phase, context, config, groups=groups,
                         eta_target=eta_target, multiplier=multiplier, masks=masks)


def stage_gradcheck(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    device = context["device"]
    phase0 = c37.load_frozen_phase(resolve_init_phase(args, config))
    eta_payload = json.loads((args.work_dir / "c37_eta_target.json").read_text(encoding="utf-8"))
    eta_target = float(eta_payload["eta_target"])
    rows = c37.balanced_rows(context["split"]["train"], context["labels"], 2)[:8]
    masks = c37.core_masks(float(config["core"]["frozen_fraction"]), device)

    variants = {
        "classification_only": [],
        "classification_focus": ["focus"],
        "classification_sidelobe": ["sidelobe"],
        "classification_focus_sidelobe_throughput": ["focus", "sidelobe", "throughput"],
        "full_E3_plus_spread": ["focus", "sidelobe", "throughput", "spread"],
    }
    records = []
    generator = torch.Generator().manual_seed(20261105)
    for name, groups in variants.items():
        variable = torch.nn.Parameter(phase0.clone().to(device))
        loss, parts = _loss_for_variable(variable, context, config, rows, groups, masks,
                                         eta_target, 0.0)
        loss.backward()
        gradient = variable.grad.detach().clone()
        finite = bool(torch.isfinite(loss) and torch.isfinite(gradient).all())
        entry = {
            "variant": name,
            "groups": groups,
            "loss": float(loss.detach()),
            "components": {key: float(value) for key, value in parts.items()},
            "gradient_norm": float(gradient.norm()),
            "gradient_max_abs": float(gradient.abs().max()),
            "finite": finite,
            "nonzero_gradient": bool(float(gradient.norm()) > 0),
        }
        if name == "classification_focus_sidelobe_throughput":
            # central finite differences on randomly selected phase pixels
            picks = torch.randint(0, c37.GRID * c37.GRID, (6,), generator=generator).tolist()
            step = 5e-3
            differences = []
            for flat in picks:
                row, column = divmod(int(flat), c37.GRID)
                values = []
                for sign in (1.0, -1.0):
                    probe = phase0.clone().to(device)
                    probe[row, column] = probe[row, column] + sign * step
                    candidate = torch.nn.Parameter(probe)
                    value, _ = _loss_for_variable(candidate, context, config, rows, groups,
                                                 masks, eta_target, 0.0)
                    values.append(float(value.detach()))
                differences.append({
                    "row": row, "column": column,
                    "finite_difference": (values[0] - values[1]) / (2 * step),
                    "autograd": float(gradient[row, column]),
                })
            relative = [abs(item["finite_difference"] - item["autograd"])
                        / max(abs(item["autograd"]), 1e-6) for item in differences]
            absolute = [abs(item["finite_difference"] - item["autograd"]) for item in differences]
            strong = [abs(item["autograd"]) > 1e-4 for item in differences]
            entry["finite_difference"] = differences
            entry["finite_difference_max_relative_error"] = float(max(relative))
            entry["finite_difference_max_absolute_error"] = float(max(absolute))
            entry["finite_difference_step_rad"] = float(step)
            entry["finite_difference_passed"] = bool(
                max(absolute) < 2e-3
                and all(abs(relative[index]) < 0.15
                        for index in range(len(relative)) if strong[index]))
        records.append(entry)

    # minimal gradient validation: one real update per family, then re-measure
    updates = []
    for name, groups in variants.items():
        variable = torch.nn.Parameter(phase0.clone().to(device))
        optimizer = torch.optim.Adam([variable], lr=float(args.learning_rate or
                                                         config["training"]["learning_rate"]))
        before = c37.evaluate_phase(context, config, variable.detach(), rows,
                                    tag=f"before_{name}", batch_size=4, verbose=False)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss_for_variable(variable, context, config, rows, groups, masks,
                                         eta_target, 0.0)
            loss.backward()
            if variable.grad is None or not torch.isfinite(variable.grad).all():
                raise RuntimeError("C3.7 gradient validation produced an invalid gradient")
            optimizer.step()
        after = c37.evaluate_phase(context, config, variable.detach(), rows,
                                   tag=f"after_{name}", batch_size=4, verbose=False)
        updates.append({
            "variant": name,
            "core_fraction_before": before["metrics"]["core_fraction"]["mean"],
            "core_fraction_after": after["metrics"]["core_fraction"]["mean"],
            "peak_localisation_before": before["metrics"]["peak_localisation_rate"],
            "peak_localisation_after": after["metrics"]["peak_localisation_rate"],
            "accuracy_before": before["metrics"]["accuracy"],
            "accuracy_after": after["metrics"]["accuracy"],
            "eta_before": before["metrics"]["eta_phase_to_cmos"]["mean"],
            "eta_after": after["metrics"]["eta_phase_to_cmos"]["mean"],
            "mean_peak_distance_before_um": before["metrics"]["peak_distance_to_target_centre_um"]["mean"],
            "mean_peak_distance_after_um": after["metrics"]["peak_distance_to_target_centre_um"]["mean"],
            "phase_delta_l2": float((variable.detach() - phase0.to(device)).norm()),
        })
    payload = {
        "stage": "gradcheck",
        "population_rows": int(len(rows)),
        "eta_target": eta_target,
        "variants": records,
        "updates": updates,
        "checks": {
            "all_gradients_finite": bool(all(item["finite"] for item in records)),
            "all_gradients_nonzero": bool(all(item["nonzero_gradient"] for item in records)),
            "finite_difference_matches_autograd": bool(next(
                item["finite_difference_passed"] for item in records
                if item["variant"] == "classification_focus_sidelobe_throughput")),
            "focus_increases_core_fraction": bool(next(
                item["core_fraction_after"] > item["core_fraction_before"] for item in updates
                if item["variant"] == "classification_focus")),
            "no_energy_collapse_in_minimal_update": bool(all(
                item["eta_after"] > 0.5 * item["eta_before"] for item in updates)),
            "no_accuracy_collapse_in_minimal_update": bool(all(
                item["accuracy_after"] > 0.5 for item in updates)),
        },
    }
    payload["all_checks_passed"] = bool(all(payload["checks"].values()))
    c37.write_json(args.work_dir / "c37_gradcheck.json", payload)
    print(json.dumps(payload["checks"], indent=2))
    print(json.dumps(updates, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _effective_phase(variable: torch.nn.Parameter, mode: str, config: dict[str, Any], *,
                     training: bool, noise_seed: int) -> torch.Tensor:
    if mode == "phase":
        return variable
    if mode != "height":
        raise ValueError(f"unsupported trainable variable mode {mode}")
    assumptions = config["fabrication_assumptions"]
    nominal = assumptions["nominal"]
    noise = c34_process._noise_field(variable, int(noise_seed), float(nominal["height_noise_nm"]))
    printed = c34_process.process_height(
        variable,
        psf_sigma_px=float(nominal["psf_sigma_px"]),
        minimum_feature_radius_px=int(nominal["minimum_feature_radius_px"]),
        height_levels=int(nominal["height_levels"]),
        maximum_height_um=float(assumptions["maximum_height_um"]),
        noise_nm=float(nominal["height_noise_nm"]),
        noise_seed=int(noise_seed),
        edge_height_bias_nm=float(nominal["edge_height_bias_nm"]),
        noise_field=noise,
        straight_through_quantization=training,
    )
    return c34.phase_from_height(printed, float(nominal["delta_n"]))


def _accumulate(variable: torch.nn.Parameter, optimizer, context, config, rows, groups,
                masks, eta_target, multiplier, mode, noise_seed) -> dict[str, float]:
    device = context["device"]
    total = int(len(rows))
    if total < 1:
        raise ValueError("an update requires at least one sample")
    microbatch = max(1, int(config["training"]["microbatch_size"]))
    while True:
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        try:
            for start in range(0, total, microbatch):
                chunk = rows[start:start + microbatch]
                weight = len(chunk) / total
                phase = _effective_phase(variable, mode, config, training=True,
                                         noise_seed=noise_seed)
                fields = c37.input_fields(context, chunk)
                result = c37.forward_batch(fields, phase, context)
                labels = context["labels"][chunk].to(device)
                loss, parts = c37.objective(result, labels, phase, context, config, groups=groups,
                                            eta_target=eta_target, multiplier=multiplier,
                                            masks=masks)
                (loss * weight).backward()
                for key, value in parts.items():
                    sums[key] = sums.get(key, 0.0) + float(value) * weight
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if microbatch <= 1:
                raise
            microbatch = max(1, microbatch // 2)
            print(f"C3.7 OOM; retrying the update with microbatch={microbatch}", flush=True)
            continue
        break
    if variable.grad is None or not torch.isfinite(variable.grad).all():
        raise RuntimeError("C3.7 update has absent or non-finite gradients")
    sums["gradient_norm"] = float(variable.grad.detach().norm())
    previous = variable.detach().clone()
    optimizer.step()
    if mode == "height":
        with torch.no_grad():
            variable.clamp_(0.0, float(config["fabrication_assumptions"]["maximum_height_um"]))
    if not torch.isfinite(variable).all():
        raise RuntimeError("C3.7 update produced non-finite parameters")
    sums["parameter_delta_l2"] = float((variable.detach() - previous).norm())
    return sums


def _run_training(context, config, phase_init, *, groups, mode, epochs, train_rows,
                  eval_rows, seed, learning_rate, arm, out_dir, eta_target,
                  log_prefix="") -> dict[str, Any]:
    device = context["device"]
    existing = out_dir / "training_summary.json"
    if existing.is_file():
        raise SystemExit(
            f"refusing to overwrite the finished run in {out_dir}; move it aside or use a "
            "different arm or seed, because a same named rerun would replace the checkpoint, "
            "history and validation records that the recorded evidence points at")
    masks = c37.core_masks(float(config["core"]["frozen_fraction"]), device)
    if mode == "height":
        variable = torch.nn.Parameter(
            c34.height_from_phase(phase_init.to(device), float(
                config["fabrication_assumptions"]["nominal"]["delta_n"])).clone())
    else:
        variable = torch.nn.Parameter(phase_init.clone().to(device))
    optimizer = torch.optim.Adam([variable], lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(seed))
    logical_batch = int(config["training"]["logical_batch_size"])
    multiplier = 0.0
    history: list[dict[str, Any]] = []
    best_phase = variable.detach().clone()
    best_key: tuple | None = None
    for epoch in range(epochs):
        order = torch.randperm(len(train_rows), generator=generator)
        start_time = time.perf_counter()
        sums: dict[str, float] = {}
        updates = 0
        for start in range(0, len(order), logical_batch):
            if args_limit_batches is not None and updates >= args_limit_batches:
                break
            selection = order[start:start + logical_batch]
            chunk = train_rows[selection]
            stats = _accumulate(variable, optimizer, context, config, chunk, groups, masks,
                                eta_target, multiplier, mode, seed * 1000 + epoch)
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value
            updates += 1
            if "eta_shortfall_mean" in stats:
                multiplier = max(0.0, multiplier + float(config["energy"]["adaptive_lagrange_rho"])
                                 * stats["eta_shortfall_mean"])
        epoch_seconds = time.perf_counter() - start_time
        phase_now = _effective_phase(variable, mode, config, training=False, noise_seed=seed)
        evaluation = c37.evaluate_phase(context, config, phase_now.detach(), eval_rows,
                                        tag=f"{arm}_seed{seed}_epoch{epoch}", batch_size=4,
                                        verbose=False)
        metrics = evaluation["metrics"]
        record = {
            "epoch": epoch,
            "arm": arm,
            "seed": int(seed),
            "groups": groups,
            "mode": mode,
            "updates": updates,
            "epoch_seconds": float(epoch_seconds),
            "loss_mean": sums.get("loss", 0.0) / max(updates, 1),
            "classification_mean": sums.get("classification", 0.0) / max(updates, 1),
            "focus_mean": sums.get("focus"),
            "sidelobe_mean": sums.get("sidelobe"),
            "spread_mean": sums.get("spread"),
            "constraint_mean": sums.get("constraint"),
            "gradient_norm_mean": sums.get("gradient_norm", 0.0) / max(updates, 1),
            "parameter_delta_l2_mean": sums.get("parameter_delta_l2", 0.0) / max(updates, 1),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "eta_mean": metrics["eta_phase_to_cmos"]["mean"],
            "eta_p10": metrics["eta_phase_to_cmos"]["p10"],
            "peak_localisation_rate": metrics["peak_localisation_rate"],
            "centroid_localisation_rate": metrics["centroid_localisation_rate"],
            "core_fraction_median": metrics["core_fraction"]["p50"],
            "target_roi_share_of_cmos_median": metrics["target_roi_share_of_cmos"]["p50"],
            "outside_target_roi_energy_fraction_mean":
                metrics["outside_target_roi_energy_fraction"]["mean"],
            "peak_distance_to_target_centre_um_median":
                metrics["peak_distance_to_target_centre_um"]["p50"],
            "eval_rows": int(len(eval_rows)),
            "eval_seconds": metrics["elapsed_seconds"],
        }
        history.append(record)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "phase_rad.npy", phase_now.detach().cpu().numpy().astype(np.float32),
                allow_pickle=False)
        np.save(out_dir / f"phase_epoch{epoch}.npy",
                phase_now.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
        if mode == "height":
            np.save(out_dir / "height_um.npy",
                    variable.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
        c37.write_json(out_dir / "history.json", {
            "arm": arm, "seed": int(seed), "groups": groups, "mode": mode,
            "epochs_completed": len(history), "history": history,
            "checkpoint_semantics": "phase_rad.npy is the latest epoch; the best epoch is "
                                    "re-selected from history at the end of the run",
        })
        print(log_prefix + json.dumps({key: record[key] for key in
                                       ("epoch", "accuracy", "eta_mean", "core_fraction_median",
                                        "peak_localisation_rate", "loss_mean")}), flush=True)
        key = (round(metrics["accuracy"], 4), round(metrics["core_fraction"]["p50"], 6))
        if best_key is None or key > best_key:
            best_key = key
            best_phase = phase_now.detach().clone()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "phase_rad.npy", best_phase.detach().cpu().numpy().astype(np.float32),
            allow_pickle=False)
    if mode == "height":
        np.save(out_dir / "height_um.npy",
                variable.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
    final = c37.evaluate_phase(context, config, best_phase, context["split"]["validation"],
                               tag=f"{arm}_seed{seed}_final", batch_size=4, verbose=True)
    c37.save_evaluation(out_dir / "validation_records.npz", final)
    payload = {
        "arm": arm, "seed": int(seed), "groups": groups, "mode": mode,
        "epochs": epochs, "learning_rate": float(learning_rate),
        "focus_weight": float(config["training"]["focus_weight"]),
        "focus_mode": str(config["training"].get("focus_mode", "mean")),
        "eta_retention_fraction": float(config["energy"]["eta_retention_fraction"]),
        "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows)),
        "history": history, "final_validation": final["metrics"],
        "phase_path": str(out_dir / "phase_rad.npy"),
        "eta_target": eta_target,
        "fabrication_nominal": config["fabrication_assumptions"]["nominal"],
    }
    c37.write_json(out_dir / "training_summary.json", payload)
    return payload


args_limit_batches: int | None = None


def stage_ablate(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    global args_limit_batches
    args_limit_batches = args.limit_batches
    phase_init = c37.load_frozen_phase(resolve_init_phase(args, config))
    eta_payload = json.loads((args.work_dir / "c37_eta_target.json").read_text(encoding="utf-8"))
    eta_target = float(eta_payload["eta_target"])
    per_class = int(args.rows_per_class or config["training"]["rows_per_class_ablation"])
    train_rows = c37.balanced_rows(context["split"]["train"], context["labels"], per_class)
    eval_rows = c37.balanced_rows(context["split"]["validation"], context["labels"],
                                  int(args.eval_per_class))
    arms = list(config["training"]["groups"].items())
    epochs = int(args.epochs or 2)
    records = []
    for arm, groups in arms:
        if arm == "E0":
            evaluation = c37.evaluate_phase(context, config, phase_init, eval_rows,
                                            tag="E0_frozen", batch_size=4, verbose=False)
            records.append({"arm": "E0", "groups": groups, "mode": "frozen",
                            "core_fraction_median": evaluation["metrics"]["core_fraction"]["p50"],
                            "accuracy": evaluation["metrics"]["accuracy"],
                            "eta_mean": evaluation["metrics"]["eta_phase_to_cmos"]["mean"],
                            "peak_localisation_rate": evaluation["metrics"]["peak_localisation_rate"],
                            "target_roi_share_of_cmos_median":
                                evaluation["metrics"]["target_roi_share_of_cmos"]["p50"],
                            "outside_target_roi_energy_fraction_mean":
                                evaluation["metrics"]["outside_target_roi_energy_fraction"]["mean"]})
            continue
        mode = "height" if arm == "E4" else "phase"
        payload = _run_training(context, config, phase_init, groups=list(groups), mode=mode,
                                epochs=epochs, train_rows=train_rows, eval_rows=eval_rows,
                                seed=int(args.seed), learning_rate=float(
                                    args.learning_rate or config["training"]["learning_rate"]),
                                arm=arm, out_dir=args.work_dir / "ablate" / arm,
                                eta_target=eta_target, log_prefix=f"ablate {arm} ")
        records.append({"arm": arm, "groups": list(groups), "mode": mode,
                        "core_fraction_median": payload["final_validation"]["core_fraction"]["p50"],
                        "accuracy": payload["final_validation"]["accuracy"],
                        "eta_mean": payload["final_validation"]["eta_phase_to_cmos"]["mean"],
                        "peak_localisation_rate": payload["final_validation"]["peak_localisation_rate"],
                        "target_roi_share_of_cmos_median":
                            payload["final_validation"]["target_roi_share_of_cmos"]["p50"],
                        "outside_target_roi_energy_fraction_mean":
                            payload["final_validation"]["outside_target_roi_energy_fraction"]["mean"]})
    baseline = next(item for item in records if item["arm"] == "E0")
    for item in records:
        item["core_fraction_gain_vs_E0"] = item["core_fraction_median"] - baseline["core_fraction_median"]
        item["eta_retention_vs_E0"] = item["eta_mean"] / baseline["eta_mean"]
    payload = {
        "stage": "ablate",
        "population": {"train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows)),
                       "epochs": epochs, "seed": int(args.seed)},
        "records": records,
        "screening_rule": "an arm enters formal training only when the core fraction median improves "
                          "over E0 while accuracy stays above 0.5 and mean efficiency stays above half "
                          "of the E0 mean",
    }
    for item in records:
        item["enters_formal_training"] = bool(
            item["core_fraction_gain_vs_E0"] > 0 and item["accuracy"] > 0.5
            and item["eta_retention_vs_E0"] > 0.5)
    c37.write_json(args.work_dir / "c37_ablation_screening.json", payload)
    print(json.dumps(records, indent=2))
    return payload


def stage_train(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if not args.arm:
        raise SystemExit("--arm is required for the train stage")
    global args_limit_batches
    args_limit_batches = args.limit_batches
    groups = list(config["training"]["groups"][args.arm])
    mode = "height" if args.arm == "E4" else "phase"
    phase_init = c37.load_frozen_phase(resolve_init_phase(args, config))
    eta_target = float(json.loads((args.work_dir / "c37_eta_target.json").read_text(
        encoding="utf-8"))["eta_target"])
    per_class = int(args.rows_per_class or config["training"]["rows_per_class_train"])
    train_rows = c37.balanced_rows(context["split"]["train"], context["labels"], per_class)
    eval_rows = c37.balanced_rows(context["split"]["validation"], context["labels"],
                                  int(args.eval_per_class))
    payload = _run_training(context, config, phase_init, groups=groups, mode=mode,
                            epochs=int(args.epochs or config["training"]["epochs"]),
                            train_rows=train_rows, eval_rows=eval_rows, seed=int(args.seed),
                            learning_rate=float(args.learning_rate
                                                or config["training"]["learning_rate"]),
                            arm=args.arm,
                            out_dir=args.work_dir / "train" / f"{args.arm}_seed{args.seed}",
                            eta_target=eta_target)
    print(json.dumps({key: payload["final_validation"][key] for key in
                      ("accuracy", "peak_localisation_rate", "centroid_localisation_rate")}, indent=2))
    print(json.dumps({"eta": payload["final_validation"]["eta_phase_to_cmos"],
                      "core_fraction": payload["final_validation"]["core_fraction"]}, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# P4 fabrication comparison
# --------------------------------------------------------------------------- #
def _candidate_phases(args, config, work_dir: Path) -> dict[str, torch.Tensor]:
    candidates: dict[str, torch.Tensor] = {}
    frozen = c37.load_frozen_phase(resolve_reference_phase(args, config))
    candidates["E0_C35_P1_high_throughput"] = frozen
    height_path, phase_path = resolve_p2_references(args, config)
    if phase_path.is_file():
        candidates["E5_C35_P2_process_aware"] = c37.load_frozen_phase(phase_path)
    elif height_path.is_file():
        height = torch.as_tensor(np.load(height_path, allow_pickle=False), dtype=torch.float32)
        candidates["E5_C35_P2_process_aware"] = c34.phase_from_height(
            height, float(config["fabrication_assumptions"]["nominal"]["delta_n"])).to(torch.float32)
    selected = args.candidates.split(",") if args.candidates else None
    train_root = work_dir / "train"
    if train_root.is_dir():
        for path in sorted(train_root.glob("*/phase_rad.npy")):
            name = f"C37_{path.parent.name}"
            candidates[name] = c37.load_frozen_phase(path)
    if selected:
        candidates = {key: value for key, value in candidates.items()
                      if any(token in key for token in selected)}
    if len(candidates) < 2:
        raise SystemExit("fabrication comparison needs at least two candidates")
    return candidates


def stage_fabricate(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    work_dir = args.work_dir
    candidates = _candidate_phases(args, config, work_dir)
    rows = c37.balanced_rows(context["split"]["validation"], context["labels"],
                             int(args.eval_per_class))
    assumptions = config["fabrication_assumptions"]
    nominal = assumptions["nominal"]
    scenarios: list[dict[str, Any]] = []
    for delta_n in assumptions["delta_n_scenarios"]:
        for sigma in assumptions["psf_sigma_px_scenarios"]:
            for radius in assumptions["minimum_feature_radius_px_scenarios"]:
                scenarios.append({
                    "name": f"dn{delta_n}_psf{sigma}_r{radius}",
                    "delta_n": float(delta_n), "psf_sigma_px": float(sigma),
                    "minimum_feature_radius_px": int(radius),
                    "height_levels": int(assumptions["height_levels"]),
                    "height_noise_nm": 0.0, "noise_seed": 0,
                })
    for seed in assumptions["height_noise_seeds"]:
        scenarios.append({
            "name": f"nominal_noise20_seed{seed}",
            "delta_n": float(nominal["delta_n"]), "psf_sigma_px": float(nominal["psf_sigma_px"]),
            "minimum_feature_radius_px": int(nominal["minimum_feature_radius_px"]),
            "height_levels": int(assumptions["height_levels"]),
            "height_noise_nm": float(assumptions["height_noise_nm"]), "noise_seed": int(seed),
        })
    records = []
    for name, phase in candidates.items():
        ideal = c37.evaluate_phase(context, config, phase, rows, tag=f"{name}_ideal",
                                   batch_size=int(args.batch_size or 4), verbose=False)
        record = {
            "candidate": name,
            "ideal": {key: ideal["metrics"][key] for key in
                      ("accuracy", "peak_localisation_rate", "centroid_localisation_rate",
                       "macro_f1")},
            "ideal_eta_mean": ideal["metrics"]["eta_phase_to_cmos"]["mean"],
            "ideal_core_fraction_median": ideal["metrics"]["core_fraction"]["p50"],
            "scenarios": [],
        }
        for scenario in scenarios:
            printed = c37.processed_phase(
                phase, config, delta_n=scenario["delta_n"],
                psf_sigma_px=scenario["psf_sigma_px"],
                minimum_feature_radius_px=scenario["minimum_feature_radius_px"],
                height_levels=scenario["height_levels"],
                height_noise_nm=scenario["height_noise_nm"],
                noise_seed=scenario["noise_seed"])
            evaluation = c37.evaluate_phase(context, config, printed, rows,
                                            tag=f"{name}_{scenario['name']}",
                                            batch_size=int(args.batch_size or 4), verbose=False)
            metrics = evaluation["metrics"]
            record["scenarios"].append({
                "scenario": scenario["name"],
                "delta_n": scenario["delta_n"],
                "psf_sigma_px": scenario["psf_sigma_px"],
                "minimum_feature_radius_px": scenario["minimum_feature_radius_px"],
                "height_noise_nm": scenario["height_noise_nm"],
                "noise_seed": scenario["noise_seed"],
                "accuracy": metrics["accuracy"],
                "accuracy_retention": metrics["accuracy"] / max(record["ideal"]["accuracy"], 1e-9),
                "peak_localisation_rate": metrics["peak_localisation_rate"],
                "centroid_localisation_rate": metrics["centroid_localisation_rate"],
                "core_fraction_median": metrics["core_fraction"]["p50"],
                "eta_mean": metrics["eta_phase_to_cmos"]["mean"],
                "eta_retention": metrics["eta_phase_to_cmos"]["mean"] / max(record["ideal_eta_mean"], 1e-12),
                "outside_target_roi_energy_fraction_mean":
                    metrics["outside_target_roi_energy_fraction"]["mean"],
                "phase_delta_l2": float((printed - phase).norm()),
            })
            print(f"fabricate {name} {scenario['name']} acc={metrics['accuracy']:.4f}", flush=True)
        accuracies = [item["accuracy"] for item in record["scenarios"]]
        peak_rates = [item["peak_localisation_rate"] for item in record["scenarios"]]
        record["worst_accuracy"] = float(min(accuracies))
        record["worst_peak_localisation_rate"] = float(min(peak_rates))
        record["process_sensitive"] = bool(min(accuracies) < float(
            config["acceptance"]["process_accuracy_retention_min"]) * record["ideal"]["accuracy"])
        records.append(record)
    payload = {
        "stage": "fabricate",
        "population_rows": int(len(rows)),
        "scenarios": scenarios,
        "candidates": records,
        "domain_note": "every scenario converts wrapped phase to physical height before blur, "
                       "minimum feature, quantisation and noise; no operator is applied to wrapped phase",
        "status": assumptions["status"],
    }
    c37.write_json(work_dir / "c37_process_comparison.json", payload)
    for record in records:
        print(json.dumps({key: record[key] for key in
                          ("candidate", "worst_accuracy", "worst_peak_localisation_rate",
                           "process_sensitive")}))
    return payload


# --------------------------------------------------------------------------- #
# visualisation inputs
# --------------------------------------------------------------------------- #
def stage_vizdata(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    work_dir = args.work_dir
    viz = work_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_phases(args, config, work_dir)
    names = list(candidates)
    spec = c37.core_spec(float(config["core"]["frozen_fraction"]))
    labels = context["labels"]
    validation = context["split"]["validation"]
    payload: dict[str, Any] = {
        "stage": "vizdata", "core_spec": spec, "candidates": names,
        "class_exemplars": [], "error_exemplars": {}, "class_averages": {},
        "average_crop": {"offset_px": 640, "side_px": 768,
                         "note": "class averages are exported from the central 768x768 pixel window, "
                                 "which is centred on the optical axis and contains the whole ROI block"},
        "intensity_maps": "full 2048x2048 single precision frames, npz key 'intensity'",
    }
    probe_per_class = 40
    probe_rows = c37.balanced_rows(validation, labels, probe_per_class)
    for name, phase in candidates.items():
        np.save(viz / f"phase_{name}.npy", phase.numpy().astype(np.float32), allow_pickle=False)
        probe = c37.evaluate_phase(context, config, phase, probe_rows,
                                   tag=f"vizprobe_{name}", batch_size=8, verbose=False)
        exemplar_rows: list[int] = []
        error_rows: list[int] = []
        for cls in range(4):
            mask = probe["labels"] == cls
            rows_cls = probe["rows"][mask]
            predictions = probe["predictions"][mask]
            correct = rows_cls[predictions == cls]
            wrong = rows_cls[predictions != cls]
            if len(correct):
                exemplar_rows.append(int(correct[0]))
            if len(wrong):
                error_rows.append(int(wrong[0]))
        chosen = exemplar_rows + [row for row in error_rows if row not in exemplar_rows]
        saved = c37.evaluate_phase(context, config, phase, torch.as_tensor(chosen),
                                   tag=f"vizmaps_{name}", batch_size=4,
                                   save_maps_rows=chosen, verbose=False)
        for row, image in saved["maps"].items():
            path = viz / f"cmos_{name}_row{row}.npz"
            with path.open("wb") as handle:
                np.savez_compressed(handle, intensity=image)
        for index, row in enumerate(saved["rows"]):
            record = dict(saved["records"][index])
            record["input_image"] = context["images"][int(row)].numpy().reshape(-1).tolist()
            item = {
                "candidate": name, "class": int(saved["labels"][index]), "row": int(row),
                "true_label": int(saved["labels"][index]),
                "prediction": int(saved["predictions"][index]),
                "roi_energies_um2": record["roi_energies_um2"],
                "record": record,
                "map_path": f"cmos_{name}_row{row}.npz",
            }
            if int(row) in exemplar_rows:
                payload["class_exemplars"].append(item)
            else:
                payload["error_exemplars"].setdefault(name, []).append(item)
        # class averages over the whole development validation split
        full = c37.evaluate_phase(context, config, phase, validation,
                                  tag=f"vizaverage_{name}", batch_size=8, verbose=False)
        correct_mask = full["predictions"] == full["labels"]
        entry: dict[str, Any] = {"rows_per_class": int(len(validation) // 4), "paths": {},
                                 "correct_paths": {}, "correct_counts": {}}
        for cls in range(4):
            class_rows = validation[labels[validation] == cls]
            class_correct = class_rows[correct_mask[full["labels"] == cls]]
            for key, subset in (("paths", class_rows), ("correct_paths", class_correct)):
                total = torch.zeros((c37.CMOS, c37.CMOS), dtype=torch.float64)
                for start in range(0, len(subset), 8):
                    chunk = subset[start:start + 8]
                    fields = c37.input_fields(context, chunk)
                    with torch.no_grad():
                        result = c37.forward_batch(fields, phase.to(context["device"]), context)
                    total += result["camera_intensity"].detach().double().sum(0).cpu()
                average = (total / max(len(subset), 1)).to(torch.float32)
                crop = average[640:640 + 768, 640:640 + 768].numpy()
                file_name = f"average_{'correct_' if key == 'correct_paths' else ''}{name}_class{cls}.npy"
                np.save(viz / file_name, crop, allow_pickle=False)
                entry[key][str(cls)] = file_name
            entry["correct_counts"][str(cls)] = int(len(class_correct))
        payload["class_averages"][name] = entry
        with (viz / f"validation_records_{name}.npz").open("wb") as handle:
            np.savez_compressed(handle, labels=full["labels"], predictions=full["predictions"],
                                roi_scores=full["roi_scores"], core_fraction=full["core_fraction"],
                                eta_phase_to_cmos=full["eta"],
                                cmos_energy_um2=full["cmos_energy"], rows=full["rows"])
        print(f"vizdata {name}: exemplars={exemplar_rows} errors={error_rows}", flush=True)
    c37.write_json(work_dir / "c37_vizdata.json", payload)
    print(json.dumps({"candidates": names,
                      "exemplars": len(payload["class_exemplars"]),
                      "error_exemplars": {k: len(v) for k, v in payload["error_exemplars"].items()}}))
    return payload


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #
def _dedupe_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per epoch; historical runs appended each epoch twice."""
    unique: dict[int, dict[str, Any]] = {}
    for item in records:
        unique[int(item["epoch"])] = item
    return [unique[key] for key in sorted(unique)]


def stage_finalize(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Close a deliberately stopped training run from its saved epoch checkpoints.

    Used only when a run is stopped before its planned epoch count because the
    selection metrics plateaued.  The epoch is chosen from the saved history with
    the same gate-then-focus rule as the formal selection, then the chosen phase is
    re-evaluated on the whole development validation split.
    """
    if not args.arm:
        raise SystemExit("--arm is required for the finalize stage")
    out_dir = args.work_dir / "train" / f"{args.arm}_seed{args.seed}"
    history_path = out_dir / "history.json"
    if not history_path.is_file():
        raise SystemExit(f"no saved history at {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    records = _dedupe_history(history["history"])
    summary_path = out_dir / "training_summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if not existing.get("stopped_early"):
            raise SystemExit("run already has a complete training summary")
    if args.epoch is not None:
        epoch = int(args.epoch)
    else:
        epoch = max(records, key=lambda item: (round(item["accuracy"], 4),
                                               round(item["core_fraction_median"], 6)))["epoch"]
    phase_path = out_dir / f"phase_epoch{epoch}.npy"
    if not phase_path.is_file():
        raise SystemExit(f"missing saved phase for epoch {epoch}")
    phase = c37.load_frozen_phase(phase_path)
    arms = config["training"]["groups"][args.arm]
    mode = "height" if args.arm == "E4" else "phase"
    aphase = phase
    if mode == "height":
        height_path = out_dir / "height_um.npy"
        if not height_path.is_file():
            raise SystemExit("height mode run has no saved height map")
        aphase = c34.phase_from_height(
            torch.as_tensor(np.load(height_path, allow_pickle=False), dtype=torch.float32),
            float(config["fabrication_assumptions"]["nominal"]["delta_n"]))
    evaluation = c37.evaluate_phase(context, config, aphase, context["split"]["validation"],
                                   tag=f"{args.arm}_seed{args.seed}_finalize_epoch{epoch}",
                                   batch_size=4, verbose=True)
    c37.save_evaluation(out_dir / "validation_records.npz", evaluation)
    payload = {
        "arm": args.arm, "seed": int(args.seed), "groups": list(arms), "mode": mode,
        "epochs": int(epoch) + 1, "planned_epochs": len(records),
        "stopped_early": True,
        "stop_reason": f"core fraction median plateaued at {records[-1]['core_fraction_median']:.4f} "
                       f"across epochs; the epoch was chosen by the documented gate-then-focus rule",
        "learning_rate": float(args.learning_rate or config["training"]["learning_rate"]),
        "focus_weight": float(config["training"]["focus_weight"]),
        "focus_mode": str(config["training"].get("focus_mode", "mean")),
        "train_rows": None, "eval_rows": int(len(context["split"]["validation"])),
        "history": records, "final_validation": evaluation["metrics"],
        "phase_path": str(out_dir / "phase_rad.npy"),
        "finalized_from_epoch_checkpoint": str(phase_path),
        "fabrication_nominal": config["fabrication_assumptions"]["nominal"],
    }
    np.save(out_dir / "phase_rad.npy", aphase.detach().cpu().numpy().astype(np.float32),
            allow_pickle=False)
    c37.write_json(summary_path, payload)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "epoch": epoch,
                      "accuracy": evaluation["metrics"]["accuracy"],
                      "core_fraction_p50": evaluation["metrics"]["core_fraction"]["p50"],
                      "peak_localisation_rate": evaluation["metrics"]["peak_localisation_rate"],
                      "eta_mean": evaluation["metrics"]["eta_phase_to_cmos"]["mean"]}, indent=2))
    return payload


def stage_spotlimit(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Single layer focal limit probe and the concentration/placement decomposition.

    ``peak_box_energy_fraction`` centres the core-sized box on the global peak, so
    it upper bounds any fixed core placement.  ``core_fraction`` uses the frozen
    ROI core.  The ratio of the two separates "the light is not concentrated"
    from "the concentrated light lands in the wrong place".
    """
    device = context["device"]
    rows_per_class = int(args.rows_per_class or 2)
    rows = c37.balanced_rows(context["split"]["validation"], context["labels"], rows_per_class)
    labels = context["labels"][rows]
    side = c37.core_side_px(float(config["core"]["frozen_fraction"]))
    fields = c37.input_fields(context, rows)
    core_mask = c37.core_masks(float(config["core"]["frozen_fraction"]), device)
    grid = (torch.arange(c37.GRID, device=device, dtype=torch.float32) - (c37.GRID - 1) / 2.0)
    radius_um = torch.sqrt(grid[:, None].square() + grid[None, :].square()) * float(
        config["geometry"]["phase_pitch_um"])
    wavelength_um = 532e-3
    reference_distance_um = float(context["lens_to_cmos_um"])
    pixel_area = c37.CMOS_PITCH_UM ** 2
    axis = torch.arange(c37.CMOS, device=device, dtype=torch.float64) - c37.FRAME_CENTER_PX
    row_index = torch.arange(len(rows), device=device)

    def measure(result: dict[str, Any]) -> dict[str, float]:
        intensity = result["camera_intensity"].double()
        total = intensity.sum((1, 2)).clamp_min(1e-30)
        flat = intensity.reshape(len(rows), -1)
        peak_index = flat.argmax(1)
        peak_row = peak_index // c37.CMOS
        peak_col = peak_index % c37.CMOS

        def box_fraction(box_side: int) -> float:
            half_box = (box_side - 1) // 2
            box = torch.zeros_like(intensity)
            for index in range(len(rows)):
                r0 = max(0, min(int(peak_row[index]) - half_box, c37.CMOS - box_side))
                c0 = max(0, min(int(peak_col[index]) - half_box, c37.CMOS - box_side))
                box[index, r0:r0 + box_side, c0:c0 + box_side] = 1.0
            return float(((intensity * box).sum((1, 2)) / total).mean())

        row_profile = intensity.sum(2)
        column_profile = intensity.sum(1)
        centroid_row = (row_profile * axis).sum(1) / total
        centroid_column = (column_profile * axis).sum(1) / total
        radial = ((row_profile * axis.square()).sum(1) + (column_profile * axis.square()).sum(1))
        rms_radius = c37.CMOS_PITCH_UM * torch.sqrt(
            (radial / total - centroid_row.square() - centroid_column.square()).clamp_min(0.0))
        core = torch.stack([(intensity * core_mask[index]).sum((1, 2)) for index in range(4)], 1)
        core_fraction = (core[row_index, labels.to(device)] * pixel_area
                         / (total * pixel_area))
        peak_box = box_fraction(side)
        return {
            "peak_box_energy_fraction_mean": peak_box,
            "peak_fwhm_box_energy_fraction_mean": box_fraction(5),
            "fixed_core_energy_fraction_mean": float(core_fraction.mean()),
            "peak_box_to_core_ratio": float(peak_box / max(float(core_fraction.mean()), 1e-9)),
            "rms_radius_um_mean": float(rms_radius.mean()),
            "eta_mean": float((result["detected_energy"] / result["input_energy"]).mean()),
        }

    curve = []
    best = None
    for coefficient in np.linspace(-12.0, 12.0, 25):
        phase = (float(coefficient) * math.pi * radius_um.square()
                 / (wavelength_um * reference_distance_um))
        with torch.no_grad():
            result = c37.forward_batch(fields, phase, context)
        entry = {"coefficient": float(coefficient), **measure(result)}
        curve.append(entry)
        if best is None or entry["peak_box_energy_fraction_mean"] > best["peak_box_energy_fraction_mean"]:
            best = entry
    reference = {}
    for name, phase in (("p1_frozen", c37.load_frozen_phase(resolve_init_phase(args, config)).to(device)),
                        ("zero", torch.zeros((c37.GRID, c37.GRID), device=device))):
        with torch.no_grad():
            reference[name] = measure(c37.forward_batch(fields, phase, context))
    aperture_mm = float(config["geometry"]["aperture_diameter_mm"])
    distance_um = float(context["lens_to_cmos_um"])
    na = (aperture_mm * 1000.0 / 2.0) / distance_um
    payload = {
        "stage": "spotlimit",
        "rows": int(len(rows)),
        "rows_per_class": rows_per_class,
        "core_fraction": float(config["core"]["frozen_fraction"]),
        "core_side_px": int(side),
        "core_side_um": float(side * c37.CMOS_PITCH_UM),
        "quadratic_phase_family": "phi = c * pi * r^2 / (lambda * 100 mm) on the 100x100 screen",
        "curve": curve,
        "best": best,
        "reference": reference,
        "analytic": {
            "pupil_diameter_mm": aperture_mm,
            "pupil_to_cmos_mm": distance_um / 1000.0,
            "numerical_na": float(na),
            "diffraction_fwhm_um": float(0.51 * wavelength_um / na),
            "screen_side_um": float(c37.GRID * float(config["geometry"]["phase_pitch_um"])),
            "object_to_spot_scale_um": float(wavelength_um * distance_um
                                             / (c37.GRID * float(config["geometry"]["phase_pitch_um"]))),
            "core_to_fwhm_ratio": float(side * c37.CMOS_PITCH_UM / (0.51 * wavelength_um / na)),
        },
        "interpretation_rule": "the peak centred box fraction upper bounds any fixed core placement; "
                               "a large peak_box_to_core_ratio means the energy is concentrated but "
                               "misplaced, while a low peak box fraction means the layer cannot "
                               "concentrate at all",
    }
    suffix = f"_{args.tag}" if args.tag else ""
    c37.write_json(args.work_dir / f"c37_spot_limit{suffix}.json", payload)
    print(json.dumps({"best": best, "reference": reference, "analytic": payload["analytic"]},
                     indent=2))
    return payload


def stage_oracle(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Class conditional focus oracle: how much core energy a phase can reach at all.

    Four separate phases are fitted, each seeing only one class and rewarded only
    for putting that class inside its own frozen core.  The four phase ensemble is
    not a single layer design and is never reported as one; it bounds what any
    single layer could reach if the four classes did not have to share one screen.
    """
    device = context["device"]
    fit_per_class = int(args.rows_per_class or 64)
    eval_per_class = int(args.eval_per_class or 125)
    steps = int(args.steps or 120)
    lr = float(args.learning_rate or config["training"]["learning_rate"])
    microbatch = int(config["training"]["microbatch_size"])
    masks = c37.core_masks(float(config["core"]["frozen_fraction"]), device)
    init = c37.load_frozen_phase(resolve_init_phase(args, config)).to(device)
    eta_target = c37.derive_eta_target(
        config, json.loads((args.work_dir / "c37_p0_spot_diagnostic.json").read_text(
            encoding="utf-8")))
    multiplier = 0.0
    train_rows = c37.balanced_rows(context["split"]["train"], context["labels"], fit_per_class)
    eval_rows = c37.balanced_rows(context["split"]["validation"], context["labels"],
                                 eval_per_class)
    records = []
    for label in range(4):
        rows = train_rows[context["labels"][train_rows] == label]
        phase = init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=lr)
        generator = torch.Generator().manual_seed(1000 + label)
        history = []
        for step in range(steps):
            index = torch.randint(0, rows.numel(), (microbatch,), generator=generator)
            batch = rows[index]
            labels = torch.full((batch.numel(),), label, device=device, dtype=torch.long)
            fields = c37.input_fields(context, batch)
            result = c37.forward_batch(fields, phase, context)
            terms = c37.spot_terms(result, labels, masks, context)
            focus = -torch.log(terms["core_fraction"].clamp_min(1e-9)).mean()
            sidelobe = (terms["wrong_roi_fraction"].mean()
                        + terms["outside_roi_fraction"].mean()
                        + 0.5 * terms["own_roi_outside_core_fraction"].mean())
            shortfall = torch.relu(eta_target - terms["eta"]).mean()
            smooth = (torch.sin((phase[..., 1:] - phase[..., :-1]) / 2.0).abs().mean()
                      + torch.sin((phase[..., 1:, :] - phase[..., :-1, :]) / 2.0).abs().mean())
            loss = (focus + sidelobe + multiplier * shortfall
                    + 0.5 * 200.0 * shortfall.square() + 0.001 * smooth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if phase.grad is None or not torch.isfinite(phase.grad).all():
                raise RuntimeError(f"oracle class {label} step {step}: bad gradient")
            optimizer.step()
            multiplier += 200.0 * float(shortfall.detach())
            if step % 20 == 0 or step == steps - 1:
                history.append({"step": step, "loss": float(loss),
                                "core_fraction": float(terms["core_fraction"].mean()),
                                "eta": float(terms["eta"].mean()),
                                "multiplier": float(multiplier)})
        subset = eval_rows[context["labels"][eval_rows] == label]
        evaluation = c37.evaluate_phase(context, config, phase.detach(), subset,
                                        tag=f"oracle_class{label}", batch_size=4, verbose=False)
        metrics = evaluation["metrics"]
        oracle_dir = args.work_dir / (f"oracle_{args.tag}" if args.tag else "oracle")
        phase_path = oracle_dir / f"phase_class{label}.npy"
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(phase_path, phase.detach().cpu().numpy().astype(np.float32), allow_pickle=False)
        records.append({
            "class": label,
            "phase_path": str(phase_path),
            "steps": steps,
            "learning_rate": lr,
            "fit_rows": int(rows.numel()),
            "eval_rows": int(subset.numel()),
            "history": history,
            "accuracy": metrics["accuracy"],
            "core_fraction_mean": metrics["core_fraction"]["mean"],
            "core_fraction_p50": metrics["core_fraction"]["p50"],
            "target_roi_share_of_cmos_p50": metrics["target_roi_share_of_cmos"]["p50"],
            "peak_localisation_rate": metrics["peak_localisation_rate"],
            "centroid_localisation_rate": metrics["centroid_localisation_rate"],
            "eta_mean": metrics["eta_phase_to_cmos"]["mean"],
            "eta_p10": metrics["eta_phase_to_cmos"]["p10"],
            "peak_distance_to_target_centre_um_p50":
                metrics["peak_distance_to_target_centre_um"]["p50"],
        })
    joint = json.loads((args.work_dir / "c37_selected_candidate.json").read_text(encoding="utf-8"))
    joint_metrics = joint["validation"]
    payload = {
        "stage": "oracle",
        "interpretation": "class conditional upper bound, not a single layer design",
        "population": {"fit_per_class": fit_per_class, "eval_per_class": eval_per_class,
                       "eval_role": "development validation split reused from C3.4/C3.5"},
        "records": records,
        "oracle_core_fraction_mean": float(np.mean([r["core_fraction_mean"] for r in records])),
        "oracle_core_fraction_p50": float(np.median([r["core_fraction_p50"] for r in records])),
        "oracle_accuracy": float(np.mean([r["accuracy"] for r in records])),
        "oracle_eta_mean": float(np.mean([r["eta_mean"] for r in records])),
        "joint_single_layer": {
            "candidate": joint["candidate"],
            "core_fraction_p50": joint_metrics["core_fraction"]["p50"],
            "core_fraction_mean": joint_metrics["core_fraction"]["mean"],
            "accuracy": joint_metrics["accuracy"],
            "eta_mean": joint_metrics["eta_phase_to_cmos"]["mean"],
        },
        "acceptance_core_fraction_median_min": float(
            config["acceptance"]["core_fraction_median_min"]),
        "verdict_rule": "if the class conditional oracle median core fraction stays below the "
                        "acceptance target, the target is not reachable on this frozen geometry; "
                        "if it clears the target, the gap is multi task interference and the "
                        "single layer schedule, not the optics",
    }
    suffix = f"_{args.tag}" if args.tag else ""
    c37.write_json(args.work_dir / f"c37_oracle_focus{suffix}.json", payload)
    print(json.dumps({key: payload[key] for key in
                      ("oracle_core_fraction_mean", "oracle_core_fraction_p50",
                       "oracle_accuracy", "oracle_eta_mean")}, indent=2))
    for record in records:
        print(f"  class {record['class']} core {record['core_fraction_mean']:.4f} "
              f"p50 {record['core_fraction_p50']:.4f} acc {record['accuracy']:.3f} "
              f"peak {record['peak_localisation_rate']:.3f} eta {record['eta_mean']:.4f} "
              f"dist {record['peak_distance_to_target_centre_um_p50']:.1f}")
    return payload


def stage_select(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """P3.5 constrained selection: classification first, throughput second, focus third."""
    work_dir = args.work_dir
    acceptance = config["acceptance"]
    p0 = json.loads((work_dir / "c37_p0_spot_diagnostic.json").read_text(encoding="utf-8"))
    baseline = p0["metrics"]
    eta_baseline = float(baseline["eta_phase_to_cmos"]["mean"])
    candidates = []
    for summary_path in sorted((work_dir / "train").glob("*/training_summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = data["final_validation"]
        candidates.append({
            "candidate": f"C37_{data['arm']}_seed{data['seed']}",
            "arm": data["arm"], "seed": int(data["seed"]), "mode": data["mode"],
            "groups": data["groups"], "epochs": data["epochs"],
            "phase_path": data["phase_path"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "core_fraction_p50": metrics["core_fraction"]["p50"],
            "core_fraction_mean": metrics["core_fraction"]["mean"],
            "target_roi_share_of_cmos_p50": metrics["target_roi_share_of_cmos"]["p50"],
            "target_roi_share_of_four_roi_p50": metrics["target_roi_share_of_four_roi"]["p50"],
            "peak_localisation_rate": metrics["peak_localisation_rate"],
            "centroid_localisation_rate": metrics["centroid_localisation_rate"],
            "outside_target_roi_energy_fraction_mean":
                metrics["outside_target_roi_energy_fraction"]["mean"],
            "eta_mean": metrics["eta_phase_to_cmos"]["mean"],
            "eta_p10": metrics["eta_phase_to_cmos"]["p10"],
            "eta_retention_vs_baseline": metrics["eta_phase_to_cmos"]["mean"] / eta_baseline,
            "core_fraction_gain_vs_baseline": (metrics["core_fraction"]["p50"]
                                               - baseline["core_fraction"]["p50"]),
            "validation": metrics,
        })
    if len(candidates) < 2:
        raise SystemExit("selection needs at least two trained candidates")
    for item in candidates:
        item["passes_classification_gate"] = bool(item["accuracy"] >= acceptance["accuracy_min"])
        item["passes_throughput_gate"] = bool(item["eta_mean"] >= 0.5 * eta_baseline)
    throughput_safe = [item for item in candidates if item["passes_throughput_gate"]]
    both_gates = [item for item in throughput_safe if item["passes_classification_gate"]]
    if both_gates:
        pool = both_gates
        selection_status = "BOTH_GATES_SATISFIED"
    elif throughput_safe:
        pool = throughput_safe
        selection_status = "CLASSIFICATION_GATE_FAILED_THROUGHPUT_SAFE"
    else:
        pool = candidates
        selection_status = "NO_GATE_SATISFIED"
    selected = max(pool, key=lambda item: (round(item["accuracy"], 6),
                                           round(item["core_fraction_p50"], 6),
                                           -item["outside_target_roi_energy_fraction_mean"]))
    accuracy_first = max(candidates, key=lambda item: (item["passes_classification_gate"],
                                                       round(item["accuracy"], 6),
                                                       round(item["core_fraction_p50"], 6)))
    # Pareto front on (accuracy, core fraction, efficiency), all maximised
    front = []
    for item in candidates:
        dominated = False
        for other in candidates:
            if other is item:
                continue
            better_or_equal = (other["accuracy"] >= item["accuracy"]
                               and other["core_fraction_p50"] >= item["core_fraction_p50"]
                               and other["eta_mean"] >= item["eta_mean"])
            strictly_better = (other["accuracy"] > item["accuracy"]
                               or other["core_fraction_p50"] > item["core_fraction_p50"]
                               or other["eta_mean"] > item["eta_mean"])
            if better_or_equal and strictly_better:
                dominated = True
                break
        front.append({"candidate": item["candidate"], "on_pareto_front": not dominated})
    gradcheck_ok = json.loads((work_dir / "c37_gradcheck.json").read_text(
        encoding="utf-8"))["all_checks_passed"]
    improved_focus = bool(
        selected["core_fraction_p50"] > baseline["core_fraction"]["p50"]
        and selected["validation"]["peak_distance_to_target_centre_um"]["p50"]
        < baseline["peak_distance_to_target_centre_um"]["p50"])
    gates = {
        "classification_gate": bool(selected["accuracy"] >= acceptance["accuracy_min"]),
        "throughput_gate": bool(selected["eta_mean"] >= 0.5 * eta_baseline),
        "focus_concentration_gate": bool(selected["core_fraction_p50"]
                                         >= acceptance["core_fraction_median_min"]),
        "peak_localisation_gate": bool(selected["peak_localisation_rate"]
                                       >= acceptance["peak_localisation_min"]),
    }
    evidence = bool(gradcheck_ok and improved_focus)
    payload = {
        "stage": "select",
        "baseline": {"candidate": "C37_E0_frozen_C35_P1", "eta_mean": eta_baseline,
                     "core_fraction_p50": baseline["core_fraction"]["p50"],
                     "peak_localisation_rate": baseline["peak_localisation_rate"],
                     "accuracy": baseline["accuracy"]},
        "bookshelf": "energy collapse is a hard rejection, not a tradeable score; within the throughput "
                     "safe set the highest accuracy wins and focus metrics only break ties",
        "selection_rule": "primary: reject every candidate whose mean efficiency is below half of the "
                          "frozen baseline (a focus gain obtained by switching light off is not a "
                          "focus gain), then take the highest accuracy, breaking ties by core "
                          "fraction and outside energy. The accuracy first ordering is reported as an "
                          "alternative because the pre-registered rule ranked accuracy first while "
                          "the data contain no candidate that satisfies both gates.",
        "selection_status": selection_status,
        "alternative_accuracy_first": accuracy_first["candidate"],
        "alternative_accuracy_first_metrics": {
            "accuracy": accuracy_first["accuracy"],
            "core_fraction_p50": accuracy_first["core_fraction_p50"],
            "eta_mean": accuracy_first["eta_mean"],
            "eta_retention_vs_baseline": accuracy_first["eta_retention_vs_baseline"],
            "passes_classification_gate": accuracy_first["passes_classification_gate"],
            "passes_throughput_gate": accuracy_first["passes_throughput_gate"],
        },
        "candidates": candidates,
        "pareto": front,
        "selected": selected["candidate"],
        "candidate": selected["candidate"],
        "phase_path": selected["phase_path"],
        "validation": selected["validation"],
        "phase_optimisation_evidence": evidence,
        "phase_optimisation_detail": {
            "gradient_checks_passed": bool(gradcheck_ok),
            "focus_improved_over_frozen_baseline": improved_focus,
            "core_fraction_p50_baseline": baseline["core_fraction"]["p50"],
            "core_fraction_p50_selected": selected["core_fraction_p50"],
            "peak_distance_p50_baseline_um":
                baseline["peak_distance_to_target_centre_um"]["p50"],
            "peak_distance_p50_selected_um":
                selected["validation"]["peak_distance_to_target_centre_um"]["p50"],
            "acceptance_gates": gates,
        },
    }
    c37.write_json(work_dir / "c37_selected_candidate.json", payload)
    print(json.dumps({"selected": payload["selected"], "evidence": evidence,
                      "accuracy": selected["accuracy"],
                      "core_fraction_p50": selected["core_fraction_p50"],
                      "peak_localisation_rate": selected["peak_localisation_rate"],
                      "eta_mean": selected["eta_mean"]}, indent=2))
    return payload


def stage_accept(args, config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    work_dir = args.work_dir
    acceptance = config["acceptance"]
    p0 = json.loads((work_dir / "c37_p0_spot_diagnostic.json").read_text(encoding="utf-8"))
    core = json.loads((work_dir / "c37_core_spec.json").read_text(encoding="utf-8"))
    grad = json.loads((work_dir / "c37_gradcheck.json").read_text(encoding="utf-8"))
    ablate = json.loads((work_dir / "c37_ablation_screening.json").read_text(encoding="utf-8"))
    process = json.loads((work_dir / "c37_process_comparison.json").read_text(encoding="utf-8"))
    final = json.loads((work_dir / "c37_selected_candidate.json").read_text(encoding="utf-8"))
    metrics = final["validation"]
    baseline = p0["metrics"]
    selected_candidate = final["candidate"]
    process_record = next((record for record in process["candidates"]
                           if selected_candidate.replace("C37_", "") in record["candidate"]), None)
    if process_record is None:
        raise SystemExit("the selected candidate has no process comparison record")
    bound_path = work_dir / "c37_spot_limit_E3_seed23.json"
    bound = json.loads(bound_path.read_text(encoding="utf-8")) if bound_path.is_file() else None
    oracle_path = work_dir / "c37_oracle_focus.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.is_file() else None
    statuses = {
        "C3.7_P0_SPOT_DIAGNOSTIC_PASS": bool(p0["coordinate_audit"]["all_checks_passed"]),
        "C3.7_TARGET_CORE_DEFINED": bool(core["all_checks_passed"]),
        "C3.7_PHASE_OPTIMIZATION_PASS": bool(final["phase_optimisation_evidence"]),
        "C3.7_ROI_CLASSIFICATION_PASS": bool(metrics["accuracy"] >= acceptance["accuracy_min"]),
        "C3.7_PEAK_LOCALIZATION_PASS": bool(
            metrics["peak_localisation_rate"] >= acceptance["peak_localisation_min"]),
        "C3.7_CENTROID_LOCALIZATION_PASS": bool(
            metrics["centroid_localisation_rate"] >= acceptance["centroid_localisation_min"]),
        "C3.7_FOCUS_CONCENTRATION_PASS": bool(
            metrics["core_fraction"]["p50"] >= acceptance["core_fraction_median_min"]
            and metrics["target_roi_share_of_cmos"]["p50"] >=
            acceptance["target_roi_share_of_cmos_median_min"]),
        "C3.7_ENERGY_CONSTRAINT_PASS": bool(
            metrics["eta_phase_to_cmos"]["mean"] >= 0.5 * baseline["eta_phase_to_cmos"]["mean"]),
        "C3.7_PROCESS_ROBUSTNESS_PASS": bool(not process_record["process_sensitive"]),
        "C3.7_VISUALIZATION_PASS": bool((work_dir / "c37_visualization_manifest.json").is_file()),
    }
    if not statuses["C3.7_FOCUS_CONCENTRATION_PASS"] and oracle is not None:
        statuses["C3.7_FOCUS_CONCENTRATION_BOUNDED_BY_GEOMETRY"] = bool(
            oracle["oracle_core_fraction_p50"] < acceptance["core_fraction_median_min"])
    elif not statuses["C3.7_FOCUS_CONCENTRATION_PASS"] and bound is not None:
        statuses["C3.7_FOCUS_CONCENTRATION_BOUNDED_BY_GEOMETRY"] = bool(
            bound["best"]["peak_box_energy_fraction_mean"]
            < acceptance["core_fraction_median_min"])
    if statuses["C3.7_ROI_CLASSIFICATION_PASS"] and not statuses["C3.7_FOCUS_CONCENTRATION_PASS"]:
        statuses["C3.7_CLASSIFICATION_PASS_FOCUS_INSUFFICIENT"] = True
    if (metrics["core_fraction"]["p50"] > baseline["core_fraction"]["p50"]
            and not statuses["C3.7_ENERGY_CONSTRAINT_PASS"]):
        statuses["C3.7_FOCUS_IMPROVED_THROUGHPUT_FAILED"] = True
    statuses["FABRICATION_READY"] = False
    statuses["HW207_CURRENT_MAPPING"] = False
    payload = {
        "stage": "accept",
        "population_roles": {
            "train": "fit only",
            "validation": "development validation reused from C3.4/C3.5 selection; used for model "
                          "selection in this configuration, therefore not an independent test set",
            "independent_eval": config["data"]["independent_eval_status"],
        },
        "statuses": statuses,
        "selected_candidate": final["candidate"],
        "selection_rule": final["selection_rule"],
        "pareto": final["pareto"],
        "candidate_table": [{key: item[key] for key in
                             ("candidate", "accuracy", "core_fraction_p50",
                              "peak_localisation_rate", "centroid_localisation_rate",
                              "target_roi_share_of_cmos_p50", "eta_mean", "eta_p10",
                              "eta_retention_vs_baseline")}
                            for item in final["candidates"]],
        "metrics": metrics,
        "baseline_metrics": {key: baseline[key] for key in
                             ("accuracy", "peak_localisation_rate", "centroid_localisation_rate")},
        "gradcheck_all_passed": grad["all_checks_passed"],
        "ablation_records": ablate["records"],
        "process_sensitive_candidates": [record["candidate"] for record in process["candidates"]
                                         if record["process_sensitive"]],
        "selected_process_record": {
            "candidate": process_record["candidate"],
            "ideal_accuracy": process_record["ideal"]["accuracy"],
            "worst_accuracy": process_record["worst_accuracy"],
            "worst_peak_localisation_rate": process_record["worst_peak_localisation_rate"],
            "process_sensitive": process_record["process_sensitive"],
        },
        "focus_oracle": None if oracle is None else {
            "interpretation": oracle["interpretation"],
            "oracle_core_fraction_p50": oracle["oracle_core_fraction_p50"],
            "oracle_core_fraction_mean": oracle["oracle_core_fraction_mean"],
            "oracle_eta_mean": oracle["oracle_eta_mean"],
            "per_class_core_fraction_mean": [record["core_fraction_mean"]
                                             for record in oracle["records"]],
            "per_class_peak_localisation_rate": [record["peak_localisation_rate"]
                                                 for record in oracle["records"]],
            "joint_core_fraction_p50": oracle["joint_single_layer"]["core_fraction_p50"],
            "joint_to_oracle_ratio": float(oracle["joint_single_layer"]["core_fraction_p50"]
                                           / max(oracle["oracle_core_fraction_p50"], 1e-9)),
            "acceptance_core_fraction_median_min":
                oracle["acceptance_core_fraction_median_min"],
            "verdict": oracle["verdict_rule"],
        },
        "focus_concentration_bound": None if bound is None else {
            "max_core_sized_peak_box_fraction_of_any_tested_phase":
                bound["best"]["peak_box_energy_fraction_mean"],
            "best_phase_family": bound["quadratic_phase_family"],
            "zero_phase_peak_box_fraction": bound["reference"]["zero"]["peak_box_energy_fraction_mean"],
            "selected_phase_peak_box_fraction":
                bound["reference"]["p1_frozen"]["peak_box_energy_fraction_mean"],
            "selected_phase_fixed_core_fraction":
                bound["reference"]["p1_frozen"]["fixed_core_energy_fraction_mean"],
            "core_to_diffraction_fwhm_ratio": bound["analytic"]["core_to_fwhm_ratio"],
            "object_to_spot_scale_um": bound["analytic"]["object_to_spot_scale_um"],
            "conclusion": "the peak centred box scan is a lower bound probe only: an extended blob "
                          "can fill a fixed core without a dominant peak, so this scan does not "
                          "upper bound the fixed core fraction. The class conditional focus oracle "
                          "in focus_oracle is the authoritative bound.",
        },
        "notes": [
            "statuses are recorded separately and never merged into one overall PASS",
            "FABRICATION_READY and HW207_CURRENT_MAPPING stay blocked: no material, printer or "
            "bench measurement was made in this round",
        ],
    }
    c37.write_json(work_dir / "c37_acceptance.json", payload)
    print(json.dumps(statuses, indent=2))
    return payload


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
STAGES = {
    "bench": stage_bench,
    "corecheck": stage_corecheck,
    "p0": stage_p0,
    "gradcheck": stage_gradcheck,
    "ablate": stage_ablate,
    "train": stage_train,
    "finalize": stage_finalize,
    "select": stage_select,
    "oracle": stage_oracle,
    "spotlimit": stage_spotlimit,
    "fabricate": stage_fabricate,
    "vizdata": stage_vizdata,
    "accept": stage_accept,
}


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = c37.load_config(args.config)
    if args.core_fraction is not None:
        config["core"]["frozen_fraction"] = float(args.core_fraction)
        print(f"C3.7 core fraction override {config['core']['frozen_fraction']}", flush=True)
    if (args.focus_weight is not None or args.focus_mode is not None
            or args.eta_retention is not None):
        config = copy.deepcopy(config)
        if args.focus_weight is not None:
            config["training"]["focus_weight"] = float(args.focus_weight)
        if args.focus_mode is not None:
            config["training"]["focus_mode"] = str(args.focus_mode)
        if args.eta_retention is not None:
            config["energy"]["eta_retention_fraction"] = float(args.eta_retention)
        print(f"C3.7 objective override focus_weight={config['training']['focus_weight']} "
              f"focus_mode={config['training'].get('focus_mode', 'mean')} "
              f"eta_retention={config['energy']['eta_retention_fraction']}", flush=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.stage == "bench":
        context = c37.build_context(config, device)
        stage_bench(args, config, context)
        return
    context = c37.build_context(config, device)
    print(f"C3.7 stage {args.stage} device={device} work_dir={args.work_dir}", flush=True)
    split = context["split"]
    cache_rows = torch.cat([split["train"], split["validation"]])
    metadata = c37.prepare_field_cache(context, cache_rows,
                                       cache_path=args.work_dir / "cache" / "input_field_cache.pt")
    print(f"input field cache rows={metadata['rows']} "
          f"seconds={metadata['elapsed_seconds']:.1f}", flush=True)
    STAGES[args.stage](args, config, context)


if __name__ == "__main__":
    main()