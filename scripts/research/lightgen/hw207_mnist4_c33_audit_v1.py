"""C3.3 P0 audit and P2 gradient/physics verification; read-only on history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as core

INIT_SCENARIOS = {
    "zero": lambda shape, device, gen: torch.zeros(shape, device=device),
    "small": lambda shape, device, gen: (torch.rand(shape, generator=gen) - 0.5).to(device) * (2 * 0.1 * math.pi),
    "uniform": lambda shape, device, gen: torch.rand(shape, generator=gen).to(device) * (2 * math.pi),
}


def abcd_1lens(d1_m: float, f_m: float, d2_m: float) -> dict:
    """P(d2) L(f) P(d1) paraxial matrix, matching the recorded tape geometry."""
    a = 1 - d2_m / f_m
    b = d1_m + d2_m - d1_m * d2_m / f_m
    c = -1 / f_m
    d = 1 - d1_m / f_m
    return {"A": a, "B_m": b, "C_per_m": c, "D": d, "magnification": a, "determinant": a * d - b * c}


def physical_audit() -> dict:
    """Static audit of where the phase screen sits and how the model propagates."""
    import inspect

    from ldm_project.v2.optics import d2nn, relays

    config = core.build_configuration(0.0, "cpu")
    base = config.base
    screen_source = inspect.getsource(d2nn.FiniteCropPhaseStack.forward)
    spacing = base.screen.spacing_m
    relay = base.relay
    real_geometry = abcd_1lens(80e-3, 30e-3, 100e-3)
    conjugate = 1.0 / (1.0 / 30e-3 - 1.0 / 80e-3)
    return {
        "identity": "HW207_C33_P0_AUDIT",
        "screen": {
            "class": "ldm_project.v2.optics.d2nn.FiniteCropPhaseStack",
            "layers": int(base.screen.layers),
            "active_shape": list(base.screen.active_shape),
            "propagation_shape": list(base.screen.propagation_shape),
            "spacing_m": float(spacing),
            "phase_shape": list(base.screen.phase.shape),
            "phase_initial": "zeros",
            "frozen_in_c31_c32": True,
            "forward_reference": "scripts/research/lightgen/hw207_mnist4_c31_c32_cmos_v1.py:85",
        },
        "planes": {
            "input_plane": base.input_plane.to_dict(),
            "relay_source": relay.source.to_dict(),
            "relay_target": relay.target.to_dict(),
            "relay_segments_m": list(relay.segments_m),
            "relay_focal_lengths_m": list(relay.focal_lengths_m),
            "relay_clear_aperture_m": float(relay.clear_aperture_m),
            "relay_magnification": float(relay.magnification),
            "relay_abcd": relay.abcd.to_dict(),
            "relay_source_cutoff_cycles_m": float(relay.source_cutoff_cycles_m),
            "relay_target_cutoff_cycles_m": float(relay.target_cutoff_cycles_m),
        },
        "camera": {"defocus_um": float(config.defocus_um), "padded_shape": [512, 512],
                   "crop": [128, 384, 128, 384], "sensor_shape": [768, 768],
                   "rois": [list(r) for r in base.rois]},
        "measured_output_geometry": {
            "source": "research_assets/lightgen/HW207_实物与距离基线_20261103/实物信息与闭环状态.md:41",
            "model": "P(100mm) L(f=30mm) P(80mm)",
            "abcd": real_geometry,
            "conjugate_image_distance_mm": conjugate * 1e3,
            "magnification_at_conjugate": -conjugate / 80e-3,
            "defocus_vs_conjugate_mm": (100e-3 - conjugate) * 1e3,
            "status": "TAPE_MEASURED_NOMINAL_NOT_CALIBRATED",
        },
        "contract_checks": {
            "screen_output_is_relay_source": base.relay.source.plane_id == "c31_screen_layer_1",
            "ideal_4f_assumption": True,
            "c31_spacing_zero_means_identity": True,
        },
        "historical_entry_points": {
            "c31_preflight": "hw207_mnist4_c31_phase_preflight_v1.py uses forward_hw207_mnist4.__wrapped__",
            "production_forward_decorator": "torch.no_grad on forward_hw207_mnist4",
            "consequence": "preflight bypassed the no_grad wrapper; C3.3 uses one undecorated forward for both",
        },
        "c33_model_changes": [
            "phase is a trainable parameter (requires_grad=True)",
            "phase applied as exp(i*phi) at the relay source plane, spacing 0 (identity)",
            "camera-plane defocus retained as an explicit free parameter",
            "single undecorated forward_field shared by preflight and training",
        ],
    }


def finite_difference(base_loss_fn, phase: torch.Tensor, entries: torch.Tensor, h: float) -> torch.Tensor:
    values = []
    with torch.no_grad():
        original = phase.flatten().clone()
        for index in entries.tolist():
            phase.flatten()[index] = original[index] + h
            plus = float(base_loss_fn())
            phase.flatten()[index] = original[index] - h
            minus = float(base_loss_fn())
            phase.flatten()[index] = original[index]
            values.append((plus - minus) / (2 * h))
    return torch.tensor(values)


def gradcheck(output: Path, device: str, defocus_list, entries: int, h: float) -> dict:
    images_all, labels_all = core.load_mnist()
    rows = core.c0_train_indices()[:64]
    images = images_all[rows].to(device)
    labels = labels_all[rows].to(device)
    report: dict = {"identity": "HW207_C33_P2_GRADIENT", "finite_difference_step_rad": h,
                    "rows": rows.tolist(), "conditions": {}}
    for defocus_um in defocus_list:
        for name, factory in INIT_SCENARIOS.items():
            generator = torch.Generator().manual_seed(7)
            phase = factory((1, 1, core.GRID, core.GRID), device, generator).requires_grad_(True)
            configuration = core.build_configuration(defocus_um, device)
            with torch.no_grad():
                zero_phase = torch.zeros_like(phase)
                baseline = core.forward_field(images, configuration, zero_phase)
                perturbed = core.forward_field(images, configuration, phase)
            identity_delta = float(
                (perturbed["camera_ideal_intensity"] - baseline["camera_ideal_intensity"])
                .abs().mean() / baseline["camera_ideal_intensity"].mean().clamp_min(1e-30))

            def loss_fn() -> torch.Tensor:
                result = core.forward_field(images, configuration, phase)
                return core.loss_from_result({**result, "phase": phase}, labels)

            loss = loss_fn()
            gradient = torch.autograd.grad(loss, phase, allow_unused=False)[0]
            flat = gradient.detach().abs().flatten()
            chosen = torch.topk(flat, min(entries, flat.numel())).indices.cpu()
            numeric = finite_difference(loss_fn, phase, chosen, h).to(device)
            analytic = gradient.flatten()[chosen.to(device)]
            scale = analytic.abs().max().clamp_min(1e-12)
            relative = (analytic - numeric).abs() / scale
            global_shift = float(
                (core.forward_field(images, configuration, phase.detach() + 1.234)
                 ["camera_ideal_intensity"] - perturbed["camera_ideal_intensity"]).abs().max()
                / perturbed["camera_ideal_intensity"].max().clamp_min(1e-30))
            key = f"defocus_{int(defocus_um)}um__{name}"
            report["conditions"][key] = {
                "defocus_um": float(defocus_um),
                "init": name,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient.norm()),
                "gradient_max_abs": float(gradient.abs().max()),
                "finite_difference_max_abs": float(numeric.abs().max()),
                "relative_error_max": float(relative.max()),
                "relative_error_median": float(relative.median()),
                "identity_intensity_change_rel": identity_delta,
                "global_phase_shift_intensity_change_rel": global_shift,
                "gradient_pass": bool(float(gradient.norm()) > 1e-5 and float(relative.max()) < 0.05),
                "saddle_at_zero_phase": bool(name == "zero" and float(gradient.norm()) < 1e-5),
            }
            print(key, json.dumps(report["conditions"][key]), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    core.write_json(output / "gradient_report.json", report)
    return report


def precision_check(device: str) -> dict:
    """FP32 vs FP64 agreement of the same forward and phase."""
    images_all, _ = core.load_mnist()
    rows = core.c0_train_indices()[:8]
    images = images_all[rows].to(device)
    configuration = core.build_configuration(0.0, device)
    generator = torch.Generator().manual_seed(3)
    phase = torch.rand((1, 1, core.GRID, core.GRID), generator=generator).to(device) * (2 * math.pi)
    with torch.no_grad():
        single = core.forward_field(images, configuration, phase)
        phase64 = phase.double()
        configuration.base.screen.double()
        configuration.base.camera_matrix = configuration.base.camera_matrix.double()
        try:
            double = core.forward_field(images, configuration, phase64)
            intensity_error = float(
                (double["camera_ideal_intensity"] - single["camera_ideal_intensity"].double()).abs().max()
                / double["camera_ideal_intensity"].max().clamp_min(1e-30))
            score_error = float(
                (double["roi_scores"] - single["roi_scores"].double()).abs().max()
                / double["roi_scores"].max().clamp_min(1e-30))
            status = "FP32_FP64_CONSISTENT" if intensity_error < 1e-5 else "FP32_FP64_DIVERGENT"
        except Exception as error:  # noqa: BLE001 - diagnostic path
            intensity_error = None
            score_error = None
            status = f"FP64_UNSUPPORTED: {type(error).__name__}: {error}"
    return {"status": status, "intensity_relative_error": intensity_error,
            "score_relative_error": score_error}


def defocus_scan(output: Path, device: str) -> dict:
    """Measure how the same random phase converts to intensity versus camera defocus."""
    images_all, labels_all = core.load_mnist()
    rows = core.c0_train_indices()[:32]
    images = images_all[rows].to(device)
    scan = {}
    for defocus_um in (0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0, 52000.0):
        configuration = core.build_configuration(defocus_um, device)
        generator = torch.Generator().manual_seed(5)
        phase = torch.rand((1, 1, core.GRID, core.GRID), generator=generator).to(device) * (2 * math.pi)
        with torch.no_grad():
            zero = core.forward_field(images, configuration, torch.zeros_like(phase))
            random = core.forward_field(images, configuration, phase)
        t0 = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        change = float((random["camera_ideal_intensity"] - zero["camera_ideal_intensity"]).abs().mean()
                       / zero["camera_ideal_intensity"].mean().clamp_min(1e-30))
        scan[f"{defocus_um:g}"] = {
            "defocus_um": defocus_um,
            "intensity_change_rel": change,
            "zero_phase_detected_over_input": float(
                (zero["detected_energy"] / zero["energy"]["lightgen_input_energy_um2"].clamp_min(1e-30)).mean()),
            "roi_scores_zero_phase": zero["roi_scores"].mean(0).tolist(),
        }
        print("defocus", defocus_um, json.dumps(scan[f"{defocus_um:g}"]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    core.write_json(output / "defocus_scan.json", {"identity": "HW207_C33_P2_DEFOCUS_SCAN", "scan": scan})
    return scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "gradient", "scan", "precision", "all"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--entries", type=int, default=8)
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--defocus", type=float, nargs="*", default=[0.0, 50.0, 1000.0, 52000.0])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("audit", "all"):
        core.write_json(args.output / "audit.json", physical_audit())
        print("audit written", flush=True)
    if args.stage in ("scan", "all"):
        defocus_scan(args.output / "scan", args.device)
    if args.stage in ("gradient", "all"):
        gradcheck(args.output / "gradient", args.device, args.defocus, args.entries, args.step)
    if args.stage in ("precision", "all"):
        core.write_json(args.output / "precision.json", precision_check(args.device))


if __name__ == "__main__":
    main()