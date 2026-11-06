"""C3.3 P6 phase export, P7 height/STL design, P8 fabrication-aware re-simulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as core

# Material inputs required by the baseline note but absent from the repository.
MATERIAL_INPUTS = {
    "resist_model": "UNKNOWN",
    "n_structure_532nm": "UNKNOWN",
    "n_substrate_532nm": "UNKNOWN",
    "surrounding_medium": "UNKNOWN",
    "printer_model": "UNKNOWN",
    "lateral_resolution_um": "UNKNOWN",
    "axial_resolution_um": "UNKNOWN",
    "max_height_um": "UNKNOWN",
    "min_feature_um": "UNKNOWN",
    "shrinkage": "UNKNOWN",
}
ASSUMED_SCENARIOS = {"delta_n": [0.50, 0.60], "max_height_um": [1.0, 2.0],
                     "note": "ASSUMPTION_ONLY, not a calibration and not a printable design"}


def circular_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    difference = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
    return float(difference.square().mean().sqrt())


def write_phase_artifacts(phase: torch.Tensor, output: Path, metadata: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    raw = phase.detach().cpu().double()
    wrapped = core.wrap_phase(raw)
    normalized = wrapped / (2 * math.pi)
    np.save(output / "phase_rad.npy", raw.numpy())
    np.save(output / "phase_wrapped.npy", wrapped.numpy())
    np.save(output / "phase_normalized.npy", normalized.numpy())
    flat = wrapped[0, 0].numpy()
    np.savetxt(output / "phase.csv", flat, delimiter=",", fmt="%.9f")
    _write_phase_png(flat, output / "phase.png")
    metadata = dict(metadata)
    metadata.update({
        "grid": [core.GRID, core.GRID],
        "pitch_um": core.PITCH_M * 1e6,
        "aperture_um": [core.GRID * core.PITCH_M * 1e6] * 2,
        "wavelength_nm": core.WAVELENGTH_M * 1e9,
        "phase_units": "radians",
        "phase_range_wrapped": [float(wrapped.min()), float(wrapped.max())],
        "screen_plane": "c31_screen_layer_1 (relay source plane, z=0)",
        "coordinate_origin": "chip centre; +x right, +y down as produced by the C2.1 affine",
        "phase_fingerprint": core.tensor_sha256(raw),
    })
    core.write_json(output / "phase_metadata.json", metadata)
    return metadata


def _write_phase_png(values: np.ndarray, path: Path) -> None:
    import struct
    import zlib

    scaled = np.clip((values / (2 * math.pi) * 255.0).round(), 0, 255).astype(np.uint8)
    height, width = scaled.shape
    raw = b"".join(b"\x00" + scaled[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path.write_bytes(png)


def height_artifacts(phase: torch.Tensor, output: Path, delta_n: float, max_height_um: float,
                     tag: str) -> dict:
    wrapped = core.wrap_phase(phase.detach().cpu().double())
    height = core.phase_to_height_um(wrapped, delta_n)
    h2pi = core.height_two_pi_um(delta_n)
    clipped = height.clamp(0, max_height_um)
    triangles, mesh_info = core.heightmap_triangles(clipped[0, 0].numpy(), substrate_um=max_height_um)
    stl_path = output / f"phase_height_{tag}.stl"
    core.write_binary_stl(stl_path, triangles.astype(np.float32).astype(np.float64))
    reread, _ = core.read_binary_stl(stl_path)
    mesh_check = core.check_mesh(reread)
    np.save(output / f"height_{tag}.npy", height[0, 0].numpy())
    row_grad = (height[0, 0, 1:, :] - height[0, 0, :-1, :]).abs()
    col_grad = (height[0, 0, :, 1:] - height[0, 0, :, :-1]).abs()
    return {
        "tag": tag,
        "delta_n": delta_n,
        "max_height_um": max_height_um,
        "two_pi_height_um": h2pi,
        "height_range_um": [float(height.min()), float(height.max())],
        "height_clipped_fraction": float((height != clipped).double().mean()),
        "max_slope_um_per_pixel": float(max(row_grad.max(), col_grad.max())),
        "max_slope_rad": float(math.atan(float(max(row_grad.max(), col_grad.max())) / (core.PITCH_M * 1e6))),
        "stl_path": str(stl_path.name),
        "stl_bytes": stl_path.stat().st_size,
        "mesh": mesh_info,
        "mesh_check": mesh_check,
        "units": "micrometre",
    }


def fab_simulation(phase: torch.Tensor, reference: torch.Tensor, configuration, images, labels, batch_size: int,
                   constraints: dict, delta_n: float, generator: torch.Generator) -> dict:
    """Apply fabrication constraints to the raw design and re-run the same optical path.

    ``phase`` is the array handed to the printer; ``reference`` is the phase the
    design was trained to realise (equal to ``phase`` unless training included a
    printer-resolution operator).
    """
    wrapped = core.wrap_phase(phase.detach().cpu().double())
    height = core.phase_to_height_um(wrapped, delta_n)
    smoothed = core.smooth_height(height, constraints["smooth_sigma_px"])
    quantized = core.quantize_height(smoothed, constraints["levels"], constraints["max_height_um"])
    opened = core.min_feature_open(quantized, constraints["min_feature_radius_px"])
    clipped = opened.clamp(0, constraints["max_height_um"])
    perturbed = core.perturb_height(clipped, constraints["print_error_um"], generator)
    height_stages = {"smoothed": smoothed, "quantized": quantized,
                     "min_feature": opened, "clipped": clipped, "perturbed": perturbed}
    results = {}

    def measure(name: str, phase_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            evaluation = core.evaluate(images, labels, configuration, phase_tensor, batch_size)
        results[name] = {
            "accuracy": evaluation["metrics"]["accuracy"],
            "macro_f1": evaluation["metrics"]["macro_f1"],
            "mean_detected_energy": evaluation["metrics"]["mean_detected_energy"],
            "outside_fraction": evaluation["metrics"]["outside_fraction"],
            "roi_energy_matrix": evaluation["metrics"]["roi_energy_matrix"],
            "phase_rms_error_vs_ideal_rad": circular_rms(
                core.wrap_phase(phase_tensor)[0, 0], wrapped.to(phase_tensor.device)[0, 0]),
            "height_clipped_fraction": float(
                (height.to(phase_tensor.device) != height.clamp(0, constraints["max_height_um"])
                 .to(phase_tensor.device)).double().mean()) if name == "ideal" else None,
        }

    measure("ideal", reference.detach().to(phase.device).to(phase.dtype))
    for name, values in height_stages.items():
        measure(name, core.height_to_phase(values, delta_n).to(phase.device).to(phase.dtype))
    return {"constraints": constraints, "delta_n": delta_n,
            "max_height_um": constraints["max_height_um"], "stages": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True, help="phase.pt written by the trainer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--levels", type=int, default=64)
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--min-feature-radius", type=int, default=1)
    parser.add_argument("--print-error-um", type=float, default=0.02)
    parser.add_argument("--max-height-um", type=float, default=1.2)
    parser.add_argument("--delta-n", type=float, nargs="*", default=[0.5, 0.6])
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()

    payload = torch.load(args.phase, map_location="cpu", weights_only=False)
    training_sigma = float(payload.get("smooth_sigma", 0.0))
    if training_sigma > 0 and payload.get("effective") is not None:
        # Training already applied the printer-resolution operator; the realised
        # design is the smoothed phase and must not be blurred a second time.
        phase = payload["effective"].double()
        printer_sigma = 0.0
    else:
        phase = payload["phase"].double()
        printer_sigma = float(args.smooth_sigma)
    reference = phase
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    metadata = {
        "training_init": payload.get("init"), "training_seed": payload.get("seed"),
        "best_epoch": payload.get("best_epoch"), "defocus_um": payload.get("defocus_um"),
        "train_rows": int(len(payload.get("train_rows", []))),
        "validation_rows": int(len(payload.get("validation_rows", []))),
        "holdout_rows": int(len(payload.get("holdout_rows", []))),
        "optical_model": "hw207_mnist4_c31_c32_cmos_v1 (ideal 4f relay + camera defocus)",
        "phase_domain": [0.0, 2 * math.pi],
    }
    export_meta = write_phase_artifacts(phase, output, metadata)

    heights, scenarios = [], []
    for delta_n in args.delta_n:
        tag = f"dn{str(delta_n).replace('.', 'p')}"
        record = height_artifacts(phase, output, delta_n, args.max_height_um, tag)
        heights.append(record)
        scenarios.append({"delta_n": delta_n, "two_pi_height_um": record["two_pi_height_um"],
                          "height_range_um": record["height_range_um"]})
    design = {"identity": "HW207_C33_PHASE_TO_HEIGHT", "material_inputs": MATERIAL_INPUTS,
              "assumed_scenarios": ASSUMED_SCENARIOS, "scenarios": heights,
              "final_printable_stl": "BLOCKED",
              "blocked_reason": "resist/printer/refractive-index inputs are UNKNOWN in the repository; "
                                "the emitted STL is an assumption scenario only",
              "units": "micrometre"}
    core.write_json(output / "height_design.json", design)

    images_all, labels_all = core.load_mnist()
    holdout = torch.tensor(payload["holdout_rows"])
    images = images_all[holdout].to(args.device)
    labels = labels_all[holdout].to(args.device)
    configuration = core.build_configuration(payload["defocus_um"], args.device)
    generator = torch.Generator().manual_seed(args.seed)
    constraints = {"levels": args.levels, "smooth_sigma_px": printer_sigma,
                   "min_feature_radius_px": args.min_feature_radius,
                   "print_error_um": args.print_error_um, "max_height_um": args.max_height_um}
    fab = fab_simulation(phase.to(args.device), reference.to(args.device), configuration, images, labels, args.batch,
                         constraints, args.delta_n[0], generator)
    ideal = fab["stages"]["ideal"]["accuracy"]
    perturbed = fab["stages"]["perturbed"]["accuracy"]
    fab["identity"] = "HW207_C33_FABRICATION_AWARE_RESIMULATION"
    fab["holdout_rows"] = int(len(holdout))
    fab["retention_pp"] = (perturbed - ideal) * 100
    fab["status"] = ("FABRICATION_AWARE_SIMULATION_PASS" if perturbed >= ideal - 0.02
                     else "FABRICATION_AWARE_SIMULATION_DEGRADED")
    core.write_json(output / "fabrication_simulation.json", fab)
    core.write_json(output / "export_summary.json", {"phase_metadata": export_meta,
                                                     "height_design": design, "fabrication": fab})
    print(json.dumps({"export": str(output), "stl_status": "PARAMETRIC_ASSUMPTION_ONLY",
                      "fab_status": fab["status"], "accuracy_retention_pp": fab["retention_pp"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()