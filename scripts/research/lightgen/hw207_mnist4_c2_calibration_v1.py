"""C2 calibration patterns and numerical sampling diagnostic, no hardware control."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_v1 import (
    calibration_patterns, coherent_pupil, energy_account, target_plane_samples,
)


def analyze(pattern, magnification, fill, device):
    results = []
    previous = None
    representative = None
    for subdivision in (4, 8, 16, 32):
        field, _ = target_plane_samples(pattern.to(device), magnification, subdivision, fill)
        integrated_intensity = F.avg_pool2d(field[:, None].square(), subdivision)[:, 0]
        energy = energy_account(pattern.to(device), field, magnification, fill)
        result = {"subdivision": subdivision,
                  "incident_um2": energy["incident_um2"].item(),
                  "mirror_on_um2": energy["on_um2"].item(),
                  "active_um2": energy["active_um2"].item(),
                  "target_um2": energy["target_window_um2"].item()}
        if previous is not None:
            denominator = energy["active_um2"].item()
            difference = (integrated_intensity - previous).abs().sum().item() * 4.0
            result["successive_intensity_l1_over_active_power"] = difference / denominator if denominator else None
        previous = integrated_intensity
        if subdivision == 8:
            representative = field
        results.append(result)
    pitch = 2.0 / 8
    pad = representative.shape[-1] // 2
    padded = F.pad(representative.to(torch.complex64), (pad, pad, pad, pad))
    area = pitch ** 2
    before = padded.abs().square().sum().item() * area
    optics = {}
    for cutoff in (0.05, 0.1, 0.2):
        for defocus in (0.0, 50.0):
            output = coherent_pupil(padded, pitch, cutoff, defocus)
            full = output.abs().square().sum().item() * area
            central = output[:, pad:-pad, pad:-pad].abs().square().sum().item() * area
            if not all(math.isfinite(x) for x in (before, full, central)):
                raise ValueError("Nonfinite pupil energy")
            optics[f"cutoff_{cutoff}_defocus_{defocus}"] = {
                "pupil_input_um2": before, "pupil_output_um2": full,
                "target_window_um2": central,
                "pupil_fraction": full / before if before else None,
                "target_fraction": central / before if before else None}
    return {"sampling": results, "pupil": optics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required for C2 numerical diagnostic")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {"identity": "C2_CALIBRATION_NUMERICAL_DIAGNOSTIC_ONLY",
              "assumptions": ["uniform illumination", "square reflective active regions",
                              "ideal magnification field normalization", "zero OFF field",
                              "periodic FFT on a 400 um padded field",
                              "cutoffs and defocus are illustrative"],
              "hardware_calibration": "BLOCKED", "coherent_dmd_micromirror_model": "BLOCKED",
              "results": {}}
    for count, magnification in ((26, 1.0), (28, 200 / (28 * 7.56))):
        patterns = calibration_patterns(count)
        report["results"][str(count)] = {}
        for name, pattern in patterns.items():
            report["results"][str(count)][name] = {}
            for fill in (1.0, 0.92):
                report["results"][str(count)][name][str(fill)] = analyze(
                    pattern[None], magnification, fill, torch.device(args.device))
            print(f"COMPLETE mirrors={count} pattern={name}", flush=True)
        torch.save(patterns, args.output / f"patterns_{count}.pt")
    with (args.output / "calibration.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("CALIBRATION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
