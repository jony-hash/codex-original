"""Pixel-integrated grating contrast proxy for an ideal axis-aligned relay."""
import argparse
import json
import math
from pathlib import Path

import torch

from hw207_mnist4_c2_v1 import (DMD_PITCH_UM, axis_aligned_pixel_integrals,
                               calibration_patterns, geometry)


def modulation(image, pitch_um, period, orientation, width_um):
    """Least-squares sinusoidal fundamental divided by fitted mean intensity."""
    if image.shape != (100, 100) or period <= 0 or pitch_um <= 0:
        raise ValueError("Invalid grating or target image")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("Invalid grating orientation")
    coords = torch.arange(100, dtype=torch.float64, device=image.device) * 2 - 99
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    mask = (x.abs() < width_um / 2 - 2) & (y.abs() < width_um / 2 - 2)
    z = x if orientation == "vertical" else y
    phase = 2 * math.pi * z[mask] / (period * pitch_um)
    design = torch.stack((torch.ones_like(phase), phase.cos(), phase.sin()), dim=1)
    coefficients = torch.linalg.lstsq(design, image[mask].double()).solution
    mean = float(coefficients[0])
    if mean <= 0:
        raise ValueError("Nonpositive fitted mean")
    return float(torch.linalg.vector_norm(coefficients[1:])) / mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"identity": "IDEAL_AXIS_GRATING_CONTRAST_PROXY_NOT_COHERENT_RELAY_MTF",
              "definition": "fundamental sinusoidal amplitude / mean intensity; fill .92 vs 1",
              "conditions": {}}
    for count, mag in ((26, 1.), (28, 200 / (28 * DMD_PITCH_UM))):
        pitch = geometry(count, mag)["pitch_um"]
        patterns = calibration_patterns(count)
        for orientation in ("vertical", "horizontal"):
            for period in (2, 4, 8):
                pattern = patterns[f"{orientation}_period_{period}"][None].cuda()
                values = []
                for fill in (1., .92):
                    image = axis_aligned_pixel_integrals(pattern, mag, fill)["mean_intensity"][0]
                    values.append(modulation(image, pitch, period, orientation, count * pitch))
                report["conditions"][f"{count}/{orientation}/{period}"] = {
                    "frequency_cycles_per_um": 1 / (period * pitch),
                    "ideal_fill_1_modulation": values[0], "fill_092_modulation": values[1],
                    "relative_modulation": values[1] / values[0]}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
