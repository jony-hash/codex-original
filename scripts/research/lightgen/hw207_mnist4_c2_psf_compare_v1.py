"""Compare spatial linear convolution and periodic pupil on a fixed counterexample."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_psf_v1 import pupil_convolve
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import DMD_PITCH_UM, calibration_patterns, coherent_pupil


def central_intensity(field, pitch, width):
    n = field.shape[-1]
    crop_n = round(200 / pitch)
    start = (n - crop_n) // 2
    center = field[:, start:start+crop_n, start:start+crop_n]
    return F.avg_pool2d(center.abs().square()[:, None], round(2 / pitch))[:, 0].double().cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pattern = calibration_patterns(28)["checkerboard"][None].cuda()
    mag = 200 / (28 * DMD_PITCH_UM)
    active = float(pattern.sum()) * DMD_PITCH_UM**2 * .92
    report = {"identity": "CIRCULAR_PUPIL_PERIODIC_VS_SPATIAL_REFERENCE", "cases": {}}
    for pitch in (.5, .25):
        images = {}
        source = padded_axis_field(pattern, mag, .92, pitch, 400).to(torch.complex64)
        for defocus in (0., 50.):
            spatial = pupil_convolve(source, pitch, .1, defocus)
            images["spatial"] = central_intensity(spatial, pitch, 400)
            refined = pupil_convolve(source, pitch, .1, defocus,
                                     radial_step=pitch/16, quadrature=4097)
            images["refined"] = central_intensity(refined, pitch, 400)
            for width in (800, 1600):
                periodic_in = padded_axis_field(pattern, mag, .92, pitch, width).to(torch.complex64)
                periodic = coherent_pupil(periodic_in, pitch, .1, defocus)
                images[str(width)] = central_intensity(periodic, pitch, width)
                del periodic_in, periodic
            key = f"pitch_{pitch}/defocus_{defocus}"
            report["cases"][key] = {
                "spatial_energy_um2": float(images["spatial"].sum()) * 4,
                "refinement_l1_over_active": float((images["refined"]-images["spatial"]).abs().sum()) * 4 / active,
                "periodic_800_l1_over_active": float((images["800"]-images["spatial"]).abs().sum()) * 4 / active,
                "periodic_1600_l1_over_active": float((images["1600"]-images["spatial"]).abs().sum()) * 4 / active}
            print(f"PSF_COMPARE {key}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
