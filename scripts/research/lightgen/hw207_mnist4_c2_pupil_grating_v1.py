"""Grating contrast of a finite assumed coherent circular-pupil relay."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_grating_v1 import modulation
from hw207_mnist4_c2_psf_v1 import pupil_convolve, pupil_transfer
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import (DMD_PITCH_UM, axis_aligned_pixel_integrals,
                               calibration_patterns, geometry)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mag = 200 / (28 * DMD_PITCH_UM)
    pitch = geometry(28, mag)["pitch_um"]
    patterns = calibration_patterns(28)
    report = {"identity": "ILLUSTRATIVE_COHERENT_GRATING_RESPONSE_NOT_GENERAL_INCOHERENT_MTF",
              "definition": "fundamental modulation after coherent pupil divided by ideal relay modulation",
              "conditions": {}}
    for cutoff, defocus in ((.05, 0.), (.1, 0.), (.2, 0.), (.1, 50.)):
        transfer = pupil_transfer(1600, .25, cutoff, defocus, device="cuda")
        for orientation in ("vertical", "horizontal"):
            for period in (2, 4, 8):
                pattern = patterns[f"{orientation}_period_{period}"][None].cuda()
                ideal = axis_aligned_pixel_integrals(pattern, mag, .92)["mean_intensity"][0]
                field = padded_axis_field(pattern, mag, .92, .25, 400).to(torch.complex64)
                out = pupil_convolve(field, .25, cutoff, defocus, transfer=transfer)
                image = F.avg_pool2d(out[:, None, 400:1200, 400:1200].abs().square(), 8)[0, 0]
                reference = modulation(ideal, pitch, period, orientation, 200)
                actual = modulation(image, pitch, period, orientation, 200)
                key = f"cutoff_{cutoff}/defocus_{defocus}/{orientation}/{period}"
                report["conditions"][key] = {"frequency_cycles_per_um": 1 / (period*pitch),
                                             "ideal_modulation": reference, "pupil_modulation": actual,
                                             "relative_modulation": actual / reference}
                print(f"PUPIL_GRATING {key}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
