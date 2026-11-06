"""Representative rotated-mirror raster convergence; no hardware calibration."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_v1 import (DMD_PITCH_UM, axis_aligned_pixel_integrals,
                               calibration_patterns, target_plane_samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"identity": "ILLUSTRATIVE_ROTATION_RASTER_DIAGNOSTIC", "conditions": {}}
    for count, mag in ((26, 1.0), (28, 200 / (28 * DMD_PITCH_UM))):
        patterns = calibration_patterns(count)
        for name in ("asymmetric", "checkerboard"):
            pattern = patterns[name][None].cuda()
            active = float(pattern.sum()) * DMD_PITCH_UM ** 2 * .92
            exact = axis_aligned_pixel_integrals(pattern, mag, .92)["mean_intensity"]
            for degrees in (0., -1., 1.):
                values, previous = {}, None
                for sub in (8, 16, 32):
                    field, _ = target_plane_samples(pattern, mag, sub, .92,
                                                    rotation_rad=math.radians(degrees))
                    intensity = F.avg_pool2d(field.square()[:, None], sub)[:, 0].double()
                    values[str(sub)] = {"target_power_um2": float(intensity.sum()) * 4,
                                        "l1_over_active_from_previous": None if previous is None else
                                        float((intensity - previous).abs().sum()) * 4 / active,
                                        "l1_over_active_from_unrotated_exact":
                                        float((intensity - exact).abs().sum()) * 4 / active}
                    previous = intensity
                    del field
                key = f"{count}/{name}/{degrees}"
                report["conditions"][key] = values
                print(f"ROTATION {key}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
