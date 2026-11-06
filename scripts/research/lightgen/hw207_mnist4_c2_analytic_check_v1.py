"""C2 analytic pixel-area reference against subpixel quadrature."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_v1 import axis_aligned_pixel_integrals, calibration_patterns, target_plane_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("CUDA required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {"identity": "ASSUMED_AXIS_ALIGNED_ANALYTIC_DIAGNOSTIC",
              "normalization": "sum absolute intensity difference times 4 um² / active mirror area",
              "results": {}}
    for count, magnification in ((26, 1.0), (28, 200 / (28 * 7.56))):
        report["results"][str(count)] = {}
        for name, source in calibration_patterns(count).items():
            pattern = source[None].cuda()
            report["results"][str(count)][name] = {}
            for fill in (1.0, 0.92):
                exact = axis_aligned_pixel_integrals(pattern, magnification, fill)["mean_intensity"]
                active_power = pattern.sum().item() * 7.56 ** 2 * fill
                values = {}
                for subdivision in (8, 16, 32, 64):
                    sampled, _ = target_plane_samples(pattern, magnification, subdivision, fill)
                    average = F.avg_pool2d(sampled[:, None].square(), subdivision)[:, 0].double()
                    error = (average - exact).abs().sum().item() * 4
                    values[str(subdivision)] = error / active_power if active_power else None
                report["results"][str(count)][name][str(fill)] = values
            print(f"COMPLETE {count} {name}", flush=True)
    with (args.output / "analytic_convergence.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
