"""Diagnostic convergence of padded C2 relay pupil; no hardware claims."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import calibration_patterns, coherent_pupil, DMD_PITCH_UM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--widths", type=int, nargs="+", default=[400, 800])
    parser.add_argument("--pitches", type=float, nargs="+", default=[1.0, .5, .25])
    args = parser.parse_args()
    if (any(p <= 0 or not math.isclose(round(2 / p) * p, 2, abs_tol=1e-8)
            for p in args.pitches) or any(w < 200 for w in args.widths)):
        raise ValueError("Pitch must divide a 2 um pixel and padded width must be at least 200 um")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the numerical diagnostic")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    result = {"identity": "ASSUMED_COHERENT_PUPIL_CONVERGENCE_DIAGNOSTIC",
              "hardware_calibration": "BLOCKED",
              "assumptions": ["square uniformly illuminated mirror apertures",
                              "field averaged over grid cells before FFT",
                              "periodic padded FFT; pupil cutoff and defocus assumed",
                              "no tilted mirror or measured relay transfer function"],
              "cases": {}}
    for count, mag in ((26, 1.0), (28, 200 / (28 * DMD_PITCH_UM))):
        patterns = calibration_patterns(count)
        for name in ("asymmetric", "checkerboard"):
            pattern = patterns[name][None].to(args.device)
            active = float(pattern.sum()) * DMD_PITCH_UM ** 2 * .92
            for offset in ((0.0, 0.0), (20.0, 0.0)):
                case_key = f"{count}/{name}/offset_{offset[0]}"
                case = {"active_power_um2": active, "runs": {}, "comparisons": {}}
                images = {}
                for width in args.widths:
                    for pitch in args.pitches:
                        samples = round(2 / pitch)
                        field = padded_axis_field(pattern, mag, .92, pitch, width, offset).to(torch.complex64)
                        n = field.shape[-1]
                        central_n = round(200 / pitch)
                        start = (n - central_n) // 2
                        input_power = float(field.abs().square().sum()) * pitch ** 2
                        for cutoff in (.05, .1, .2):
                            for defocus in (0.0, 50.0):
                                out = coherent_pupil(field, pitch, cutoff, defocus)
                                total = float(out.abs().square().sum()) * pitch ** 2
                                central = out[:, start:start + central_n, start:start + central_n]
                                pixels = F.avg_pool2d(central.abs().square()[:, None], samples)[:, 0]
                                image = pixels.cpu().double()
                                window = float(image.sum()) * 4
                                if not all(math.isfinite(x) for x in (input_power, total, window)):
                                    raise ValueError("Nonfinite pupil result")
                                key = f"width_{width}/pitch_{pitch}/cutoff_{cutoff}/defocus_{defocus}"
                                case["runs"][key] = {"input_power_um2": input_power,
                                                     "pupil_output_power_um2": total,
                                                     "window_power_um2": window,
                                                     "input_grid_over_active": input_power / active,
                                                     "pupil_over_input": total / input_power}
                                images[(width, pitch, cutoff, defocus)] = image
                        del field
                        print(f"COMPLETE {case_key} width={width} pitch={pitch}", flush=True)
                for cutoff in (.05, .1, .2):
                    for defocus in (0.0, 50.0):
                        pairs = [((width, a), (width, b)) for width in args.widths
                                 for a, b in zip(args.pitches, args.pitches[1:])]
                        pairs += [((a, pitch), (b, pitch)) for pitch in args.pitches
                                  for a, b in zip(args.widths, args.widths[1:])]
                        for left, right in pairs:
                            a = images[(*left, cutoff, defocus)]
                            b = images[(*right, cutoff, defocus)]
                            label = f"{left}_to_{right}/cutoff_{cutoff}/defocus_{defocus}"
                            case["comparisons"][label] = float((a - b).abs().sum()) * 4 / active
                result["cases"][case_key] = case
    with (args.output / "pupil_convergence.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("PUPIL_CONVERGENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
