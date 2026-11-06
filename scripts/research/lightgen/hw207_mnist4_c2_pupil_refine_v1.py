"""Fine-grid check on one retained validation sample per class."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_psf_v1 import pupil_convolve, pupil_transfer
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import DMD_PITCH_UM, encode


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    selection = torch.cat([(labels == k).nonzero().flatten()[:4] for k in range(4)]).sort().values
    labels, rows, images = labels[selection], rows[selection], images[selection]
    four = torch.stack([(labels == k).nonzero().flatten()[0] for k in range(4)]).sort().values
    labels, rows, images = labels[four], rows[four], images[four]
    mag = 200 / (28 * DMD_PITCH_UM)
    commands = encode(features(images, True).cuda(), 28, "nearest", magnification=mag)
    active = commands.sum((1, 2)).cpu() * DMD_PITCH_UM**2 * .92
    report = {"identity": "SPATIAL_PSF_FINE_GRID_CHECK", "rows": rows.tolist(),
              "labels": labels.tolist(), "conditions": {}}
    for cutoff, defocus in ((.05, 0.), (.1, 0.), (.2, 0.), (.1, 50.)):
        transfer = pupil_transfer(3200, .125, cutoff, defocus, radial_step=.125/16,
                                  quadrature=4097, device="cuda")
        result = []
        for command in commands.split(1):
            field = padded_axis_field(command, mag, .92, .125, 400).to(torch.complex64)
            output = pupil_convolve(field, .125, cutoff, defocus, transfer=transfer)
            central = output[:, 800:2400, 800:2400]
            result.append(F.avg_pool2d(central.abs().square()[:, None], 16)[:, 0].cpu())
        fine = torch.cat(result)
        key = f"cutoff_{cutoff}_defocus_{defocus}"
        prior = torch.load(args.input / f"{key}.pt", map_location="cpu", weights_only=True)
        if not torch.equal(prior["rows"][four], rows):
            raise ValueError("Validation identity mismatch")
        diff = (fine - prior["intensity"][four]).abs().sum((1, 2)) * 4 / active
        report["conditions"][key] = {"l1_over_active_per_row": diff.tolist(), "maximum": float(diff.max())}
        print(f"FINE_GRID {key} max={float(diff.max())}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
