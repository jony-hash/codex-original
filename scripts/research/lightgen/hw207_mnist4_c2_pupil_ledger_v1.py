"""Finite-sample energy ledger with analytic mirror-spectrum pupil power."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import features
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_pupil_energy_v1 import circular_pupil_energy
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
    images, labels, rows = images[selection], labels[selection], rows[selection]
    mag = 200 / (28 * DMD_PITCH_UM)
    command = encode(features(images, True).cuda(), 28, "nearest", magnification=mag)
    on = command.sum((1, 2)).double().cpu() * DMD_PITCH_UM**2
    active = on * .92
    report = {"identity": "ASSUMED_COHERENT_PUPIL_ENERGY_LEDGER_SUBSET", "rows": rows.tolist(),
              "labels": labels.tolist(), "incident_full_array_um2": 28**2 * DMD_PITCH_UM**2,
              "on_um2": on.tolist(), "active_um2": active.tolist(), "conditions": {}}
    for cutoff in (.05, .1, .2):
        powers = {}
        for window in (800, 1600, 3200):
            parts = [circular_pupil_energy(batch, mag, .92, cutoff, window).cpu()
                     for batch in command.split(4)]
            powers[window] = torch.cat(parts)
        intensity = torch.load(args.input / f"cutoff_{cutoff}_defocus_0.0.pt",
                               map_location="cpu", weights_only=True)
        if not torch.equal(intensity["rows"], rows):
            raise ValueError("Sample rows do not match saved intensities")
        target = intensity["intensity"].double().sum((1, 2)) * 4
        report["conditions"][str(cutoff)] = {
            "pupil_power_um2": powers[3200].tolist(), "target_power_um2": target.tolist(),
            "pupil_over_active_mean": float((powers[3200]/active).mean()),
            "target_over_active_mean": float((target/active).mean()),
            "max_target_minus_pupil_over_active": float(((target-powers[3200])/active).max()),
            "quadrature_800_to_1600_max_over_active": float(((powers[1600]-powers[800]).abs()/active).max()),
            "quadrature_1600_to_3200_max_over_active": float(((powers[3200]-powers[1600]).abs()/active).max())}
        print(f"LEDGER cutoff={cutoff} rows={len(rows)}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
