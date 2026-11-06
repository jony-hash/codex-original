"""Finite pupil sensitivity on fixed validation rows, spatial PSF reference."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_psf_v1 import pupil_convolve, pupil_transfer
from hw207_mnist4_c2_pupil_v1 import padded_axis_field
from hw207_mnist4_c2_v1 import (DMD_PITCH_UM, axis_aligned_pixel_integrals,
                               bind_identity_binary_heads, encode, frozen_scores, intensity_metrics)


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "c1-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output directory must be new")
    args.output.mkdir(parents=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    chosen = torch.cat([(labels == k).nonzero().flatten()[:4] for k in range(4)]).sort().values
    images, labels, rows = images[chosen], labels[chosen], rows[chosen]
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cuda"))
    mag = 200 / (28 * DMD_PITCH_UM)
    source = features(images, True).cuda()
    commands = encode(source, 28, "nearest", magnification=mag)
    reference = map_image(source, True)
    ideal = axis_aligned_pixel_integrals(commands, mag, .92)["mean_intensity"].float()
    pitch = .25
    report = {"identity": "ILLUSTRATIVE_SPATIAL_PSF_VALIDATION_SUBSET",
              "population": "first four validation samples per class, restored validation order",
              "rows": rows.tolist(), "labels": labels.tolist(), "pitch_um": pitch,
              "scope": "target-window energy only; full transmitted pupil power not inferred",
              "conditions": {}}
    for cutoff, defocus in ((.05, 0.), (.1, 0.), (.2, 0.), (.1, 50.)):
        transfer = pupil_transfer(1600, pitch, cutoff, defocus, device="cuda")
        outputs = []
        for command in commands.split(2):
            field = padded_axis_field(command, mag, .92, pitch, 400).to(torch.complex64)
            propagated = pupil_convolve(field, pitch, cutoff, defocus, transfer=transfer)
            central = propagated[:, 400:1200, 400:1200]
            outputs.append(F.avg_pool2d(central.abs().square()[:, None], 8)[:, 0].cpu())
        intensity = torch.cat(outputs)
        metrics = intensity_metrics(reference.cpu(), intensity)
        active = commands.cpu().sum((1, 2)) * DMD_PITCH_UM**2 * .92
        change = (intensity - ideal.cpu()).abs().sum((1, 2)) * 4 / active
        item = {"n": len(rows), "mean_nrmse": float(metrics["nrmse"].mean()),
                "mean_l1_over_active_vs_ideal": float(change.mean()),
                "mean_target_over_active": float((intensity.sum((1, 2)) * 4 / active).mean()),
                "readouts": {}}
        for mode in ("raw", "normalized"):
            scores, _ = frozen_scores(intensity, heads[mode], mode == "normalized")
            item["readouts"][mode] = [{"seed": score["seed"],
                                       "prediction": score["prediction"].cpu().tolist(),
                                       "margin": score["margin"].cpu().tolist(),
                                       "accuracy": float((score["prediction"].cpu() == labels).float().mean())}
                                      for score in scores]
        key = f"cutoff_{cutoff}_defocus_{defocus}"
        torch.save({"intensity": intensity, "rows": rows, "labels": labels}, args.output / f"{key}.pt")
        report["conditions"][key] = item
        print(f"SPATIAL_PUPIL {key} rows={len(rows)}", flush=True)
    with (args.output / "pupil_population.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
