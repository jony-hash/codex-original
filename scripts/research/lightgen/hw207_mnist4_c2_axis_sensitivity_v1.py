"""Validation-only, axis-aligned relay offset and magnification sensitivity."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_v1 import (axis_aligned_pixel_integrals, bind_identity_binary_heads,
                               encode, frozen_scores, intensity_metrics)


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "c1-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cuda"))
    conditions = [("baseline", 1., (0., 0.)), ("x_minus_2um", 1., (-2., 0.)),
                  ("x_plus_2um", 1., (2., 0.)), ("y_plus_2um", 1., (0., 2.)),
                  ("x_plus_10um", 1., (10., 0.)), ("mag_minus_5pct", .95, (0., 0.)),
                  ("mag_plus_5pct", 1.05, (0., 0.))]
    report = {"identity": "ILLUSTRATIVE_ASSUMPTION_ONLY", "selection": "NONE",
              "population": "validation", "conditions": {}}
    count = 28
    nominal = 200 / (28 * 7.56)
    for name, factor, offset in conditions:
        stats = {"rows": 0, "nrmse_sum": 0., "energy_sum": 0., "change_l1_sum": 0.,
                 "correct": {m: [0, 0, 0] for m in ("raw", "normalized")}}
        for start in range(0, len(rows), 32):
            end = min(start + 32, len(rows))
            source = features(images[start:end], True).cuda()
            commands = encode(source, count, "nearest", magnification=nominal)
            base = axis_aligned_pixel_integrals(commands, nominal, .92)["mean_intensity"].float()
            actual = axis_aligned_pixel_integrals(commands, nominal * factor, .92, offset)["mean_intensity"].float()
            metrics = intensity_metrics(map_image(source, True), actual)
            active = commands.sum((1, 2)) * (7.56 ** 2) * .92
            if not torch.isfinite(metrics["nrmse"]).all() or (active == 0).any():
                raise ValueError("Undefined validation metric")
            stats["rows"] += end - start
            stats["nrmse_sum"] += float(metrics["nrmse"].sum())
            stats["energy_sum"] += float((actual.sum((1, 2)) * 4 / active).sum())
            stats["change_l1_sum"] += float(((actual-base).abs().sum((1, 2)) * 4 / active).sum())
            for mode in ("raw", "normalized"):
                scores, _ = frozen_scores(actual, heads[mode], mode == "normalized")
                for i, score in enumerate(scores):
                    stats["correct"][mode][i] += int((score["prediction"].cpu() == labels[start:end]).sum())
        for key in ("nrmse_sum", "energy_sum", "change_l1_sum"):
            stats[key.replace("_sum", "_mean")] = stats.pop(key) / stats["rows"]
        stats["accuracy"] = {m: [n / stats["rows"] for n in values]
                             for m, values in stats.pop("correct").items()}
        report["conditions"][name] = stats
        print(f"SENSITIVITY {name} rows={stats['rows']}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
