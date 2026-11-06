"""Validation-only fixed-grid threshold sensitivity; no test-set selection."""
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images, labels, rows = population(args.indices, args.data, args.train, args.test)["validation"]
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cuda"))
    report = {"identity": "VALIDATION_ONLY_AREA_THRESHOLD_SENSITIVITY",
              "thresholds": [.3, .4, .5, .6, .7], "selection": "NONE", "conditions": {}}
    for count, mag in ((26, 1.0), (28, 200 / (28 * 7.56))):
        for threshold in report["thresholds"]:
            stats = {"rows": 0, "nrmse_sum": 0.0, "foreground_sum": 0.0,
                     "correct": {mode: [0, 0, 0] for mode in ("raw", "normalized")}}
            for start in range(0, len(rows), 32):
                end = min(start + 32, len(rows))
                image = features(images[start:end], True).cuda()
                command = encode(image, count, "area", threshold, magnification=mag)
                intensity = axis_aligned_pixel_integrals(command, mag, .92)["mean_intensity"].float()
                reference = map_image(image, True)
                metric = intensity_metrics(reference, intensity)
                if not torch.isfinite(metric["nrmse"]).all():
                    raise ValueError("Undefined validation NRMSE")
                stats["rows"] += end-start
                stats["nrmse_sum"] += float(metric["nrmse"].sum())
                stats["foreground_sum"] += float(command.sum()) / (count ** 2)
                for mode in ("raw", "normalized"):
                    scores, _ = frozen_scores(intensity, heads[mode], mode == "normalized")
                    for i, score in enumerate(scores):
                        stats["correct"][mode][i] += int((score["prediction"].cpu() == labels[start:end]).sum())
            stats["mean_nrmse"] = stats.pop("nrmse_sum") / stats["rows"]
            stats["mean_foreground_fraction"] = stats.pop("foreground_sum") / stats["rows"]
            stats["accuracy"] = {mode: [n / stats["rows"] for n in values]
                                 for mode, values in stats.pop("correct").items()}
            report["conditions"][f"{count}/{threshold}"] = stats
            print(f"THRESHOLD {count}/{threshold} rows={stats['rows']}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
