"""Validation-only exact square-aperture rotation sensitivity."""
import argparse
import json
import math
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_rotation_exact_v1 import rotated_coverage_matrix, rotated_intensity
from hw207_mnist4_c2_v1 import bind_identity_binary_heads, encode, frozen_scores, intensity_metrics


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
    mag = 200 / (28 * 7.56)
    report = {"identity": "ILLUSTRATIVE_ROTATION_SENSITIVITY", "population": "validation",
              "selection": "NONE", "conditions": {}}
    for degrees in (0., -1., 1., 5.):
        matrix = rotated_coverage_matrix(28, mag, .92, math.radians(degrees)).cuda()
        stats = {"rows": 0, "nrmse_sum": 0., "correct": {m: [0, 0, 0] for m in ("raw", "normalized")}}
        for start in range(0, len(rows), 32):
            end = min(start+32, len(rows))
            source = features(images[start:end], True).cuda()
            command = encode(source, 28, "nearest", magnification=mag)
            intensity = rotated_intensity(command, mag, matrix).float()
            metrics = intensity_metrics(map_image(source, True), intensity)
            if not torch.isfinite(metrics["nrmse"]).all():
                raise ValueError("Undefined NRMSE")
            stats["rows"] += end-start
            stats["nrmse_sum"] += float(metrics["nrmse"].sum())
            for mode in ("raw", "normalized"):
                scores, _ = frozen_scores(intensity, heads[mode], mode == "normalized")
                for i, score in enumerate(scores):
                    stats["correct"][mode][i] += int((score["prediction"].cpu() == labels[start:end]).sum())
        stats["mean_nrmse"] = stats.pop("nrmse_sum") / stats["rows"]
        stats["accuracy"] = {m: [n / stats["rows"] for n in values]
                             for m, values in stats.pop("correct").items()}
        report["conditions"][str(degrees)] = stats
        print(f"ROTATION_VALIDATION {degrees} rows={stats['rows']}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
