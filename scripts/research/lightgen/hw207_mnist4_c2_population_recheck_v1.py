"""Recompute every saved C2 axis population command, intensity and readout."""
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
    for name in ("indices", "data", "train", "test", "c1-root", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full population recheck")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = population(args.indices, args.data, args.train, args.test)
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cuda"))
    report = {"identity": "C2_AXIS_POPULATION_RECOMPUTATION", "conditions": {}}
    for count, mag in ((26, 1.0), (28, 200 / (28 * 7.56))):
        for method in ("nearest", "area", "c0_nearest"):
            for fill in (1.0, .92):
                for role, (images, labels, rows) in samples.items():
                    condition = {"rows": 0, "different_mirror_commands": 0,
                                 "different_predictions": 0, "max_nrmse_abs_difference": 0.0,
                                 "max_energy_abs_difference": 0.0}
                    for start in range(0, len(rows), 256):
                        end = min(start + 256, len(rows))
                        path = args.input / "axis" / str(count) / method / str(fill) / role / f"{start:06d}.pt"
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (not torch.equal(saved["rows"], rows[start:end]) or
                                not torch.equal(saved["labels"], labels[start:end])):
                            raise ValueError(f"Population mismatch: {path}")
                        intensities = []
                        for j in range(start, end, 32):
                            k = min(j + 32, end)
                            image = features(images[j:k], True).cuda()
                            if method == "c0_nearest":
                                image = map_image(image, True)
                            command = encode(image, count, "nearest" if method == "c0_nearest" else method,
                                             magnification=mag)
                            original = saved["pattern"][j-start:k-start].cuda()
                            condition["different_mirror_commands"] += int((command != original).sum())
                            intensities.append(axis_aligned_pixel_integrals(original, mag, fill)["mean_intensity"].float())
                        intensity = torch.cat(intensities)
                        reference = map_image(features(images[start:end], True), True)
                        metrics = intensity_metrics(reference, intensity.cpu())
                        condition["max_nrmse_abs_difference"] = max(condition["max_nrmse_abs_difference"],
                            float((metrics["nrmse"] - saved["nrmse"]).abs().max()))
                        condition["max_energy_abs_difference"] = max(condition["max_energy_abs_difference"],
                            float((metrics["candidate_sum"] - saved["candidate_sum"]).abs().max()))
                        for mode in ("raw", "normalized"):
                            scores, _ = frozen_scores(intensity, heads[mode], mode == "normalized")
                            for score, old in zip(scores, saved["frozen"][mode]["runs"]):
                                condition["different_predictions"] += int((score["prediction"].cpu() != old["prediction"]).sum())
                        condition["rows"] += end - start
                    report["conditions"][f"{count}/{method}/{fill}/{role}"] = condition
                    print(f"RECHECK {count}/{method}/{fill}/{role} {condition}", flush=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
