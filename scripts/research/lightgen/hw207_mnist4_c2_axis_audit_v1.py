"""Audit C2 axis-aligned population chunks and summarize frozen predictions."""
import argparse
import json
import math
from pathlib import Path

import torch

from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_v1 import axis_aligned_pixel_integrals, bind_identity_binary_heads, frozen_scores


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "c1_root", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    receipt = json.loads((args.output / "axis_preflight.json").read_text())
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cpu"))
    for mode in ("raw", "normalized"):
        if [m["sha256"] for m in heads[mode]["models"]] != [m["sha256"] for m in receipt["heads"][mode]["models"]]:
            raise ValueError("Frozen C1 head changed")
    report = {"identity": "C2_AXIS_ALIGNED_DIAGNOSTIC_ONLY", "conditions": {},
              "hardware_mapping": "BLOCKED", "coherent_dmd": "BLOCKED"}
    for count, magnification in ((26, 1.0), (28, 200 / (28 * 7.56))):
        for method in ("nearest", "area", "c0_nearest"):
            for fill in (1.0, 0.92):
                for role, (_, labels, rows) in samples.items():
                    matrices = {mode: [torch.zeros((4, 4), dtype=torch.int64) for _ in range(3)]
                                for mode in ("raw", "normalized")}
                    nrmse, correlation, energy, foreground = [], [], [], []
                    representative_differences = []
                    for start in range(0, len(rows), 256):
                        end = min(start + 256, len(rows))
                        path = args.output / "axis" / str(count) / method / str(fill) / role / f"{start:06d}.pt"
                        chunk = torch.load(path, map_location="cpu", weights_only=True)
                        if ((chunk["role"], chunk["count"], chunk["method"], chunk["fill_factor"], chunk["magnification"])
                                != (role, count, method, fill, magnification)):
                            raise ValueError(f"Condition mismatch: {path}")
                        if not torch.equal(chunk["rows"], rows[start:end]) or not torch.equal(chunk["labels"], labels[start:end]):
                            raise ValueError(f"Population mismatch: {path}")
                        pattern = chunk["pattern"]
                        if pattern.shape != (end-start, count, count) or not ((pattern == 0) | (pattern == 1)).all():
                            raise ValueError(f"Invalid mirror pattern: {path}")
                        recomputed = axis_aligned_pixel_integrals(pattern[:1], magnification, fill)["mean_intensity"].float()
                        torch.testing.assert_close(recomputed, chunk["representative_intensity"], atol=1e-6, rtol=1e-6)
                        nrmse.extend(chunk["nrmse"].tolist())
                        correlation.extend(chunk["normalized_correlation"].tolist())
                        energy.extend((chunk["candidate_sum"] * 4 / (count ** 2 * 7.56 ** 2)).tolist())
                        foreground.extend(pattern.float().mean((1, 2)).tolist())
                        for mode in ("raw", "normalized"):
                            independently_scored, _ = frozen_scores(recomputed, heads[mode], mode == "normalized")
                            for i, run in enumerate(chunk["frozen"][mode]["runs"]):
                                predicted = run["prediction"]
                                if run["seed"] != (11, 23, 37)[i] or len(predicted) != end-start:
                                    raise ValueError(f"Readout identity mismatch: {path}")
                                check = independently_scored[i]
                                if check["prediction"][0].item() != predicted[0].item():
                                    representative_differences.append({"original_index": rows[start].item(),
                                        "mode": mode, "seed": run["seed"], "saved": predicted[0].item(),
                                        "cpu": check["prediction"][0].item(), "cpu_margin": check["margin"][0].item()})
                                matrices[mode][i] += torch.bincount(labels[start:end] * 4 + predicted, minlength=16).reshape(4, 4)
                    key = f"{count}/{method}/{fill}/{role}"
                    finite_nrmse = [v for v in nrmse if math.isfinite(v)]
                    finite_correlation = [v for v in correlation if math.isfinite(v)]
                    report["conditions"][key] = {"count": len(rows),
                        "mean_nrmse": sum(finite_nrmse)/len(finite_nrmse) if finite_nrmse else None,
                        "undefined_nrmse": len(rows)-len(finite_nrmse),
                        "mean_correlation": sum(finite_correlation)/len(finite_correlation) if finite_correlation else None,
                        "undefined_correlation": len(rows)-len(finite_correlation), "mean_target_over_incident": sum(energy)/len(rows),
                        "mean_foreground_mirror_fraction": sum(foreground)/len(rows),
                        "representative_readout_differences": representative_differences,
                        "readouts": {mode: [{"seed": seed, "confusion": matrix.tolist(),
                                              "accuracy": matrix.diag().sum().item()/len(rows)}
                                             for seed, matrix in zip((11, 23, 37), values)]
                                     for mode, values in matrices.items()}}
                    print(f"AUDIT {key} rows={len(rows)}", flush=True)
    with (args.output / "axis_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
