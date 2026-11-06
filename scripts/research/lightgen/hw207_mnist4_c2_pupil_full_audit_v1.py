"""CPU replay of all saved pupil intensities, input rows and frozen heads."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import features, map_image
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_v1 import bind_identity_binary_heads, frozen_scores, intensity_metrics


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "c1-root", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cpu"))
    config = json.loads((args.input / "configuration.json").read_text())
    for mode, models in config["heads"].items():
        if [m["sha256"] for m in models] != [m["sha256"] for m in heads[mode]["models"]]:
            raise ValueError("Frozen head identity mismatch")
    report = {"identity": "CPU_PUPIL_POPULATION_REPLAY", "conditions": {}}
    for role, conditions in config["roles"].items():
        images, labels, rows = samples[role]
        for cutoff, defocus in conditions:
            key = f"cutoff_{cutoff}_defocus_{defocus}"
            matrices = {mode: [torch.zeros((4, 4), dtype=torch.int64) for _ in range(3)]
                        for mode in ("raw", "normalized")}
            nrmse_sum, energy_sum, differences = 0., 0., []
            max_metric_error = 0.
            for start in range(0, len(rows), 32):
                end = min(start+32, len(rows))
                path = args.input / role / key / f"{start:06d}.pt"
                item = torch.load(path, map_location="cpu", weights_only=True)
                if ((item["role"], item["cutoff"], item["defocus"]) != (role, cutoff, defocus)
                        or not torch.equal(item["rows"], rows[start:end])
                        or not torch.equal(item["labels"], labels[start:end])
                        or item["intensity"].shape != (end-start,100,100)):
                    raise ValueError(f"Population or shape mismatch: {path}")
                reference = map_image(features(images[start:end], True), True)
                metrics = intensity_metrics(reference, item["intensity"])
                max_metric_error = max(max_metric_error, float((metrics["nrmse"]-item["nrmse"]).abs().max()),
                                       float((metrics["candidate_sum"]-item["target_sum"]).abs().max()))
                nrmse_sum += float(metrics["nrmse"].sum())
                energy_sum += float(metrics["candidate_sum"].sum()) * 4
                for mode in ("raw", "normalized"):
                    scores, _ = frozen_scores(item["intensity"], heads[mode], mode == "normalized")
                    for i, score in enumerate(scores):
                        pred = score["prediction"]
                        saved = item["scores"][mode]["runs"][i]
                        if score["seed"] != saved["seed"]:
                            raise ValueError("Head seed mismatch")
                        for j in (pred != saved["prediction"]).nonzero().flatten().tolist():
                            differences.append({"row": int(rows[start+j]), "mode": mode, "seed": score["seed"],
                                                "saved": int(saved["prediction"][j]), "cpu": int(pred[j]),
                                                "cpu_margin": float(score["margin"][j])})
                        matrices[mode][i] += torch.bincount(labels[start:end]*4+pred, minlength=16).reshape(4,4)
            report["conditions"][f"{role}/{key}"] = {"rows": len(rows), "mean_nrmse": nrmse_sum/len(rows),
                "mean_target_power_um2": energy_sum/len(rows), "max_metric_abs_error": max_metric_error,
                "cpu_prediction_differences": differences,
                "readouts": {mode: [{"seed": seed, "accuracy": float(m.diag().sum())/len(rows),
                                      "confusion": m.tolist()} for seed, m in zip((11,23,37), ms)]
                             for mode, ms in matrices.items()}}
            print(f"PUPIL_CPU_AUDIT {role}/{key} rows={len(rows)} differences={len(differences)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
