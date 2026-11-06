"""Summarize saved frozen-head test margins and named prior tie-sensitive rows."""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"identity": "SAVED_FROZEN_HEAD_TEST_MARGIN_AUDIT",
              "near_tie_definition": "top-1 minus top-2 logit <= 1e-4; diagnostic threshold",
              "conditions": {}}
    for count in (26, 28):
        for method in ("nearest", "area", "c0_nearest"):
            for fill in (1., .92):
                directory = args.input / "axis" / str(count) / method / str(fill) / "test"
                chunks = [torch.load(path, map_location="cpu", weights_only=True) for path in sorted(directory.glob("*.pt"))]
                rows = torch.cat([x["rows"] for x in chunks])
                labels = torch.cat([x["labels"] for x in chunks])
                if len(rows) != 4157 or len(rows.unique()) != len(rows):
                    raise ValueError("Unexpected test population")
                results = {}
                for mode in ("raw", "normalized"):
                    results[mode] = []
                    for i, seed in enumerate((11, 23, 37)):
                        predictions = torch.cat([x["frozen"][mode]["runs"][i]["prediction"] for x in chunks])
                        margins = torch.cat([x["frozen"][mode]["runs"][i]["margin"] for x in chunks])
                        matrix = torch.bincount(labels*4 + predictions, minlength=16).reshape(4,4)
                        hits = matrix.diag().double()
                        recall = hits / matrix.sum(1).clamp_min(1)
                        f1 = 2 * hits / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
                        special = []
                        for row in (4848, 6744):
                            at = (rows == row).nonzero().flatten()
                            if len(at) != 1:
                                raise ValueError(f"Named original test row missing: {row}")
                            j = int(at[0])
                            special.append({"row": row, "label": int(labels[j]), "prediction": int(predictions[j]),
                                            "margin": float(margins[j])})
                        results[mode].append({"seed": seed, "accuracy": float(hits.sum()/len(rows)),
                            "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
                            "per_class_recall": recall.tolist(), "near_tie_count": int((margins <= 1e-4).sum()),
                            "mean_margin": float(margins.double().mean()), "named_prior_tie_rows": special})
                report["conditions"][f"{count}/{method}/{fill}"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
