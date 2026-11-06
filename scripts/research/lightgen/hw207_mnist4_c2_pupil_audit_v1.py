"""Recompute saved pupil-subset predictions on CPU from frozen C1 weights."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c2_v1 import bind_identity_binary_heads, frozen_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--c1-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads((args.input / "pupil_population.json").read_text(encoding="utf-8"))
    heads = bind_identity_binary_heads(args.c1_root, torch.device("cpu"))
    report = {"identity": "CPU_FROZEN_PUPIL_READOUT_AUDIT", "conditions": {}}
    for key, condition in saved["conditions"].items():
        data = torch.load(args.input / f"{key}.pt", map_location="cpu", weights_only=True)
        if data["rows"].tolist() != saved["rows"] or data["labels"].tolist() != saved["labels"]:
            raise ValueError("Saved row identity mismatch")
        mismatches = []
        for mode in ("raw", "normalized"):
            computed, _ = frozen_scores(data["intensity"], heads[mode], mode == "normalized")
            for score, old in zip(computed, condition["readouts"][mode]):
                if score["seed"] != old["seed"]:
                    raise ValueError("Seed mismatch")
                for index, (a, b) in enumerate(zip(score["prediction"].tolist(), old["prediction"])):
                    if a != b:
                        mismatches.append({"row": saved["rows"][index], "mode": mode,
                                           "seed": score["seed"], "saved": b, "cpu": a})
        report["conditions"][key] = {"rows": len(data["rows"]), "differences": mismatches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
