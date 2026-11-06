"""CPU replay of selected C2 auxiliary readouts using saved mirror commands."""
import argparse
import json
from pathlib import Path

import torch

from hw207_mnist4_c0_v1 import metrics
from hw207_mnist4_c1_v1 import population
from hw207_mnist4_c2_readout_v1 import load_role


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "readouts", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    selected = json.loads((args.readouts / "selection.json").read_text())
    saved_results = json.loads((args.readouts / "test_results.json").read_text())
    report = {"identity": "CPU_REPLAY_SELECTED_AUXILIARY_READOUTS", "conditions": {}}
    for count in (26, 28):
        choice = selected[str(count)]
        best = max(choice["candidates"], key=lambda x: (x["mean_validation_accuracy"], -x["decay"]))
        if best["decay"] != choice["decay"]:
            raise ValueError("Validation selection mismatch")
        train, _, _ = load_role(args.input, samples, count, "train", torch.device("cpu"))
        scale = float(train.square().sum(1).mean().sqrt())
        del train
        test, labels, rows = load_role(args.input, samples, count, "test", torch.device("cpu"))
        if rows.tolist() != saved_results[str(count)]["rows"]:
            raise ValueError("Test population mismatch")
        item = {"train_scale_recomputed": scale, "train_scale_saved": choice["scale"], "runs": []}
        for seed, old in zip((11, 23, 37), saved_results[str(count)]["runs"]):
            model_file = args.readouts / f"{count}_decay_{choice['decay']}_seed_{seed}.pt"
            candidate = torch.load(model_file, map_location="cpu", weights_only=True)
            if (candidate["seed"] != seed or candidate["decay"] != choice["decay"]
                    or candidate["scale"] != choice["scale"] or len(candidate["training"]["history"]) != 30):
                raise ValueError("Selected checkpoint mismatch")
            model = torch.nn.Linear(10000, 4)
            model.load_state_dict(candidate["state"])
            with torch.no_grad():
                logits = model(test / choice["scale"])
            predictions = logits.argmax(1).tolist()
            differences = [{"row": rows[i].item(), "saved": old["metrics"]["prediction"][i], "cpu": prediction}
                           for i, prediction in enumerate(predictions) if prediction != old["metrics"]["prediction"][i]]
            item["runs"].append({"seed": seed, "differences": differences,
                                 "cpu_metrics": metrics(logits, labels)})
        report["conditions"][str(count)] = item
        print(f"CPU_REPLAY {count} test_rows={len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
