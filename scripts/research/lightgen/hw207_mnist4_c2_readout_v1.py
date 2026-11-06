"""New auxiliary linear readouts for fixed C2 conditions; C1 stays frozen."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import metrics
from hw207_mnist4_c1_v1 import population, save_chunk
from hw207_mnist4_c2_v1 import axis_aligned_pixel_integrals


def fit(train, labels, valid, valid_labels, seed, decay, max_steps=None):
    """Same SGD path for three-step preflight, label control and training."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = torch.nn.Linear(train.shape[1], 4, device=train.device)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.15, weight_decay=decay)
    history, steps = [], 0
    for epoch in range(30):
        total = 0.
        for indices in torch.randperm(len(labels), generator=generator).split(256):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train[indices]), labels[indices])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite loss or gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite weights")
            total += float(loss) * len(indices)
            steps += 1
            if max_steps is not None and steps == max_steps:
                updated = {name: not torch.equal(initial[name], value) for name, value in model.named_parameters()}
                if not all(updated.values()):
                    raise ValueError("Trainable parameter did not update")
                return model, {"steps": steps, "updated": updated}
        with torch.no_grad():
            validation = metrics(model(valid), valid_labels)
        history.append({"epoch": epoch + 1, "loss": total / len(labels), "validation": validation})
    return model, {"steps": steps, "history": history}


def load_role(root, samples, count, role, device):
    _, labels, rows = samples[role]
    mag = 1. if count == 26 else 200/(28*7.56)
    tensors = []
    for start in range(0, len(rows), 256):
        end = min(start+256, len(rows))
        file = root / "axis" / str(count) / "nearest" / "0.92" / role / f"{start:06d}.pt"
        saved = torch.load(file, map_location="cpu", weights_only=True)
        if (not torch.equal(saved["rows"], rows[start:end]) or not torch.equal(saved["labels"], labels[start:end])
                or (saved["count"], saved["method"], saved["fill_factor"], saved["magnification"], saved["role"])
                != (count, "nearest", .92, mag, role)):
            raise ValueError(f"C2 population or condition mismatch: {file}")
        chunks = [axis_aligned_pixel_integrals(p.to(device), mag, .92)["mean_intensity"].float().flatten(1)
                  for p in saved["pattern"].split(32)]
        tensors.append(torch.cat(chunks))
    return torch.cat(tensors), labels.to(device), rows


def main():
    p = argparse.ArgumentParser()
    for name in ("indices", "data", "train", "test", "input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("CUDA required and output must be new")
    args.output.mkdir(parents=True)
    samples = population(args.indices, args.data, args.train, args.test)
    device = torch.device("cuda")
    selection = {}
    for count in (26, 28):
        train, y, _ = load_role(args.input, samples, count, "train", device)
        valid, vy, _ = load_role(args.input, samples, count, "validation", device)
        scale = train.square().sum(1).mean().sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid train-only scale")
        train, valid = train/scale, valid/scale
        _, three_steps = fit(train, y, valid, vy, 11, 0., max_steps=3)
        shuffled = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(207)).to(device)]
        _, negative = fit(train, shuffled, valid, vy, 11, 0.)
        negative_accuracy = negative["history"][-1]["validation"]["accuracy"]
        receipt = {"condition": f"{count}/nearest/0.92/raw", "scale": float(scale),
                   "three_steps": three_steps, "random_label_validation_accuracy": negative_accuracy,
                   "random_label_history": negative["history"], "seed": 11, "decay": 0.}
        with (args.output / f"preflight_{count}.json").open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2, allow_nan=False)
        if negative_accuracy >= .4:
            raise ValueError("Random-label control did not pass")
        if args.preflight_only:
            del train, valid
            continue
        candidates = []
        for decay in (0., .0001, .001):
            accuracies = []
            for seed in (11, 23, 37):
                model, training = fit(train, y, valid, vy, seed, decay)
                candidate = {"condition": receipt["condition"], "seed": seed, "decay": decay,
                             "scale": float(scale), "training": training,
                             "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
                save_chunk(args.output / f"{count}_decay_{decay}_seed_{seed}.pt", candidate)
                accuracies.append(training["history"][-1]["validation"]["accuracy"])
                print(f"TRAIN count={count} decay={decay} seed={seed}", flush=True)
            candidates.append({"decay": decay, "validation_accuracies": accuracies,
                               "mean_validation_accuracy": sum(accuracies)/len(accuracies)})
        best = max(candidates, key=lambda x: (x["mean_validation_accuracy"], -x["decay"]))
        selection[str(count)] = {"decay": best["decay"], "scale": float(scale), "candidates": candidates}
        del train, valid
    if args.preflight_only:
        return
    with (args.output / "selection.json").open("x", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, allow_nan=False)
    # Both selections are frozen before either test set is evaluated.
    test_results = {}
    for count in (26, 28):
        test, labels, rows = load_role(args.input, samples, count, "test", device)
        chosen = selection[str(count)]
        runs = []
        for seed in (11, 23, 37):
            path = args.output / f"{count}_decay_{chosen['decay']}_seed_{seed}.pt"
            candidate = torch.load(path, map_location="cpu", weights_only=True)
            model = torch.nn.Linear(test.shape[1], 4, device=device)
            model.load_state_dict(candidate["state"])
            with torch.no_grad():
                logits = model(test/chosen["scale"])
                item = metrics(logits, labels)
                item["prediction"] = logits.argmax(1).cpu().tolist()
            runs.append({"seed": seed, "metrics": item})
        test_results[str(count)] = {"rows": rows.tolist(), "runs": runs}
    with (args.output / "test_results.json").open("x", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
