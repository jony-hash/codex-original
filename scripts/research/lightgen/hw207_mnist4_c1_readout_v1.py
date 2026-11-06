"""C1 linear readouts using committed propagation chunks."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c0_v1 import metrics
from hw207_mnist4_c1_run_v1 import readout_features
from hw207_mnist4_c1_v1 import checked_chunk, population, save_chunk


def load_role(root, name, binary, distance, window, role, labels, rows):
    parts = []
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        path = root / "chunks" / name / str(distance) / role / f"{start:06d}.pt"
        chunk = checked_chunk(path, binary, distance, window, labels[start:end], rows[start:end], role)
        parts.append(chunk["intensity"])
    return torch.cat(parts)


def fit_candidate(train, labels, valid, valid_labels, seed, decay, device):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = torch.nn.Linear(train.shape[1], 4, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15, weight_decay=decay)
    history = []
    for epoch in range(30):
        loss_sum = 0.0
        for indices in torch.randperm(len(labels), generator=generator).split(256):
            optimizer.zero_grad()
            logits = model(train[indices])
            loss = F.cross_entropy(logits, labels[indices])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite loss or gradients")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite model parameter")
            loss_sum += loss.item() * len(indices)
        with torch.no_grad():
            validation = metrics(model(valid), valid_labels)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(labels),
                        "validation": validation})
    return {"seed": seed, "decay": decay, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history": history}


def results_path(root, name, distance, size, normalize):
    return root / "readouts" / name / str(distance) / str(size) / ("normalized" if normalize else "raw")


def select_all(samples, receipt, root, device):
    selection = {}
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.001):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                continue
            train_images, train_labels, train_rows = samples["train"]
            valid_images, valid_labels, valid_rows = samples["validation"]
            del train_images, valid_images
            train = load_role(root, name, binary, distance, window, "train", train_labels, train_rows)
            valid = load_role(root, name, binary, distance, window, "validation", valid_labels, valid_rows)
            train_labels = train_labels.to(device)
            valid_labels = valid_labels.to(device)
            for size in (100, 25):
                for normalize in (False, True):
                    directory = results_path(root, name, distance, size, normalize)
                    directory.mkdir(parents=True, exist_ok=True)
                    x_train = readout_features(train.to(device), size, normalize)
                    x_valid = readout_features(valid.to(device), size, normalize)
                    scale = x_train.square().sum(1).mean().sqrt()
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Invalid train-only scale")
                    x_train /= scale
                    x_valid /= scale
                    candidates = []
                    for decay in (0.0, 0.0001, 0.001):
                        runs = []
                        for seed in (11, 23, 37):
                            path = directory / f"decay_{decay}_seed_{seed}.pt"
                            if path.exists():
                                candidate = torch.load(path, map_location="cpu", weights_only=True)
                                if candidate["seed"] != seed or candidate["decay"] != decay or candidate["scale"] != scale.item() or len(candidate["history"]) != 30:
                                    raise ValueError(f"Candidate mismatch {path}")
                            else:
                                candidate = fit_candidate(x_train, train_labels, x_valid, valid_labels, seed, decay, device)
                                candidate["scale"] = scale.item()
                                save_chunk(path, candidate)
                            runs.append(candidate["history"][-1]["validation"]["accuracy"])
                        candidates.append({"decay": decay, "validation_accuracies": runs,
                                           "mean_validation_accuracy": sum(runs) / len(runs)})
                    best = max(candidates, key=lambda c: (c["mean_validation_accuracy"], -c["decay"]))
                    key = f"{name}/{distance}/{size}/{'normalized' if normalize else 'raw'}"
                    selection[key] = {"decay": best["decay"], "scale": scale.item(), "candidates": candidates}
                    print(f"SELECTED {key} decay={best['decay']}", flush=True)
                    del x_train, x_valid
            del train, valid
    selected_path = root / "readout_selection.json"
    if selected_path.exists():
        previous = json.loads(selected_path.read_text(encoding="utf-8"))
        if previous != selection:
            raise ValueError("Frozen model selection changed")
    else:
        with selected_path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    return selection


def evaluate_all(samples, receipt, root, selection, device):
    results = {}
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.001):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                continue
            _, labels, rows = samples["test"]
            images = load_role(root, name, binary, distance, window, "test", labels, rows).to(device)
            labels = labels.to(device)
            for size in (100, 25):
                for normalize in (False, True):
                    key = f"{name}/{distance}/{size}/{'normalized' if normalize else 'raw'}"
                    chosen = selection[key]
                    x = readout_features(images, size, normalize) / chosen["scale"]
                    runs = []
                    for seed in (11, 23, 37):
                        directory = results_path(root, name, distance, size, normalize)
                        path = directory / f"decay_{chosen['decay']}_seed_{seed}.pt"
                        candidate = torch.load(path, map_location="cpu", weights_only=True)
                        model = torch.nn.Linear(x.shape[1], 4, device=device)
                        model.load_state_dict(candidate["state"])
                        with torch.no_grad():
                            logits = model(x)
                            result = metrics(logits, labels)
                            result["prediction"] = logits.argmax(1).cpu().tolist()
                        runs.append({"seed": seed, "metrics": result})
                    results[key] = {"runs": runs}
            del images
    path = root / "readout_test.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != results:
            raise ValueError("Existing frozen test results differ")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    samples = population(args.indices, args.data, args.train, args.test)
    receipt = json.loads((args.output / "propagation_preflight.json").read_text(encoding="utf-8"))
    expected = 4 * sum(receipt["windows"][name][str(distance)] is not None
                       for name in ("gray", "binary") for distance in (0, 0.001))
    if receipt["status"] != "READY_PROPAGATION" or len(receipt["readout_checks"]) != expected:
        raise ValueError("Readout preflight incomplete")
    selection = select_all(samples, receipt, args.output, torch.device("cuda"))
    evaluate_all(samples, receipt, args.output, selection, torch.device("cuda"))


if __name__ == "__main__":
    main()
