"""Independent local verification of returned C1 population and frozen readouts."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c1_audit_v1 import aggregate
from hw207_mnist4_c1_v1 import population


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "data_root", "analysis_root", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    receipt = json.loads((args.data_root / "propagation_preflight.json").read_text(encoding="utf-8"))
    selection = json.loads((args.analysis_root / "readout_selection.json").read_text(encoding="utf-8"))
    readout_test = json.loads((args.analysis_root / "readout_test.json").read_text(encoding="utf-8"))
    verification = {"conditions": {}, "readouts": {}, "status": "CHECKING"}
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.0005, 0.001, 0.002):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                continue
            for role, (_, labels, rows) in samples.items():
                result = aggregate(args.data_root, name, binary, distance, window, role, labels, rows)
                saved = torch.load(args.data_root / "roi_audit" / name / str(distance) / (role + ".pt"),
                                   map_location="cpu", weights_only=True)
                if result["count"] != saved["count"] or result["counts"] != saved["counts"]:
                    raise ValueError("ROI counts differ")
                for kind in ("a", "b"):
                    for field in ("confusion", "prediction"):
                        if result["roi"][kind][field] != saved["roi"][kind][field]:
                            raise ValueError(f"ROI {field} mismatch")
                for field in ("input_energy", "full_energy", "crop_energy", "roi_b_outside"):
                    if not torch.allclose(torch.tensor(result[field]), torch.tensor(saved[field]), rtol=1e-6, atol=1e-5):
                        raise ValueError(f"Energy mismatch: {field}")
                for field in ("class_mean", "class_variance"):
                    torch.testing.assert_close(result[field], saved[field], rtol=1e-6, atol=1e-6)
                verification["conditions"][f"{name}/{distance}/{role}"] = {
                    "count": result["count"],
                    "roi_a_accuracy": result["roi"]["a"]["accuracy"],
                    "roi_b_accuracy": result["roi"]["b"]["accuracy"],
                    "roi_a_coverage": result["roi"]["a"]["coverage"],
                    "crop_fraction_mean": (torch.tensor(result["crop_energy"]) / torch.tensor(result["input_energy"])).mean().item(),
                }
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.001):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                continue
            _, labels, rows = samples["test"]
            for size in (100, 25):
                for normalize in (False, True):
                    mode = "normalized" if normalize else "raw"
                    key = f"{name}/{distance}/{size}/{mode}"
                    chosen = selection[key]
                    _, train_labels, train_rows = samples["train"]
                    sum_squared = 0.0
                    zero_rows = 0
                    for start in range(0, len(train_rows), 256):
                        chunk = torch.load(args.data_root / "chunks" / name / str(distance) / "train" / f"{start:06d}.pt",
                                           map_location="cpu", weights_only=True)
                        image = chunk["intensity"]
                        energy = image.sum((1, 2), keepdim=True)
                        zero_rows += (energy == 0).sum().item()
                        if normalize:
                            image = image / torch.where(energy > 0, energy, torch.ones_like(energy))
                        if size == 25:
                            image = F.avg_pool2d(image[:, None], 4)[:, 0]
                        sum_squared += image.double().square().sum().item()
                    computed_scale = (sum_squared / len(train_labels)) ** 0.5
                    if abs(computed_scale - chosen["scale"]) > 1e-5 * computed_scale:
                        raise ValueError(f"Train-only scale mismatch: {key}")
                    models = []
                    directory = args.analysis_root / "readouts" / name / str(distance) / str(size) / mode
                    for seed in (11, 23, 37):
                        candidate = torch.load(directory / f"decay_{chosen['decay']}_seed_{seed}.pt",
                                               map_location="cpu", weights_only=True)
                        models.append(candidate["state"])
                    for run in readout_test[key]["runs"]:
                        measured = run["metrics"]
                        predicted = torch.tensor(measured["prediction"], dtype=torch.int64)
                        if len(predicted) != len(labels) or (predicted < 0).any() or (predicted > 3).any():
                            raise ValueError(f"Invalid saved predictions: {key}")
                        matrix = torch.bincount(labels * 4 + predicted, minlength=16).reshape(4, 4)
                        if matrix.tolist() != measured["confusion"]:
                            raise ValueError(f"Saved confusion mismatch: {key}")
                        hit = matrix.diag()
                        accuracy = hit.sum().item() / len(labels)
                        recall = hit / matrix.sum(1).clamp_min(1)
                        f1 = (2 * hit / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)).mean().item()
                        if (abs(accuracy - measured["accuracy"]) > 1e-12
                                or abs(recall.mean().item() - measured["balanced_accuracy"]) > 1e-6
                                or abs(f1 - measured["macro_f1"]) > 1e-6):
                            raise ValueError(f"Saved metric mismatch: {key}")
                    mismatches = [0, 0, 0]
                    correct = [0, 0, 0]
                    differences = [[], [], []]
                    test_images = []
                    for start in range(0, len(rows), 256):
                        end = min(start + 256, len(rows))
                        chunk = torch.load(args.data_root / "chunks" / name / str(distance) / "test" / f"{start:06d}.pt",
                                           map_location="cpu", weights_only=True)
                        image = chunk["intensity"]
                        if args.device == "cuda":
                            test_images.append(image)
                        if normalize:
                            energy = image.sum((1, 2), keepdim=True)
                            image = image / torch.where(energy > 0, energy, torch.ones_like(energy))
                        if size == 25:
                            image = F.avg_pool2d(image[:, None], 4)[:, 0]
                        x = image.flatten(1) / chosen["scale"]
                        for i, state in enumerate(models):
                            predicted = (x @ state["weight"].T + state["bias"]).argmax(1)
                            old = torch.tensor(readout_test[key]["runs"][i]["metrics"]["prediction"][start:end])
                            mismatches[i] += (predicted != old).sum().item()
                            correct[i] += (predicted == labels[start:end]).sum().item()
                            for index in torch.where(predicted != old)[0].tolist():
                                scores = (x[index].double() @ state["weight"].double().T + state["bias"].double())
                                top = scores.topk(2).values
                                differences[i].append({"test_position": start + index,
                                    "original_index": rows[start + index].item(),
                                    "label": labels[start + index].item(),
                                    "saved": old[index].item(), "cpu": predicted[index].item(),
                                    "fp64_top_margin": (top[0] - top[1]).item()})
                    gpu_mismatches = None
                    if args.device == "cuda":
                        image = torch.cat(test_images).cuda()
                        if normalize:
                            energy = image.sum((1, 2), keepdim=True)
                            image = image / torch.where(energy > 0, energy, torch.ones_like(energy))
                        if size == 25:
                            image = F.avg_pool2d(image[:, None], 4)[:, 0]
                        full_x = image.flatten(1) / chosen["scale"]
                        gpu_mismatches = []
                        for i, state in enumerate(models):
                            model = torch.nn.Linear(full_x.shape[1], 4, device="cuda")
                            model.load_state_dict(state)
                            with torch.no_grad():
                                predicted = model(full_x).argmax(1).cpu()
                            expected = torch.tensor(readout_test[key]["runs"][i]["metrics"]["prediction"])
                            gpu_mismatches.append((predicted != expected).sum().item())
                        if any(gpu_mismatches):
                            raise ValueError(f"Full-batch CUDA readout mismatch: {key} {gpu_mismatches}")
                    verification["readouts"][key] = {"mismatches": mismatches, "correct": correct,
                                                       "differences": differences, "cuda_mismatches": gpu_mismatches,
                                                       "train_zero_energy": zero_rows, "scale": computed_scale}
    verification["status"] = "PASS_WITH_CPU_DIFFERENCES" if any(
        any(r["mismatches"]) for r in verification["readouts"].values()) else "PASS"
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, indent=2, allow_nan=False)
    print(verification["status"], len(verification["conditions"]), len(verification["readouts"]))


if __name__ == "__main__":
    main()
