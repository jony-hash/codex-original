"""Independently recompute C1 ROI readout and intensity summaries."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

from hw207_mnist4_c1_v1 import population, save_chunk


def aggregate(root, name, binary, distance, window, role, labels, rows):
    confusion = {"a": torch.zeros(4, 5, dtype=torch.int64),
                 "b": torch.zeros(4, 5, dtype=torch.int64)}
    energy_in, energy_full, energy_crop, outside = [], [], [], []
    margins = {"a": [], "b": []}
    sums = torch.zeros((4, 100, 100), dtype=torch.float64)
    squares = torch.zeros_like(sums)
    counts = torch.zeros(4, dtype=torch.int64)
    predictions = {"a": [], "b": []}
    for start in range(0, len(rows), 256):
        end = min(start + 256, len(rows))
        path = root / "chunks" / name / str(distance) / role / f"{start:06d}.pt"
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        if (chunk["binary"], chunk["distance_m"], chunk["window"], chunk["role"]) != (binary, distance, window, role):
            raise ValueError(f"Condition mismatch: {path}")
        if not torch.equal(chunk["rows"], rows[start:end]) or not torch.equal(chunk["labels"], labels[start:end]):
            raise ValueError(f"Population mismatch: {path}")
        intensity = chunk["intensity"]
        if intensity.shape != (end-start, 100, 100) or not torch.isfinite(intensity).all() or (intensity < 0).any():
            raise ValueError(f"Invalid intensity: {path}")
        crop = intensity.sum((1, 2))
        if not torch.allclose(crop, chunk["crop_energy"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"Crop energy mismatch: {path}")
        input_energy = chunk["input_energy"]
        full_energy = chunk["full_energy"]
        if (not torch.isfinite(input_energy).all() or not torch.isfinite(full_energy).all()
                or (input_energy <= 0).any() or (full_energy < 0).any()):
            raise ValueError(f"Invalid energy: {path}")
        if ((full_energy - input_energy).abs() / input_energy > 1e-5).any():
            raise ValueError(f"Full-window energy mismatch: {path}")
        if (crop > full_energy + 1e-5 * input_energy).any():
            raise ValueError(f"Crop exceeds full-window energy: {path}")
        for window_name, (y0, y1, x0, x1) in {
            "a0": (0, 50, 0, 50), "a1": (0, 50, 50, 100),
            "a2": (50, 100, 0, 50), "a3": (50, 100, 50, 100),
            "b0": (5, 45, 5, 45), "b1": (5, 45, 55, 95),
            "b2": (55, 95, 5, 45), "b3": (55, 95, 55, 95),
        }.items():
            if window_name.endswith("0"):
                scores = []
            scores.append(intensity[:, y0:y1, x0:x1].sum((1, 2)))
            if window_name.endswith("3"):
                kind = window_name[0]
                scores = torch.stack(scores, dim=1)
                if not torch.allclose(scores, chunk["roi_" + kind], rtol=1e-6, atol=1e-5):
                    raise ValueError(f"ROI mismatch: {path}")
                top, _ = scores.topk(2, dim=1)
                predicted = scores.argmax(1)
                predicted[(top[:, 0] == 0) | (top[:, 0] == top[:, 1])] = 4
                truth = labels[start:end]
                confusion[kind] += torch.bincount(truth * 5 + predicted, minlength=20).reshape(4, 5)
                predictions[kind].extend(predicted.tolist())
                correct = scores.gather(1, truth[:, None])[:, 0]
                wrong = scores.clone()
                wrong.scatter_(1, truth[:, None], -1)
                margins[kind].extend((correct - wrong.max(1).values).tolist())
                if kind == "b":
                    outside.extend((crop - scores.sum(1)).tolist())
        for label in range(4):
            selected = intensity[labels[start:end] == label].double()
            sums[label] += selected.sum(0)
            squares[label] += selected.square().sum(0)
            counts[label] += len(selected)
        energy_in.extend(chunk["input_energy"].tolist())
        energy_full.extend(chunk["full_energy"].tolist())
        energy_crop.extend(crop.tolist())
    results = {}
    for kind, matrix in confusion.items():
        tp = matrix[:, :4].diag()
        precision_denominator = matrix[:, :4].sum(0)
        recall_denominator = matrix.sum(1)
        f1 = 2 * tp / (precision_denominator + recall_denominator).clamp_min(1)
        results[kind] = {"confusion": matrix.tolist(), "accuracy": tp.sum().item() / len(rows),
                         "coverage": 1 - matrix[:, 4].sum().item() / len(rows),
                         "macro_f1": f1.mean().item(), "prediction": predictions[kind],
                         "correct_minus_wrong": margins[kind]}
    counts_float = counts[:, None, None].double()
    if (counts == 0).any():
        raise ValueError("Every class must have samples")
    means = sums / counts_float
    variance = (squares / counts_float - means.square()).clamp_min(0)
    pairs = {f"{a}-{b}": torch.linalg.vector_norm(means[a] - means[b]).item()
             for a, b in combinations(range(4), 2)}
    return {"count": len(rows), "counts": counts.tolist(), "roi": results, "input_energy": energy_in,
            "full_energy": energy_full, "crop_energy": energy_crop, "roi_b_outside": outside,
            "class_mean": means.float(), "class_variance": variance.float(),
            "class_mean_pair_l2": pairs}


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    samples = population(args.indices, args.data, args.train, args.test)
    receipt = json.loads((args.output / "propagation_preflight.json").read_text(encoding="utf-8"))
    if receipt["status"] != "READY_PROPAGATION":
        raise ValueError("Missing valid propagation preflight")
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.0005, 0.001, 0.002):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                continue
            for role, (_, labels, rows) in samples.items():
                path = args.output / "roi_audit" / name / str(distance) / (role + ".pt")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError(path)
                result = aggregate(args.output, name, binary, distance, window, role, labels, rows)
                save_chunk(path, result)
                print(f"ROI_AUDIT {name} {distance} {role} {result['count']}", flush=True)


if __name__ == "__main__":
    main()
