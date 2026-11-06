"""C1 fixed-population propagation and ROI measurement."""
import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from hw207_mnist4_c1_v1 import (
    checked_chunk, numerical_probes, population, produce_chunk, save_chunk,
)


def condition_window(convergence, name, distance):
    if distance == 0:
        return 100
    result = convergence["results"][name][str(distance)]
    if not result["window"] or not result["precision"] or not result["precision"]["passed"]:
        return None
    comparisons = result["comparisons"]
    if not comparisons or not comparisons[-1]["passed"] or comparisons[-1]["large"] != result["window"]:
        raise ValueError("Invalid convergence selection")
    if max(comparisons[-1]["image_errors"] + comparisons[-1]["roi_errors"]) > 1e-3:
        raise ValueError("Invalid convergence errors")
    if max(result["precision"]["complex_errors"] + result["precision"]["energy_errors"]) > 1e-5:
        raise ValueError("Invalid precision errors")
    return result["window"]


def readout_features(intensity, size, normalize):
    if normalize:
        energy = intensity.sum((-2, -1), keepdim=True)
        intensity = intensity / torch.where(energy > 0, energy, torch.ones_like(energy))
    if size == 25:
        intensity = F.avg_pool2d(intensity[:, None], 4)[:, 0]
    return intensity.flatten(1)


def readout_fit(x, y, seed, decay, epochs, batch, steps=None):
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], 4, device=x.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.15, weight_decay=decay)
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}
    step = 0
    for _ in range(epochs):
        for ids in torch.randperm(len(y), generator=generator).split(batch):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Invalid readout loss or gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Invalid readout parameter")
            step += 1
            if steps == step:
                if any(torch.equal(initial[name], param) for name, param in model.named_parameters()):
                    raise ValueError("Readout trainable parameter did not update")
                return model
    return model


def readout_preflight(samples, windows, device):
    selected = {}
    for role, per_class in (("train", 128), ("validation", 16)):
        images, labels, rows = samples[role]
        indices = torch.cat([torch.where(labels == label)[0][:per_class] for label in range(4)])
        selected[role] = (images[indices], labels[indices].to(device), rows[indices])
    checks = {}
    for binary in (False, True):
        name = "binary" if binary else "gray"
        for distance in (0, 0.001):
            window = windows[name][str(distance)]
            if window is None:
                continue
            features_by_role = {}
            for role, (images, labels, rows) in selected.items():
                parts = []
                for start in range(0, len(rows), 256):
                    stop = min(start + 256, len(rows))
                    chunk = produce_chunk(images[start:stop], labels[start:stop], rows[start:stop],
                                          binary, distance, window, device, 8)
                    parts.append(chunk["intensity"])
                features_by_role[role] = torch.cat(parts).to(device)
            for size in (100, 25):
                for normalize in (False, True):
                    train = readout_features(features_by_role["train"], size, normalize)
                    valid = readout_features(features_by_role["validation"], size, normalize)
                    scale = train.square().sum(1).mean().sqrt()
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Invalid train-only RMS scale")
                    train, valid = train / scale, valid / scale
                    train_labels = selected["train"][1]
                    readout_fit(train, train_labels, 11, 0.0, 1, 64, steps=3)
                    key = f"{name}_{distance}_{size}_{normalize}"
                    checks[key] = {"three_steps": "PASS", "train_rms_scale": scale.item()}
                    if binary and distance == 0.001 and not normalize:
                        permutation = torch.randperm(len(train_labels), generator=torch.Generator().manual_seed(20701))
                        shuffled = train_labels[permutation]
                        negative = readout_fit(train, shuffled, 11, 0.0, 30, 256)
                        with torch.no_grad():
                            accuracy = (negative(valid).argmax(1) == selected["validation"][1]).float().mean().item()
                        checks[key]["negative_accuracy"] = accuracy
                        if accuracy >= 0.4:
                            raise ValueError(f"Shuffled-label negative control failed: {key}")
    return checks


def preflight(samples, convergence, device):
    images, labels, rows = samples["validation"]
    selected = torch.cat([torch.where(labels == label)[0][:16] for label in range(4)])
    if rows[selected].tolist() != convergence["validation_original_indices"] or labels[selected].tolist() != convergence["labels"]:
        raise ValueError("Convergence population does not match C0 validation rows")
    probes = numerical_probes(device)
    if not probes["passed"]:
        raise ValueError("Production numerical probes failed")
    windows = {}
    for binary in (False, True):
        name = "binary" if binary else "gray"
        windows[name] = {str(d): condition_window(convergence, name, d) for d in (0, 0.0005, 0.001, 0.002)}
        for distance in (0, 0.0005, 0.001, 0.002):
            window = windows[name][str(distance)]
            if window is None:
                continue
            sample = selected[:4]
            whole = produce_chunk(images[sample], labels[sample], rows[sample], binary, distance, window, device, 2)
            parts = [produce_chunk(images[part], labels[part], rows[part], binary, distance, window, device, 1)
                     for part in sample.split(2)]
            with tempfile.TemporaryDirectory() as temporary:
                checkpoint = Path(temporary) / "chunk.pt"
                save_chunk(checkpoint, parts[0])
                parts[0] = checked_chunk(checkpoint, binary, distance, window,
                                         labels[sample[:2]], rows[sample[:2]])
            for key in ("rows", "labels", "intensity", "input_energy", "full_energy", "crop_energy", "roi_a", "roi_b"):
                combined = torch.cat([part[key] for part in parts])
                if not torch.allclose(whole[key], combined, rtol=1e-5, atol=1e-5):
                    raise ValueError(f"Interrupted chunk mismatch: {name} {distance} {key}")
    readout_checks = readout_preflight(samples, windows, device)
    return {"status": "READY_PROPAGATION", "probes": probes, "windows": windows,
            "readout_checks": readout_checks, "validation_original_indices": rows[selected].tolist()}


def measure(samples, receipt, output, device):
    for name, binary in (("gray", False), ("binary", True)):
        for distance in (0, 0.0005, 0.001, 0.002):
            window = receipt["windows"][name][str(distance)]
            if window is None:
                print(f"SKIP {name} {distance} convergence blocked", flush=True)
                continue
            for role, (images, labels, rows) in samples.items():
                directory = output / "chunks" / name / str(distance) / role
                directory.mkdir(parents=True, exist_ok=True)
                for start in range(0, len(rows), 256):
                    stop = min(start + 256, len(rows))
                    path = directory / f"{start:06d}.pt"
                    if path.exists():
                        checked_chunk(path, binary, distance, window, labels[start:stop], rows[start:stop], role)
                        continue
                    chunk = produce_chunk(images[start:stop], labels[start:stop], rows[start:stop],
                                          binary, distance, window, device, 8, role)
                    save_chunk(path, chunk)
                    checked_chunk(path, binary, distance, window, labels[start:stop], rows[start:stop], role)
                print(f"COMPLETE {name} {distance} {role} rows={len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ("indices", "data", "train", "test", "convergence", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("C1 full production requires CUDA")
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    samples = population(args.indices, args.data, args.train, args.test)
    convergence = json.loads(args.convergence.read_text(encoding="utf-8"))
    receipt = preflight(samples, convergence, torch.device(args.device))
    receipt["inputs"] = {key: str(getattr(args, key).resolve()) for key in ("indices", "data", "train", "test", "convergence")}
    receipt_path = args.output / "propagation_preflight.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old["windows"] != receipt["windows"] or old["inputs"] != receipt["inputs"]:
            raise ValueError("Resume inputs do not match")
    else:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    measure(samples, receipt, args.output, torch.device(args.device))
    print("PROPAGATION_FINISHED", flush=True)


if __name__ == "__main__":
    main()
