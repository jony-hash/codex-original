"""HW207 C0 digital experiment. CPU only, no optical runtime."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

TARGET = {"identity": "HW207_TARGET_DIAGNOSTIC", "assumption": "ASSUMED_FULL_FIELD_DIAGNOSTIC",
          "pitch_um": 2, "wavelength_nm": 532, "field_um": [200, 200],
          "source_grid": [28, 28], "target_grid": [100, 100], "source_features": 784}
CURRENT = {"identity": "HW207_CURRENT_MAPPING", "status": "BLOCKED",
           "missing": ["DMD integration", "ON/OFF collection direction", "input relay conjugacy and coordinate mapping"]}


def load(path, size):
    x, y = torch.load(path, map_location="cpu", weights_only=True)
    if x.shape != (size, 28, 28) or x.dtype != torch.uint8 or y.shape != (size,) or y.dtype != torch.int64 or not ((0 <= y) & (y <= 9)).all():
        raise ValueError(f"Unexpected MNIST format: {path}")
    return x, y


def split(y, seed):
    rng = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in range(4):
        rows = torch.where(y == label)[0]
        rows = rows[torch.randperm(len(rows), generator=rng)]
        n = round(len(rows) / 10)
        train.append(rows[n:])
        val.append(rows[:n])
    return torch.cat(train), torch.cat(val)


def features(x, binary):
    return (x >= 128).float() if binary else x.float() / 255


def metrics(logits, y):
    y = y.to(logits.device)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    cm = torch.bincount(4 * y + logits.argmax(1), minlength=16).reshape(4, 4)
    hit = cm.diag()
    recall = hit / cm.sum(1).clamp_min(1)
    return dict(accuracy=hit.sum().item() / len(y), balanced_accuracy=recall.mean().item(),
                macro_f1=(2 * hit / (cm.sum(0) + cm.sum(1)).clamp_min(1)).mean().item(),
                precision=(hit / cm.sum(0).clamp_min(1)).tolist(), recall=recall.tolist(),
                confusion=cm.tolist(), support=cm.sum(1).tolist())


def fit(x, y, seed, decay, cfg, steps=None):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    device = torch.device(cfg.get("device", "cpu"))
    x, y = x.to(device), y.to(device)
    model = torch.nn.Linear(784, 4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=decay)
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}
    step = 0
    for _ in range(cfg["epochs"]):
        for ids in torch.randperm(len(y), generator=rng).split(cfg["batch"]):
            opt.zero_grad()
            loss = F.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite loss or gradient")
            opt.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite parameters")
            step += 1
            if steps == step:
                if any(torch.equal(initial[name], param) for name, param in model.named_parameters()):
                    raise ValueError("A trainable group did not update")
                return model
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite parameters")
    return model


def select(x, y, v, vy, cfg):
    candidates = []
    for decay in cfg["decays"]:
        runs = []
        for seed in cfg["seeds"]:
            start = time.perf_counter()
            model = fit(x, y, seed, decay, cfg)
            elapsed = time.perf_counter() - start
            runs.append((model, {"seed": seed, "train_seconds": elapsed,
                                 "validation": metrics(model(v.to(model.weight.device)).detach(), vy)}))
        mean = sum(r["validation"]["accuracy"] for _, r in runs) / len(runs)
        candidates.append((mean, decay, runs))
    best = max(candidates, key=lambda c: (c[0], -c[1]))
    return best, [{"decay": d, "mean_validation_accuracy": m,
                   "runs": [r for _, r in runs]} for m, d, runs in candidates]


def map_image(x, binary):
    if binary:
        return F.interpolate(x[:, None], size=(100, 100), mode="nearest-exact")[:, 0]
    return F.interpolate(x[:, None], size=(100, 100), mode="bilinear", align_corners=False)[:, 0]


def morphology(x):
    mask = x >= 0.5
    side = x.shape[-1]
    axes = []
    for occupied in (mask.any(1), mask.any(2)):
        i = torch.arange(side)
        lo = torch.where(occupied, i, side).amin(1)
        hi = torch.where(occupied, i, -1).amax(1)
        axes.append((hi - lo + 1).clamp_min(0) / side)
    edge = ((mask[:, 1:] != mask[:, :-1]).sum((1, 2)) +
            (mask[:, :, 1:] != mask[:, :, :-1]).sum((1, 2))) / side
    return torch.stack((mask.float().mean((1, 2)), *axes, edge), 1)


def diagnose(x, binary, batch):
    measurements = []
    for raw in x.split(batch):
        src = features(raw, binary)
        dst = map_image(src, binary)
        if not ((dst >= 0) & (dst <= 1)).all() or (binary and not ((dst == 0) | (dst == 1)).all()):
            raise ValueError("Invalid target values")
        back = F.interpolate(dst[:, None], size=(28, 28), mode="area")[:, 0]
        before, after = morphology(src), morphology(dst)
        mae = (src - back).abs().mean((1, 2))
        measurements.append(torch.cat((before, after, after - before, mae[:, None]), 1))
    values = torch.cat(measurements)
    fields = ["foreground_fraction", "bbox_width_fraction", "bbox_height_fraction", "edge_length_over_field_side"]
    names = [f"{prefix}_{field}" for prefix in ("source", "target", "delta") for field in fields]
    names.append("area_roundtrip_mae")
    return {"count": len(x), "mode": "nearest-exact" if binary else "bilinear align_corners=False",
            "target_bbox_mean_um": (values[:, 5:7].mean(0) * 200).tolist(),
            "statistics": {name: {"mean": col.mean().item(), "min": col.min().item(),
                                  "max": col.max().item(), "mean_abs": col.abs().mean().item()}
                           for name, col in zip(names, values.T)}}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("training", "test", "config", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["device"] = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    x, y = load(args.training, 60000)
    t, ty = load(args.test, 10000)
    a, b = split(y, cfg["split_seed"])
    c = torch.where(ty <= 3)[0]
    sources = {"train": (x, y, a, args.training), "validation": (x, y, b, args.training),
               "test": (t, ty, c, args.test)}
    counts = {k: torch.bincount(labels[ids], minlength=4).tolist()
              for k, (_, labels, ids, _) in sources.items()}
    if len(torch.unique(torch.cat((a, b)))) != 24754 or sum(counts["test"]) != 4157 or len(a) + len(b) != 24754:
        raise ValueError("Invalid population or split overlap")
    seen = set()
    for images, _, ids, _ in sources.values():
        for row in images[ids].reshape(-1, 784).tolist():
            pixel_key = bytes(row)
            if pixel_key in seen:
                raise ValueError("Duplicate pixel content")
            seen.add(pixel_key)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "CHECKING",
        "purpose": "C0 fixed train/validation split and validation-only selection",
        "expected_negative_control": "shuffled training labels, validation accuracy < 0.40",
        "negative_seed": 20701,
        "training": str(args.training.resolve()),
        "test": str(args.test.resolve()),
        "config": str(args.config.resolve()),
        "output": str(args.output.resolve()),
        "python": sys.version,
        "torch": torch.__version__,
        "device": args.device,
        "config_values": cfg,
        "counts": counts,
        "trainable_groups": ["weight", "bias"],
        "evaluator": "metrics only; no parameters",
        "test_role": "evaluation only after both selections",
        "resumability": "not claimed",
        "checks": {},
    }
    save(args.output / "preflight.json", receipt)
    started = time.perf_counter()
    try:
        for binary in (False, True):
            name = "binary" if binary else "gray"
            fx = features(x[a], binary).flatten(1)
            vx = features(x[b], binary).flatten(1)
            if not torch.isfinite(fx).all() or not torch.isfinite(vx).all():
                raise ValueError("Nonfinite input")
            model = fit(fx, y[a], cfg["seeds"][0], cfg["decays"][0], cfg, steps=3)
            real_metrics = metrics(model(vx.to(model.weight.device)).detach(), y[b])
            shuffled = y[a][torch.randperm(len(a), generator=torch.Generator().manual_seed(20701))]
            negative_model = fit(fx, shuffled, cfg["seeds"][0], cfg["decays"][0], cfg)
            negative = metrics(negative_model(vx.to(negative_model.weight.device)).detach(), y[b])
            if negative["accuracy"] >= 0.40:
                raise ValueError(f"{name} shuffled-label negative control was not rejected")
            diagnose(x[a[:cfg["batch"]]], binary, cfg["batch"])
            receipt["checks"][name] = {"three_real_updates": "PASS",
                "finite_input_loss_gradients_parameters": "PASS",
                "weight_and_bias_updated": "PASS", "validation_after_three_steps": real_metrics,
                "negative_control_validation": negative, "negative_control": "PASS",
                "mapping_batch": "PASS"}
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["status"] = "READY"
        save(args.output / "preflight.json", receipt)
    except Exception as error:
        receipt["status"] = "INVALID_HARNESS"
        receipt["error"] = repr(error)
        save(args.output / "preflight.json", receipt)
        raise
    print("READY: both preflights and shuffled-label controls passed", flush=True)
    if args.preflight:
        return
    save(args.output / "indices.json", {k: {"source": str(path.resolve()), "original_indices": ids.tolist()}
                                        for k, (_, _, ids, path) in sources.items()})
    result = {"identity": "HW207_MNIST4_C0", "config": cfg, "torch": torch.__version__,
              "counts": counts, "pixel_duplicates": 0,
              "sources_sha256": {p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                 for p in (args.training, args.test)},
              "target": TARGET, "current": CURRENT}
    save(args.output / "data.json", result)
    errors = {}
    for binary in (False, True):
        name = "binary" if binary else "gray"
        try:
            start = time.perf_counter()
            target = {k: diagnose(images[ids], binary, cfg["batch"])
                      for k, (images, _, ids, _) in sources.items()}
            target["seconds"] = time.perf_counter() - start
            save(args.output / f"mapping_{name}.json", target)
            print(f"Mapped {name}: {len(a)+len(b)+len(c)}", flush=True)
        except Exception as error:
            errors[f"mapping_{name}"] = repr(error)
    chosen = {}
    for binary in (False, True):
        name = "binary" if binary else "gray"
        try:
            model_selection = select(features(x[a], binary).flatten(1), y[a],
                                     features(x[b], binary).flatten(1), y[b], cfg)
            chosen[name] = (binary, model_selection)
        except Exception as error:
            errors[f"selection_{name}"] = repr(error)
    save(args.output / "selection.json", {name: {"decay": best[1], "candidates": grid}
                                          for name, (_, (best, grid)) in chosen.items()})
    for name, (binary, (best, grid)) in chosen.items():
        try:
            test = features(t[c], binary).flatten(1)
            for model, run in best[2]:
                with torch.no_grad():
                    start = time.perf_counter()
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    logits = model(test.to(model.weight.device))
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    run["inference_seconds"] = time.perf_counter() - start
                run["test"] = metrics(logits, ty[c])
            save(args.output / f"classification_{name}.json", {"decay": best[1], "parameters": 3140,
                 "runs": [run for _, run in best[2]]})
            print(f"{name} test accuracies: {[run['test']['accuracy'] for _, run in best[2]]}", flush=True)
        except Exception as error:
            errors[f"evaluation_{name}"] = repr(error)
    save(args.output / "execution.json", {"errors": errors})
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
