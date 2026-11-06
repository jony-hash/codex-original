"""C3.3 P4/P5/P6 single-layer phase training, frozen evaluation and export."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from scripts.research.lightgen import hw207_mnist4_c33_phase_v1 as core

INITS = ("zero", "small", "uniform")


def make_phase(name: str, shape, device: str, generator: torch.Generator) -> torch.Tensor:
    if name == "zero":
        return torch.zeros(shape, device=device)
    if name == "small":
        return (torch.rand(shape, generator=generator, dtype=torch.float32).to(device) - 0.5) * (0.2 * math.pi)
    if name == "uniform":
        return torch.rand(shape, generator=generator, dtype=torch.float32).to(device) * (2 * math.pi)
    raise ValueError(f"unknown init {name}")


def evaluate_split(configuration, phase, images, labels, batch_size: int, weights: dict | None = None) -> dict:
    return core.evaluate(images, labels, configuration, phase, batch_size, weights)


def train(args) -> dict:
    device = args.device
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    images_all, labels_all = core.load_mnist()
    train_rows, validation_rows, holdout_rows = core.stratified_split(
        core.c0_train_indices(), labels_all, seed=20261104)
    if args.limit and args.limit < len(train_rows):
        permutation = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(20261104))
        train_rows = torch.sort(train_rows[permutation[: args.limit]])[0]

    configuration = core.build_configuration(args.defocus, device)
    phase = torch.nn.Parameter(make_phase(args.init, (1, 1, core.GRID, core.GRID), device, generator))

    def effective_phase() -> torch.Tensor:
        """Optionally train a laterally smoothed (manufacturable) phase."""
        if args.smooth_sigma > 0:
            return core.smooth_height(phase, args.smooth_sigma)
        return phase

    optimizer = torch.optim.Adam([phase], lr=args.lr)
    weights = dict(core.DEFAULT_WEIGHTS)
    weights["tv"] = args.tv

    train_images, train_labels = images_all[train_rows].to(device), labels_all[train_rows].to(device)
    validation_images = images_all[validation_rows].to(device)
    validation_labels = labels_all[validation_rows].to(device)

    history = []
    best = {"accuracy": -1.0, "epoch": -1, "loss": float("inf"), "state": None}
    started = time.time()
    for epoch in range(args.epochs):
        configuration.base.screen.train()
        order = torch.randperm(len(train_rows), generator=generator)
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), args.batch):
            index = order[start: start + args.batch]
            current = effective_phase()
            result = core.forward_field(train_images[index], configuration, current)
            loss = core.loss_from_result({**result, "phase": current}, train_labels[index], weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if phase.grad is None or not torch.isfinite(phase.grad).all():
                raise RuntimeError("non-finite phase gradient")
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        evaluation = evaluate_split(configuration, effective_phase().detach(), validation_images,
                                   validation_labels, args.batch, weights)
        metrics = evaluation["metrics"]
        entry = {"epoch": epoch + 1, "train_loss": epoch_loss / max(batches, 1),
                 "val_loss": evaluation["loss"], "val_accuracy": metrics["accuracy"],
                 "val_macro_f1": metrics["macro_f1"], "phase_l2": float(phase.detach().norm()),
                 "phase_tv": float(core.phase_tv(phase.detach())),
                 "mean_detected_energy": metrics["mean_detected_energy"],
                 "elapsed_s": time.time() - started}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        improved = (metrics["accuracy"] > best["accuracy"] + 1e-9
                    or (abs(metrics["accuracy"] - best["accuracy"]) <= 1e-9
                        and evaluation["loss"] < best["loss"]))
        if improved:
            best = {"accuracy": metrics["accuracy"], "epoch": epoch + 1,
                    "loss": evaluation["loss"], "state": phase.detach().clone()}
        if args.patience and epoch + 1 - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch + 1}", flush=True)
            break

    if best["state"] is None:
        best["state"] = phase.detach().clone()
        best["epoch"] = len(history)
        best["accuracy"] = history[-1]["val_accuracy"] if history else 0.0

    final_phase = best["state"]
    effective = (core.smooth_height(final_phase.to(device), args.smooth_sigma).detach()
                 if args.smooth_sigma > 0 else final_phase)
    evaluation = evaluate_split(configuration, effective, validation_images, validation_labels, args.batch)
    holdout = evaluate_split(configuration, effective,
                             images_all[holdout_rows].to(device), labels_all[holdout_rows].to(device), args.batch)
    train_eval = evaluate_split(configuration, effective,
                                images_all[train_rows].to(device), labels_all[train_rows].to(device), args.batch)

    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({"phase": final_phase.cpu(), "effective": effective.cpu(),
                "smooth_sigma": args.smooth_sigma, "defocus_um": args.defocus, "init": args.init,
                "seed": args.seed, "best_epoch": best["epoch"],
                "train_rows": train_rows, "validation_rows": validation_rows,
                "holdout_rows": holdout_rows}, args.output / "phase.pt")

    summary = {
        "identity": "HW207_C33_SINGLE_PHASE_TRAINING",
        "defocus_um": args.defocus,
        "init": args.init,
        "seed": args.seed,
        "optimizer": {"name": "Adam", "lr": args.lr, "epochs_requested": args.epochs,
                      "batch": args.batch, "limit": args.limit, "patience": args.patience},
        "weights": weights,
        "rows": {"train": int(len(train_rows)), "validation": int(len(validation_rows)),
                 "holdout": int(len(holdout_rows))},
        "best_epoch": best["epoch"],
        "history": history,
        "validation": evaluation["metrics"],
        "holdout": holdout["metrics"],
        "train": train_eval["metrics"],
        "smooth_sigma": args.smooth_sigma, "phase": {"l2": float(final_phase.norm()), "tv": float(core.phase_tv(final_phase)),
                  "wrapped_min": float(core.wrap_phase(final_phase).min()),
                  "wrapped_max": float(core.wrap_phase(final_phase).max()),
                  "fingerprint": core.tensor_sha256(final_phase)},
        "elapsed_s": time.time() - started,
        "device": device,
        "trainable_parameters": int(phase.numel()),
    }
    core.write_json(args.output / "training_summary.json", summary)
    print(json.dumps({"done": True, "output": str(args.output),
                      "val_acc": summary["validation"]["accuracy"],
                      "holdout_acc": summary["holdout"]["accuracy"]}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", choices=INITS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--defocus", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tv", type=float, default=1e-3)
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="train through a lateral Gaussian of this many pixels (manufacturable arm)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()