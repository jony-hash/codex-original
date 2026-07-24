#!/usr/bin/env python3
"""Diagnostic-only software readout upper bound over the frozen v61 cache.

The readouts never participate in the single-exposure model: they answer only
whether a cached plane retains paired target information.  Validation selects
ridge strength; the test split is evaluated once after that selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
SEEDS = (1234, 1235, 1236)
STRICT = {"corr": 0.78, "dice": 0.72, "saturation_error": 0.08}


def _feature(values: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(values):
        values = torch.cat((values.real, values.imag), dim=1)
    return F.adaptive_avg_pool2d(values.float(), (32, 32))


def _target(values: torch.Tensor) -> torch.Tensor:
    return values.float().clamp_min(0.0)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    scale = torch.quantile(image.flatten(1), 0.995, dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
    return (image / scale).clamp(0.0, 1.0)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred, truth = _normalize(prediction), _normalize(target)
    flat_pred, flat_truth = pred.flatten(1), truth.flatten(1)
    p_centered, t_centered = flat_pred - flat_pred.mean(1, keepdim=True), flat_truth - flat_truth.mean(1, keepdim=True)
    corr = (p_centered * t_centered).sum(1) / (p_centered.norm(dim=1) * t_centered.norm(dim=1)).clamp_min(1e-8)
    p_mask, t_mask = pred >= 0.5, truth >= 0.5
    dice = (2 * (p_mask & t_mask).flatten(1).sum(1).float()) / (p_mask.flatten(1).sum(1) + t_mask.flatten(1).sum(1)).clamp_min(1)
    saturation_error = ((pred >= 0.99).flatten(1).float().mean(1) - (truth >= 0.99).flatten(1).float().mean(1)).abs()
    return {"corr": corr, "dice": dice, "saturation_error": saturation_error}


def _summary(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    metrics = _metrics(prediction, target)
    norm = _normalize(prediction)
    diversity = torch.pdist(norm.flatten(1), p=2).square().mean() if len(norm) > 1 else torch.tensor(0.0)
    return {"corr": float(metrics["corr"].mean()), "dice": float(metrics["dice"].mean()), "saturation_error": float(metrics["saturation_error"].mean()), "diversity": float(diversity), "per_sample": {name: value.tolist() for name, value in metrics.items()}}


def paired_bootstrap_delta(candidate: list[float], control: list[float], *, seed: int, iterations: int = 2000) -> dict[str, float]:
    left, right = torch.tensor(candidate), torch.tensor(control)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(left), (iterations, len(left)), generator=generator)
    draws = (left[indices] - right[indices]).mean(1)
    return {"mean": float((left - right).mean()), "ci_low": float(torch.quantile(draws, 0.025)), "ci_high": float(torch.quantile(draws, 0.975))}


class TinyUNetReadout(nn.Module):
    """Fixed small diagnostic readout; never exported into optical inference."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1), nn.GELU())
        self.out = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Softplus())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=(200, 200), mode="bilinear", align_corners=False)
        skip = self.down(value)
        hidden = F.interpolate(self.bottleneck(self.pool(skip)), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat((skip, hidden), dim=1))


def _pca(train: torch.Tensor, *others: torch.Tensor, dimensions: int = 64) -> tuple[torch.Tensor, ...]:
    mean = train.mean(0, keepdim=True)
    _, _, basis = torch.pca_lowrank(train - mean, q=min(dimensions, train.shape[0] - 1, train.shape[1]), center=False)
    return tuple((value - mean) @ basis for value in (train, *others))


def _ridge_fit(train_x: torch.Tensor, train_y: torch.Tensor, strength: float) -> torch.Tensor:
    augmented = torch.cat((train_x, torch.ones(len(train_x), 1)), dim=1)
    eye = torch.eye(augmented.shape[1])
    eye[-1, -1] = 0.0
    return torch.linalg.solve(augmented.T @ augmented + strength * eye, augmented.T @ train_y)


def ridge_probe(train_x: torch.Tensor, val_x: torch.Tensor, test_x: torch.Tensor, train_y: torch.Tensor, val_y: torch.Tensor, test_y: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    train_x, val_x, test_x = _pca(train_x, val_x, test_x)
    train_y_small = F.adaptive_avg_pool2d(train_y, (32, 32)).flatten(1)
    val_y_small = F.adaptive_avg_pool2d(val_y, (32, 32)).flatten(1)
    chosen, best = None, float("-inf")
    for strength in RIDGE_GRID:
        weight = _ridge_fit(train_x, train_y_small, strength)
        val_prediction = torch.cat((val_x, torch.ones(len(val_x), 1)), dim=1) @ weight
        score = _metrics(val_prediction.view(-1, 1, 32, 32), val_y_small.view(-1, 1, 32, 32))["corr"].mean().item()
        if score > best:
            chosen, best = strength, score
    weight = _ridge_fit(train_x, train_y_small, float(chosen))
    prediction = torch.cat((test_x, torch.ones(len(test_x), 1)), dim=1) @ weight
    prediction = F.interpolate(prediction.view(-1, 1, 32, 32), size=(200, 200), mode="bilinear", align_corners=False)
    return prediction, {"selected_ridge": chosen, "validation_corr": best}


def nonlinear_probe(train_x: torch.Tensor, val_x: torch.Tensor, test_x: torch.Tensor, train_y: torch.Tensor, val_y: torch.Tensor, test_y: torch.Tensor, *, seed: int, epochs: int, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    model = TinyUNetReadout(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for start in range(0, len(train_x), 32):
            prediction = model(train_x[start:start + 32].to(device))
            loss = F.mse_loss(_normalize(prediction), _normalize(train_y[start:start + 32].to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = F.mse_loss(_normalize(model(val_x.to(device))), _normalize(val_y.to(device))).item()
        if loss < best:
            best, best_state = loss, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state or {})
    model.eval()
    with torch.no_grad():
        return model(test_x.to(device)).cpu(), {"seed": seed, "epochs": epochs, "validation_mse": best}


def _controls(train: dict[str, Any], test: dict[str, Any]) -> dict[str, torch.Tensor]:
    labels, target = train["labels"], train["target"]
    prototypes = torch.stack([target[labels == label].mean(0) for label in range(10)])
    label_only = prototypes[test["labels"]]
    shuffled = test["target"].clone()
    for label in range(10):
        indices = torch.nonzero(test["labels"] == label, as_tuple=False).flatten()
        shuffled[indices] = test["target"][indices.roll(1)]
    return {"label_only": label_only, "shuffled_target": shuffled}


def _candidate_planes(report: dict[str, Any]) -> tuple[str, ...]:
    first = report.get("decision", {}).get("first_plane")
    order = tuple(report.get("planes", {}).get("test", {}))
    if first in order and order.index(first) > 0:
        return (order[order.index(first) - 1], first)
    return ("chip", "camera_intensity")


def _input_from_cache(split: dict[str, Any], plane: str, mode: str) -> torch.Tensor:
    if mode == "code_only":
        return split["codes"]
    if mode == "code_label":
        one_hot = F.one_hot(split["labels"], 10).float()
        values = torch.cat((split["codes"], one_hot), dim=1)
        return values
    return _feature(split["planes"][plane])


def run(*, cache: Path, audit_report: Path, output: Path, device: str = "cuda", nonlinear_epochs: int = 60) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v61 nonlinear upper bound requires CUDA")
    bundle = torch.load(cache, map_location="cpu", weights_only=False)
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    cached = bundle["cached"]
    if bundle.get("parameter_hash") != report.get("invariants", {}).get("parameter_hash_before"):
        raise ValueError("cache and robust audit do not describe the same frozen model")
    controls = _controls(cached["train"], cached["test"])
    results: dict[str, Any] = {"identity": _summary(cached["test"]["target"], cached["test"]["target"]), "controls": {name: _summary(value, cached["test"]["target"]) for name, value in controls.items()}, "linear": {}, "nonlinear": {}}
    for plane in tuple(report["planes"]["test"]):
        train_x, val_x, test_x = (_input_from_cache(cached[name], plane, "plane").flatten(1) for name in ("train", "val", "test"))
        prediction, selection = ridge_probe(train_x, val_x, test_x, cached["train"]["target"], cached["val"]["target"], cached["test"]["target"])
        results["linear"][plane] = {"selection": selection, "metrics": _summary(prediction, cached["test"]["target"])}
    for mode in ("code_only", "code_label"):
        train_x, val_x, test_x = (_input_from_cache(cached[name], "slm_command", mode).flatten(1) for name in ("train", "val", "test"))
        prediction, selection = ridge_probe(train_x, val_x, test_x, cached["train"]["target"], cached["val"]["target"], cached["test"]["target"])
        results["linear"][mode] = {"selection": selection, "metrics": _summary(prediction, cached["test"]["target"])}
    for plane in _candidate_planes(report):
        seed_results = []
        for seed in SEEDS:
            prediction, details = nonlinear_probe(*(_input_from_cache(cached[name], plane, "plane") for name in ("train", "val", "test")), cached["train"]["target"], cached["val"]["target"], cached["test"]["target"], seed=seed, epochs=nonlinear_epochs, device=torch.device(device))
            seed_results.append({"details": details, "metrics": _summary(prediction, cached["test"]["target"])} )
        results["nonlinear"][plane] = seed_results
    verdicts = {}
    for kind in ("linear", "nonlinear"):
        entries = results[kind]
        for name, entry in entries.items():
            metrics = entry["metrics"] if kind == "linear" else entry[0]["metrics"]
            label, shuffled = results["controls"]["label_only"], results["controls"]["shuffled_target"]
            verdicts[f"{kind}:{name}"] = {"strict_readable": metrics["corr"] >= STRICT["corr"] and metrics["dice"] >= STRICT["dice"] and metrics["saturation_error"] <= STRICT["saturation_error"], "corr_vs_label_only": paired_bootstrap_delta(metrics["per_sample"]["corr"], label["per_sample"]["corr"], seed=111), "dice_vs_label_only": paired_bootstrap_delta(metrics["per_sample"]["dice"], label["per_sample"]["dice"], seed=112), "corr_vs_shuffled": paired_bootstrap_delta(metrics["per_sample"]["corr"], shuffled["per_sample"]["corr"], seed=113), "dice_vs_shuffled": paired_bootstrap_delta(metrics["per_sample"]["dice"], shuffled["per_sample"]["dice"], seed=114)}
    final = {"status": "manual-review-required", "diagnostic_only": True, "accepted_summary_written": False, "invariants": {"frozen_cache_parameter_hash": bundle["parameter_hash"], "parameter_update_count": 0, "formal_contract_changed": False, "test_evaluated_once_after_validation_selection": True}, "inputs": {"cache": str(cache), "audit_report": str(audit_report)}, "pre_registered": {"ridge_grid": RIDGE_GRID, "seeds": SEEDS, "strict": STRICT, "nonlinear_epochs": nonlinear_epochs}, "results": results, "verdicts": verdicts}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nonlinear-epochs", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
