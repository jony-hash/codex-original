#!/usr/bin/env python3
"""Frozen v60 full-split optical-field audit with robust collapse statistics.

This diagnostic is deliberately outside the formal inference path.  It caches
read-only forward fields for the v61 software upper bound and never writes a
student checkpoint or changes the physical-forward contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config


DEFAULT_CHECKPOINT = Path("outputs/20260715_v60/runs/20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/checkpoints/one_shot_distill_best.pt")
DEFAULT_TRAIN = Path("outputs/20260514_unversioned/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/teacher_distill_dataset_readable_train.pt")
DEFAULT_VAL = DEFAULT_TRAIN.with_name("teacher_distill_dataset_readable_val.pt")
DEFAULT_TEST = DEFAULT_TRAIN.with_name("teacher_distill_dataset_readable_test.pt")
DEFAULT_CONFIG = Path("configs/eval_v56_coherent_relay_from_scratch.yaml")
PLANE_ORDER = ("slm_command", "slm_output", "pre_encoder_relay_image", "encoder_input", "encoder_layer_1", "encoder_layer_2", "encoder_output", "ols_input", "ols_output", "decoder_input", "decoder_layer_1", "decoder_layer_2", "decoder_output", "chip", "camera_field", "camera_intensity")
SEPARATION_METRICS = ("amplitude", "complex", "intensity")


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _complex(record: dict[str, Any]) -> torch.Tensor:
    real, imag = record.get("real"), record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag) or real.shape != imag.shape:
        raise ValueError("strict optical field is missing matching real/imag tensors")
    return torch.complex(real, imag).detach().cpu()


def _load_split(path: Path, expected: str, count: int) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid readable payload: {path}")
    codes, labels, target = payload.get("codes"), payload.get("class_labels"), payload.get("teacher_camera")
    if not all(torch.is_tensor(value) for value in (codes, labels, target)):
        raise ValueError(f"{expected} payload lacks codes/labels/teacher_camera")
    counts = {label: int((labels == label).sum()) for label in range(10)}
    if payload.get("readability_split") != expected or len(labels) != count or any(value != count // 10 for value in counts.values()):
        raise ValueError(f"{expected} payload is not the required balanced readable split: {counts}")
    if payload.get("config", {}).get("filter", {}).get("teacher_filter") != "readability_strict_split":
        raise ValueError(f"{expected} payload is not readability_strict_split")
    return {"codes": codes.float(), "labels": labels.long(), "target": target.float()}


def _forward(model: torch.nn.Module, split: dict[str, torch.Tensor], device: torch.device, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    values: dict[str, list[torch.Tensor]] = {name: [] for name in PLANE_ORDER}
    provenance: dict[str, Any] | None = None
    with torch.no_grad():
        for start in range(0, len(split["labels"]), batch_size):
            output = model(split["codes"][start:start + batch_size].to(device), split["labels"][start:start + batch_size].to(device), return_optical_fields=True)
            fields = output.get("optical_fields")
            if not isinstance(fields, dict):
                raise ValueError("model did not return strict optical fields")
            if provenance is None:
                provenance = {name: {key: value for key, value in record.items() if key not in {"real", "imag"}} for name, record in fields.items()}
            values["slm_command"].append(output["slm_command"].detach().cpu().float())
            values["camera_intensity"].append(output["physical_camera_plane"].detach().cpu().float())
            for name in PLANE_ORDER:
                if name in {"slm_command", "camera_intensity"}:
                    continue
                values[name].append(_complex(fields[name]))
    result = {name: torch.cat(parts) for name, parts in values.items()}
    return result, provenance or {}


def _distance(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a, b = left.to(torch.complex128), right.to(torch.complex128)
    aa, bb = a.abs(), b.abs()
    def relative(x: torch.Tensor, y: torch.Tensor) -> float:
        return float((x - y).norm() / ((x.norm() + y.norm()).mul(0.5).clamp_min(1e-12)))
    phase = torch.angle((a.conj() * b).sum())
    aligned = a * torch.exp(1j * phase).to(a.dtype)
    return {"amplitude": relative(aa, bb), "complex": relative(aligned, b), "intensity": relative(aa.square(), bb.square())}


def _pair_signals(values: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    same = {name: [] for name in SEPARATION_METRICS}
    cross = {name: [] for name in SEPARATION_METRICS}
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            destination = same if int(labels[left]) == int(labels[right]) else cross
            for name, value in _distance(values[left, 0], values[right, 0]).items():
                destination[name].append(value)
    return same, cross


def _diagnostic_reduce(values: torch.Tensor) -> torch.Tensor:
    """Use a fixed 32² analysis grid; cached formal fields remain untouched."""
    if tuple(values.shape[-2:]) == (32, 32):
        return values
    if torch.is_complex(values):
        return torch.complex(F.adaptive_avg_pool2d(values.real, (32, 32)), F.adaptive_avg_pool2d(values.imag, (32, 32)))
    return F.adaptive_avg_pool2d(values, (32, 32))


def bootstrap_ci(values: list[float], *, iterations: int, seed: int, draw_size: int = 512) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    generator = torch.Generator().manual_seed(seed)
    # A full 4,500-pair bootstrap for every complex plane would allocate many
    # gigabytes.  Fixed-size pair resampling keeps the 2,000 replicate rule
    # while making the diagnostic feasible on the project's 6 GB CUDA host.
    draws = torch.randint(tensor.numel(), (iterations, min(draw_size, tensor.numel())), generator=generator)
    medians = tensor[draws].median(dim=1).values
    return {"median": float(tensor.median()), "ci_low": float(torch.quantile(medians, 0.025)), "ci_high": float(torch.quantile(medians, 0.975)), "draw_size": min(draw_size, tensor.numel())}


def summarize_plane(values: torch.Tensor, labels: torch.Tensor, *, iterations: int, seed: int) -> dict[str, Any]:
    reduced = _diagnostic_reduce(values)
    same, cross = _pair_signals(reduced, labels)
    result: dict[str, Any] = {"sample_count": int(len(labels)), "analysis_shape": list(reduced.shape[-2:]), "separation": {}}
    for offset, name in enumerate(SEPARATION_METRICS):
        same_ci, cross_ci = bootstrap_ci(same[name], iterations=iterations, seed=seed + offset), bootstrap_ci(cross[name], iterations=iterations, seed=seed + 100 + offset)
        # Independent pair resampling is conservative diagnostic uncertainty;
        # label permutation below separately tests class association.
        result["separation"][name] = {"same": same_ci, "cross": cross_ci, "signal": cross_ci["median"] - same_ci["median"], "signal_ci_low": cross_ci["ci_low"] - same_ci["ci_high"], "signal_ci_high": cross_ci["ci_high"] - same_ci["ci_low"]}
    return result


def _features(values: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(values):
        values = torch.cat((values.real, values.imag), dim=1)
    return F.adaptive_avg_pool2d(values.float(), (16, 16)).flatten(1)


def permutation_null(values: torch.Tensor, labels: torch.Tensor, *, permutations: int, seed: int) -> dict[str, float]:
    feature = _features(values)
    classes = torch.arange(10)
    def accuracy(candidate: torch.Tensor) -> float:
        centroids = torch.stack([feature[candidate == label].mean(0) for label in classes])
        scores = F.normalize(feature, dim=1) @ F.normalize(centroids, dim=1).T
        return float((scores.argmax(1) == labels).float().mean())
    observed = accuracy(labels)
    generator = torch.Generator().manual_seed(seed)
    null = torch.tensor([accuracy(labels[torch.randperm(len(labels), generator=generator)]) for _ in range(permutations)])
    return {"observed_accuracy": observed, "null_p95": float(torch.quantile(null, 0.95)), "p_value": float(((null >= observed).sum() + 1) / (permutations + 1))}


def decision(val: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    planes = list(val)
    for plane in planes:
        failed = []
        for metric in SEPARATION_METRICS:
            if val[plane]["relative_retention"][metric]["ci_high"] < 0.10 and test[plane]["relative_retention"][metric]["ci_high"] < 0.10:
                failed.append(metric)
        if len(failed) >= 2:
            return {"decision": "robust-localized-collapse", "first_plane": plane, "metrics": failed}
    chip, camera = val.get("chip"), val.get("camera_field")
    if chip and camera and chip["relative_retention"]["complex"]["ci_low"] >= 0.10 and camera["relative_retention"]["intensity"]["ci_high"] < 0.10 and test["chip"]["relative_retention"]["complex"]["ci_low"] >= 0.10 and test["camera_field"]["relative_retention"]["intensity"]["ci_high"] < 0.10:
        return {"decision": "robust-phase-to-intensity-readout-collapse", "first_plane": "camera_field"}
    return {"decision": "no-robust-localized-collapse"}


def _add_relative_retention(summary: dict[str, dict[str, Any]]) -> None:
    baseline = summary["slm_command"]["separation"]
    for plane in summary.values():
        plane["relative_retention"] = {}
        for metric in SEPARATION_METRICS:
            denominator = max(abs(float(baseline[metric]["signal"])), 1e-12)
            signal = plane["separation"][metric]
            plane["relative_retention"][metric] = {key: float(signal[f"signal_{key}"]) / denominator for key in ("ci_low", "ci_high")}
            plane["relative_retention"][metric]["median"] = float(signal["signal"]) / denominator


def run(*, checkpoint: Path = DEFAULT_CHECKPOINT, train_dataset: Path = DEFAULT_TRAIN, val_dataset: Path = DEFAULT_VAL, test_dataset: Path = DEFAULT_TEST, config: Path = DEFAULT_CONFIG, output: Path, device: str = "cuda", batch_size: int = 4, bootstrap_iterations: int = 2000, permutations: int = 128) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v61 formal forward requires CUDA; CPU is diagnostic-only and not accepted")
    splits = {"train": _load_split(train_dataset, "train", 800), "val": _load_split(val_dataset, "val", 100), "test": _load_split(test_dataset, "test", 100)}
    cfg = load_training_config(config)
    model = OneShotOpticalGenerator(cfg.physics, cfg.one_shot, image_size=cfg.first_stage.image_size).to(device).eval()
    load_one_shot_checkpoint(model, torch.load(checkpoint, map_location="cpu", weights_only=False))
    before = _state_hash(model)
    cached: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for name, split in splits.items():
        planes, metadata = _forward(model, split, torch.device(device), batch_size)
        cached[name] = {**split, "planes": planes}
        provenance[name] = metadata
    after = _state_hash(model)
    if before != after:
        raise RuntimeError("frozen-model hash changed during v61 forward")
    val = {plane: summarize_plane(cached["val"]["planes"][plane], cached["val"]["labels"], iterations=bootstrap_iterations, seed=1234 + index * 17) for index, plane in enumerate(PLANE_ORDER)}
    test = {plane: summarize_plane(cached["test"]["planes"][plane], cached["test"]["labels"], iterations=bootstrap_iterations, seed=2234 + index * 17) for index, plane in enumerate(PLANE_ORDER)}
    _add_relative_retention(val)
    _add_relative_retention(test)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_name("v61_frozen_optical_cache.pt")
    torch.save({"cached": cached, "provenance": provenance, "checkpoint": str(checkpoint), "parameter_hash": before}, cache_path)
    report = {"status": "manual-review-required", "diagnostic_only": True, "accepted_summary_written": False, "invariants": {"parameter_update_count": 0, "parameter_hash_before": before, "parameter_hash_after": after, "formal_contract_changed": False}, "inputs": {"checkpoint": str(checkpoint), "config": str(config), "train": str(train_dataset), "val": str(val_dataset), "test": str(test_dataset), "cache": str(cache_path)}, "bootstrap": {"iterations": bootstrap_iterations, "method": "pair-median resampling", "seed": 1234}, "permutation_null": {plane: permutation_null(cached["test"]["planes"][plane], cached["test"]["labels"], permutations=permutations, seed=3234 + index) for index, plane in enumerate(PLANE_ORDER)}, "planes": {"val": val, "test": test}, "decision": decision(val, test)}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
