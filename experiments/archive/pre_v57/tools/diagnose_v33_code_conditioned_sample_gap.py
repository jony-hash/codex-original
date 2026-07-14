#!/usr/bin/env python3
"""v33 code-conditioned generator/sample-gap diagnostic.

This diagnostic reads existing v33 eval/sample tensor bundles.  It does not
train, sample, use CUDA, or write accepted summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch
import torch.nn.functional as F

from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output


PLANE_KEYS = (
    "target_code_amplitude",
    "decoder_input_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
BRIDGE_PLANES = {"target_code_amplitude", "decoder_input_amplitude"}
DECODER_PLANES = {"decoder_output_amplitude", "chip_intensity_primary"}
PHYSICAL_PLANE = "physical_camera_plane"


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v33_code_conditioned_sample_gap_{timestamp}.json"


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"tensor bundle must contain a dict payload: {path}")
    return payload


def _bundle_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_tensors.pt" if kind == "eval" else "one_shot_sample_tensors.pt"
    return run / "artifacts" / filename


def _load_bundle(run: Path, kind: str) -> dict[str, Any]:
    path = _bundle_path(run, kind)
    if not path.is_file():
        raise FileNotFoundError(path)
    return _torch_load(path)


def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4:
        raise ValueError(f"expected 2D/3D/4D image tensor, got shape {tuple(value.shape)}")
    if value.shape[1] != 1:
        value = value[:, :1]
    return value


def _labels(bundle: dict[str, Any]) -> torch.Tensor:
    labels = bundle.get("class_labels")
    if not torch.is_tensor(labels):
        labels = bundle.get("labels")
    if not torch.is_tensor(labels):
        raise ValueError("tensor bundle missing class_labels/labels")
    return labels.detach().long().cpu().view(-1)


def _reference(bundle: dict[str, Any], kind: str) -> tuple[str, torch.Tensor]:
    keys = ("eval_target", "teacher_camera", "reference_images") if kind == "eval" else ("reference_images",)
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, _as_nchw(value)
    raise ValueError(f"{kind} tensor bundle missing reference tensor")


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = _as_nchw(tensor)
    dims = tuple(range(1, value.ndim))
    min_v = value.amin(dim=dims, keepdim=True)
    max_v = value.amax(dim=dims, keepdim=True)
    return ((value - min_v) / (max_v - min_v).clamp_min(1.0e-6)).clamp(0.0, 1.0)


def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = _as_nchw(tensor)
    target = _as_nchw(reference)
    if value.shape[-2:] != target.shape[-2:]:
        value = F.interpolate(value, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return value


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    left_nchw = _as_nchw(a)
    right_nchw = _resize_like(b, left_nchw)
    count = min(left_nchw.shape[0], right_nchw.shape[0])
    if count <= 0:
        return float("nan")
    left = left_nchw[:count].reshape(count, -1)
    right = right_nchw[:count].reshape(count, -1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    denom = left.norm(dim=1) * right.norm(dim=1)
    valid = denom > 1.0e-8
    if not bool(valid.any()):
        return 0.0
    return float(((left[valid] * right[valid]).sum(dim=1) / denom[valid]).mean().item())


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    left = _as_nchw(a)
    right = _resize_like(b, left)
    count = min(left.shape[0], right.shape[0])
    if count <= 0:
        return float("nan")
    return float(F.mse_loss(left[:count], right[:count]).item())


def _class_means(tensor: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = _as_nchw(tensor)
    labels = labels[: value.shape[0]]
    classes = torch.unique(labels, sorted=True)
    means = []
    kept = []
    for label in classes:
        mask = labels == label
        if bool(mask.any()):
            means.append(value[mask].mean(dim=0))
            kept.append(int(label.item()))
    if not means:
        raise ValueError("no class means could be computed")
    return torch.tensor(kept, dtype=torch.long), torch.stack(means, dim=0)


def _class_similarity(a: torch.Tensor, a_labels: torch.Tensor, b: torch.Tensor, b_labels: torch.Tensor) -> dict[str, Any]:
    a_classes, a_means = _class_means(a, a_labels)
    b_classes, b_means = _class_means(b, b_labels)
    common = sorted(set(a_classes.tolist()) & set(b_classes.tolist()))
    if not common:
        raise ValueError("no common labels for class similarity")
    a_map = {int(label): idx for idx, label in enumerate(a_classes.tolist())}
    b_map = {int(label): idx for idx, label in enumerate(b_classes.tolist())}
    left = torch.stack([a_means[a_map[label]] for label in common], dim=0)
    right = torch.stack([b_means[b_map[label]] for label in common], dim=0)
    return {"labels": common, "corr": _corr(left, right), "mse": _mse(left, right)}


def _same_label_diversity(tensor: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    value = _as_nchw(tensor)
    labels = labels[: value.shape[0]]
    entries = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        if int(mask.sum().item()) < 2:
            continue
        group = value[mask].reshape(int(mask.sum().item()), -1)
        centered = group - group.mean(dim=0, keepdim=True)
        entries.append(
            {
                "label": int(label.item()),
                "count": int(mask.sum().item()),
                "mean_pixel_std": float(group.std(dim=0, unbiased=False).mean().item()),
                "mean_mse_to_class_mean": float((centered.square().mean(dim=1)).mean().item()),
            }
        )
    return {
        "labels_with_repeats": len(entries),
        "mean_pixel_std": float(sum(item["mean_pixel_std"] for item in entries) / len(entries)) if entries else None,
        "mean_mse_to_class_mean": (
            float(sum(item["mean_mse_to_class_mean"] for item in entries) / len(entries)) if entries else None
        ),
        "per_label": entries,
    }


def _residual_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    residual = bundle.get("generator_residual_map")
    if not torch.is_tensor(residual):
        return {"available": False, "abs_mean": None, "same_label_diversity": None}
    labels = _labels(bundle)
    value = _as_nchw(residual)
    return {
        "available": True,
        "abs_mean": float(value.abs().mean().item()),
        "abs_max": float(value.abs().max().item()),
        "same_label_diversity": _same_label_diversity(value, labels),
    }


def _plane_records(bundle: dict[str, Any], labels: torch.Tensor, reference: torch.Tensor) -> tuple[dict[str, Any], list[str]]:
    records = {}
    missing = []
    for key in PLANE_KEYS:
        value = bundle.get(key)
        if not torch.is_tensor(value):
            missing.append(key)
            continue
        normalized = _normalize(value)
        ref = _normalize(reference)
        records[key] = {
            "reference_corr": _class_similarity(normalized, labels, ref, labels)["corr"],
            "reference_mse": _class_similarity(normalized, labels, ref, labels)["mse"],
            "same_label_diversity": _same_label_diversity(normalized, labels),
        }
    return records, missing


def _run_label(path: Path) -> str:
    name = path.name
    for marker in ("_train_v33_", "v33_"):
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def analyze_run(run: Path, *, kind: str) -> dict[str, Any]:
    bundle = _load_bundle(run, kind)
    labels = _labels(bundle)
    reference_key, reference = _reference(bundle, kind)
    residual = _residual_summary(bundle)
    planes, missing = _plane_records(bundle, labels, reference)
    physical = bundle.get(PHYSICAL_PLANE)
    if not torch.is_tensor(physical):
        return {
            "run": run.as_posix(),
            "run_type": kind,
            "label": _run_label(run),
            "status": "harness-invalid",
            "classification": "harness-invalid",
            "reason": "missing physical_camera_plane",
            "missing_planes": missing,
        }
    if not residual.get("available"):
        return {
            "run": run.as_posix(),
            "run_type": kind,
            "label": _run_label(run),
            "reference_key": reference_key,
            "reference_source": None,
            "status": "harness-invalid",
            "classification": "harness-invalid",
            "reason": "missing generator_residual_map",
            "checks": {
                "residual_active": False,
                "bridge_pass": False,
                "decoder_pass": False,
                "physical_cmos_pass": False,
                "sample_reference_metric_invalid": False,
            },
            "metrics": {
                "residual": residual,
                "planes": planes,
            },
            "missing_planes": missing,
            "human_readability_required": True,
            "accepted_summary_written": False,
        }
    physical_ref = _class_similarity(_normalize(physical), labels, _normalize(reference), labels)
    target_ref = planes.get("target_code_amplitude", {})
    decoder_ref = planes.get("decoder_output_amplitude", {})
    bridge_pass = any(planes.get(key, {}).get("reference_corr", 0.0) >= 0.75 for key in BRIDGE_PLANES)
    decoder_pass = any(planes.get(key, {}).get("reference_corr", 0.0) >= 0.70 for key in DECODER_PLANES)
    physical_pass = physical_ref["corr"] >= 0.70
    residual_abs = float(residual["abs_mean"] or 0.0)
    residual_div = None
    if isinstance(residual.get("same_label_diversity"), dict):
        residual_div = residual["same_label_diversity"].get("mean_mse_to_class_mean")
    residual_active = residual_abs > 1.0e-4 and (residual_div is None or float(residual_div or 0.0) > 1.0e-10)

    sample_metric_invalid = False
    sample_ref_source = None
    if kind == "sample":
        ref_source = bundle.get("reference_source")
        sample_ref_source = str(ref_source) if ref_source is not None else None
        codes = bundle.get("codes")
        sample_metric_invalid = torch.is_tensor(codes) and sample_ref_source != "paired_distill_code_reference"

    if not residual_active:
        status = "residual-inactive"
        reason = "generator residual is missing or effectively constant"
    elif kind == "sample" and sample_metric_invalid and not physical_pass:
        status = "sample-reference-metric-invalid"
        reason = "sample exact metrics compare random/generated codes to unpaired reference images"
    elif bridge_pass and decoder_pass and physical_pass:
        status = "code-conditioned-target-rescue"
        reason = "code-conditioned planes and physical CMOS are readable against the active reference"
    elif bridge_pass and not decoder_pass:
        status = "bridge-after-code-failure"
        reason = "code-conditioned bridge plane is readable but decoder/chip is not"
    elif decoder_pass and not physical_pass:
        status = "camera-after-code-failure"
        reason = "decoder/chip plane is readable but physical CMOS is not"
    elif not bridge_pass:
        status = "code-target-unlearnable"
        reason = "target_code/decoder_input does not preserve the code-conditioned target"
    else:
        status = "harness-invalid"
        reason = "diagnostic predicates were inconclusive"
    return {
        "run": run.as_posix(),
        "run_type": kind,
        "label": _run_label(run),
        "reference_key": reference_key,
        "reference_source": sample_ref_source,
        "status": status,
        "classification": status,
        "reason": reason,
        "checks": {
            "residual_active": residual_active,
            "bridge_pass": bridge_pass,
            "decoder_pass": decoder_pass,
            "physical_cmos_pass": physical_pass,
            "sample_reference_metric_invalid": sample_metric_invalid,
        },
        "metrics": {
            "residual": residual,
            "physical_camera_vs_reference": physical_ref,
            "target_code_vs_reference": target_ref,
            "decoder_output_vs_reference": decoder_ref,
            "planes": planes,
        },
        "missing_planes": missing,
        "human_readability_required": True,
        "accepted_summary_written": False,
    }


def _extract_runs_from_summary(summary_payload: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    eval_runs = []
    sample_runs = []
    for probe in summary_payload.get("probes", []):
        if not isinstance(probe, dict):
            continue
        for key in ("same_batch_eval_run", "test_eval_run", "eval_run"):
            value = probe.get(key)
            if value:
                eval_runs.append(Path(str(value)))
        if probe.get("sample_run"):
            sample_runs.append(Path(str(probe["sample_run"])))
    return _dedupe(eval_runs), _dedupe(sample_runs)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    if by_status.get("code-conditioned-target-rescue"):
        status = "code-conditioned-target-rescue"
    elif by_status.get("sample-reference-metric-invalid"):
        status = "sample-reference-metric-invalid"
    elif by_status.get("camera-after-code-failure"):
        status = "camera-after-code-failure"
    elif by_status.get("bridge-after-code-failure"):
        status = "bridge-after-code-failure"
    elif by_status.get("harness-invalid"):
        status = "harness-invalid"
    elif by_status.get("residual-inactive"):
        status = "residual-inactive"
    elif by_status.get("code-target-unlearnable"):
        status = "code-target-unlearnable"
    else:
        status = "harness-invalid"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def run_v33_code_conditioned_sample_gap(
    *,
    summary: Path | None = None,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_eval = list(eval_runs or [])
    resolved_sample = list(sample_runs or [])
    source_summary = None
    if summary is not None:
        summary_payload = _load_json(summary)
        assert summary_payload is not None
        source_summary = summary.as_posix()
        if not (resolved_eval or resolved_sample):
            resolved_eval, resolved_sample = _extract_runs_from_summary(summary_payload)
    records = []
    errors = []
    for run in _dedupe(resolved_eval):
        try:
            records.append(analyze_run(run, kind="eval"))
        except Exception as exc:
            errors.append({"run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            records.append({"run": run.as_posix(), "run_type": "eval", "status": "harness-invalid", "classification": "harness-invalid"})
    for run in _dedupe(resolved_sample):
        try:
            records.append(analyze_run(run, kind="sample"))
        except Exception as exc:
            errors.append({"run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            records.append({"run": run.as_posix(), "run_type": "sample", "status": "harness-invalid", "classification": "harness-invalid"})
    if not records:
        raise ValueError("at least one eval/sample run or summary with runs is required")
    decision = _decision(records)
    payload = {
        "version": "v33",
        "objective": "code_conditioned_target_sample_gap_diagnostic",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": source_summary,
        "eval_runs": [run.as_posix() for run in _dedupe(resolved_eval)],
        "sample_runs": [run.as_posix() for run in _dedupe(resolved_sample)],
        "runs": records,
        "errors": errors,
        "status": decision["status"],
        "decision": decision,
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--eval-run", action="append", type=Path, default=None)
    parser.add_argument("--sample-run", action="append", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v33_code_conditioned_sample_gap(
        summary=args.summary,
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
