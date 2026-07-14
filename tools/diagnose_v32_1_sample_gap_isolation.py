#!/usr/bin/env python3
"""v32.1 offline sample-gap isolation diagnostic.

This tool reads existing v32 bridge redesign eval/sample artifacts and decides
whether eval-positive bridge outputs fail samples because of a reference/code
gap, a collapsed sample bridge plane, or a downstream camera gap. It never
trains, samples, uses CUDA, or writes accepted summaries.
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
    "slm_command",
    "encoded_amplitude",
    "target_code_amplitude",
    "decoder_input_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
TARGET_CODE_KEY = "target_code_amplitude"
PHYSICAL_KEY = "physical_camera_plane"
PROBE_PREFIXES = (
    "probe_a_ols_lowfreq_deep_bridge_same_batch",
    "probe_b_ols_spatial_multiscale_bridge_same_batch",
)


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v32_1_sample_gap_isolation_{timestamp}.json"


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
            return key, _as_nchw(value.detach().float().cpu())
    raise ValueError(f"{kind} tensor bundle missing reference tensor")


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


def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = _as_nchw(tensor)
    target = _as_nchw(reference)
    if value.shape[-2:] != target.shape[-2:]:
        value = F.interpolate(value, size=target.shape[-2:], mode="bilinear", align_corners=False)
    if value.shape[0] != target.shape[0]:
        if value.shape[0] == 1:
            value = value.expand(target.shape[0], -1, -1, -1)
        else:
            count = min(value.shape[0], target.shape[0])
            value = value[:count]
    return value


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    value = _as_nchw(tensor)
    dims = tuple(range(1, value.ndim))
    min_v = value.amin(dim=dims, keepdim=True)
    max_v = value.amax(dim=dims, keepdim=True)
    return (value - min_v) / (max_v - min_v).clamp_min(1e-6)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    left_nchw = _as_nchw(a)
    right_nchw = _as_nchw(b)
    left = left_nchw.reshape(left_nchw.shape[0], -1)
    right = right_nchw.reshape(right_nchw.shape[0], -1)
    count = min(left.shape[0], right.shape[0])
    if count == 0:
        return float("nan")
    left = left[:count] - left[:count].mean(dim=1, keepdim=True)
    right = right[:count] - right[:count].mean(dim=1, keepdim=True)
    denom = left.norm(dim=1) * right.norm(dim=1)
    valid = denom > 1e-8
    if not bool(valid.any()):
        return 0.0
    values = (left[valid] * right[valid]).sum(dim=1) / denom[valid]
    return float(values.mean().item())


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    left = _as_nchw(a)
    right = _as_nchw(b)
    count = min(left.shape[0], right.shape[0])
    if count == 0:
        return float("nan")
    return float(F.mse_loss(left[:count], right[:count]).item())


def _class_means(tensor: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = _as_nchw(tensor)
    labels = labels[: value.shape[0]]
    unique = torch.unique(labels, sorted=True)
    means = [value[labels == label].mean(dim=0) for label in unique if bool((labels == label).any())]
    if not means:
        raise ValueError("no class means could be computed")
    return unique[: len(means)], torch.stack(means, dim=0)


def _align_class_means(
    eval_tensor: torch.Tensor,
    eval_labels: torch.Tensor,
    sample_tensor: torch.Tensor,
    sample_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    eval_classes, eval_means = _class_means(eval_tensor, eval_labels)
    sample_classes, sample_means = _class_means(sample_tensor, sample_labels)
    common = sorted(set(int(label) for label in eval_classes.tolist()) & set(int(label) for label in sample_classes.tolist()))
    if not common:
        raise ValueError("eval/sample bundles have no labels in common")
    eval_map = {int(label): index for index, label in enumerate(eval_classes.tolist())}
    sample_map = {int(label): index for index, label in enumerate(sample_classes.tolist())}
    eval_aligned = torch.stack([eval_means[eval_map[label]] for label in common], dim=0)
    sample_aligned = torch.stack([sample_means[sample_map[label]] for label in common], dim=0)
    return eval_aligned, sample_aligned, common


def _class_mean_similarity(
    eval_tensor: torch.Tensor,
    eval_labels: torch.Tensor,
    sample_tensor: torch.Tensor,
    sample_labels: torch.Tensor,
) -> dict[str, Any]:
    eval_aligned, sample_aligned, labels = _align_class_means(eval_tensor, eval_labels, sample_tensor, sample_labels)
    sample_aligned = _resize_like(sample_aligned, eval_aligned)
    return {
        "labels": labels,
        "corr": _corr(eval_aligned, sample_aligned),
        "mse": _mse(eval_aligned, sample_aligned),
    }


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


def _code_diversity(bundle: dict[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    codes = bundle.get("codes")
    if not torch.is_tensor(codes):
        return {"available": False, "same_label_mean_std": None}
    value = codes.detach().float().cpu().reshape(codes.shape[0], -1)
    labels = labels[: value.shape[0]]
    stds = []
    for label in torch.unique(labels, sorted=True):
        group = value[labels == label]
        if group.shape[0] >= 2:
            stds.append(float(group.std(dim=0, unbiased=False).mean().item()))
    return {
        "available": True,
        "same_label_mean_std": float(sum(stds) / len(stds)) if stds else 0.0,
    }


def _plane_metrics(
    eval_bundle: dict[str, Any],
    sample_bundle: dict[str, Any],
    eval_labels: torch.Tensor,
    sample_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[str]]:
    records: dict[str, Any] = {}
    missing = []
    for key in PLANE_KEYS:
        eval_value = eval_bundle.get(key)
        sample_value = sample_bundle.get(key)
        if not torch.is_tensor(eval_value) or not torch.is_tensor(sample_value):
            missing.append(key)
            continue
        sim = _class_mean_similarity(_normalize(eval_value), eval_labels, _normalize(sample_value), sample_labels)
        diversity = _same_label_diversity(_normalize(sample_value), sample_labels)
        records[key] = {
            "eval_sample_class_mean_similarity": sim,
            "sample_same_label_diversity": diversity,
        }
    return records, missing


def _probe_label(probe: dict[str, Any]) -> str:
    label = str(probe.get("label") or "")
    if label:
        return label
    run = str(probe.get("test_eval_run") or probe.get("same_batch_eval_run") or "")
    for prefix in PROBE_PREFIXES:
        if prefix in run:
            return prefix
    return Path(run).name


def _eligible_probe(probe: dict[str, Any]) -> bool:
    label = _probe_label(probe)
    return any(label.startswith(prefix) for prefix in PROBE_PREFIXES) and bool(probe.get("test_eval_run")) and bool(
        probe.get("sample_run")
    )


def _status_priority(status: str) -> int:
    return {
        "harness-invalid": 0,
        "sample-plane-collapse": 1,
        "label-template-reference-mismatch": 2,
        "code-insensitive-generator-gap": 3,
        "camera-after-sample-bridge-gap": 4,
    }.get(status, 5)


def analyze_probe(probe: dict[str, Any]) -> dict[str, Any]:
    label = _probe_label(probe)
    eval_run = Path(str(probe["test_eval_run"]))
    sample_run = Path(str(probe["sample_run"]))
    eval_bundle = _load_bundle(eval_run, "eval")
    sample_bundle = _load_bundle(sample_run, "sample")
    eval_labels = _labels(eval_bundle)
    sample_labels = _labels(sample_bundle)
    eval_ref_key, eval_ref = _reference(eval_bundle, "eval")
    sample_ref_key, sample_ref = _reference(sample_bundle, "sample")
    plane_metrics, missing_planes = _plane_metrics(eval_bundle, sample_bundle, eval_labels, sample_labels)

    physical_eval = eval_bundle.get(PHYSICAL_KEY)
    physical_sample = sample_bundle.get(PHYSICAL_KEY)
    target_sample = sample_bundle.get(TARGET_CODE_KEY)
    if not torch.is_tensor(physical_eval) or not torch.is_tensor(physical_sample) or not torch.is_tensor(target_sample):
        return {
            "label": label,
            "status": "harness-invalid",
            "reason": "missing required physical/target-code planes",
            "eval_run": eval_run.as_posix(),
            "sample_run": sample_run.as_posix(),
            "missing_planes": missing_planes,
        }

    physical_eval_norm = _normalize(physical_eval)
    physical_sample_norm = _normalize(physical_sample)
    target_sample_norm = _normalize(target_sample)
    eval_ref_norm = _normalize(eval_ref)
    sample_ref_norm = _normalize(sample_ref)

    generated_eval_sample = _class_mean_similarity(physical_eval_norm, eval_labels, physical_sample_norm, sample_labels)
    eval_ref_sample_ref = _class_mean_similarity(eval_ref_norm, eval_labels, sample_ref_norm, sample_labels)
    sample_generated_sample_ref = _class_mean_similarity(
        physical_sample_norm,
        sample_labels,
        sample_ref_norm,
        sample_labels,
    )
    sample_generated_eval_ref = _class_mean_similarity(physical_sample_norm, sample_labels, eval_ref_norm, eval_labels)
    target_sample_ref = _class_mean_similarity(target_sample_norm, sample_labels, sample_ref_norm, sample_labels)
    physical_sample_ref = _class_mean_similarity(physical_sample_norm, sample_labels, sample_ref_norm, sample_labels)
    code_diversity = _code_diversity(sample_bundle, sample_labels)
    physical_diversity = _same_label_diversity(physical_sample_norm, sample_labels)

    generated_matches_eval = generated_eval_sample["corr"] >= 0.95 or generated_eval_sample["mse"] <= 1.0e-4
    generated_fails_sample_ref = sample_generated_sample_ref["corr"] < 0.75
    refs_mismatch = eval_ref_sample_ref["corr"] < 0.85 and sample_generated_eval_ref["corr"] >= sample_generated_sample_ref["corr"]
    code_insensitive = (
        bool(code_diversity["available"])
        and float(code_diversity["same_label_mean_std"] or 0.0) > 1.0e-6
        and physical_diversity["mean_mse_to_class_mean"] is not None
        and float(physical_diversity["mean_mse_to_class_mean"]) <= 1.0e-8
    )
    sample_target_collapse = target_sample_ref["corr"] < 0.50
    camera_after_bridge = target_sample_ref["corr"] >= 0.75 and physical_sample_ref["corr"] < 0.60

    if generated_matches_eval and generated_fails_sample_ref and refs_mismatch:
        status = "label-template-reference-mismatch"
        reason = "sample generated planes match eval templates but fail against sample references"
    elif code_insensitive:
        status = "code-insensitive-generator-gap"
        reason = "different sampled codes produce near-identical same-label outputs"
    elif sample_target_collapse:
        status = "sample-plane-collapse"
        reason = "sample target_code_amplitude is not readable against sample references"
    elif camera_after_bridge:
        status = "camera-after-sample-bridge-gap"
        reason = "sample bridge plane is readable but physical CMOS is not"
    else:
        status = "harness-invalid"
        reason = "diagnostic predicates were inconclusive; inspect tensor bundle contract"

    return {
        "label": label,
        "status": status,
        "reason": reason,
        "eval_run": eval_run.as_posix(),
        "sample_run": sample_run.as_posix(),
        "eval_reference_key": eval_ref_key,
        "sample_reference_key": sample_ref_key,
        "missing_planes": missing_planes,
        "checks": {
            "generated_matches_eval": generated_matches_eval,
            "generated_fails_sample_reference": generated_fails_sample_ref,
            "eval_sample_reference_mismatch": refs_mismatch,
            "code_insensitive": code_insensitive,
            "sample_target_code_collapse": sample_target_collapse,
            "camera_after_sample_bridge_gap": camera_after_bridge,
        },
        "metrics": {
            "generated_eval_vs_sample": generated_eval_sample,
            "eval_reference_vs_sample_reference": eval_ref_sample_ref,
            "sample_generated_vs_sample_reference": sample_generated_sample_ref,
            "sample_generated_vs_eval_reference": sample_generated_eval_ref,
            "sample_target_code_vs_sample_reference": target_sample_ref,
            "sample_physical_cmos_vs_sample_reference": physical_sample_ref,
            "sample_code_diversity": code_diversity,
            "sample_physical_same_label_diversity": physical_diversity,
            "planes": plane_metrics,
        },
    }


def _decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        status = "harness-invalid"
    else:
        status = sorted((record["status"] for record in records), key=_status_priority)[0]
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["label"])
    return {
        "status": status,
        "status_probe_labels": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def run_v32_1_sample_gap_isolation(
    *,
    summary: Path,
    bridge_report: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    summary_payload = _load_json(summary)
    bridge_payload = _load_json(bridge_report)
    assert summary_payload is not None
    assert bridge_payload is not None
    probes = [probe for probe in summary_payload.get("probes", []) if isinstance(probe, dict) and _eligible_probe(probe)]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for probe in probes:
        try:
            records.append(analyze_probe(probe))
        except Exception as exc:
            errors.append({"label": _probe_label(probe), "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": _probe_label(probe),
                    "status": "harness-invalid",
                    "reason": "failed to analyze probe tensor bundles",
                    "error": f"{type(exc).__name__}: {exc}",
                    "eval_run": str(probe.get("test_eval_run")),
                    "sample_run": str(probe.get("sample_run")),
                }
            )
    payload = {
        "version": "v32.1",
        "objective": "sample_gap_isolation_no_retrain",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": summary.as_posix(),
        "bridge_report": bridge_report.as_posix(),
        "bridge_report_status": bridge_payload.get("decision", {}).get("status"),
        "eligible_probe_labels": [_probe_label(probe) for probe in probes],
        "runs": records,
        "errors": errors,
        "status": _decision(records)["status"],
        "decision": _decision(records),
        "human_readability_required": True,
        "notes": [
            "v32.1 is diagnostic-only and does not replace true final CMOS v4 preview human review.",
            "No accepted summary is written by this tool.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="v32 bridge redesign launcher summary JSON.")
    parser.add_argument("--bridge-report", type=Path, required=True, help="v32 bridge topology report JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_v32_1_sample_gap_isolation(
        summary=args.summary,
        bridge_report=args.bridge_report,
        output=args.output,
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
