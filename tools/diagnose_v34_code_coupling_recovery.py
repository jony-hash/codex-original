#!/usr/bin/env python3
"""v34 sample-time code coupling recovery diagnostic.

Reads no-retrain sample tensor bundles from a code residual scale sweep and
locates the first plane where same-label code variation collapses. It never
trains, promotes checkpoints, or writes accepted summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch

from tools.diagnose_v21_physical_cmos_readout import _load_json, _require_report_output


CHAIN_KEYS = (
    "generator_desired_intensity",
    "slm_command",
    "encoded_amplitude",
    "ols_output_amplitude",
    "target_code_amplitude",
    "decoder_output_amplitude",
    "physical_camera_plane",
)
COMMAND_PLANE = "slm_command"
ENCODED_PLANE = "encoded_amplitude"
OLS_PLANE = "ols_output_amplitude"
TARGET_PLANE = "target_code_amplitude"
DECODER_PLANE = "decoder_output_amplitude"
PHYSICAL_PLANE = "physical_camera_plane"

CODE_STD_THRESHOLD = 1.0e-5
RESIDUAL_MSE_THRESHOLD = 1.0e-5
RESIDUAL_STD_THRESHOLD = 1.0e-3
PLANE_MSE_THRESHOLD = 1.0e-6
PLANE_STD_THRESHOLD = 1.0e-4


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v34_code_coupling_recovery_{timestamp}.json"


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


def _load_sample_bundle(run: Path) -> dict[str, Any]:
    path = run / "artifacts" / "one_shot_sample_tensors.pt"
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


def _as_samples(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    if value.ndim == 1:
        value = value.unsqueeze(1)
    elif value.ndim == 2:
        pass
    elif value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4:
        pass
    else:
        raise ValueError(f"expected sample tensor with 1-4 dims, got {tuple(value.shape)}")
    return value.reshape(value.shape[0], -1)


def _same_label_diversity(tensor: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    samples = _as_samples(tensor)
    labels = labels[: samples.shape[0]]
    entries = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        if int(mask.sum().item()) < 2:
            continue
        group = samples[mask]
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


def _active(diversity: dict[str, Any], *, mse_threshold: float, std_threshold: float) -> bool:
    mse = diversity.get("mean_mse_to_class_mean")
    std = diversity.get("mean_pixel_std")
    return (
        mse is not None
        and std is not None
        and float(mse) >= mse_threshold
        and float(std) >= std_threshold
    )


def _tensor_summary(bundle: dict[str, Any], labels: torch.Tensor, key: str) -> dict[str, Any]:
    value = bundle.get(key)
    if not torch.is_tensor(value):
        return {"available": False, "active": False, "same_label_diversity": None}
    diversity = _same_label_diversity(value, labels)
    threshold = (
        {"mse_threshold": RESIDUAL_MSE_THRESHOLD, "std_threshold": RESIDUAL_STD_THRESHOLD}
        if key == "generator_residual_map"
        else {"mse_threshold": PLANE_MSE_THRESHOLD, "std_threshold": PLANE_STD_THRESHOLD}
    )
    samples = _as_samples(value)
    return {
        "available": True,
        "shape": list(value.shape),
        "global_std": float(samples.std(unbiased=False).item()),
        "same_label_diversity": diversity,
        "thresholds": threshold,
        "active": _active(diversity, mse_threshold=threshold["mse_threshold"], std_threshold=threshold["std_threshold"]),
    }


def _code_summary(bundle: dict[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    codes = bundle.get("codes")
    if not torch.is_tensor(codes):
        return {"available": False, "active": False, "same_label_mean_std": None}
    samples = _as_samples(codes)
    labels = labels[: samples.shape[0]]
    stds = []
    for label in torch.unique(labels, sorted=True):
        group = samples[labels == label]
        if group.shape[0] >= 2:
            stds.append(float(group.std(dim=0, unbiased=False).mean().item()))
    mean_std = float(sum(stds) / len(stds)) if stds else 0.0
    return {
        "available": True,
        "shape": list(codes.shape),
        "same_label_mean_std": mean_std,
        "active": mean_std >= CODE_STD_THRESHOLD,
    }


def _preview_paths(run: Path) -> dict[str, Any]:
    image = run / "images" / "one_shot_sample_100x100_cell_chain_preview.png"
    artifact = run / "artifacts" / "one_shot_sample_100x100_cell_chain_preview.json"
    return {
        "v4_style_preview": image.as_posix(),
        "v4_style_preview_exists": image.is_file(),
        "v4_style_preview_artifact": artifact.as_posix(),
        "v4_style_preview_artifact_exists": artifact.is_file(),
        "final_human_gate_plane": "physical_camera_plane",
        "final_human_gate_label": "CMOS intensity",
    }


def _classification(chain: dict[str, Any]) -> tuple[str, str]:
    if not chain["codes"]["active"] or not chain["generator_residual_map"]["active"]:
        return "harness-invalid", "codes or generator residual variation is inactive"
    if not chain[COMMAND_PLANE]["active"]:
        return "residual-to-command-bottleneck", "residual variation does not reach SLM command"
    if not chain[ENCODED_PLANE]["active"] or not chain[OLS_PLANE]["active"]:
        return "command-to-ols-bottleneck", "SLM command variation does not survive encoder/OLS"
    if not chain[TARGET_PLANE]["active"]:
        return "ols-to-target-code-bottleneck", "OLS variation does not reach target code amplitude"
    if not chain[DECODER_PLANE]["active"]:
        return "decoder-after-code-bottleneck", "target/code variation does not survive decoder"
    if not chain[PHYSICAL_PLANE]["active"]:
        return "camera-after-code-bottleneck", "decoder variation does not survive true CMOS plane"
    return "code-coupling-recovered-needs-human-review", "sample-time code variation reaches true CMOS preview"


def analyze_sample_run(run: Path, *, scale: float, label: str | None = None) -> dict[str, Any]:
    bundle = _load_sample_bundle(run)
    labels = _labels(bundle)
    chain: dict[str, Any] = {
        "codes": _code_summary(bundle, labels),
        "generator_residual_map": _tensor_summary(bundle, labels, "generator_residual_map"),
    }
    missing = []
    for key in CHAIN_KEYS:
        record = _tensor_summary(bundle, labels, key)
        chain[key] = record
        if not record["available"]:
            missing.append(key)
    if missing:
        status = "harness-invalid"
        reason = f"missing chain tensors: {', '.join(missing)}"
    else:
        status, reason = _classification(chain)
    return {
        "label": label or run.name,
        "scale": float(scale),
        "run": run.as_posix(),
        "status": status,
        "classification": status,
        "reason": reason,
        "chain": chain,
        "missing_planes": missing,
        "preview_package": _preview_paths(run),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _decision(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {}
    for record in records:
        by_status.setdefault(record["status"], []).append(record["run"])
    if errors or by_status.get("harness-invalid"):
        status = "harness-invalid"
    elif by_status.get("code-coupling-recovered-needs-human-review"):
        status = "code-coupling-recovered-needs-human-review"
    elif by_status.get("camera-after-code-bottleneck"):
        status = "camera-after-code-bottleneck"
    elif by_status.get("decoder-after-code-bottleneck"):
        status = "decoder-after-code-bottleneck"
    elif by_status.get("ols-to-target-code-bottleneck"):
        status = "ols-to-target-code-bottleneck"
    elif by_status.get("command-to-ols-bottleneck"):
        status = "command-to-ols-bottleneck"
    else:
        status = "residual-to-command-bottleneck"
    return {
        "status": status,
        "status_runs": by_status,
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": (
            "human_v4_cmos_preview_review"
            if status == "code-coupling-recovered-needs-human-review"
            else "diagnostic_followup_before_human_acceptance"
        ),
    }


def run_v34_code_coupling_recovery(
    *,
    summary: Path | None = None,
    sample_runs: list[Path],
    scales: list[float],
    labels: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    if len(sample_runs) != len(scales):
        raise ValueError("sample_runs and scales must have the same length")
    if labels is not None and len(labels) != len(sample_runs):
        raise ValueError("labels must have the same length as sample_runs")
    output_path = output or _default_output()
    _require_report_output(output_path)
    source_summary = None
    source_status = None
    if summary is not None:
        payload = _load_json(summary)
        assert payload is not None
        source_summary = summary.as_posix()
        source_status = payload.get("status")
    records = []
    errors = []
    for index, (run, scale) in enumerate(zip(sample_runs, scales, strict=True)):
        label = labels[index] if labels is not None else None
        try:
            records.append(analyze_sample_run(run, scale=scale, label=label))
        except Exception as exc:
            errors.append({"run": run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            records.append(
                {
                    "label": label or run.name,
                    "scale": float(scale),
                    "run": run.as_posix(),
                    "status": "harness-invalid",
                    "classification": "harness-invalid",
                    "reason": "sample analysis failed",
                    "accepted_summary_written": False,
                    "human_readability_required": True,
                }
            )
    decision = _decision(records, errors)
    result = {
        "version": "v34",
        "objective": "sample_time_code_coupling_recovery",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": source_summary,
        "source_status": source_status,
        "thresholds": {
            "code_same_label_std": CODE_STD_THRESHOLD,
            "residual_same_label_mse": RESIDUAL_MSE_THRESHOLD,
            "residual_same_label_std": RESIDUAL_STD_THRESHOLD,
            "plane_same_label_mse": PLANE_MSE_THRESHOLD,
            "plane_same_label_std": PLANE_STD_THRESHOLD,
        },
        "runs": records,
        "errors": errors,
        "decision": decision,
        "status": decision["status"],
        "human_readability_required": True,
        "human_gate_note": "Only true final CMOS intensity / physical_camera_plane v4-style previews can satisfy human review.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--sample-run", action="append", type=Path, required=True)
    parser.add_argument("--scale", action="append", type=float, required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v34_code_coupling_recovery(
        summary=args.summary,
        sample_runs=args.sample_run,
        scales=args.scale,
        labels=args.label,
        output=args.output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
