#!/usr/bin/env python3
"""v33.2 real-OLS candidate preview/sample-gap diagnostic.

This diagnostic reads existing v33.1 eval/sample artifacts plus a no-retrain
real-OLS sample run. It never trains, uses exact random-sample metrics as an
acceptance gate, or writes accepted summaries.
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
BRIDGE_PLANES = ("target_code_amplitude", "decoder_input_amplitude")
DECODER_PLANES = ("decoder_output_amplitude", "chip_intensity_primary")
PHYSICAL_PLANE = "physical_camera_plane"
DIVERSITY_MSE_THRESHOLD = 1.0e-8
DIVERSITY_STD_THRESHOLD = 1.0e-5


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v33_2_real_ols_candidate_preview_{timestamp}.json"


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


def _diversity_pass(diversity: dict[str, Any]) -> bool:
    mse = diversity.get("mean_mse_to_class_mean")
    std = diversity.get("mean_pixel_std")
    return (
        mse is not None
        and std is not None
        and float(mse) > DIVERSITY_MSE_THRESHOLD
        and float(std) > DIVERSITY_STD_THRESHOLD
    )


def _code_diversity(bundle: dict[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    codes = bundle.get("codes")
    if not torch.is_tensor(codes):
        return {"available": False, "same_label_mean_std": None, "active": False}
    value = codes.detach().float().cpu().reshape(codes.shape[0], -1)
    labels = labels[: value.shape[0]]
    stds = []
    for label in torch.unique(labels, sorted=True):
        group = value[labels == label]
        if group.shape[0] >= 2:
            stds.append(float(group.std(dim=0, unbiased=False).mean().item()))
    mean_std = float(sum(stds) / len(stds)) if stds else 0.0
    return {"available": True, "same_label_mean_std": mean_std, "active": mean_std > 1.0e-5}


def _residual_summary(bundle: dict[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    residual = bundle.get("generator_residual_map")
    if not torch.is_tensor(residual):
        return {"available": False, "abs_mean": None, "abs_max": None, "same_label_diversity": None, "active": False}
    value = _as_nchw(residual)
    diversity = _same_label_diversity(value, labels)
    abs_mean = float(value.abs().mean().item())
    return {
        "available": True,
        "abs_mean": abs_mean,
        "abs_max": float(value.abs().max().item()),
        "same_label_diversity": diversity,
        "active": abs_mean > 1.0e-4 and _diversity_pass(diversity),
    }


def _reference(bundle: dict[str, Any]) -> tuple[str | None, torch.Tensor | None]:
    value = bundle.get("reference_images")
    if torch.is_tensor(value):
        return "reference_images", _as_nchw(value)
    return None, None


def _plane_records(bundle: dict[str, Any], labels: torch.Tensor) -> tuple[dict[str, Any], list[str]]:
    records: dict[str, Any] = {}
    missing = []
    reference_key, reference = _reference(bundle)
    for key in PLANE_KEYS:
        value = bundle.get(key)
        if not torch.is_tensor(value):
            missing.append(key)
            continue
        normalized = _normalize(value)
        record = {"same_label_diversity": _same_label_diversity(normalized, labels)}
        if reference is not None:
            ref = _normalize(reference)
            record["reference_corr"] = _corr(normalized, ref)
            record["reference_mse"] = _mse(normalized, ref)
            record["reference_key"] = reference_key
        records[key] = record
    return records, missing


def _sample_reference_metric_invalid(bundle: dict[str, Any]) -> bool:
    source = bundle.get("reference_source")
    source_text = str(source) if source is not None else None
    return torch.is_tensor(bundle.get("codes")) and source_text != "paired_distill_code_reference"


def _preview_paths(run: Path) -> dict[str, str]:
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


def _extract_probe(summary: dict[str, Any], label_contains: str) -> dict[str, Any] | None:
    for probe in summary.get("probes", []):
        if isinstance(probe, dict) and label_contains in str(probe.get("label", "")):
            return probe
    return None


def _classify_candidate(sample_run: Path, bundle: dict[str, Any], labels: torch.Tensor) -> tuple[str, str, dict[str, Any]]:
    planes, missing = _plane_records(bundle, labels)
    previews = _preview_paths(sample_run)
    residual = _residual_summary(bundle, labels)
    codes = _code_diversity(bundle, labels)
    checks = {
        "sample_reference_metric_invalid": _sample_reference_metric_invalid(bundle),
        "residual_active": residual["active"],
        "code_diversity_active": codes["active"],
        "v4_style_preview_exists": previews["v4_style_preview_exists"],
        "missing_planes": missing,
    }
    if missing or not previews["v4_style_preview_exists"]:
        return "harness-invalid", "missing required sample planes or v4-style preview", checks
    if not residual["active"] or not codes["active"]:
        return "sample-code-distribution-gap", "sample code/residual variation is inactive", checks

    bridge_ok = all(_diversity_pass(planes[key]["same_label_diversity"]) for key in BRIDGE_PLANES)
    decoder_ok = all(_diversity_pass(planes[key]["same_label_diversity"]) for key in DECODER_PLANES)
    physical_ok = _diversity_pass(planes[PHYSICAL_PLANE]["same_label_diversity"])
    checks.update(
        {
            "bridge_sample_diversity_ok": bridge_ok,
            "decoder_sample_diversity_ok": decoder_ok,
            "physical_cmos_sample_diversity_ok": physical_ok,
        }
    )
    if not bridge_ok:
        return "sample-code-distribution-gap", "target_code/decoder_input sample planes collapsed", checks
    if not decoder_ok:
        return "bridge-after-code-failure", "bridge/code planes vary but decoder/chip sample planes collapsed", checks
    if not physical_ok:
        return "camera-after-code-failure", "decoder/chip sample planes vary but true CMOS sample plane collapsed", checks
    return "manual-preview-required", "real OLS sample preview must be judged by human CMOS readability", checks


def analyze_candidate(*, sample_run: Path) -> dict[str, Any]:
    bundle = _load_bundle(sample_run, "sample")
    labels = _labels(bundle)
    planes, missing = _plane_records(bundle, labels)
    residual = _residual_summary(bundle, labels)
    codes = _code_diversity(bundle, labels)
    status, reason, checks = _classify_candidate(sample_run, bundle, labels)
    return {
        "run": sample_run.as_posix(),
        "run_type": "sample",
        "role": "real_ols_candidate",
        "status": status,
        "classification": status,
        "reason": reason,
        "checks": checks,
        "metrics": {
            "codes": codes,
            "residual": residual,
            "planes": planes,
        },
        "missing_planes": missing,
        "preview_package": _preview_paths(sample_run),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _control_record(sample_run: Path | None) -> dict[str, Any] | None:
    if sample_run is None:
        return None
    if not sample_run.is_dir():
        return {
            "run": sample_run.as_posix(),
            "role": "clean_teacher_control",
            "status": "harness-invalid",
            "reason": "control sample run missing",
        }
    return {
        "run": sample_run.as_posix(),
        "role": "clean_teacher_control",
        "status": "control-sample-available",
        "preview_package": _preview_paths(sample_run),
    }


def _decision(candidate: dict[str, Any], errors: list[dict[str, str]]) -> dict[str, Any]:
    status = "harness-invalid" if errors else candidate["status"]
    return {
        "status": status,
        "candidate_run": candidate.get("run"),
        "accepted_summary_written": False,
        "human_readability_required": True,
        "next_gate": (
            "human_v4_cmos_preview_review"
            if status == "manual-preview-required"
            else "diagnostic_followup_before_human_acceptance"
        ),
    }


def run_v33_2_real_ols_candidate_preview(
    *,
    summary: Path,
    candidate_sample_run: Path,
    control_sample_run: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    summary_payload = _load_json(summary)
    assert summary_payload is not None
    errors: list[dict[str, str]] = []
    candidate: dict[str, Any]
    try:
        candidate = analyze_candidate(sample_run=candidate_sample_run)
    except Exception as exc:
        errors.append({"run": candidate_sample_run.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
        candidate = {
            "run": candidate_sample_run.as_posix(),
            "role": "real_ols_candidate",
            "status": "harness-invalid",
            "classification": "harness-invalid",
            "reason": "candidate sample analysis failed",
            "accepted_summary_written": False,
            "human_readability_required": True,
        }
    d_probe = _extract_probe(summary_payload, "probe_d_mnist_sample_ols_bridge")
    c_probe = _extract_probe(summary_payload, "probe_c_clean_teacher")
    payload = {
        "version": "v33.2",
        "objective": "real_ols_candidate_preview_sample_gap_gate",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "source_summary": summary.as_posix(),
        "source_v33_1_status": summary_payload.get("status"),
        "real_ols_probe": d_probe,
        "clean_teacher_probe": c_probe,
        "candidate": candidate,
        "control": _control_record(control_sample_run),
        "errors": errors,
        "decision": _decision(candidate, errors),
        "status": _decision(candidate, errors)["status"],
        "human_readability_required": True,
        "human_gate_note": (
            "Only the v4-style true final CMOS intensity / physical_camera_plane preview can satisfy human review; "
            "target_code, decoder_input, decoder_output, and exact random-sample metrics remain diagnostic-only."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-sample-run", required=True, type=Path)
    parser.add_argument("--control-sample-run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or _default_output()
    result = run_v33_2_real_ols_candidate_preview(
        summary=args.summary,
        candidate_sample_run=args.candidate_sample_run,
        control_sample_run=args.control_sample_run,
        output=output,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
