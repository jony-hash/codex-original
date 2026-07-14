#!/usr/bin/env python3
"""Offline v26 OLS decodability software-oracle diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any


import torch
import torch.nn.functional as F

from tools.diagnose_v21_physical_cmos_readout import (
    FOCUS_CLASSES,
    _load_json,
    _metric_record,
    _prototype_label_audit,
    _require_report_output,
    _resize_like,
    _torch_load,
)


PLANE_ORDER = (
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
AMPLITUDE_FEATURE_KEYS = (
    "encoded_amplitude",
    "ols_output_amplitude",
    "decoder_output_amplitude",
    "chip_intensity_primary",
    "physical_camera_plane",
)
PHASE_FEATURE_KEYS = (
    "encoded_phase",
    "ols_output_phase",
    "decoder_output_phase",
)
REFERENCE_KEYS = (
    "reference_images",
    "eval_target",
    "teacher_camera",
    "original_digits",
    "target_images",
    "targets",
)
LABEL_KEYS = ("class_labels", "labels", "sample_labels")
DECODE_ACC_PASS = 0.70
DECODE_ACC_FAIL = 0.50
FOCUS_ACC_PASS = 0.60
PROXY_AUDIT_HINT = 0.60


def _default_output() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v26_ols_decodability_oracle_{timestamp}.json"


def _default_preview_path(output: Path) -> Path:
    return output.with_suffix(".preview.png")


def _tensor_bundle_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_tensors.pt" if kind == "eval" else "one_shot_sample_tensors.pt"
    return run / "artifacts" / filename


def _metrics_path(run: Path, kind: str) -> Path:
    filename = "one_shot_distill_eval_metrics.json" if kind == "eval" else "one_shot_sample_metrics.json"
    return run / "artifacts" / filename


def _v4_preview_path(run: Path, kind: str) -> Path:
    filename = (
        "one_shot_distill_eval_100x100_cell_chain_preview.png"
        if kind == "eval"
        else "one_shot_sample_100x100_cell_chain_preview.png"
    )
    return run / "images" / filename


def _first_tensor(bundle: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, torch.Tensor | None]:
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return key, value
    return None, None


def _load_bundle(run: Path, kind: str) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, str, str]:
    path = _tensor_bundle_path(run, kind)
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = _torch_load(path)
    label_key, labels = _first_tensor(bundle, LABEL_KEYS)
    reference_key, reference = _first_tensor(bundle, REFERENCE_KEYS)
    if labels is None or reference is None:
        missing = []
        if labels is None:
            missing.append("class labels")
        if reference is None:
            missing.append("reference/target images")
        raise ValueError(f"{kind} tensor bundle is missing required {' and '.join(missing)}: {path}")
    return bundle, labels.detach().cpu().long(), reference.detach().float().cpu(), str(label_key), str(reference_key)


def _plane_record(key: str, tensor: torch.Tensor, reference: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    plane = _resize_like(tensor, reference)
    metrics = _metric_record(plane, reference, labels)
    audit = _prototype_label_audit(plane, reference, labels)
    return {
        "plane": key,
        "shape": list(tensor.shape),
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "focus_classes": metrics["focus_classes"],
        "prototype_label_audit_accuracy": audit.get("correct_ratio"),
        "prototype_label_confusion": audit.get("confusion"),
        "prototype_label_audit": audit,
    }


def _extract_run_paths(payload: Any, eval_runs: list[Path], sample_runs: list[Path]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "eval_run" and value:
                eval_runs.append(Path(str(value)))
            elif key == "sample_run" and value:
                sample_runs.append(Path(str(value)))
            else:
                _extract_run_paths(value, eval_runs, sample_runs)
    elif isinstance(payload, list):
        for item in payload:
            _extract_run_paths(item, eval_runs, sample_runs)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _runs_from_summaries(summary_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    eval_runs: list[Path] = []
    sample_runs: list[Path] = []
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        assert summary is not None
        _extract_run_paths(summary, eval_runs, sample_runs)
    return _dedupe(eval_runs), _dedupe(sample_runs)


def _resize_feature(tensor: torch.Tensor, *, size: int) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    if value.ndim != 4:
        raise ValueError(f"feature tensor must have shape [N,C,H,W], got {tuple(value.shape)}")
    if value.shape[-2:] != (size, size):
        value = F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
    return value.flatten(1)


def _feature_source(
    bundle: dict[str, Any],
    keys: tuple[str, ...],
    *,
    feature_size: int,
) -> tuple[torch.Tensor | None, list[str]]:
    parts: list[torch.Tensor] = []
    used: list[str] = []
    for key in keys:
        value = bundle.get(key)
        if not torch.is_tensor(value):
            continue
        parts.append(_resize_feature(value, size=feature_size))
        used.append(key)
    if not parts:
        return None, []
    return torch.cat(parts, dim=1), used


def _feature_sources(bundle: dict[str, Any], *, feature_size: int) -> dict[str, tuple[torch.Tensor, list[str]]]:
    sources: dict[str, tuple[torch.Tensor, list[str]]] = {}
    for key in AMPLITUDE_FEATURE_KEYS:
        value = bundle.get(key)
        if torch.is_tensor(value):
            features, used = _feature_source(bundle, (key,), feature_size=feature_size)
            assert features is not None
            sources[key] = (features, used)
    ols_amp_phase_keys = tuple(key for key in ("ols_output_amplitude", "ols_output_phase") if torch.is_tensor(bundle.get(key)))
    if ols_amp_phase_keys:
        features, used = _feature_source(bundle, ols_amp_phase_keys, feature_size=feature_size)
        assert features is not None
        sources["ols_output_amp_phase"] = (features, used)
    encoded_ols_keys = tuple(
        key
        for key in ("encoded_amplitude", "encoded_phase", "ols_output_amplitude", "ols_output_phase")
        if torch.is_tensor(bundle.get(key))
    )
    if encoded_ols_keys:
        features, used = _feature_source(bundle, encoded_ols_keys, feature_size=feature_size)
        assert features is not None
        sources["encoded_ols_combined"] = (features, used)
    phase_keys = tuple(key for key in PHASE_FEATURE_KEYS if torch.is_tensor(bundle.get(key)))
    if phase_keys:
        parts: list[torch.Tensor] = []
        used: list[str] = []
        for key in phase_keys:
            value = bundle[key].detach().float().cpu()
            parts.extend([
                _resize_feature(torch.sin(value), size=feature_size),
                _resize_feature(torch.cos(value), size=feature_size),
            ])
            used.append(key)
        sources["phase_sincos"] = (torch.cat(parts, dim=1), used)
    return sources


def _standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _pca_project(train_x: torch.Tensor, test_x: torch.Tensor, *, pca_dim: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    train_centered = train_x - train_x.mean(dim=0, keepdim=True)
    test_centered = test_x - train_x.mean(dim=0, keepdim=True)
    max_dim = min(int(pca_dim), train_x.shape[0] - 1, train_x.shape[1])
    if max_dim < 1:
        raise ValueError("at least two training examples are required for PCA")
    _u, _s, vh = torch.linalg.svd(train_centered, full_matrices=False)
    components = vh[:max_dim].t()
    return train_centered @ components, test_centered @ components, max_dim


def _ridge_classify(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    *,
    class_labels: list[int],
    ridge: float,
) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(class_labels)}
    y = torch.zeros((train_y.numel(), len(class_labels)), dtype=train_x.dtype)
    for row, label in enumerate(train_y.tolist()):
        y[row, label_to_index[int(label)]] = 1.0
    design = torch.cat([train_x, torch.ones((train_x.shape[0], 1), dtype=train_x.dtype)], dim=1)
    test_design = torch.cat([test_x, torch.ones((test_x.shape[0], 1), dtype=test_x.dtype)], dim=1)
    identity = torch.eye(design.shape[1], dtype=design.dtype)
    identity[-1, -1] = 0.0
    weights = torch.linalg.solve(design.t() @ design + ridge * identity, design.t() @ y)
    predicted_indices = (test_design @ weights).argmax(dim=1)
    return torch.tensor([class_labels[int(index)] for index in predicted_indices.tolist()], dtype=torch.long)


def _accuracy_record(predicted: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    correct = predicted == labels
    confusion: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted({int(item) for item in labels.tolist()}):
        mask = labels == label
        counts: dict[str, int] = {}
        for item in predicted[mask].tolist():
            counts[str(int(item))] = counts.get(str(int(item)), 0) + 1
        confusion[str(label)] = counts
        per_class[str(label)] = {
            "count": int(mask.sum().item()),
            "accuracy": float(correct[mask].float().mean().item()) if bool(mask.any()) else 0.0,
            "predicted_label_counts": counts,
        }
    focus = {
        str(label): per_class.get(str(label), {"count": 0, "accuracy": None})
        for label in FOCUS_CLASSES
    }
    return {
        "accuracy": float(correct.float().mean().item()) if correct.numel() else 0.0,
        "confusion": confusion,
        "per_class": per_class,
        "focus_classes": focus,
        "focus_failures": [
            {
                "class": label,
                "accuracy": focus[str(label)]["accuracy"],
                "threshold": FOCUS_ACC_PASS,
            }
            for label in FOCUS_CLASSES
            if focus[str(label)].get("accuracy") is None or float(focus[str(label)]["accuracy"]) < FOCUS_ACC_PASS
        ],
    }


def _run_linear_probe(
    rows: list[dict[str, Any]],
    *,
    feature_name: str,
    pca_dim: int,
    ridge: float,
) -> dict[str, Any]:
    run_ids = sorted({str(row["run"]) for row in rows})
    if len(run_ids) < 2:
        return {
            "feature_name": feature_name,
            "valid": False,
            "status": "insufficient-oracle-data",
            "reason": "leave-run-out requires at least two independent runs",
            "run_count": len(run_ids),
        }
    labels_all = torch.cat([row["labels"] for row in rows])
    class_labels = sorted({int(item) for item in labels_all.tolist()})
    predictions: list[torch.Tensor] = []
    truth: list[torch.Tensor] = []
    folds: list[dict[str, Any]] = []
    for held_run in run_ids:
        train_rows = [row for row in rows if str(row["run"]) != held_run]
        test_rows = [row for row in rows if str(row["run"]) == held_run]
        train_x = torch.cat([row["features"] for row in train_rows], dim=0)
        train_y = torch.cat([row["labels"] for row in train_rows], dim=0)
        test_x = torch.cat([row["features"] for row in test_rows], dim=0)
        test_y = torch.cat([row["labels"] for row in test_rows], dim=0)
        if len({int(item) for item in train_y.tolist()}) < 2:
            return {
                "feature_name": feature_name,
                "valid": False,
                "status": "insufficient-oracle-data",
                "reason": "training folds require at least two labels",
                "run_count": len(run_ids),
            }
        train_z, test_z = _standardize(train_x, test_x)
        train_pca, test_pca, used_dim = _pca_project(train_z, test_z, pca_dim=pca_dim)
        pred = _ridge_classify(train_pca, train_y, test_pca, class_labels=class_labels, ridge=ridge)
        fold_record = _accuracy_record(pred, test_y)
        folds.append(
            {
                "held_out_run": held_run,
                "train_count": int(train_y.numel()),
                "test_count": int(test_y.numel()),
                "pca_dim": used_dim,
                "accuracy": fold_record["accuracy"],
                "focus_failures": fold_record["focus_failures"],
            }
        )
        predictions.append(pred)
        truth.append(test_y)
    overall = _accuracy_record(torch.cat(predictions), torch.cat(truth))
    return {
        "feature_name": feature_name,
        "valid": True,
        "status": "oracle-evaluated",
        "run_count": len(run_ids),
        "sample_count": int(torch.cat(truth).numel()),
        "folds": folds,
        **overall,
    }


def _run_reconstruction_oracle(
    rows: list[dict[str, Any]],
    *,
    feature_name: str,
    pca_dim: int,
    ridge: float,
) -> dict[str, Any]:
    run_ids = sorted({str(row["run"]) for row in rows})
    if len(run_ids) < 2:
        return {
            "feature_name": feature_name,
            "valid": False,
            "status": "insufficient-oracle-data",
            "reason": "leave-run-out requires at least two independent runs",
        }
    predictions: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for held_run in run_ids:
        train_rows = [row for row in rows if str(row["run"]) != held_run]
        test_rows = [row for row in rows if str(row["run"]) == held_run]
        train_x = torch.cat([row["features"] for row in train_rows], dim=0)
        test_x = torch.cat([row["features"] for row in test_rows], dim=0)
        train_ref = torch.cat([row["reference_small"] for row in train_rows], dim=0).flatten(1)
        test_ref = torch.cat([row["reference_small"] for row in test_rows], dim=0)
        train_z, test_z = _standardize(train_x, test_x)
        train_pca, test_pca, _used_dim = _pca_project(train_z, test_z, pca_dim=pca_dim)
        design = torch.cat([train_pca, torch.ones((train_pca.shape[0], 1), dtype=train_pca.dtype)], dim=1)
        test_design = torch.cat([test_pca, torch.ones((test_pca.shape[0], 1), dtype=test_pca.dtype)], dim=1)
        identity = torch.eye(design.shape[1], dtype=design.dtype)
        identity[-1, -1] = 0.0
        weights = torch.linalg.solve(design.t() @ design + ridge * identity, design.t() @ train_ref)
        pred = (test_design @ weights).reshape_as(test_ref).clamp(0.0, 1.0)
        predictions.append(pred)
        references.append(test_ref)
        labels.append(torch.cat([row["labels"] for row in test_rows], dim=0))
    pred_all = torch.cat(predictions, dim=0)
    ref_all = torch.cat(references, dim=0)
    labels_all = torch.cat(labels, dim=0)
    metrics = _metric_record(pred_all, ref_all, labels_all)
    audit = _prototype_label_audit(pred_all, ref_all, labels_all)
    return {
        "feature_name": feature_name,
        "valid": True,
        "status": "diagnostic-upper-bound",
        "overall": metrics["overall"],
        "focus_classes": metrics["focus_classes"],
        "prototype_label_audit_accuracy": audit.get("correct_ratio"),
        "prototype_label_confusion": audit.get("confusion"),
        "preview_note": "software ridge reconstruction is diagnostic-only and cannot satisfy CMOS acceptance",
    }


def analyze_run(run: Path, *, kind: str, feature_size: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if kind not in {"eval", "sample"}:
        raise ValueError(f"kind must be 'eval' or 'sample', got {kind!r}")
    bundle, labels, reference, label_key, reference_key = _load_bundle(run, kind)
    plane_records = []
    missing_planes = []
    for key in PLANE_ORDER:
        value = bundle.get(key)
        if torch.is_tensor(value):
            plane_records.append(_plane_record(key, value.detach().float().cpu(), reference, labels))
        else:
            missing_planes.append(key)
    feature_sources = _feature_sources(bundle, feature_size=feature_size)
    feature_rows: dict[str, dict[str, Any]] = {}
    reference_small = F.interpolate(reference, size=(feature_size, feature_size), mode="bilinear", align_corners=False)
    for feature_name, (features, used_keys) in feature_sources.items():
        feature_rows[feature_name] = {
            "run": run.as_posix(),
            "run_type": kind,
            "features": features,
            "labels": labels,
            "reference_small": reference_small,
            "source_tensor_keys": used_keys,
        }
    record = {
        "run_type": kind,
        "run": run.as_posix(),
        "tensor_bundle": _tensor_bundle_path(run, kind).as_posix(),
        "metrics_path": _metrics_path(run, kind).as_posix(),
        "stored_metrics": _load_json(_metrics_path(run, kind), required=False),
        "v4_preview": _v4_preview_path(run, kind).as_posix(),
        "label_key": label_key,
        "reference_key": reference_key,
        "plane_order": list(PLANE_ORDER),
        "missing_planes": missing_planes,
        "available_feature_sources": {
            feature_name: row["source_tensor_keys"]
            for feature_name, row in feature_rows.items()
        },
        "planes": plane_records,
        "accepted_summary_written": False,
    }
    return record, feature_rows


def _best_valid_oracle(oracles: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    valid = [record for record in oracles.values() if record.get("valid")]
    if not valid:
        return None
    return max(valid, key=lambda record: float(record.get("accuracy") or 0.0))


def _max_plane_audit(run_records: list[dict[str, Any]], names: tuple[str, ...]) -> float | None:
    values = []
    for run in run_records:
        for plane in run.get("planes", []):
            if plane.get("plane") in names and plane.get("prototype_label_audit_accuracy") is not None:
                values.append(float(plane["prototype_label_audit_accuracy"]))
    return max(values) if values else None


def _decision(oracles: dict[str, dict[str, Any]], run_records: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_valid_oracle(oracles)
    if best is None:
        status = "insufficient-oracle-data"
    else:
        accuracy = float(best.get("accuracy") or 0.0)
        focus_failures = best.get("focus_failures") or []
        proxy_audit = _max_plane_audit(run_records, ("encoded_amplitude", "ols_output_amplitude"))
        if accuracy >= DECODE_ACC_PASS and not focus_failures:
            status = "ols-decodable"
        elif accuracy < DECODE_ACC_FAIL or focus_failures:
            status = "ols-not-decodable"
        elif proxy_audit is not None and proxy_audit >= PROXY_AUDIT_HINT and accuracy < DECODE_ACC_PASS:
            status = "metric-proxy-misleading"
        else:
            status = "metric-proxy-misleading"
    return {
        "status": status,
        "best_oracle_feature": None if best is None else best["feature_name"],
        "best_oracle_accuracy": None if best is None else best.get("accuracy"),
        "best_oracle_focus_failures": [] if best is None else best.get("focus_failures", []),
        "accepted_summary_written": False,
        "human_readability_required": True,
    }


def _write_preview(
    *,
    preview_path: Path,
    run_records: list[dict[str, Any]],
    reconstruction: dict[str, Any] | None,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rows = []
    for run in run_records[:3]:
        planes = {plane["plane"]: plane for plane in run.get("planes", [])}
        rows.append(
            [
                run["run_type"],
                planes.get("encoded_amplitude"),
                planes.get("ols_output_amplitude"),
                planes.get("decoder_output_amplitude"),
                planes.get("physical_camera_plane"),
            ]
        )
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), 5, figsize=(10, 2.4 * len(rows)), squeeze=False)
    headers = ("run", "encoded audit", "OLS audit", "decoder audit", "CMOS audit")
    for column, header in enumerate(headers):
        axes[0][column].set_title(header)
    for row_index, row in enumerate(rows):
        axes[row_index][0].text(0.0, 0.5, str(row[0]), fontsize=9)
        axes[row_index][0].axis("off")
        for column, plane in enumerate(row[1:], start=1):
            axes[row_index][column].axis("off")
            if plane is None:
                axes[row_index][column].text(0.2, 0.5, "missing", fontsize=8)
                continue
            audit = plane.get("prototype_label_audit_accuracy")
            overall = plane.get("overall", {})
            axes[row_index][column].text(
                0.02,
                0.58,
                f"audit={audit:.2f}" if audit is not None else "audit=n/a",
                fontsize=8,
            )
            axes[row_index][column].text(
                0.02,
                0.36,
                f"corr={overall.get('corr'):.2f}" if overall.get("corr") is not None else "corr=n/a",
                fontsize=8,
            )
    if reconstruction is not None:
        fig.suptitle(
            f"v26 OLS oracle: {reconstruction.get('feature_name')} reconstruction is diagnostic-only",
            fontsize=10,
        )
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    return preview_path.as_posix()


def run_ols_decodability_oracle(
    *,
    eval_runs: list[Path] | None = None,
    sample_runs: list[Path] | None = None,
    summaries: list[Path] | None = None,
    output: Path | None = None,
    feature_size: int = 16,
    pca_dim: int = 16,
    ridge: float = 1.0e-2,
) -> dict[str, Any]:
    output_path = output or _default_output()
    _require_report_output(output_path)
    resolved_eval_runs = list(eval_runs or [])
    resolved_sample_runs = list(sample_runs or [])
    if summaries:
        summary_eval, summary_sample = _runs_from_summaries(summaries)
        resolved_eval_runs.extend(summary_eval)
        resolved_sample_runs.extend(summary_sample)
    resolved_eval_runs = _dedupe(resolved_eval_runs)
    resolved_sample_runs = _dedupe(resolved_sample_runs)
    if not resolved_eval_runs and not resolved_sample_runs:
        raise ValueError("at least one --eval-run, --sample-run, or --summary with runs is required")

    run_records = []
    feature_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for run in resolved_eval_runs:
        record, feature_rows = analyze_run(run, kind="eval", feature_size=feature_size)
        run_records.append(record)
        for feature_name, row in feature_rows.items():
            feature_rows_by_name.setdefault(feature_name, []).append(row)
    for run in resolved_sample_runs:
        record, feature_rows = analyze_run(run, kind="sample", feature_size=feature_size)
        run_records.append(record)
        for feature_name, row in feature_rows.items():
            feature_rows_by_name.setdefault(feature_name, []).append(row)

    linear_oracles = {
        feature_name: _run_linear_probe(rows, feature_name=feature_name, pca_dim=pca_dim, ridge=ridge)
        for feature_name, rows in sorted(feature_rows_by_name.items())
    }
    best = _best_valid_oracle(linear_oracles)
    reconstruction = None
    if best is not None:
        rows = feature_rows_by_name[best["feature_name"]]
        reconstruction = _run_reconstruction_oracle(rows, feature_name=best["feature_name"], pca_dim=pca_dim, ridge=ridge)
    decision = _decision(linear_oracles, run_records)
    preview_path = _default_preview_path(output_path)
    preview_written = _write_preview(preview_path=preview_path, run_records=run_records, reconstruction=reconstruction)
    payload = {
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "version": "v26",
        "objective": "ols_representation_decodability_software_oracle",
        "source_summaries": [path.as_posix() for path in summaries or []],
        "eval_runs": [path.as_posix() for path in resolved_eval_runs],
        "sample_runs": [path.as_posix() for path in resolved_sample_runs],
        "thresholds": {
            "ols_decodable_accuracy": DECODE_ACC_PASS,
            "ols_not_decodable_accuracy": DECODE_ACC_FAIL,
            "focus_accuracy": FOCUS_ACC_PASS,
        },
        "oracle_config": {
            "feature_size": feature_size,
            "pca_dim": pca_dim,
            "ridge": ridge,
            "split_policy": "leave-run-out",
            "same_run_train_test_allowed": False,
        },
        "runs": run_records,
        "linear_oracles": linear_oracles,
        "reconstruction_oracle": reconstruction,
        "preview": preview_written,
        "decision": decision,
        "human_readability_required": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, default=None, help="Launcher summary JSON.")
    parser.add_argument("--eval-run", action="append", type=Path, default=None, help="Existing one-shot eval run directory.")
    parser.add_argument("--sample-run", action="append", type=Path, default=None, help="Existing one-shot sample run directory.")
    parser.add_argument("--output", type=Path, default=None, help="Relative output JSON path under outputs/reports/.")
    parser.add_argument("--feature-size", type=int, default=16, help="Spatial size for CPU oracle features.")
    parser.add_argument("--pca-dim", type=int, default=16, help="Maximum PCA dimension for low-capacity oracle.")
    parser.add_argument("--ridge", type=float, default=1.0e-2, help="Ridge value for linear classifiers/reconstruction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ols_decodability_oracle(
        eval_runs=args.eval_run,
        sample_runs=args.sample_run,
        summaries=args.summary,
        output=args.output,
        feature_size=args.feature_size,
        pca_dim=args.pca_dim,
        ridge=args.ridge,
    )
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
