"""Read-only fixed-exposure raw-CMOS audit for the v68 diagnostic checkpoint.

The audit never trains or updates a model.  It uses one global exposure scale
for the complete validation split, so its preview is not confused with the
per-sample normalized display used by historical training figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ldm_project import load_training_config
from ldm_project.student.checkpoints import (
    checkpoint_boundary_contract,
    load_one_shot_checkpoint,
)
from ldm_project.student.experimental.model import OneShotOpticalGenerator
try:
    from tools.diagnose_v68_transparent_memory import _build_prototype_bank
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from diagnose_v68_transparent_memory import _build_prototype_bank


REPORT_SCHEMA = "v68_raw_cmos_acceptance_v2"


def _tensor_hash(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor], prefix: str) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key.startswith(prefix):
            digest.update(key.encode())
            digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _correlation_matrix(prediction: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    pred = prediction.flatten(1).float()
    proto = prototypes.flatten(1).float()
    pred = pred - pred.mean(dim=1, keepdim=True)
    proto = proto - proto.mean(dim=1, keepdim=True)
    numerator = pred @ proto.T / pred.shape[1]
    denominator = pred.pow(2).mean(dim=1, keepdim=True).sqrt() * proto.pow(2).mean(dim=1).sqrt()
    return numerator / denominator.clamp_min(1.0e-8)


def global_exposure_scale(raw: torch.Tensor) -> float:
    """Return the single 99.5-percentile exposure scale for a whole split."""
    if raw.numel() == 0:
        raise ValueError("raw camera tensor must not be empty")
    scale = float(torch.quantile(raw.detach().float().flatten(), 0.995).item())
    return max(scale, 1.0e-12)


def fixed_exposure_view(raw: torch.Tensor, scale: float) -> torch.Tensor:
    """Map raw intensity with one registered scale; no per-sample enhancement."""
    if not scale > 0.0:
        raise ValueError(f"exposure scale must be positive, got {scale}")
    return (raw / scale).clamp(0.0, 1.0)


def per_sample_normalized_view(raw: torch.Tensor) -> torch.Tensor:
    """Historical display comparator; each sample gets its own 99.5% scale."""
    if raw.ndim < 2:
        raise ValueError("raw camera tensor must include a sample dimension")
    scales = torch.quantile(raw.detach().float().flatten(1), 0.995, dim=1)
    return (raw / scales.clamp_min(1.0e-12).view(-1, *([1] * (raw.ndim - 1)))).clamp(
        0.0, 1.0
    )


def _sample_correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_flat = prediction.detach().float().flatten(1)
    target_flat = target.detach().float().flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    denominator = (
        prediction_centered.pow(2).mean(dim=1).sqrt()
        * target_centered.pow(2).mean(dim=1).sqrt()
    )
    return (prediction_centered * target_centered).mean(dim=1) / denominator.clamp_min(
        1.0e-8
    )


def _sample_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_mask = prediction.detach().float().clamp(0.0, 1.0) > 0.2
    target_mask = target.detach().float().clamp(0.0, 1.0) > 0.2
    dims = tuple(range(1, prediction_mask.ndim))
    intersection = (prediction_mask & target_mask).float().sum(dim=dims)
    denominator = prediction_mask.float().sum(dim=dims) + target_mask.float().sum(dim=dims)
    return 2.0 * intersection / denominator.clamp_min(1.0)


def _class_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(int(value) for value in labels.unique().tolist()):
        indices = labels == label
        result[str(label)] = {
            "sample_count": int(indices.sum().item()),
            "corr": float(_sample_correlation(prediction[indices], target[indices]).mean()),
            "dice": float(_sample_dice(prediction[indices], target[indices]).mean()),
        }
    return result


def _preview(
    path: Path,
    clean_targets: torch.Tensor,
    teacher: torch.Tensor,
    fixed_raw: torch.Tensor,
    normalized: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    chosen: list[int] = []
    for label in range(10):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten()
        if matches.numel():
            chosen.append(int(matches[0]))
    figure, axes = plt.subplots(10, 4, figsize=(8, 18), squeeze=False)
    for row, index in enumerate(chosen):
        images = (
            clean_targets[index, 0],
            teacher[index, 0],
            fixed_raw[index, 0],
            normalized[index, 0],
        )
        for column, image in enumerate(images):
            axes[row][column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row][column].axis("off")
        axes[row][0].set_ylabel(str(int(labels[index])), rotation=0, labelpad=12)
    for column, title in enumerate(
        ("clean prototype", "teacher camera", "raw-linear-locked-scale", "per-sample-normalized")
    ):
        axes[0][column].set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(
    *,
    config: str | Path,
    checkpoint: str | Path,
    validation_dataset: str | Path,
    output: str | Path,
    device: str = "cuda",
    max_samples: int | None = None,
    exposure_mode: str | None = None,
    numerical_report: str | None = None,
    training_admission_report: str | None = None,
) -> dict[str, Any]:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v68 raw-CMOS audit")
    train_cfg = load_training_config(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dataset = torch.load(validation_dataset, map_location="cpu", weights_only=False)
    codes = dataset["codes"]
    labels = dataset["class_labels"].long()
    teacher = dataset.get("teacher_camera", dataset.get("teacher_image"))
    if teacher is None:
        raise ValueError("validation dataset requires teacher_camera or teacher_image")
    count = min(len(labels), int(max_samples)) if max_samples is not None else len(labels)
    codes, labels, teacher = codes[:count], labels[:count], teacher[:count]
    prototypes = _build_prototype_bank(train_cfg).float()
    clean_targets = prototypes.index_select(0, labels)
    model = OneShotOpticalGenerator(
        replace(train_cfg.physics, physical_forward_contract=True),
        train_cfg.one_shot,
        image_size=train_cfg.first_stage.image_size,
    ).to(resolved).eval()
    load_one_shot_checkpoint(model, payload)
    raw_values: list[torch.Tensor] = []
    energy_records: list[dict[str, float]] = []
    with torch.no_grad():
        for code, label in zip(codes, labels):
            output_values = model(
                code[None].to(resolved), label[None].to(resolved), return_optical_fields=False
            )
            raw_values.append(output_values["physical_camera_intensity_raw"].detach().cpu())
            energy_records.append(
                {
                    key: float(value.detach().float().mean().cpu())
                    for key, value in output_values.items()
                    if key.endswith("energy_ratio") or key.endswith("retained_energy")
                }
            )
            del output_values
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    raw = torch.cat(raw_values)
    normalized = per_sample_normalized_view(raw)
    inferred_mode = "locked" if train_cfg.one_shot.diagnostic_camera_normalization == "fixed_exposure" else "calibrate"
    exposure_mode = exposure_mode or inferred_mode
    if exposure_mode not in {"calibrate", "locked"}:
        raise ValueError("exposure_mode must be calibrate or locked")
    if exposure_mode == "locked":
        if train_cfg.one_shot.diagnostic_camera_normalization != "fixed_exposure":
            raise ValueError("locked exposure requires diagnostic_camera_normalization=fixed_exposure")
        configured_scale = train_cfg.one_shot.diagnostic_camera_exposure_scale
        if configured_scale is None:
            raise ValueError("locked exposure requires diagnostic_camera_exposure_scale")
        exposure_scale = float(configured_scale)
    else:
        if train_cfg.one_shot.diagnostic_camera_normalization == "fixed_exposure":
            raise ValueError("fixed-exposure configs must use locked audit mode")
        exposure_scale = global_exposure_scale(raw)
    fixed_raw = fixed_exposure_view(raw, exposure_scale)
    prototype_scores = _correlation_matrix(fixed_raw, prototypes)
    predicted_labels = prototype_scores.argmax(dim=1)
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for target, predicted in zip(labels, predicted_labels):
        confusion[int(target), int(predicted)] += 1
    clean_corr_samples = _sample_correlation(fixed_raw, clean_targets)
    clean_dice_samples = _sample_dice(fixed_raw, clean_targets)
    teacher_corr_samples = _sample_correlation(fixed_raw, teacher.clamp(0.0, 1.0))
    teacher_dice_samples = _sample_dice(fixed_raw, teacher.clamp(0.0, 1.0))
    normalized_clean_corr = _sample_correlation(normalized, clean_targets)
    normalized_clean_dice = _sample_dice(normalized, clean_targets)
    normalized_teacher_corr = _sample_correlation(normalized, teacher.clamp(0.0, 1.0))
    normalized_teacher_dice = _sample_dice(normalized, teacher.clamp(0.0, 1.0))
    high_saturation = float((fixed_raw >= 0.99).float().mean().item())
    low_saturation = float((fixed_raw <= 0.01).float().mean().item())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "diagnostic_only": True,
        "status": "raw-audit-complete",
        "manual_review_required": True,
        "sample_count": count,
        "exposure_mode": exposure_mode,
        "exposure_scale_global_99_5": exposure_scale,
        "exposure_scale_hash": hashlib.sha256(str(exposure_scale).encode()).hexdigest(),
        "normalization_contract": "one_global_scale_gamma_1_no_smoothing",
        "target_provenance": {
            "clean_target": "mnist_prototype",
            "teacher_target": "teacher_camera",
            "primary_gate_target": "mnist_prototype",
        },
        "metrics": {
            "clean_prototype": {
                "fixed_exposure_corr": float(clean_corr_samples.mean()),
                "fixed_exposure_dice": float(clean_dice_samples.mean()),
                "per_sample_normalized_corr": float(normalized_clean_corr.mean()),
                "per_sample_normalized_dice": float(normalized_clean_dice.mean()),
            },
            "teacher_camera": {
                "fixed_exposure_corr": float(teacher_corr_samples.mean()),
                "fixed_exposure_dice": float(teacher_dice_samples.mean()),
                "per_sample_normalized_corr": float(normalized_teacher_corr.mean()),
                "per_sample_normalized_dice": float(normalized_teacher_dice.mean()),
            },
            "prototype_label_accuracy": float((predicted_labels == labels).float().mean().item()),
            "high_saturation_ratio": high_saturation,
            "low_saturation_ratio": low_saturation,
        },
        "prototype_confusion": confusion.tolist(),
        "per_class_metrics": {
            "clean_prototype": _class_metrics(fixed_raw, clean_targets, labels),
            "teacher_camera": _class_metrics(fixed_raw, teacher.clamp(0.0, 1.0), labels),
        },
        "energy_records_mean": {
            key: sum(record.get(key, 0.0) for record in energy_records) / max(len(energy_records), 1)
            for key in sorted({key for record in energy_records for key in record})
        },
        "checkpoint_hashes": {
            "generator_e": _state_hash(payload["model_state_dict"], "generator_e."),
            "encoder": _state_hash(payload["model_state_dict"], "encoder."),
            "decoder": _state_hash(payload["model_state_dict"], "decoder."),
        },
        "optical_boundary_contract": checkpoint_boundary_contract(payload),
        "checkpoint_contract_explicit": isinstance(
            payload.get("optical_boundary_contract"), str
        ),
        "automated_gate": {
            "clean_raw_corr": float(clean_corr_samples.mean()) >= 0.70,
            "clean_raw_dice": float(clean_dice_samples.mean()) >= 0.68,
            "high_saturation": high_saturation <= 0.08,
            "prototype_accuracy": float((predicted_labels == labels).float().mean().item()) >= 0.80,
        },
    }
    report["automated_gate_passed"] = all(report["automated_gate"].values())
    report["linked_reports"] = {
        "numerical_report": str(numerical_report) if numerical_report is not None else None,
        "training_admission_report": (
            str(training_admission_report)
            if training_admission_report is not None
            else None
        ),
    }
    root = Path(output)
    root.parent.mkdir(parents=True, exist_ok=True)
    preview_path = root.with_name(root.stem + "_raw_preview.png")
    _preview(preview_path, clean_targets, teacher, fixed_raw, normalized, labels)
    report["raw_preview"] = str(preview_path)
    root.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--exposure-mode", choices=("calibrate", "locked"))
    parser.add_argument("--numerical-report", type=Path)
    parser.add_argument("--training-admission-report", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
