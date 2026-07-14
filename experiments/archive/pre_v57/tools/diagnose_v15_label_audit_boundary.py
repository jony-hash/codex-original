#!/usr/bin/env python3
"""Offline v15 label-audit boundary diagnostics.

This script compares class-average, multi-prototype, and ROI/edge-weighted
label audits on saved one-shot eval tensor bundles. It does not train models
or run samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from ldm_project.teacher_distill import _build_mnist_label_prototype_bank
from ldm_project.visualization_core import readable_camera_for_display


DEFAULT_Z1_RUN = Path(
    "outputs/20260625_v15/runs/"
    "20260625_205632_822704_train_v15_probe_z1_prototype_ce_from_x_e5_eval"
)
DEFAULT_Z2_RUN = Path(
    "outputs/20260625_v15/runs/"
    "20260625_205949_528408_train_v15_probe_z2_prototype_ce_from_s50_e5_eval"
)
DEFAULT_FOCUS_CLASSES = (0, 1, 3, 4, 7, 9)
DEFAULT_CONFUSION_PAIRS = ((3, 5), (7, 9), (4, 9), (1, 8), (0, 5), (9, 8))


@dataclass(frozen=True)
class AuditResult:
    nearest_labels: torch.Tensor
    correlations: torch.Tensor
    margins: torch.Tensor
    top_labels: list[list[int]]
    top_scores: list[list[float]]
    correct_ratio: float


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _centered_features(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().float().flatten(1)
    centered = flat - flat.mean(dim=1, keepdim=True)
    scaled = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return F.normalize(scaled, dim=1, eps=1.0e-6)


def prototype_correlation(samples: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return _centered_features(samples) @ _centered_features(prototypes).T


def build_class_average_prototypes(
    samples: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_labels: Iterable[int],
) -> torch.Tensor:
    labels = labels.detach().long().reshape(-1)
    prototypes = []
    for label in class_labels:
        mask = labels == int(label)
        if not bool(mask.any().item()):
            raise ValueError(f"cannot build prototype for missing class {int(label)}")
        prototypes.append(samples.index_select(0, torch.nonzero(mask, as_tuple=False).flatten()).mean(dim=0))
    return torch.stack(prototypes, dim=0)


def class_average_audit(
    samples: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    class_labels: Iterable[int],
    top_k: int = 3,
) -> AuditResult:
    class_label_list = [int(label) for label in class_labels]
    label_to_index = {label: index for index, label in enumerate(class_label_list)}
    labels_cpu = labels.detach().long().cpu().reshape(-1)
    requested_indices = torch.tensor([label_to_index[int(label)] for label in labels_cpu.tolist()], dtype=torch.long)
    correlations = prototype_correlation(samples, prototypes)
    nearest_indices = correlations.argmax(dim=1).cpu()
    nearest_labels = torch.tensor([class_label_list[int(index)] for index in nearest_indices], dtype=torch.long)
    other = correlations.clone()
    other[torch.arange(other.shape[0]), requested_indices] = -2.0
    margins = correlations.gather(1, requested_indices[:, None]).squeeze(1).cpu() - other.max(dim=1).values.cpu()
    top_scores_raw, top_indices = correlations.topk(k=min(int(top_k), correlations.shape[1]), dim=1)
    top_labels = [
        [class_label_list[int(index)] for index in row.detach().cpu().tolist()]
        for row in top_indices
    ]
    top_scores = [[float(score) for score in row.detach().cpu().tolist()] for row in top_scores_raw]
    correct_ratio = float((nearest_labels == labels_cpu).float().mean().item())
    return AuditResult(
        nearest_labels=nearest_labels,
        correlations=correlations.detach().cpu(),
        margins=margins,
        top_labels=top_labels,
        top_scores=top_scores,
        correct_ratio=correct_ratio,
    )


def multi_prototype_vote_audit(
    samples: torch.Tensor,
    labels: torch.Tensor,
    prototype_samples: torch.Tensor,
    prototype_labels: torch.Tensor,
    *,
    class_labels: Iterable[int],
    neighbors: int = 5,
    top_k: int = 3,
) -> AuditResult:
    class_label_list = [int(label) for label in class_labels]
    labels_cpu = labels.detach().long().cpu().reshape(-1)
    prototype_labels_cpu = prototype_labels.detach().long().cpu().reshape(-1)
    correlations_to_samples = prototype_correlation(samples, prototype_samples)
    k = min(int(neighbors), correlations_to_samples.shape[1])
    top_scores_raw, top_indices = correlations_to_samples.topk(k=k, dim=1)
    nearest_labels = []
    top_labels: list[list[int]] = []
    top_scores: list[list[float]] = []
    class_score_rows = []
    for sample_index in range(correlations_to_samples.shape[0]):
        label_scores: dict[int, list[float]] = {label: [] for label in class_label_list}
        for score, proto_index in zip(top_scores_raw[sample_index], top_indices[sample_index]):
            label = int(prototype_labels_cpu[int(proto_index)].item())
            if label in label_scores:
                label_scores[label].append(float(score.item()))
        class_scores = []
        for label in class_label_list:
            scores = label_scores[label]
            class_scores.append(sum(scores) / float(len(scores)) if scores else -2.0)
        class_score_tensor = torch.tensor(class_scores, dtype=torch.float32)
        class_score_rows.append(class_score_tensor)
        sorted_scores, sorted_indices = class_score_tensor.topk(k=min(int(top_k), len(class_label_list)))
        top_labels.append([class_label_list[int(index)] for index in sorted_indices.tolist()])
        top_scores.append([float(score) for score in sorted_scores.tolist()])
        nearest_labels.append(class_label_list[int(sorted_indices[0].item())])
    class_scores_tensor = torch.stack(class_score_rows, dim=0)
    nearest_label_tensor = torch.tensor(nearest_labels, dtype=torch.long)
    label_to_index = {label: index for index, label in enumerate(class_label_list)}
    requested_indices = torch.tensor([label_to_index[int(label)] for label in labels_cpu.tolist()], dtype=torch.long)
    other = class_scores_tensor.clone()
    other[torch.arange(other.shape[0]), requested_indices] = -2.0
    margins = class_scores_tensor.gather(1, requested_indices[:, None]).squeeze(1) - other.max(dim=1).values
    return AuditResult(
        nearest_labels=nearest_label_tensor,
        correlations=class_scores_tensor,
        margins=margins,
        top_labels=top_labels,
        top_scores=top_scores,
        correct_ratio=float((nearest_label_tensor == labels_cpu).float().mean().item()),
    )


def foreground_roi(samples: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    requested = prototypes.index_select(0, labels.detach().long().cpu())
    mask = (requested > 0.20).float()
    return samples * mask, prototypes * (prototypes > 0.20).float()


def edge_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"edge_map expects [N, C, H, W], got {tuple(tensor.shape)}")
    sobel_x = tensor.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = tensor.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    channels = tensor.shape[1]
    grad_x = F.conv2d(tensor, sobel_x.expand(channels, 1, 3, 3), padding=1, groups=channels)
    grad_y = F.conv2d(tensor, sobel_y.expand(channels, 1, 3, 3), padding=1, groups=channels)
    return (grad_x.pow(2) + grad_y.pow(2)).sqrt()


def center_crop(tensor: torch.Tensor, crop_fraction: float = 0.80) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    crop_h = max(1, int(round(height * float(crop_fraction))))
    crop_w = max(1, int(round(width * float(crop_fraction))))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return tensor[..., top:top + crop_h, left:left + crop_w]


def normalize_readable(tensor: torch.Tensor) -> torch.Tensor:
    return readable_camera_for_display(tensor).detach().float().cpu()


def per_class_summary(
    result: AuditResult,
    labels: torch.Tensor,
    *,
    class_labels: Iterable[int],
    report_class_labels: Iterable[int] | None = None,
    pairs: Iterable[tuple[int, int]],
) -> dict[str, object]:
    labels_cpu = labels.detach().long().cpu().reshape(-1)
    report_labels = [int(label) for label in (report_class_labels or class_labels)]
    out = {}
    for label in report_labels:
        mask = labels_cpu == int(label)
        if not bool(mask.any().item()):
            continue
        nearest = result.nearest_labels[mask]
        out[str(int(label))] = {
            "count": int(mask.sum().item()),
            "correct_ratio": float((nearest == int(label)).float().mean().item()),
            "nearest_label_counts": {str(k): int(v) for k, v in Counter(nearest.tolist()).items()},
            "margin_mean": float(result.margins[mask].mean().item()),
            "top3_first_sample": result.top_labels[int(torch.nonzero(mask, as_tuple=False).flatten()[0].item())],
        }
    pair_out = {}
    class_label_list = [int(label) for label in class_labels]
    label_to_index = {label: index for index, label in enumerate(class_label_list)}
    for target, negative in pairs:
        mask = labels_cpu == int(target)
        if not bool(mask.any().item()) or int(negative) not in label_to_index or int(target) not in label_to_index:
            continue
        target_corr = result.correlations[mask, label_to_index[int(target)]]
        negative_corr = result.correlations[mask, label_to_index[int(negative)]]
        pair_out[f"{int(target)}->{int(negative)}"] = {
            "target_corr_mean": float(target_corr.mean().item()),
            "negative_corr_mean": float(negative_corr.mean().item()),
            "target_minus_negative_mean": float((target_corr - negative_corr).mean().item()),
            "fixed_by_this_audit": bool((result.nearest_labels[mask] == int(target)).all().item()),
        }
    return {"classes": out, "pairs": pair_out}


def _load_mnist_samples(torch_module, *, image_size: int, root: str, per_class: int, class_labels: Iterable[int]):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for v15 label-audit boundary diagnostics.") from exc

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    requested = tuple(int(label) for label in class_labels)
    buckets = {label: [] for label in requested}
    for image, label in dataset:
        label = int(label)
        if label not in buckets or len(buckets[label]) >= int(per_class):
            continue
        buckets[label].append(image)
        if all(len(items) >= int(per_class) for items in buckets.values()):
            break
    missing = [label for label, items in buckets.items() if not items]
    if missing:
        raise ValueError(f"MNIST sample bank is missing labels: {missing}")
    images = []
    labels = []
    for label in requested:
        images.extend(buckets[label])
        labels.extend([label] * len(buckets[label]))
    return torch_module.stack(images, dim=0), torch_module.tensor(labels, dtype=torch_module.long)


def _validate_input_path(path: Path) -> None:
    text = str(path)
    if "outputs/20260625_v15/runs/" in text:
        return
    if "_v16/runs/" in text:
        return
    if "outputs/20260514_unversioned/runs/20260625_095311_822080_train_twophoton_100x100_2um_probe_s_phase_only_full_connect_label_margin_extend_e40" in text:
        return
    raise ValueError(
        "input tensor bundles must come from outputs/20260625_v15/runs/, outputs/*_v16/runs/, "
        "or the explicit pre-v15 S-50 milestone"
    )


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/reports") / f"twophoton_100x100_v15_label_audit_boundary_summary_{timestamp}.json"


def _audit_tensor_bundle(
    *,
    bundle_path: Path,
    metrics_path: Path | None,
    class_average_prototypes: torch.Tensor,
    multi_samples: torch.Tensor,
    multi_labels: torch.Tensor,
    class_labels: tuple[int, ...],
    focus_classes: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
    neighbors: int,
) -> dict[str, object]:
    payload = _load_torch(bundle_path)
    labels = payload["class_labels"].detach().long().cpu()
    teacher = normalize_readable(payload.get("eval_target", payload["teacher_camera"]))
    student = normalize_readable(payload.get("student_camera_raw", payload["student_camera_display"]))
    original = normalize_readable(payload.get("original_digits", teacher))

    class_avg_results = {
        "teacher": class_average_audit(teacher, labels, class_average_prototypes, class_labels=class_labels),
        "student": class_average_audit(student, labels, class_average_prototypes, class_labels=class_labels),
        "original_digits": class_average_audit(original, labels, class_average_prototypes, class_labels=class_labels),
    }
    multi_student = multi_prototype_vote_audit(
        student,
        labels,
        multi_samples,
        multi_labels,
        class_labels=class_labels,
        neighbors=neighbors,
    )
    roi_student, roi_prototypes = foreground_roi(student, class_average_prototypes, labels)
    roi_result = class_average_audit(roi_student, labels, roi_prototypes, class_labels=class_labels)
    edge_result = class_average_audit(edge_map(student), labels, edge_map(class_average_prototypes), class_labels=class_labels)
    crop_result = class_average_audit(
        center_crop(student),
        labels,
        center_crop(class_average_prototypes),
        class_labels=class_labels,
    )

    metric_overall = {}
    if metrics_path is not None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_overall = {
            key: metrics.get("overall", {}).get(key)
            for key in (
                "paired_corr",
                "paired_dice",
                "student_high_saturation_ratio",
                "student_label_correct_ratio",
                "min_class_student_label_correct_ratio",
            )
        }

    audits = {
        "class_average_teacher": per_class_summary(
            class_avg_results["teacher"], labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "class_average_student": per_class_summary(
            class_avg_results["student"], labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "class_average_original_digits": per_class_summary(
            class_avg_results["original_digits"], labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "multi_prototype_student": per_class_summary(
            multi_student, labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "roi_student": per_class_summary(
            roi_result, labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "edge_student": per_class_summary(
            edge_result, labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
        "center_crop_student": per_class_summary(
            crop_result, labels, class_labels=class_labels, report_class_labels=focus_classes, pairs=pairs,
        ),
    }
    pair_diagnosis = {}
    for target, negative in pairs:
        key = f"{int(target)}->{int(negative)}"
        pair_diagnosis[key] = {
            "teacher_class_average_fixed": audits["class_average_teacher"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
            "student_class_average_fixed": audits["class_average_student"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
            "student_multi_prototype_fixed": audits["multi_prototype_student"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
            "student_roi_fixed": audits["roi_student"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
            "student_edge_fixed": audits["edge_student"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
            "student_center_crop_fixed": audits["center_crop_student"]["pairs"].get(key, {}).get("fixed_by_this_audit"),
        }

    return {
        "tensor_bundle": str(bundle_path),
        "metrics": str(metrics_path) if metrics_path else None,
        "metrics_overall": metric_overall,
        "audit_correct_ratio": {
            "class_average_teacher": class_avg_results["teacher"].correct_ratio,
            "class_average_student": class_avg_results["student"].correct_ratio,
            "class_average_original_digits": class_avg_results["original_digits"].correct_ratio,
            "multi_prototype_student": multi_student.correct_ratio,
            "roi_student": roi_result.correct_ratio,
            "edge_student": edge_result.correct_ratio,
            "center_crop_student": crop_result.correct_ratio,
        },
        "audits": audits,
        "pair_diagnosis": pair_diagnosis,
    }


def _decision(runs: list[dict[str, object]]) -> dict[str, object]:
    teacher_bad = any(
        run["pair_diagnosis"].get("3->5", {}).get("teacher_class_average_fixed") is False
        for run in runs
    )
    multi_fix = all(
        run["pair_diagnosis"].get(pair, {}).get("student_multi_prototype_fixed") is True
        for run in runs
        for pair in ("3->5", "7->9")
        if pair in run["pair_diagnosis"]
    )
    roi_fix = any(
        diagnosis.get("student_roi_fixed") is True or diagnosis.get("student_edge_fixed") is True
        for run in runs
        for diagnosis in run["pair_diagnosis"].values()
    )
    if teacher_bad:
        recommendation = "Teacher target also fails class-average audit; prioritize redefining label audit."
    elif multi_fix:
        recommendation = "Multi-prototype audit fixes key failures; implement multi_prototype_ce / hard_negative_ce."
    elif roi_fix:
        recommendation = "ROI/edge audit fixes at least one failure; implement ROI/edge-weighted label objective or audit."
    else:
        recommendation = "All offline audits still fail key pairs; consider stronger readout targets or alternating training."
    return {
        "sample_allowed": False,
        "accepted_summary_written": False,
        "recommendation": recommendation,
        "teacher_class_average_failure_detected": teacher_bad,
        "multi_prototype_fixes_key_pairs": multi_fix,
        "roi_or_edge_fixes_any_pair": roi_fix,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tensor-bundle",
        action="append",
        type=Path,
        default=None,
        help="Eval tensor bundle. May be supplied multiple times. Defaults to Z1 and Z2.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        type=Path,
        default=None,
        help="Matching eval metrics JSON. May be supplied multiple times.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Summary JSON path under outputs/reports.")
    parser.add_argument("--data-root", type=str, default="data", help="MNIST data root.")
    parser.add_argument("--image-size", type=int, default=200)
    parser.add_argument("--mnist-prototypes-per-class", type=int, default=128)
    parser.add_argument("--mnist-multiprototype-per-class", type=int, default=32)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--focus-classes", type=int, nargs="*", default=list(DEFAULT_FOCUS_CLASSES))
    parser.add_argument("--version", type=str, default="v15", help="Version label to record in the summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor_bundles = args.tensor_bundle or [
        DEFAULT_Z1_RUN / "artifacts/one_shot_distill_eval_tensors.pt",
        DEFAULT_Z2_RUN / "artifacts/one_shot_distill_eval_tensors.pt",
    ]
    metrics_paths = args.metrics or [
        DEFAULT_Z1_RUN / "artifacts/one_shot_distill_eval_metrics.json",
        DEFAULT_Z2_RUN / "artifacts/one_shot_distill_eval_metrics.json",
    ]
    if len(metrics_paths) != len(tensor_bundles):
        raise ValueError("--metrics count must match --tensor-bundle count")
    output = args.output or _default_output_path()
    if not str(output).startswith("outputs/reports/"):
        raise ValueError("diagnostic output must be under outputs/reports/")
    for path in tensor_bundles:
        _validate_input_path(path)

    class_labels = tuple(range(10))
    focus_classes = tuple(int(label) for label in args.focus_classes)
    class_average_prototypes = _build_mnist_label_prototype_bank(
        torch,
        class_labels=class_labels,
        image_size=int(args.image_size),
        root=str(args.data_root),
        max_per_class=int(args.mnist_prototypes_per_class),
    ).contiguous()
    multi_samples, multi_labels = _load_mnist_samples(
        torch,
        image_size=int(args.image_size),
        root=str(args.data_root),
        per_class=int(args.mnist_multiprototype_per_class),
        class_labels=class_labels,
    )
    multi_samples = normalize_readable(multi_samples)
    pairs = tuple((int(a), int(b)) for a, b in DEFAULT_CONFUSION_PAIRS)
    runs = [
        _audit_tensor_bundle(
            bundle_path=bundle,
            metrics_path=metrics,
            class_average_prototypes=class_average_prototypes,
            multi_samples=multi_samples,
            multi_labels=multi_labels,
            class_labels=class_labels,
            focus_classes=focus_classes,
            pairs=pairs,
            neighbors=int(args.neighbors),
        )
        for bundle, metrics in zip(tensor_bundles, metrics_paths)
    ]
    summary = {
        "diagnostic_only": True,
        "sample_run_executed": False,
        "accepted_summary_written": False,
        "version": str(args.version),
        "focus_classes": list(focus_classes),
        "confusion_pairs": [[a, b] for a, b in pairs],
        "class_average_prototypes_per_class": int(args.mnist_prototypes_per_class),
        "multi_prototype_samples_per_class": int(args.mnist_multiprototype_per_class),
        "multi_prototype_neighbors": int(args.neighbors),
        "runs": runs,
        "decision": _decision(runs),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["decision"], indent=2, ensure_ascii=False))
    print(f"summary_path={output}")


if __name__ == "__main__":
    main()
