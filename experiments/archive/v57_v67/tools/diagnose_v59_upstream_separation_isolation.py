#!/usr/bin/env python3
"""Zero-training v59 upstream class-separation broad-isolation diagnostic.

The diagnostic reuses the v56 checkpoint and the readable validation split.  It
does not optimize parameters, change the formal relay, or write an accepted
summary.  Three model forwards are compared: the paired input, a deterministic
label derangement, and a within-class code rotation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config


DEFAULT_CHECKPOINT = Path(
    "outputs/20260714_v56/runs/20260714_204958_697374_train_v56_coherent_relay_from_scratch/"
    "checkpoints/one_shot_distill_best.pt"
)
DEFAULT_DATASET = Path(
    "outputs/20260514_unversioned/runs/20260610_151747_768931_train_teacher_distill_15k_"
    "acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts/"
    "teacher_distill_dataset_readable_val.pt"
)
DEFAULT_CONFIG = Path("configs/eval_v56_coherent_relay_from_scratch.yaml")
DEFAULT_V58_REPORT = Path("outputs/reports/v58_aperture_throughput_upper_bound.json")
DEFAULT_OUTPUT = Path("outputs/reports/v59_upstream_separation_isolation.json")
PLANE_ORDER = ("slm_command", "slm_output", "pre_encoder_relay_image", "physical_camera_plane")
PLANE_PIXEL_SIZE = {
    "slm_command": 2.5e-6,
    "slm_output": 2.5e-6,
    "pre_encoder_relay_image": 2.0e-6,
    "physical_camera_plane": 2.5e-6,
}
METRICS = ("amplitude", "complex", "intensity")


def _load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a dictionary tensor bundle: {path}")
    return payload


def _field(record: dict[str, Any]) -> torch.Tensor:
    real, imag = record.get("real"), record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag) or real.shape != imag.shape:
        raise ValueError("optical field record must contain matching real/imag tensors")
    return torch.complex(real, imag).detach().cpu()


def _balanced_labels(labels: torch.Tensor) -> dict[str, Any]:
    labels = labels.detach().cpu().long().view(-1)
    counts = {str(label): int((labels == label).sum()) for label in range(10)}
    return {"sample_count": int(labels.numel()), "counts": counts, "valid": counts == {str(label): 10 for label in range(10)}}


def deterministic_label_derangement(labels: torch.Tensor) -> torch.Tensor:
    """Shift every class label by one while preserving the input order."""
    labels = labels.detach().clone().long()
    if labels.numel() == 0 or not bool(((labels >= 0) & (labels < 10)).all()):
        raise ValueError("labels must contain only class ids 0-9")
    return labels.add(1).remainder(10)


def deterministic_within_class_code_rotation(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rotate codes inside each class, preserving labels and the code multiset."""
    if codes.ndim < 2 or codes.shape[0] != labels.shape[0]:
        raise ValueError("codes and labels must have aligned leading dimensions")
    rotated = codes.detach().clone()
    for label in torch.unique(labels.detach().cpu().long(), sorted=True):
        indices = torch.nonzero(labels.detach().cpu().long() == label, as_tuple=False).flatten()
        if indices.numel() > 1:
            rotated.index_copy_(0, indices, codes.index_select(0, indices.roll(1)))
    return rotated


def _validate_dataset(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    codes, labels, target = payload.get("codes"), payload.get("class_labels"), payload.get("teacher_camera")
    if not torch.is_tensor(codes) or codes.ndim != 2 or codes.shape[1] != 128:
        reasons.append("validation dataset codes must have shape [N,128]")
    if not torch.is_tensor(labels) or labels.ndim != 1:
        reasons.append("validation dataset class_labels must be one-dimensional")
    if not torch.is_tensor(target) or target.ndim != 4 or tuple(target.shape[-2:]) != (200, 200):
        reasons.append("validation dataset teacher_camera must have shape [N,1,200,200]")
    if torch.is_tensor(codes) and torch.is_tensor(labels) and codes.shape[0] != labels.shape[0]:
        reasons.append("codes and class_labels have different sample counts")
    if torch.is_tensor(labels):
        balance = _balanced_labels(labels)
        if not balance["valid"]:
            reasons.append(f"expected exactly 10 samples per class, got {balance['counts']}")
    if torch.is_tensor(target) and not bool(torch.isfinite(target).all()):
        reasons.append("teacher_camera contains non-finite values")
    if payload.get("readability_split") != "val":
        reasons.append(f"readability_split must be 'val', got {payload.get('readability_split')!r}")
    config = payload.get("config")
    filter_config = config.get("filter", {}) if isinstance(config, dict) else {}
    if filter_config.get("teacher_filter") != "readability_strict_split":
        reasons.append("dataset was not produced by readability_strict_split")
    nearest, margin = payload.get("label_audit_nearest_label"), payload.get("label_audit_margin")
    if not torch.is_tensor(nearest) or not torch.is_tensor(labels) or not torch.equal(nearest.long(), labels.long()):
        reasons.append("readability nearest-label audit does not match requested labels")
    if not torch.is_tensor(margin) or not bool(torch.isfinite(margin).all()) or bool((margin <= 0).any()):
        reasons.append("readability label margins are missing, non-finite, or non-positive")
    return reasons


def _validate_v58_report(path: Path, checkpoint: Path) -> list[str]:
    reasons: list[str] = []
    if not path.is_file():
        return [f"v58 report does not exist: {path}"]
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("diagnostic_only") is not True or report.get("accepted_summary_written") is not False:
        reasons.append("v58 report acceptance metadata is invalid")
    if report.get("decision") != "aperture-upper-bound-no-separation-recovery":
        reasons.append("v58 report does not contain the required negative aperture decision")
    report_checkpoint = Path(str(report.get("inputs", {}).get("checkpoint", "")))
    if report_checkpoint != checkpoint:
        reasons.append(f"v58 checkpoint mismatch: {report_checkpoint} != {checkpoint}")
    return reasons


def _validate_contract(train_cfg) -> list[str]:
    physics, one_shot = train_cfg.physics, train_cfg.one_shot
    reasons: list[str] = []
    if physics is None:
        return ["training config does not contain physics"]
    if one_shot.d2nn is None:
        return ["training config does not contain explicit one_shot.d2nn geometry"]
    macro = physics.macro_optics
    checks = {
        "physical_forward_contract": physics.physical_forward_contract is True,
        "front_path_model": physics.front_path_model == "coherent_imaging_4f",
        "back_path_model": physics.back_path_model == "free_space",
        "identity_ols": one_shot.ols_kind == "identity",
        "d2nn_grid_size": int(one_shot.d2nn.grid_size) == 100,
        "d2nn_pixel_size": math.isclose(float(one_shot.d2nn.pixel_size), 2.0e-6, rel_tol=0.0, abs_tol=1e-15),
        "d2nn_spacing": math.isclose(float(one_shot.d2nn.layer_spacing), 1.5e-3, rel_tol=0.0, abs_tol=1e-12),
        "d2nn_encoder_layers": int(one_shot.d2nn.encoder_layers) == 2,
        "d2nn_decoder_layers": int(one_shot.d2nn.decoder_layers) == 2,
        "front_lens1_focal_length": math.isclose(float(macro.front_lens1_focal_length), 0.22321428571428573, rel_tol=0.0, abs_tol=1e-12),
        "front_lens2_focal_length": math.isclose(float(macro.front_lens2_focal_length), 0.0892857142857143, rel_tol=0.0, abs_tol=1e-12),
        "front_lens_clear_aperture": math.isclose(float(macro.front_lens1_clear_aperture), 0.02375, rel_tol=0.0, abs_tol=1e-12) and math.isclose(float(macro.front_lens2_clear_aperture), 0.02375, rel_tol=0.0, abs_tol=1e-12),
        "back_relay_geometry": all(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-11) for actual, expected in ((macro.dist_lg_to_relay_lens, 0.100), (macro.dist_relay_lens_to_cmos, 0.250), (macro.f_relay, 0.0714285714))),
        "back_relay_clear_aperture": math.isclose(float(macro.back_relay_clear_aperture), 0.0266, rel_tol=0.0, abs_tol=1e-12),
    }
    reasons.extend(f"formal contract check failed: {key}" for key, value in checks.items() if not value)
    return reasons


def _extract_batch(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    fields = outputs.get("optical_fields")
    if not isinstance(fields, dict):
        raise ValueError("model did not return strict optical fields")
    return {
        "slm_command": outputs["slm_command"].detach().cpu().float(),
        "slm_output": _field(fields["slm_output"]),
        "pre_encoder_relay_image": _field(fields["pre_encoder_relay_image"]),
        "physical_camera_plane": outputs["physical_camera_plane"].detach().cpu().float(),
    }


def _forward_variant(model, codes: torch.Tensor, labels: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    collected: dict[str, list[torch.Tensor]] = {key: [] for key in PLANE_ORDER}
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            outputs = model(
                codes[start : start + batch_size].to(device),
                labels[start : start + batch_size].to(device),
                return_optical_fields=True,
            )
            batch = _extract_batch(outputs)
            for key in PLANE_ORDER:
                collected[key].append(batch[key])
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def _pair_distance(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.detach().to(torch.complex128), b.detach().to(torch.complex128)
    a_amp, b_amp = a.abs(), b.abs()
    def relative(left: torch.Tensor, right: torch.Tensor) -> float:
        denom = (left.norm() + right.norm()).mul(0.5).clamp_min(1e-12)
        return float((left - right).norm() / denom)
    phase = torch.angle((a.conj() * b).sum())
    aligned = a * torch.exp(1j * phase).to(a.dtype)
    return {"amplitude": relative(a_amp, b_amp), "complex": relative(aligned, b), "intensity": relative(a_amp.square(), b_amp.square())}


def summarize_plane(values: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    same: dict[str, list[float]] = {metric: [] for metric in METRICS}
    cross: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            destination = same if int(labels[left]) == int(labels[right]) else cross
            for metric, value in _pair_distance(values[left, 0], values[right, 0]).items():
                destination[metric].append(value)
    same_median = {metric: median(values) if values else None for metric, values in same.items()}
    cross_median = {metric: median(values) if values else None for metric, values in cross.items()}
    return {
        "sample_count": int(values.shape[0]),
        "same_class_pair_count": len(same["complex"]),
        "cross_class_pair_count": len(cross["complex"]),
        "same_class_median": same_median,
        "cross_class_median": cross_median,
        "separation_signal": {
            metric: cross_median[metric] - same_median[metric]
            if same_median[metric] is not None and cross_median[metric] is not None else None
            for metric in METRICS
        },
    }


def _feature_matrix(values: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(values):
        value = torch.cat([values.real.float().cpu(), values.imag.float().cpu()], dim=1)
    else:
        value = values.detach().float().cpu()
    if value.ndim == 4:
        value = F.adaptive_avg_pool2d(value, (32, 32))
    value = value.flatten(1)
    value = value - value.mean(dim=1, keepdim=True)
    return value / value.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _fold_ids(labels: torch.Tensor, folds: int) -> torch.Tensor:
    result = torch.empty_like(labels)
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        result[indices] = torch.arange(indices.numel()) % folds
    return result


def prototype_cv_accuracy(values: torch.Tensor, labels: torch.Tensor, *, folds: int = 5) -> float:
    features = _feature_matrix(values)
    fold_ids = _fold_ids(labels, folds)
    correct, total = 0, 0
    classes = torch.unique(labels, sorted=True)
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        centroids = []
        for label in classes:
            centroid = features[train & (labels == label)].mean(dim=0)
            centroids.append(centroid / centroid.norm().clamp_min(1e-12))
        scores = features[test] @ torch.stack(centroids).T
        predictions = classes[scores.argmax(dim=1)]
        correct += int((predictions == labels[test]).sum())
        total += int(test.sum())
    return float(correct / max(total, 1))


def permutation_null(values: torch.Tensor, labels: torch.Tensor, *, permutations: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observed = prototype_cv_accuracy(values, labels)
    null = []
    for _ in range(permutations):
        null.append(prototype_cv_accuracy(values, labels[torch.randperm(labels.numel(), generator=generator)]))
    null_tensor = torch.tensor(null)
    exceed = int((null_tensor >= observed).sum())
    return {
        "observed_accuracy": observed,
        "permutation_count": permutations,
        "seed": seed,
        "null_mean": float(null_tensor.mean()),
        "null_std": float(null_tensor.std(unbiased=False)),
        "null_p95": float(torch.quantile(null_tensor, 0.95)),
        "p_value": float((exceed + 1) / (permutations + 1)),
    }


def _paired_target_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target_original = target
    pred, target = prediction.float().flatten(1), target.float().flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    corr = (pred * target).sum(dim=1) / (pred.norm(dim=1) * target.norm(dim=1)).clamp_min(1e-12)
    raw_pred = prediction.float().flatten(1)
    raw_target = target_original.float().flatten(1)
    return {"corr_mean": float(corr.mean()), "mse_mean": float((raw_pred - raw_target).square().mean())}


def _sensitivity(baseline: dict[str, torch.Tensor], variant: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for key in PLANE_ORDER:
        left, right = baseline[key], variant[key]
        delta = (right - left).flatten(1).norm(dim=1)
        denom = left.flatten(1).norm(dim=1).clamp_min(1e-12)
        result[key] = float((delta / denom).mean())
    return result


def _spectrum(values: torch.Tensor, labels: torch.Tensor, *, pixel_size: float, cutoff: float) -> dict[str, Any]:
    value = values.abs().float() if torch.is_complex(values) else values.float()
    value = value[:, 0]
    global_mean = value.mean(dim=0)
    between, within = [], []
    for label in torch.unique(labels, sorted=True):
        group = value[labels == label]
        class_mean = group.mean(dim=0)
        between.append(class_mean - global_mean)
        within.extend(group - class_mean)
    between_fft = torch.fft.fftshift(torch.fft.fft2(torch.stack(between)), dim=(-2, -1)).abs().square().mean(0)
    within_fft = torch.fft.fftshift(torch.fft.fft2(torch.stack(within)), dim=(-2, -1)).abs().square().mean(0)
    height, width = value.shape[-2:]
    fy = (torch.arange(height) - height // 2) / (height * pixel_size)
    fx = (torch.arange(width) - width // 2) / (width * pixel_size)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_y.square() + grid_x.square())
    bands = {"low": radius <= 0.5 * cutoff, "passband": radius <= cutoff, "out_of_passband": radius > cutoff}
    result = {"pixel_size_m": pixel_size, "cutoff_cycles_per_m": cutoff}
    for name, mask in bands.items():
        between_energy = float(between_fft[mask].sum())
        within_energy = float(within_fft[mask].sum())
        result[name] = {
            "between_energy": between_energy,
            "within_energy": within_energy,
            "between_to_within": between_energy / max(within_energy, 1e-12),
        }
    total_between = float(between_fft.sum())
    result["between_out_of_passband_fraction"] = float(between_fft[bands["out_of_passband"]].sum()) / max(total_between, 1e-12)
    return result


def _render_preview(output: Path, target: torch.Tensor, variants: dict[str, dict[str, torch.Tensor]], labels: torch.Tensor) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the v59 class-balanced preview") from exc
    indices = torch.tensor([int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)])
    columns = (
        ("teacher target", target),
        ("baseline SLM", variants["baseline"]["slm_command"]),
        ("baseline 4f |E|", variants["baseline"]["pre_encoder_relay_image"].abs()),
        ("baseline CMOS", variants["baseline"]["physical_camera_plane"]),
        ("label-shift CMOS", variants["label_shift"]["physical_camera_plane"]),
        ("code-rotate CMOS", variants["code_rotation"]["physical_camera_plane"]),
    )
    figure, axes = plt.subplots(10, len(columns), figsize=(15, 24), squeeze=False)
    for row, index in enumerate(indices.tolist()):
        for column, (title, tensor) in enumerate(columns):
            image = tensor[index, 0].float().cpu()
            image = image / image.quantile(0.995).clamp_min(1e-12)
            axes[row][column].imshow(image.clamp(0, 1), cmap="gray", interpolation="nearest")
            axes[row][column].axis("off")
            if row == 0:
                axes[row][column].set_title(title)
        axes[row][0].set_ylabel(f"class {int(labels[index])}")
    figure.tight_layout()
    preview = output.with_name(f"{output.stem}_preview.png")
    figure.savefig(preview, dpi=160)
    plt.close(figure)
    return preview


def _decision(
    teacher: dict[str, Any],
    planes: dict[str, dict[str, Any]],
    sensitivities: dict[str, dict[str, float]],
    target_metrics: dict[str, dict[str, float]],
    spectra: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    teacher_strong = teacher["prototype_cv"]["p_value"] < 0.01 and teacher["prototype_cv"]["observed_accuracy"] > teacher["prototype_cv"]["null_p95"]
    slm_signal = planes["baseline"]["slm_command"]["prototype_cv"]["p_value"] < 0.01
    front_signal = planes["baseline"]["pre_encoder_relay_image"]["prototype_cv"]["p_value"] < 0.01
    teacher_complex_signal = float(teacher["separation"]["separation_signal"]["complex"] or 0.0)
    slm_complex_signal = float(planes["baseline"]["slm_command"]["separation"]["separation_signal"]["complex"] or 0.0)
    front_complex_signal = float(planes["baseline"]["pre_encoder_relay_image"]["separation"]["separation_signal"]["complex"] or 0.0)
    slm_relative_to_teacher = slm_complex_signal / max(teacher_complex_signal, 1e-12)
    front_relative_to_slm = front_complex_signal / max(slm_complex_signal, 1e-12)
    label_effect = sensitivities["label_shift"]["slm_command"]
    code_effect = sensitivities["code_rotation"]["slm_command"]
    baseline_mse = max(float(target_metrics["baseline"]["mse_mean"]), 1e-12)
    label_target_mse_delta = abs(float(target_metrics["label_shift"]["mse_mean"]) - float(target_metrics["baseline"]["mse_mean"])) / baseline_mse
    code_target_mse_delta = abs(float(target_metrics["code_rotation"]["mse_mean"]) - float(target_metrics["baseline"]["mse_mean"])) / baseline_mse
    outside_passband = float(spectra["slm_command"]["between_out_of_passband_fraction"])
    bandwidth_candidate = slm_signal and front_relative_to_slm < 0.10 and outside_passband > 0.50
    slm_weak = slm_relative_to_teacher < 0.10
    if not teacher_strong:
        status = "teacher-target-separation-weak"
        next_gate = "inspect teacher readable split and target construction before student changes"
    elif bandwidth_candidate:
        status = "generator-relay-bandwidth-mismatch-candidate"
        next_gate = "compare formal pupil-band spectral energy before considering any contract redesign"
    elif slm_weak and label_effect <= 0.01 and label_target_mse_delta <= 0.01:
        status = "generator-label-conditioning-or-average-template"
        next_gate = "offline generator label/template audit; no decoder or loss tuning"
    elif slm_weak and code_effect <= 0.01 and code_target_mse_delta <= 0.01:
        status = "generator-code-conditioning-instance-collapse"
        next_gate = "offline generator code sensitivity and target pairing audit"
    else:
        status = "objective-representation-mismatch-candidate"
        next_gate = "offline decodability upper bound; do not tune decoder or losses"
    return {
        "status": status,
        "teacher_target_strong": teacher_strong,
        "slm_separation_significant": slm_signal,
        "front_separation_significant": front_signal,
        "slm_complex_separation_relative_to_teacher": slm_relative_to_teacher,
        "front_complex_separation_relative_to_slm": front_relative_to_slm,
        "slm_separation_weak": slm_weak,
        "front_bandwidth_candidate": bandwidth_candidate,
        "slm_between_energy_out_of_passband_fraction": outside_passband,
        "label_target_mse_relative_delta": label_target_mse_delta,
        "code_target_mse_relative_delta": code_target_mse_delta,
        "label_to_code_slm_sensitivity_ratio": label_effect / max(code_effect, 1e-12),
        "next_gate": next_gate,
        "diagnostic_only": True,
        "accepted_summary_written": False,
    }


def run(*, train_config: Path = DEFAULT_CONFIG, checkpoint: Path = DEFAULT_CHECKPOINT, validation_dataset: Path = DEFAULT_DATASET, v58_report: Path = DEFAULT_V58_REPORT, output: Path = DEFAULT_OUTPUT, device: str = "cuda", batch_size: int = 4, permutations: int = 128) -> dict[str, Any]:
    paths = {"train_config": Path(train_config), "checkpoint": Path(checkpoint), "validation_dataset": Path(validation_dataset), "v58_report": Path(v58_report), "output": Path(output)}
    reasons = [f"missing input: {name}={path}" for name, path in paths.items() if name != "output" and not path.is_file()]
    if reasons:
        report = {"status": "harness-invalid", "diagnostic_only": True, "accepted_summary_written": False, "reasons": reasons, "inputs": {key: str(value) for key, value in paths.items()}}
        paths["output"].parent.mkdir(parents=True, exist_ok=True)
        paths["output"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    train_cfg = load_training_config(paths["train_config"])
    dataset = _load(paths["validation_dataset"])
    reasons.extend(_validate_dataset(dataset))
    reasons.extend(_validate_v58_report(paths["v58_report"], paths["checkpoint"]))
    reasons.extend(_validate_contract(train_cfg))
    labels = dataset.get("class_labels").detach().cpu().long()
    codes = dataset.get("codes").detach().cpu().float()
    target = dataset.get("teacher_camera").detach().cpu().float()
    report: dict[str, Any] = {
        "status": "harness-invalid" if reasons else "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "v58_manual_cmos_verdict": "baseline-and-upper-bound-unreadable",
        "inputs": {key: str(value) for key, value in paths.items()},
        "invariants": {"zero_training": True, "same_checkpoint": True, "same_validation_codes": True, "formal_contract_changed": False, "parameter_update_count": 0},
        "dataset": {**_balanced_labels(labels), "readability_split": dataset.get("readability_split"), "teacher_filter": dataset.get("config", {}).get("filter", {}).get("teacher_filter")},
        "reasons": reasons,
    }
    if reasons:
        paths["output"].parent.mkdir(parents=True, exist_ok=True)
        paths["output"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v59 full-geometry diagnostic; CPU fallback is not valid")
    model = OneShotOpticalGenerator(train_cfg.physics, train_cfg.one_shot, image_size=train_cfg.first_stage.image_size).to(resolved_device)
    checkpoint_payload = torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=False)
    load_one_shot_checkpoint(model, checkpoint_payload)
    model.eval()
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    label_shift = deterministic_label_derangement(labels)
    code_rotation = deterministic_within_class_code_rotation(codes, labels)
    variants = {
        "baseline": _forward_variant(model, codes, labels, resolved_device, batch_size),
        "label_shift": _forward_variant(model, codes, label_shift, resolved_device, batch_size),
        "code_rotation": _forward_variant(model, code_rotation, labels, resolved_device, batch_size),
    }
    changed = [key for key, value in model.state_dict().items() if not torch.equal(before[key], value.detach().cpu())]
    report["invariants"]["parameter_update_count"] = len(changed)
    if changed:
        report["status"] = "harness-invalid"
        report["reasons"].append(f"model parameters changed during diagnostic: {changed[:5]}")
    plane_reports: dict[str, dict[str, Any]] = {}
    for variant_name, variant in variants.items():
        plane_reports[variant_name] = {}
        for plane in PLANE_ORDER:
            plane_reports[variant_name][plane] = {"separation": summarize_plane(variant[plane], labels), "prototype_cv": permutation_null(variant[plane], labels, permutations=permutations, seed=20260714)}
    teacher_cv = permutation_null(target, labels, permutations=permutations, seed=20260714)
    target_report = {"separation": summarize_plane(target, labels), "prototype_cv": teacher_cv}
    target_metrics = {name: _paired_target_metrics(variant["physical_camera_plane"], target) for name, variant in variants.items()}
    sensitivities = {name: _sensitivity(variants["baseline"], variant) for name, variant in variants.items() if name != "baseline"}
    cutoff_front_output = float(train_cfg.physics.macro_optics.front_lens2_clear_aperture) / (2.0 * float(train_cfg.physics.macro_optics.front_lens2_focal_length) * float(train_cfg.physics.optics.wavelength))
    spectra = {}
    for plane in PLANE_ORDER:
        cutoff = cutoff_front_output * 0.4 if plane in {"slm_command", "slm_output"} else cutoff_front_output
        if plane == "physical_camera_plane":
            cutoff = float(train_cfg.physics.macro_optics.back_relay_clear_aperture) / (2.0 * float(train_cfg.physics.macro_optics.f_relay) * float(train_cfg.physics.optics.wavelength))
        spectra[plane] = _spectrum(variants["baseline"][plane], labels, pixel_size=PLANE_PIXEL_SIZE[plane], cutoff=cutoff)
    for variant_report in plane_reports.values():
        for plane, plane_report in variant_report.items():
            teacher_signal = target_report["separation"]["separation_signal"]
            plane_signal = plane_report["separation"]["separation_signal"]
            plane_report["relative_complex_separation_to_teacher"] = float(plane_signal["complex"] or 0.0) / max(float(teacher_signal["complex"] or 0.0), 1e-12)
    report.update({"teacher_target": target_report, "planes": plane_reports, "target_metrics": target_metrics, "variant_sensitivity": sensitivities, "spectral_audit": spectra})
    report["decision"] = _decision(target_report, plane_reports, sensitivities, target_metrics, spectra)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"class_labels": labels, "teacher_target": target, **{name: {key: value for key, value in variant.items()} for name, variant in variants.items()}}, paths["output"].with_suffix(".pt"))
    report["preview"] = str(_render_preview(paths["output"], target, variants, labels))
    paths["output"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v58-report", type=Path, default=DEFAULT_V58_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=128)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result.get("status"), "decision": result.get("decision"), "reasons": result.get("reasons", [])}, sort_keys=True))
    raise SystemExit(0 if result.get("status") != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
