#!/usr/bin/env python3
"""Zero-training semantic-band carriage audit for the frozen v60 student.

The audit keeps the formal optical forward unchanged and only applies native
sampling Fourier masks to detector-normalized CMOS intensity for offline,
diagnostic band attribution.  The band probes are held-out software upper
bounds; they are not an inference path or a promotion gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ldm_project import OneShotOpticalGenerator, load_one_shot_checkpoint, load_training_config


REPO_ROOT = Path(__file__).resolve().parents[1]
V61_TOOL = REPO_ROOT / "tools/diagnose_v61_offline_decodability_upper_bound.py"
V62_TOOL = REPO_ROOT / "tools/diagnose_v62_native_conditional_control_audit.py"
V63_TOOL = REPO_ROOT / "tools/diagnose_v63_code_conditioning_provenance_audit.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/20260715_v60/runs/"
    "20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/"
    "checkpoints/one_shot_distill_best.pt"
)
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs/train_v56_coherent_relay_from_scratch.yaml"
DEFAULT_REFERENCE_REPORT = REPO_ROOT / "outputs/reports/v58_3_b0_relay_cuda_preflight.json"
DEFAULT_V61_REPORT = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"
DEFAULT_V61_TENSORS = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.pt"
DEFAULT_V63_REPORT = REPO_ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.json"
DEFAULT_V63_TENSORS = REPO_ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.pt"
DEFAULT_DATASET_ROOT = (
    REPO_ROOT
    / "outputs/20260514_unversioned/runs/"
    "20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_"
    "diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts"
)
DEFAULT_DATASETS = {
    "train": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_train.pt",
    "val": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_val.pt",
    "test": DEFAULT_DATASET_ROOT / "teacher_distill_dataset_readable_test.pt",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs/reports/v64_semantic_band_carriage_audit.json"

CAMERA_SHAPE = (200, 200)
CAMERA_PITCH = 2.5e-6
SEMANTIC_CUTOFF_CYCLES_MM = 20.0
MID_CUTOFF_CYCLES_MM = 80.0
TEACHER_LOW_ENERGY_MIN = 0.90
CODE_LOW_ENERGY_MIN = 0.50
PROBE_RETENTION_MIN = 0.90
ROTATION_CI_MIN = 0.0
ROTATION_BOOTSTRAP = 2000
RIDGE_PENALTIES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V61 = _load_module("v61_band_helpers", V61_TOOL)
V62 = _load_module("v62_band_helpers", V62_TOOL)
V63 = _load_module("v63_band_helpers", V63_TOOL)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _invalid_report(inputs: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v64-semantic-band-carriage-audit",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "reasons": reasons,
    }


def _frequency_grid(shape: tuple[int, int] = CAMERA_SHAPE, pitch: float = CAMERA_PITCH) -> torch.Tensor:
    fy = torch.fft.fftshift(torch.fft.fftfreq(shape[0], d=pitch)) / 1000.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(shape[1], d=pitch)) / 1000.0
    return torch.sqrt(fy[:, None].square() + fx[None, :].square())


def frequency_band_masks(
    shape: tuple[int, int] = CAMERA_SHAPE,
    pitch: float = CAMERA_PITCH,
) -> dict[str, torch.Tensor]:
    radius = _frequency_grid(shape, pitch)
    masks = {
        "semantic_0_20_cycles_mm": radius <= SEMANTIC_CUTOFF_CYCLES_MM,
        "mid_20_80_cycles_mm": (radius > SEMANTIC_CUTOFF_CYCLES_MM) & (radius <= MID_CUTOFF_CYCLES_MM),
        "high_above_80_cycles_mm": radius > MID_CUTOFF_CYCLES_MM,
    }
    union = torch.zeros_like(radius, dtype=torch.bool)
    for mask in masks.values():
        if bool((union & mask).any()):
            raise RuntimeError("frequency masks overlap")
        union |= mask
    if not bool(union.all()):
        raise RuntimeError("frequency masks do not cover the native DFT lattice")
    return masks


def apply_frequency_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or tuple(values.shape[-2:]) != CAMERA_SHAPE:
        raise ValueError(f"native camera values must be [N,C,200,200], got {tuple(values.shape)}")
    if tuple(mask.shape) != CAMERA_SHAPE:
        raise ValueError(f"frequency mask must be 200x200, got {tuple(mask.shape)}")
    centered = values.float() - values.float().mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered), dim=(-2, -1))
    filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask.to(spectrum.device), dim=(-2, -1))).real
    return filtered + values.float().mean(dim=(-2, -1), keepdim=True)


def spectral_energy_fractions(values: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict[str, float]:
    if values.ndim != 4 or tuple(values.shape[-2:]) != CAMERA_SHAPE:
        raise ValueError("spectral values must use native 200x200 camera sampling")
    centered = values.float() - values.float().mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered), dim=(-2, -1)).abs().square()
    energy = {name: float(spectrum[..., mask].sum()) for name, mask in masks.items()}
    total = sum(energy.values())
    if total <= 0.0:
        return {name: 0.0 for name in masks}
    return {name: value / total for name, value in energy.items()}


def teacher_cutoff_report(target: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    fractions = spectral_energy_fractions(target.float(), masks)
    low = fractions["semantic_0_20_cycles_mm"]
    return {
        "fractions": fractions,
        "semantic_low_energy_fraction": low,
        "cutoff_valid": low >= TEACHER_LOW_ENERGY_MIN,
        "cutoff_cycles_mm": SEMANTIC_CUTOFF_CYCLES_MM,
        "required_fraction": TEACHER_LOW_ENERGY_MIN,
    }


def _factor_band_report(v63_tensors: Mapping[str, Any], masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    means = v63_tensors["factor_means"]["camera_intensity"]
    global_mean = means["global_mean"]
    reports: dict[str, Any] = {}
    for factor_name in ("code_mean", "label_mean"):
        factor = means[factor_name] - global_mean
        reports[factor_name] = spectral_energy_fractions(factor, masks)
    return reports


def _validate_artifacts(
    *,
    v61_report_path: Path,
    v61_tensors_path: Path,
    v63_report_path: Path,
    v63_tensors_path: Path,
    checkpoint: Path,
    test_payload: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    reasons: list[str] = []
    v61_report = json.loads(v61_report_path.read_text(encoding="utf-8"))
    v63_report = json.loads(v63_report_path.read_text(encoding="utf-8"))
    if v61_report.get("schema_version") != "v61-offline-linear-decodability-upper-bound":
        reasons.append("v61 report schema is invalid")
    if v63_report.get("schema_version") != "v63-code-conditioning-provenance-audit":
        reasons.append("v63 report schema is invalid")
    v61_bundle = torch.load(v61_tensors_path, map_location="cpu", weights_only=False)
    v63_bundle = torch.load(v63_tensors_path, map_location="cpu", weights_only=False)
    expected_labels = test_payload.get("class_labels")
    expected_target = test_payload.get("teacher_camera")
    if not torch.equal(v61_bundle.get("class_labels"), expected_labels):
        reasons.append("v61 test labels do not match readable test dataset")
    if not torch.equal(v61_bundle.get("teacher_target"), expected_target):
        reasons.append("v61 teacher target does not match readable test dataset")
    if not torch.equal(v63_bundle.get("test_class_labels"), expected_labels):
        reasons.append("v63 test labels do not match readable test dataset")
    if v63_bundle.get("schema_version") != "v63-code-conditioning-provenance-audit":
        reasons.append("v63 tensor schema is invalid")
    factors = v63_bundle.get("factor_means", {}).get("camera_intensity", {})
    for key in ("global_mean", "code_mean", "label_mean"):
        if not torch.is_tensor(factors.get(key)) or tuple(factors[key].shape[-2:]) != CAMERA_SHAPE:
            reasons.append(f"v63 camera factor tensor {key} is invalid")
    expected_checkpoint = v61_report.get("inputs", {}).get("checkpoint")
    if expected_checkpoint and Path(expected_checkpoint).resolve() != checkpoint.resolve():
        reasons.append("v61 report does not reference the fixed v60 checkpoint")
    if v63_report.get("decision", {}).get("status") != "native-code-conditioned-subvisual-representation-confirmed":
        reasons.append("v63 report does not contain the required confirmed decision")
    return reasons, v61_report, v63_bundle


def _extract_native_camera(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    config: Any,
    device: torch.device,
    sink: np.memmap,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    codes = payload["codes"]
    labels = payload["class_labels"]
    target = payload["teacher_camera"].detach().cpu().float()
    errors = {"raw_camera_intensity_field_max_error": 0.0, "detector_normalization_max_error": 0.0}
    for start in range(0, int(codes.shape[0]), batch_size):
        end = min(start + batch_size, int(codes.shape[0]))
        with torch.no_grad():
            outputs = model(codes[start:end].to(device), labels[start:end].to(device), return_optical_fields=True)
        camera, batch_errors = V62._validate_forward(outputs, config=config)
        sink[start:end] = camera.flatten(1).numpy()
        for name, value in batch_errors.items():
            errors[name] = max(errors[name], value)
        del outputs, camera
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return target, errors


def _make_band_memmaps(root: Path, split: str, count: int) -> dict[str, np.memmap]:
    masks = ("semantic_0_20_cycles_mm", "nonsemantic_above_20_cycles_mm")
    return {
        name: np.memmap(root / f"{name}_{split}.f32", mode="w+", dtype=np.float32, shape=(count, 200 * 200))
        for name in masks
    }


def _write_band_features(raw: np.memmap, bands: Mapping[str, np.memmap], masks: Mapping[str, torch.Tensor], block: int = 16) -> None:
    low = masks["semantic_0_20_cycles_mm"]
    high = ~low
    for start in range(0, raw.shape[0], block):
        end = min(start + block, raw.shape[0])
        values = torch.from_numpy(np.asarray(raw[start:end]).reshape(-1, 1, 200, 200))
        bands["semantic_0_20_cycles_mm"][start:end] = apply_frequency_mask(values, low).flatten(1).numpy()
        bands["nonsemantic_above_20_cycles_mm"][start:end] = apply_frequency_mask(values, high).flatten(1).numpy()
    for band in bands.values():
        band.flush()


def _rotation_report(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, Any]:
    observed = V61._correlation_per_sample(prediction, target)
    rotated_parts = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            raise ValueError(f"rotation requires at least two samples for class {label}")
        rotated_parts.append((indices, torch.roll(indices, shifts=1, dims=0)))
    rotated_target = target.clone()
    for indices, source in rotated_parts:
        rotated_target[indices] = target[source]
    rotated = V61._correlation_per_sample(prediction, rotated_target)
    delta = observed - rotated
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples = []
    for _ in range(ROTATION_BOOTSTRAP):
        sample = torch.randint(delta.numel(), (delta.numel(),), generator=generator)
        samples.append(delta[sample].mean())
    bootstrap = torch.stack(samples)
    return {
        "paired_corr_mean": float(observed.mean()),
        "rotated_target_corr_mean": float(rotated.mean()),
        "paired_corr_drop": float(delta.mean()),
        "bootstrap_iterations": ROTATION_BOOTSTRAP,
        "bootstrap_seed": seed,
        "ci_low": float(torch.quantile(bootstrap, 0.025)),
        "ci_high": float(torch.quantile(bootstrap, 0.975)),
        "ci_low_positive": bool(float(torch.quantile(bootstrap, 0.025)) > ROTATION_CI_MIN),
    }


def _probe_retained(probe: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    test = probe["test"]
    return (
        float(test["target_corr_mean"]) >= PROBE_RETENTION_MIN * float(baseline["target_corr_mean"])
        and float(test["thresholded_dice_mean"]) >= PROBE_RETENTION_MIN * float(baseline["thresholded_dice_mean"])
        and bool(probe["paired_rotation"]["ci_low_positive"])
    )


def decide(
    *,
    cutoff_valid: bool,
    code_low_fraction: float,
    semantic_probe_retained: bool,
    nonsemantic_probe_retained: bool,
) -> dict[str, Any]:
    if not cutoff_valid:
        return {"status": "harness-invalid", "reason": "teacher semantic cutoff did not reach the pre-registered energy fraction"}
    semantic_carrier = code_low_fraction >= CODE_LOW_ENERGY_MIN
    if semantic_probe_retained and nonsemantic_probe_retained:
        return {"status": "distributed-subvisual-carrier", "semantic_carrier": semantic_carrier, "semantic_probe_retained": True, "nonsemantic_probe_retained": True}
    if semantic_carrier and semantic_probe_retained:
        return {"status": "semantic-band-carrier-retained", "semantic_carrier": True, "semantic_probe_retained": True, "nonsemantic_probe_retained": False}
    if nonsemantic_probe_retained:
        return {"status": "out-of-semantic-band-subvisual-carrier-candidate", "semantic_carrier": semantic_carrier, "semantic_probe_retained": semantic_probe_retained, "nonsemantic_probe_retained": True}
    return {"status": "distributed-subvisual-carrier", "semantic_carrier": semantic_carrier, "semantic_probe_retained": semantic_probe_retained, "nonsemantic_probe_retained": False}


def _render_preview(output: Path, target: torch.Tensor, raw: torch.Tensor, low: torch.Tensor, high: torch.Tensor, labels: torch.Tensor) -> Path:
    import matplotlib.pyplot as plt

    selected = []
    for label in range(10):
        selected.append(int(torch.nonzero(labels == label, as_tuple=False).flatten()[0]))
    figure, axes = plt.subplots(4, 10, figsize=(16, 7), squeeze=False)
    rows = (("teacher", target), ("raw_cmos", raw), ("semantic_0_20", low), ("nonsemantic", high))
    for row, (name, values) in enumerate(rows):
        for column, index in enumerate(selected):
            image = values[index, 0].float().clamp(0.0, 1.0)
            axes[row, column].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axes[row, column].set_title(f"{name}\nlabel {int(labels[index])}", fontsize=7)
            axes[row, column].axis("off")
    figure.suptitle("v64 native CMOS semantic-band audit — diagnostic only")
    figure.tight_layout()
    preview = output.with_name(f"{output.stem}_preview.png")
    figure.savefig(preview, dpi=160)
    plt.close(figure)
    return preview


def run(
    *,
    train_config: str | Path = DEFAULT_TRAIN_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    reference_report: str | Path = DEFAULT_REFERENCE_REPORT,
    v61_report: str | Path = DEFAULT_V61_REPORT,
    v61_tensors: str | Path = DEFAULT_V61_TENSORS,
    v63_report: str | Path = DEFAULT_V63_REPORT,
    v63_tensors: str | Path = DEFAULT_V63_TENSORS,
    train_dataset: str | Path = DEFAULT_DATASETS["train"],
    val_dataset: str | Path = DEFAULT_DATASETS["val"],
    test_dataset: str | Path = DEFAULT_DATASETS["test"],
    output: str | Path = DEFAULT_OUTPUT,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    paths = {name: Path(value) for name, value in {
        "train_config": train_config, "checkpoint": checkpoint, "reference_report": reference_report,
        "v61_report": v61_report, "v61_tensors": v61_tensors, "v63_report": v63_report,
        "v63_tensors": v63_tensors, "train_dataset": train_dataset, "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }.items()}
    output_path = Path(output)
    reasons = [f"missing input: {name}={path}" for name, path in paths.items() if not path.is_file()]
    if paths["checkpoint"].resolve() != DEFAULT_CHECKPOINT.resolve():
        reasons.append("v64 accepts only the pre-registered v60 epoch-5 checkpoint")
    if device != "cuda":
        reasons.append("v64 full-geometry diagnostic requires --device cuda")
    if not torch.cuda.is_available():
        reasons.append("CUDA is unavailable")
    if reasons:
        report = _invalid_report(paths, reasons)
        _json_dump(output_path, report)
        return report
    config = load_training_config(paths["train_config"])
    reasons.extend(V62._validate_contract(config))
    reasons.extend(V62._validate_v58_3_report(paths["reference_report"]))
    datasets = {split: V62._load_bundle(paths[f"{split}_dataset"]) for split in ("train", "val", "test")}
    for split, payload in datasets.items():
        reasons.extend(V62._validate_readable_split(payload, split))
    artifact_reasons, v61_report, v63_bundle = _validate_artifacts(
        v61_report_path=paths["v61_report"], v61_tensors_path=paths["v61_tensors"],
        v63_report_path=paths["v63_report"], v63_tensors_path=paths["v63_tensors"],
        checkpoint=paths["checkpoint"], test_payload=datasets["test"],
    )
    reasons.extend(artifact_reasons)
    provenance_reasons, provenance, _ = V63._validated_provenance(datasets)
    reasons.extend(provenance_reasons)
    if reasons:
        report = _invalid_report(paths, reasons)
        _json_dump(output_path, report)
        return report
    masks = frequency_band_masks()
    cutoff = teacher_cutoff_report(datasets["train"]["teacher_camera"], masks)
    band_factor = _factor_band_report(v63_bundle, masks)
    baseline = v61_report["modalities"]["camera_intensity"]["test"]
    if not cutoff["cutoff_valid"]:
        report = _invalid_report(paths, ["teacher semantic cutoff did not reach the pre-registered energy fraction"])
        report["teacher_cutoff"] = cutoff
        _json_dump(output_path, report)
        return report
    report: dict[str, Any] = {
        "schema_version": "v64-semantic-band-carriage-audit",
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in paths.items()},
        "invariants": {"student_training": False, "optimizer_created": False, "parameter_update_count": 0, "formal_contract_changed": False, "native_sampling_only": True, "software_readout_refit": "band probes only; v61 raw probe frozen", "resize_or_interpolation": False},
        "provenance": provenance,
        "frequency_contract": {"camera_shape": list(CAMERA_SHAPE), "camera_pitch_um": CAMERA_PITCH * 1.0e6, "masks": {name: int(mask.sum()) for name, mask in masks.items()}},
        "teacher_cutoff": cutoff,
        "factor_band_energy": band_factor,
        "baseline_v61_camera_intensity": baseline,
        "pre_registered": {"teacher_low_energy_min": TEACHER_LOW_ENERGY_MIN, "code_low_energy_min": CODE_LOW_ENERGY_MIN, "probe_retention_min": PROBE_RETENTION_MIN, "semantic_cutoff_cycles_mm": SEMANTIC_CUTOFF_CYCLES_MM, "mid_cutoff_cycles_mm": MID_CUTOFF_CYCLES_MM},
        "reasons": [],
    }
    _json_dump(output_path, report)
    resolved_device = torch.device("cuda")
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(resolved_device)
    load_one_shot_checkpoint(model, torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=False))
    model.eval()
    before = V62._state_snapshot(model)
    try:
        with tempfile.TemporaryDirectory(prefix="v64_native_camera_") as temp:
            root = Path(temp)
            raw_sinks: dict[str, np.memmap] = {}
            band_sinks: dict[str, dict[str, np.memmap]] = {}
            targets: dict[str, torch.Tensor] = {}
            labels: dict[str, torch.Tensor] = {}
            semantic_errors = {"raw_camera_intensity_field_max_error": 0.0, "detector_normalization_max_error": 0.0}
            for split, payload in datasets.items():
                count = int(payload["codes"].shape[0])
                raw_sinks[split] = np.memmap(root / f"raw_{split}.f32", mode="w+", dtype=np.float32, shape=(count, 200 * 200))
                band_sinks[split] = _make_band_memmaps(root, split, count)
                targets[split], errors = _extract_native_camera(model, payload, config=config, device=resolved_device, sink=raw_sinks[split], batch_size=batch_size)
                labels[split] = payload["class_labels"].detach().cpu().long()
                for name, value in errors.items():
                    semantic_errors[name] = max(semantic_errors[name], value)
                _write_band_features(raw_sinks[split], band_sinks[split], masks)
            report["field_semantic_checks"] = semantic_errors
            changed = V62._state_changes(model, before)
            report["invariants"]["parameter_update_count"] = len(changed)
            if changed:
                report["status"] = "harness-invalid"
                report["reasons"].append(f"model parameters changed during diagnostic: {changed[:5]}")
                return report
            model.cpu()
            del model
            torch.cuda.empty_cache()
            probe_reports: dict[str, Any] = {}
            predictions: dict[str, torch.Tensor] = {}
            for band_name in ("semantic_0_20_cycles_mm", "nonsemantic_above_20_cycles_mm"):
                features = {split: band_sinks[split][band_name] for split in datasets}
                probe, prediction = V61._decode_native_memmap_modality(features, targets, labels, compute_device=resolved_device)
                probe["paired_rotation"] = _rotation_report(prediction, targets["test"], labels["test"], seed=6400 + len(probe_reports))
                probe_reports[band_name] = probe
                predictions[band_name] = prediction
                torch.cuda.empty_cache()
            report["band_probes"] = probe_reports
            code_low = band_factor["code_mean"]["semantic_0_20_cycles_mm"]
            semantic_retained = _probe_retained(probe_reports["semantic_0_20_cycles_mm"], baseline)
            nonsemantic_retained = _probe_retained(probe_reports["nonsemantic_above_20_cycles_mm"], baseline)
            report["decision"] = decide(cutoff_valid=bool(cutoff["cutoff_valid"]), code_low_fraction=code_low, semantic_probe_retained=semantic_retained, nonsemantic_probe_retained=nonsemantic_retained)
            raw_test = torch.from_numpy(np.asarray(raw_sinks["test"])).reshape(-1, 1, 200, 200)
            low_test = apply_frequency_mask(raw_test, masks["semantic_0_20_cycles_mm"])
            high_test = apply_frequency_mask(raw_test, ~masks["semantic_0_20_cycles_mm"])
            preview = _render_preview(output_path, targets["test"], raw_test, low_test, high_test, labels["test"])
            tensor_path = output_path.with_suffix(".pt")
            torch.save({"schema_version": report["schema_version"], "diagnostic_only": True, "accepted_summary_written": False, "class_labels": labels["test"], "teacher_target": targets["test"], "raw_camera": raw_test, "semantic_camera": low_test, "nonsemantic_camera": high_test}, tensor_path)
            report["artifacts"] = {"tensors": str(tensor_path), "preview": str(preview)}
    except (RuntimeError, ValueError, OSError) as exc:
        report["status"] = "harness-invalid"
        report["reasons"].append(str(exc))
    finally:
        _json_dump(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("train-config", DEFAULT_TRAIN_CONFIG), ("checkpoint", DEFAULT_CHECKPOINT), ("reference-report", DEFAULT_REFERENCE_REPORT), ("v61-report", DEFAULT_V61_REPORT), ("v61-tensors", DEFAULT_V61_TENSORS), ("v63-report", DEFAULT_V63_REPORT), ("v63-tensors", DEFAULT_V63_TENSORS), ("train-dataset", DEFAULT_DATASETS["train"]), ("val-dataset", DEFAULT_DATASETS["val"]), ("test-dataset", DEFAULT_DATASETS["test"]), ("output", DEFAULT_OUTPUT)):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "reasons": result.get("reasons", [])}, sort_keys=True))
    raise SystemExit(0 if result["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
