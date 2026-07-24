#!/usr/bin/env python3
"""Zero-training common-mode and readout-locality attribution audit.

All observers in this module are offline diagnostics over native CMOS
intensity.  The frozen formal optical forward is never changed and no
observer is a deployment or promotion path.
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
import torch.nn.functional as F

from ldm_project import (
    OneShotOpticalGenerator,
    load_one_shot_checkpoint,
    load_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
V61_PATH = ROOT / "tools/diagnose_v61_offline_decodability_upper_bound.py"
V62_PATH = ROOT / "tools/diagnose_v62_native_conditional_control_audit.py"
V63_PATH = ROOT / "tools/diagnose_v63_code_conditioning_provenance_audit.py"
V64_PATH = ROOT / "tools/diagnose_v64_semantic_band_carriage_audit.py"
CHECKPOINT = (
    ROOT
    / "outputs/20260715_v60/runs/20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/checkpoints/one_shot_distill_best.pt"
)
CONFIG = ROOT / "configs/train_v56_coherent_relay_from_scratch.yaml"
REFERENCE = ROOT / "outputs/reports/v58_3_b0_relay_cuda_preflight.json"
V61_REPORT = ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"
V61_TENSORS = ROOT / "outputs/reports/v61_offline_decodability_upper_bound.pt"
V63_REPORT = ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.json"
V63_TENSORS = ROOT / "outputs/reports/v63_code_conditioning_provenance_audit.pt"
V64_REPORT = ROOT / "outputs/reports/v64_semantic_band_carriage_audit.json"
V64_TENSORS = ROOT / "outputs/reports/v64_semantic_band_carriage_audit.pt"
DATA_ROOT = (
    ROOT
    / "outputs/20260514_unversioned/runs/20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts"
)
DATASETS = {
    name: DATA_ROOT / f"teacher_distill_dataset_readable_{name}.pt"
    for name in ("train", "val", "test")
}
OUTPUT = ROOT / "outputs/reports/v65_common_mode_readout_locality_audit.json"

PENALTIES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
KERNEL_SIZES = (3, 7, 15, 31)
RETENTION = 0.90
BOOTSTRAP = 2000


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V61 = _module("v65_v61", V61_PATH)
V62 = _module("v65_v62", V62_PATH)
V63 = _module("v65_v63", V63_PATH)
V64 = _module("v65_v64", V64_PATH)


def _dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _invalid(inputs: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v65-common-mode-readout-locality-audit",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {k: str(v) for k, v in inputs.items()},
        "reasons": reasons,
    }


def fixed_common_mode_display(
    train: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if (
        train.ndim != 4
        or values.ndim != 4
        or tuple(train.shape[-2:]) != (200, 200)
        or tuple(values.shape[-2:]) != (200, 200)
    ):
        raise ValueError("common-mode observer requires native [N,1,200,200] tensors")
    mean = train.float().mean(0, keepdim=True)
    residual = train.float() - mean
    # ``torch.quantile`` has an implementation limit for very large flattened
    # tensors (the native 800 x 200 x 200 train split is ~32M values).  The
    # feature tensor is already materialized on CPU by the read-only extractor,
    # so use NumPy's deterministic selection implementation without changing
    # the statistic or the formal optical path.
    residual_abs = residual.abs().detach().cpu().numpy().reshape(-1)
    scale = float(max(np.quantile(residual_abs, 0.995), 1.0e-8))
    gain = 0.45 / scale
    display = (0.5 + gain * (values.float() - mean)).clamp(0.0, 1.0)
    return display, {
        "train_common_mean_rms": float(mean.square().mean().sqrt()),
        "train_residual_abs_q995": scale,
        "global_gain": float(gain),
    }


def _pointwise_fit(
    train: torch.Tensor,
    target: torch.Tensor,
    val: torch.Tensor,
    val_target: torch.Tensor,
    penalties: tuple[float, ...] = PENALTIES,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    x = train.flatten(1).float()
    y = target.flatten(1).float()
    xv = val.flatten(1).float()
    yt = val_target.flatten(1).float()
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    var = xc.square().mean(0)
    cov = (xc * yc).mean(0)
    validation = []
    for penalty in penalties:
        pred = (xv - xm) * cov / (var + penalty * var.mean().clamp_min(1.0e-8)) + ym
        validation.append(
            {"penalty": penalty, "raw_mse_mean": float((pred - yt).square().mean())}
        )
    selected = min(validation, key=lambda item: item["raw_mse_mean"])["penalty"]
    all_x, all_y = (
        torch.cat((train, val)).flatten(1).float(),
        torch.cat((target, val_target)).flatten(1).float(),
    )
    am, bm = all_x.mean(0), all_y.mean(0)
    ac, bc = all_x - am, all_y - bm
    slope = (ac * bc).mean(0) / (
        ac.square().mean(0) + selected * ac.square().mean().clamp_min(1.0e-8)
    )
    return {
        "probe": {
            "kind": "held_out_pointwise_affine",
            "selected_penalty": selected,
            "validation": validation,
            "native_sampling": True,
        }
    }, {"mean": am, "target_mean": bm, "slope": slope}


def _pointwise_predict(
    model: Mapping[str, torch.Tensor], values: torch.Tensor
) -> torch.Tensor:
    return (
        (values.flatten(1).float() - model["mean"]) * model["slope"]
        + model["target_mean"]
    ).reshape(-1, 1, 200, 200)


def _conv_stats(
    values: torch.Tensor, target: torch.Tensor, size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = size // 2
    xtx = torch.zeros((size * size + 1, size * size + 1), device=device)
    xty = torch.zeros((size * size + 1, 1), device=device)
    for start in range(0, values.shape[0], 4):
        image = values[start : start + 4].to(device)
        y = target[start : start + 4, 0].to(device).flatten()
        patches = (
            F.unfold(image, size, padding=radius)
            .transpose(1, 2)
            .reshape(-1, size * size)
        )
        patches = torch.cat(
            (patches, torch.ones((patches.shape[0], 1), device=device)), dim=1
        )
        xtx.add_(patches.T @ patches)
        xty.add_(patches.T @ y[:, None])
    return xtx, xty


def _conv_predict(
    values: torch.Tensor, weight: torch.Tensor, size: int, device: torch.device
) -> torch.Tensor:
    radius = size // 2
    outputs = []
    for start in range(0, values.shape[0], 4):
        image = values[start : start + 4].to(device)
        patches = F.unfold(image, size, padding=radius).transpose(1, 2)
        pred = patches @ weight[:-1] + weight[-1]
        outputs.append(pred.reshape(-1, 1, 200, 200).detach().cpu())
    return torch.cat(outputs)


def _conv_fit(
    train: torch.Tensor,
    target: torch.Tensor,
    val: torch.Tensor,
    val_target: torch.Tensor,
    size: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    xtx, xty = _conv_stats(train, target, size, device)
    validation = []
    for penalty in PENALTIES:
        system = xtx + penalty * torch.eye(xtx.shape[0], device=device)
        weight = torch.linalg.solve(system, xty).flatten()
        pred = _conv_predict(val, weight, size, device)
        validation.append(
            {
                "penalty": penalty,
                "raw_mse_mean": float((pred - val_target).square().mean()),
            }
        )
    selected = min(validation, key=lambda item: item["raw_mse_mean"])["penalty"]
    all_x, all_y = torch.cat((train, val)), torch.cat((target, val_target))
    xtx, xty = _conv_stats(all_x, all_y, size, device)
    weight = (
        torch.linalg.solve(xtx + selected * torch.eye(xtx.shape[0], device=device), xty)
        .flatten()
        .detach()
        .cpu()
    )
    return {
        "probe": {
            "kind": "held_out_shared_local_convolution",
            "kernel_size": size,
            "selected_penalty": selected,
            "validation": validation,
            "padding": "zero",
        },
    }, weight


def _fourier_fit(
    train: torch.Tensor,
    target: torch.Tensor,
    val: torch.Tensor,
    val_target: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    def stats(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        numerator = torch.zeros((200, 200), dtype=torch.complex64, device=device)
        power = torch.zeros((200, 200), dtype=torch.float32, device=device)
        for start in range(0, x.shape[0], 8):
            xf = torch.fft.fft2(x[start : start + 8, 0].to(device))
            yf = torch.fft.fft2(y[start : start + 8, 0].to(device))
            numerator += (xf.conj() * yf).sum(0)
            power += xf.abs().square().sum(0)
        return numerator, power

    num, power = stats(train, target)
    validation = []
    scale = power.mean().clamp_min(1.0e-8)
    for penalty in PENALTIES:
        h = num / (power + penalty * scale)
        pred = []
        for start in range(0, val.shape[0], 8):
            pred.append(
                torch.fft.ifft2(
                    torch.fft.fft2(val[start : start + 8, 0].to(device)) * h
                ).real.cpu()
            )
        prediction = torch.cat(pred)[:, None]
        validation.append(
            {
                "penalty": penalty,
                "raw_mse_mean": float((prediction - val_target).square().mean()),
            }
        )
    selected = min(validation, key=lambda item: item["raw_mse_mean"])["penalty"]
    all_num, all_power = stats(torch.cat((train, val)), torch.cat((target, val_target)))
    h = (
        (all_num / (all_power + selected * all_power.mean().clamp_min(1.0e-8)))
        .detach()
        .cpu()
    )
    return {
        "probe": {
            "kind": "held_out_fourier_diagonal",
            "selected_penalty": selected,
            "validation": validation,
            "support": "full translation-equivariant",
        },
    }, h


def _fourier_predict(
    values: torch.Tensor, transfer: torch.Tensor, device: torch.device
) -> torch.Tensor:
    output = []
    for start in range(0, values.shape[0], 8):
        output.append(
            torch.fft.ifft2(
                torch.fft.fft2(values[start : start + 8, 0].to(device))
                * transfer.to(device)
            ).real.cpu()
        )
    return torch.cat(output)[:, None]


def _metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    name: str,
    prediction_kind: str,
) -> dict[str, Any]:
    result = V61._summarize_reconstruction(pred, target, labels, prototypes)
    result["observer"] = name
    result["prediction_kind"] = prediction_kind
    result["paired_rotation"] = V64._rotation_report(
        pred, target, labels, seed=6500 + sum(ord(char) for char in name) % 100
    )
    return result


def _retained(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return (
        float(metrics["target_corr_mean"])
        >= RETENTION * float(baseline["target_corr_mean"])
        and float(metrics["thresholded_dice_mean"])
        >= RETENTION * float(baseline["thresholded_dice_mean"])
        and bool(metrics["paired_rotation"]["ci_low_positive"])
    )


def decide(results: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    common = results["common_mode_display"]
    if (
        float(common["target_corr_mean"]) >= 0.78
        and float(common["thresholded_dice_mean"]) >= 0.72
        and float(common["prototype_label_accuracy"]) >= 0.70
    ):
        return {"status": "common-mode-display-limited", "human_review_required": True}
    if _retained(results["pointwise_affine"], baseline):
        return {"status": "photometric-local-readout-candidate"}
    passing = [
        size
        for size in KERNEL_SIZES
        if _retained(results[f"local_conv_{size}"], baseline)
    ]
    if passing:
        return {
            "status": "local-spatial-readout-mismatch-candidate",
            "smallest_kernel_size": min(passing),
        }
    if _retained(results["fourier_wiener"], baseline):
        return {"status": "translation-equivariant-global-readout-candidate"}
    return {
        "status": "nonlocal-codebook-objective-representation-mismatch-confirmed",
        "dense_reference": "v61 camera-intensity probe remains decodable",
    }


def _validate_inputs(
    paths: Mapping[str, Path], datasets: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    reasons: list[str] = []
    reasons.extend(V62._validate_contract(load_training_config(paths["config"])))
    reasons.extend(V62._validate_v58_3_report(paths["reference"]))
    for split, payload in datasets.items():
        reasons.extend(V62._validate_readable_split(payload, split))
    reasons.extend(V63._validated_provenance(datasets)[0])
    v61_report = json.loads(paths["v61_report"].read_text(encoding="utf-8"))
    v63_report = json.loads(paths["v63_report"].read_text(encoding="utf-8"))
    v64_report = json.loads(paths["v64_report"].read_text(encoding="utf-8"))
    if v61_report.get("decision", {}).get("status") != "camera-intensity-decodable":
        reasons.append("v61 raw camera reference is not valid")
    if (
        v63_report.get("decision", {}).get("status")
        != "native-code-conditioned-subvisual-representation-confirmed"
    ):
        reasons.append("v63 provenance decision is not valid")
    if v64_report.get("decision", {}).get("status") != "distributed-subvisual-carrier":
        reasons.append("v64 semantic-band decision is not valid")
    v61_bundle = torch.load(
        paths["v61_tensors"], map_location="cpu", weights_only=False
    )
    v64_bundle = torch.load(
        paths["v64_tensors"], map_location="cpu", weights_only=False
    )
    if not torch.equal(v61_bundle["class_labels"], datasets["test"]["class_labels"]):
        reasons.append("v61 labels differ from test dataset")
    if not torch.equal(
        v61_bundle["teacher_target"], datasets["test"]["teacher_camera"]
    ):
        reasons.append("v61 teacher target differs from test dataset")
    if not torch.equal(v64_bundle["class_labels"], datasets["test"]["class_labels"]):
        reasons.append("v64 labels differ from test dataset")
    return reasons, v61_report, v64_bundle


def _preview(
    path: Path,
    target: torch.Tensor,
    observers: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> Path:
    import matplotlib.pyplot as plt

    chosen = [
        int(torch.nonzero(labels == k, as_tuple=False).flatten()[0]) for k in range(10)
    ]
    rows = [("teacher", target)] + list(observers.items())
    fig, axes = plt.subplots(
        len(rows), 10, figsize=(16, 2.2 * len(rows)), squeeze=False
    )
    for i, (name, values) in enumerate(rows):
        for j, index in enumerate(chosen):
            axes[i, j].imshow(values[index, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{name}\nlabel {int(labels[index])}", fontsize=7)
            axes[i, j].axis("off")
    fig.suptitle("v65 common-mode/readout locality audit — diagnostic only")
    fig.tight_layout()
    out = path.with_name(f"{path.stem}_preview.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run(
    *, device: str = "cuda", batch_size: int = 4, output: str | Path = OUTPUT
) -> dict[str, Any]:
    paths = {
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "reference": REFERENCE,
        "v61_report": V61_REPORT,
        "v61_tensors": V61_TENSORS,
        "v63_report": V63_REPORT,
        "v63_tensors": V63_TENSORS,
        "v64_report": V64_REPORT,
        "v64_tensors": V64_TENSORS,
        **{f"{name}_dataset": value for name, value in DATASETS.items()},
    }
    output = Path(output)
    reasons = [
        f"missing input: {name}" for name, path in paths.items() if not path.is_file()
    ]
    if device != "cuda":
        reasons.append("v65 requires CUDA")
    if not torch.cuda.is_available():
        reasons.append("CUDA unavailable")
    if reasons:
        report = _invalid(paths, reasons)
        _dump(output, report)
        return report
    datasets = {name: V62._load_bundle(path) for name, path in DATASETS.items()}
    reasons, v61_report, v64_bundle = _validate_inputs(paths, datasets)
    if reasons:
        report = _invalid(paths, reasons)
        _dump(output, report)
        return report
    report: dict[str, Any] = {
        "schema_version": "v65-common-mode-readout-locality-audit",
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {k: str(v) for k, v in paths.items()},
        "invariants": {
            "student_training": False,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "formal_contract_changed": False,
            "native_camera_sampling": "200x200 @ 2.5 um",
            "target_code_or_bridge": "disabled",
            "feature_storage": "temporary native float32 memmap; deleted after run",
            "v61_raw_probe_refit": False,
        },
        "baseline_v61": v61_report["modalities"]["camera_intensity"]["test"],
        "reasons": [],
    }
    _dump(output, report)
    config = load_training_config(CONFIG)
    model = (
        OneShotOpticalGenerator(
            config.physics, config.one_shot, image_size=config.first_stage.image_size
        )
        .cuda()
        .eval()
    )
    load_one_shot_checkpoint(
        model, torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    )
    before = V62._state_snapshot(model)
    try:
        with tempfile.TemporaryDirectory(prefix="v65_native_camera_") as temp:
            raw: dict[str, np.memmap] = {}
            targets: dict[str, torch.Tensor] = {}
            labels: dict[str, torch.Tensor] = {}
            for split, payload in datasets.items():
                count = int(payload["codes"].shape[0])
                raw[split] = np.memmap(
                    Path(temp) / f"raw_{split}.f32",
                    mode="w+",
                    dtype=np.float32,
                    shape=(count, 40000),
                )
                targets[split], _ = V64._extract_native_camera(
                    model,
                    payload,
                    config=config,
                    device=torch.device("cuda"),
                    sink=raw[split],
                    batch_size=batch_size,
                )
                labels[split] = payload["class_labels"].cpu().long()
            changed = V62._state_changes(model, before)
            if changed:
                report["status"] = "harness-invalid"
                report["reasons"].append(f"model parameters changed: {changed[:5]}")
                return report
            model.cpu()
            del model
            torch.cuda.empty_cache()
            tensors = {
                split: torch.from_numpy(np.asarray(raw[split]))
                .reshape(-1, 1, 200, 200)
                .float()
                for split in raw
            }
            common, common_params = fixed_common_mode_display(
                tensors["train"], tensors["test"]
            )
            prototypes = V61._teacher_prototypes(
                torch.cat((targets["train"], targets["val"])),
                torch.cat((labels["train"], labels["val"])),
            )
            results: dict[str, Any] = {
                "common_mode_display": _metrics(
                    common,
                    targets["test"],
                    labels["test"],
                    prototypes,
                    name="common_mode_display",
                    prediction_kind="fixed_train_mean_and_gain",
                )
            }
            predictions = {"raw_cmos": tensors["test"], "common_mode_display": common}
            point_report, point_model = _pointwise_fit(
                tensors["train"], targets["train"], tensors["val"], targets["val"]
            )
            point_pred = _pointwise_predict(point_model, tensors["test"])
            results["pointwise_affine"] = {
                **point_report,
                **_metrics(
                    point_pred,
                    targets["test"],
                    labels["test"],
                    prototypes,
                    name="pointwise_affine",
                    prediction_kind="train_pixelwise_affine",
                ),
            }
            predictions["pointwise_affine"] = point_pred
            for size in KERNEL_SIZES:
                kernel_report, weight = _conv_fit(
                    tensors["train"],
                    targets["train"],
                    tensors["val"],
                    targets["val"],
                    size,
                    torch.device("cuda"),
                )
                pred = _conv_predict(
                    tensors["test"], weight.to("cuda"), size, torch.device("cuda")
                )
                results[f"local_conv_{size}"] = {
                    **kernel_report,
                    **_metrics(
                        pred,
                        targets["test"],
                        labels["test"],
                        prototypes,
                        name=f"local_conv_{size}",
                        prediction_kind="shared_zero_padded_kernel",
                    ),
                }
                predictions[f"local_conv_{size}"] = pred
            fourier_report, transfer = _fourier_fit(
                tensors["train"],
                targets["train"],
                tensors["val"],
                targets["val"],
                torch.device("cuda"),
            )
            fourier_pred = _fourier_predict(
                tensors["test"], transfer, torch.device("cuda")
            )
            results["fourier_wiener"] = {
                **fourier_report,
                **_metrics(
                    fourier_pred,
                    targets["test"],
                    labels["test"],
                    prototypes,
                    name="fourier_wiener",
                    prediction_kind="full_support_frequency_diagonal",
                ),
            }
            predictions["fourier_wiener"] = fourier_pred
            report["common_mode_parameters"] = common_params
            report["observers"] = results
            report["decision"] = decide(results, report["baseline_v61"])
            preview = _preview(output, targets["test"], predictions, labels["test"])
            tensor_path = output.with_suffix(".pt")
            torch.save(
                {
                    "schema_version": report["schema_version"],
                    "diagnostic_only": True,
                    "accepted_summary_written": False,
                    "class_labels": labels["test"],
                    "teacher_target": targets["test"],
                    "raw_camera": tensors["test"],
                    "common_mode_display": common,
                    "observer_predictions": {
                        k: v for k, v in predictions.items() if k != "raw_cmos"
                    },
                },
                tensor_path,
            )
            report["artifacts"] = {"tensors": str(tensor_path), "preview": str(preview)}
    except (RuntimeError, ValueError, OSError) as exc:
        report["status"] = "harness-invalid"
        report["reasons"].append(str(exc))
    finally:
        _dump(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(**vars(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result.get("decision"),
                "reasons": result.get("reasons", []),
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
