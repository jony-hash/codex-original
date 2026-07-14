#!/usr/bin/env python3
"""Offline sweep of readout calibration parameters on a phase-only D2NN checkpoint.

Loads a saved eval tensor bundle (from one_shot_distill_eval), or runs a fresh
eval, then sweeps:
  - foreground_threshold (for Dice / hole / speckle / contrast)
  - normalize_percentile
  - center crop / readout window (simulated via clamping)

Outputs a JSON summary showing how metrics shift with each parameter, helping
answer the question: is Q's dice=0.644 limited by readout calibration or by
the D2NN phase distribution itself?
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path


import torch

from ldm_project.config import (
    OneShotConfig,
    PhysicsConfig,
    D2NNConfig,
    TrainingConfig,
    DataConfig,
    FirstStageConfig,
    DistillationConfig,
)
from ldm_project.one_shot_model import OneShotOpticalGenerator
from ldm_project.one_shot_eval_metrics import (
    compute_one_shot_distill_eval_metrics,
    add_student_label_audit_metrics,
)
from ldm_project.one_shot_distill_data import _load_distill_payload
from ldm_project.teacher_distill import (
    _build_mnist_label_prototype_bank,
    _teacher_label_audit_json,
    compute_teacher_distill_label_audit_batched,
)


def _load_model(checkpoint_path: str, device: torch.device) -> OneShotOpticalGenerator:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    physics = PhysicsConfig(
        front_path_model="relay_identity",
        back_path_model="relay_identity",
    )
    physics.front_relay_psf_sigma_px = 0.0
    physics.readout.relay_psf_sigma_px = 0.0
    cfg = OneShotConfig(
        code_dim=128, label_embed_dim=32, hidden_dim=256, num_classes=10,
        generator_e_arch="label_template",
        generator_command_transform="intensity_to_amplitude",
        generator_template_init="mnist_prototype",
        optical_core_arch="diffractive_phase_only",
        slm_background_floor=0.05, ols_kind="identity",
        d2nn=D2NNConfig(
            wavelength=5.32e-7, pixel_size=2.0e-6,
            grid_size=100, propagation_grid_size=1024,
            encoder_layers=2, decoder_layers=2, layer_spacing=1.50e-3,
            modulation="phase_only", phase_init="small",
        ),
        distill_target_source="mnist_prototype",
        eval_samples_per_class=10, eval_preview_per_class=2,
    )
    model = OneShotOpticalGenerator(physics, cfg, image_size=200).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _make_train_cfg() -> TrainingConfig:
    return TrainingConfig(
        one_shot=OneShotConfig(
            code_dim=128, label_embed_dim=32, hidden_dim=256, num_classes=10,
            generator_e_arch="label_template",
            optical_core_arch="diffractive_phase_only",
            distill_target_source="mnist_prototype",
        ),
        data=DataConfig(root="data"),
        first_stage=FirstStageConfig(image_size=200),
        distillation=DistillationConfig(
            label_audit_prototypes_per_class=20,
            label_audit_batch_size=50,
        ),
    )


def run_sweep(
    model: OneShotOpticalGenerator,
    codes: torch.Tensor,
    labels: torch.Tensor,
    teacher_camera: torch.Tensor,
    device: torch.device,
) -> dict:
    """Run the model once, then sweep readout calibration parameters."""
    with torch.no_grad():
        outputs = model(codes.to(device), labels.to(device))
    student_camera = outputs["camera_plane"].detach().cpu()
    slm_amplitude = outputs["slm_amplitude"].detach().cpu()

    results: dict[str, object] = {}

    # ── Sweep 1: foreground_threshold ──
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    threshold_results: list[dict] = []
    for thr in thresholds:
        m = compute_one_shot_distill_eval_metrics(
            torch,
            teacher_camera=teacher_camera,
            student_camera=student_camera,
            slm_amplitude=slm_amplitude,
            class_labels=labels,
            foreground_threshold=thr,
        )
        ov = m.get("overall", {})
        threshold_results.append({
            "foreground_threshold": thr,
            "paired_corr": ov.get("paired_corr"),
            "paired_dice": ov.get("paired_dice"),
            "foreground_hole_ratio": ov.get("foreground_hole_ratio"),
            "background_speckle_ratio": ov.get("background_speckle_ratio"),
            "readable_contrast": ov.get("readable_contrast"),
            "foreground_ratio": ov.get("foreground_ratio"),
        })
    results["threshold_sweep"] = threshold_results

    # ── Sweep 2: intensity clamping (simulate normalization / readout window) ──
    # Clamp student camera to [low, high] range before metric computation
    clamp_configs = [
        {"low": 0.0, "high": 1.0, "label": "full_range"},
        {"low": 0.0, "high": 0.95, "label": "cap_0_95"},
        {"low": 0.0, "high": 0.90, "label": "cap_0_90"},
        {"low": 0.05, "high": 1.0, "label": "floor_005"},
        {"low": 0.10, "high": 1.0, "label": "floor_010"},
        {"low": 0.02, "high": 0.98, "label": "tight_002_098"},
        {"low": 0.05, "high": 0.95, "label": "tight_005_095"},
    ]
    clamp_results: list[dict] = []
    for cc in clamp_configs:
        clamped = student_camera.clone().clamp(cc["low"], cc["high"])
        m = compute_one_shot_distill_eval_metrics(
            torch,
            teacher_camera=teacher_camera,
            student_camera=clamped,
            slm_amplitude=slm_amplitude,
            class_labels=labels,
        )
        ov = m.get("overall", {})
        clamp_results.append({
            "clamp_label": cc["label"],
            "low": cc["low"],
            "high": cc["high"],
            "paired_corr": ov.get("paired_corr"),
            "paired_dice": ov.get("paired_dice"),
            "foreground_hole_ratio": ov.get("foreground_hole_ratio"),
            "background_speckle_ratio": ov.get("background_speckle_ratio"),
            "readable_contrast": ov.get("readable_contrast"),
        })
    results["clamp_sweep"] = clamp_results

    # ── Sweep 3: label audit (to see if margin/clamping improves label_acc) ──
    prototypes = _build_mnist_label_prototype_bank(
        torch,
        class_labels=tuple(range(10)),
        image_size=200,
        root="data",
        max_per_class=20,
    )

    # Measure label accuracy on raw student_camera
    base_metrics = compute_one_shot_distill_eval_metrics(
        torch,
        teacher_camera=teacher_camera,
        student_camera=student_camera,
        slm_amplitude=slm_amplitude,
        class_labels=labels,
    )
    audit_tensors = compute_teacher_distill_label_audit_batched(
        torch, student_camera, labels, prototypes,
        class_labels=tuple(range(10)), batch_size=50, device=device,
    )
    audit_json = _teacher_label_audit_json(labels, audit_tensors, class_labels=tuple(range(10)))
    add_student_label_audit_metrics(base_metrics, audit_json)
    ov = base_metrics.get("overall", {})

    label_audit_summary = {
        "student_label_correct_ratio": ov.get("student_label_correct_ratio"),
        "student_label_margin_mean": ov.get("student_label_margin_mean"),
        "min_class_label_correct_ratio": ov.get("min_class_student_label_correct_ratio"),
        "min_class_label_margin_mean": ov.get("min_class_student_label_margin_mean"),
    }
    per_class_labels = {}
    pc = base_metrics.get("per_class", {})
    for lbl in range(10):
        entry = pc.get(str(lbl), {})
        per_class_labels[str(lbl)] = {
            "correct_ratio": entry.get("student_label_correct_ratio"),
            "margin_mean": entry.get("student_label_margin_mean"),
            "confusion": entry.get("student_nearest_label_confusion", {}),
        }
    label_audit_summary["per_class"] = per_class_labels
    results["label_audit"] = label_audit_summary

    # ── Score baseline at standard threshold (0.20) ──
    baseline_m = compute_one_shot_distill_eval_metrics(
        torch,
        teacher_camera=teacher_camera,
        student_camera=student_camera,
        slm_amplitude=slm_amplitude,
        class_labels=labels,
        foreground_threshold=0.20,
    )
    ov = baseline_m.get("overall", {})
    results["baseline_threshold_020"] = {
        "paired_corr": ov.get("paired_corr"),
        "paired_dice": ov.get("paired_dice"),
        "foreground_hole_ratio": ov.get("foreground_hole_ratio"),
        "background_speckle_ratio": ov.get("background_speckle_ratio"),
        "readable_contrast": ov.get("readable_contrast"),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline sweep of phase-only readout calibration")
    parser.add_argument("--checkpoint", required=True, help="Q phase-only checkpoint path")
    parser.add_argument("--distill-dataset", required=True, help="Distill test dataset (100 samples)")
    default_output = (
        Path("outputs/reports")
        / f"twophoton_100x100_phase_only_readout_sweep_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    parser.add_argument("--output", default=str(default_output), help="Output JSON path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model = _load_model(args.checkpoint, device)

    print("Loading distill dataset...")
    payload = _load_distill_payload(torch, args.distill_dataset)
    codes = payload["codes"].float()
    labels = payload["class_labels"].long()
    teacher_camera = payload["teacher_camera"].float()
    print(f"Samples: {len(labels)}, distribution: "
          f"{tuple((labels == i).sum().item() for i in range(10))}")

    print("Running sweep...")
    results = run_sweep(model, codes, labels, teacher_camera, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n=== Baseline (threshold=0.20) ===")
    b = results["baseline_threshold_020"]
    print(f"  corr={b['paired_corr']:.4f}  dice={b['paired_dice']:.4f}  "
          f"hole={b['foreground_hole_ratio']:.4f}  "
          f"contrast={b['readable_contrast']:.2f}")

    print("\n=== Threshold Sweep ===")
    for r in results["threshold_sweep"]:
        print(f"  thr={r['foreground_threshold']:.2f}  corr={r['paired_corr']:.4f}  "
              f"dice={r['paired_dice']:.4f}  hole={r['foreground_hole_ratio']:.4f}  "
              f"contrast={r['readable_contrast']:.2f}")

    print("\n=== Clamp Sweep ===")
    for r in results["clamp_sweep"]:
        print(f"  {r['clamp_label']:<20}  corr={r['paired_corr']:.4f}  "
              f"dice={r['paired_dice']:.4f}  hole={r['foreground_hole_ratio']:.4f}  "
              f"contrast={r['readable_contrast']:.2f}")

    print("\n=== Label Audit ===")
    la = results["label_audit"]
    print(f"  label_acc={la['student_label_correct_ratio']:.4f}  "
          f"margin_mean={la['student_label_margin_mean']:.4f}")
    for lbl, entry in sorted(la["per_class"].items(), key=lambda x: int(x[0])):
        print(f"  class {lbl}: acc={entry['correct_ratio']:.2f}  "
              f"margin={entry['margin_mean']:.4f}  confusion={entry['confusion']}")

    print(f"\nSaved sweep results: {output_path}")


if __name__ == "__main__":
    main()
