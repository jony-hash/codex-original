#!/usr/bin/env python3
"""v9.8 全类别受控复验主脚本.

流程:
  Phase 1: 构建 diagnostic checkpoint (v8.2c generator + identity D2NN)
           — 使用 relay_identity front/back 消除传播畸变，只测 D2NN identity
  Phase 2: 运行 4 组 PSF 变体的 distill eval (100 样本, 每类 10)
           — 通过动态切换 front/back path model 实现正确 PSF 模拟
  Phase 3: 输出对比表，评估 decision gates
  Phase 4: 可选 —— 如果 gates 通过，运行 3-epoch controlled training

用法:
  python tools/run_v9_8_controlled_reverification.py \
      --v8-2c-checkpoint <path> \
      --distill-dataset <path> \
      [--output-dir /tmp/v9_8_reverification] \
      [--device cuda] \
      [--run-training]
"""

from __future__ import annotations

import json
import time
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
from ldm_project.one_shot_model import OneShotOpticalGenerator, ONE_SHOT_MODEL_VERSION
from ldm_project.one_shot_eval_metrics import (
    STRICT_STUDENT_READABILITY_THRESHOLDS,
    compute_one_shot_distill_eval_metrics,
    add_student_label_audit_metrics,
    compute_strict_student_readability_acceptance,
)
from ldm_project.one_shot_distill_data import _load_distill_payload
from ldm_project.teacher_distill import (
    _build_mnist_label_prototype_bank,
    _teacher_label_audit_json,
    compute_teacher_distill_label_audit_batched,
)

# ─────────────────────────────────────────────
# Phase 1: Build diagnostic checkpoint
# ─────────────────────────────────────────────

def build_diagnostic_checkpoint(
    v8_2c_checkpoint: str,
    output_path: str,
    device: torch.device,
    seed: int = 1234,
) -> OneShotOpticalGenerator:
    """构建并保存 diagnostic checkpoint，返回 model 实例.

    使用 relay_identity front/back 以消除自由空间传播畸变，
    确保 D2NN identity 时 camera = SLM command 的直接成像.
    """
    physics = PhysicsConfig(
        front_path_model="relay_identity",
        back_path_model="relay_identity",
    )
    physics.front_relay_psf_sigma_px = 0.0
    physics.readout.relay_psf_sigma_px = 0.0

    one_shot_cfg = OneShotConfig(
        code_dim=128,
        label_embed_dim=32,
        hidden_dim=256,
        num_classes=10,
        generator_e_arch="mlp",
        optical_core_arch="phase_amplitude_d2nn",
        slm_background_floor=0.05,
        ols_kind="identity",
        metasurface_grid_size=40,
        d2nn=D2NNConfig(
            wavelength=physics.optics.wavelength,
            pixel_size=physics.optics.pixel_size,
            grid_size=40,
            propagation_grid_size=1024,
            encoder_layers=1,
            decoder_layers=1,
            layer_spacing=0.005,
            modulation="phase_amplitude",
            phase_init="small",
            amplitude_logit_init=4.0,
        ),
        distill_target_source="teacher_camera",
        eval_samples_per_class=20,
        eval_preview_per_class=2,
    )
    print("Building OneShotOpticalGenerator with v9.8 D2NN config (relay_identity)...")
    model = OneShotOpticalGenerator(physics, one_shot_cfg, image_size=200).to(device)
    model.eval()

    # Load v8.2c generator_e.* warm-start
    payload = torch.load(v8_2c_checkpoint, map_location=device, weights_only=False)
    ckpt_state = payload["model_state_dict"]
    generator_keys = {k: v for k, v in ckpt_state.items() if k.startswith("generator_e.")}
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in generator_keys.items():
        target = model_state.get(k)
        if target is not None and tuple(target.shape) == tuple(v.shape):
            target.copy_(v.to(device=target.device, dtype=target.dtype))
            loaded += 1
        else:
            skipped += 1
            print(f"  SKIPPED: {k}")
    print(f"Generator loaded: {loaded}, skipped: {skipped}")

    # Init D2NN as identity
    with torch.no_grad():
        for name in ("encoder", "decoder"):
            stack = getattr(model, name, None)
            if stack is not None and hasattr(stack, "phase"):
                stack.phase.zero_()
                if hasattr(stack, "amplitude_logits"):
                    stack.amplitude_logits.fill_(20.0)
                print(f"  {name}: identity init (phase=0, amp≈1)")

    checkpoint = {
        "model_version": ONE_SHOT_MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "config": {
            "optical_core_arch": "phase_amplitude_d2nn",
            "generator_e_arch": "mlp",
            "code_dim": 128,
            "num_classes": 10,
            "d2nn_layers": 1,
            "front_path_model": "relay_identity",
            "back_path_model": "relay_identity",
            "seed": seed,
        },
        "generator_warm_start_summary": {"loaded": loaded, "skipped": skipped},
        "description": "v9.8 diagnostic: v8.2c generator_e + identity D2NN (relay_identity front/back)",
    }
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path_p)
    print(f"Saved diagnostic checkpoint: {output_path_p}")
    return model


# ─────────────────────────────────────────────
# Phase 2: Run 4 PSF variants
# ─────────────────────────────────────────────


def _make_training_config(one_shot_cfg: OneShotConfig) -> TrainingConfig:
    """为 label audit 构造最小 TrainingConfig."""
    return TrainingConfig(
        one_shot=one_shot_cfg,
        data=DataConfig(root="data"),
        first_stage=FirstStageConfig(image_size=200),
        distillation=DistillationConfig(
            label_audit_prototypes_per_class=20,
            label_audit_batch_size=50,
        ),
    )


def _set_front_psf(model: OneShotOpticalGenerator, sigma: float) -> None:
    """动态切换 front path model.

    sigma=0 → relay_identity (ideal imaging, no PSF)
    sigma>0 → relay_coherent_psf (coherent Gaussian blur on SLM field)
    """
    if sigma <= 1e-6:
        model.front_path.front_path_model = "relay_identity"
    else:
        model.front_path.front_path_model = "relay_coherent_psf"
        model.front_path.physics.front_relay_psf_sigma_px = sigma


def _set_back_psf(model: OneShotOpticalGenerator, sigma: float) -> None:
    """动态切换 back path model.

    sigma=0 → relay_identity (ideal imaging, no PSF)
    sigma>0 → relay_psf (intensity-domain Gaussian blur on chip)
    """
    if sigma <= 1e-6:
        model.back_path.physics.back_path_model = "relay_identity"
    else:
        model.back_path.physics.back_path_model = "relay_psf"
        model.back_path.physics.readout.relay_psf_sigma_px = sigma


def run_psf_variant(
    model: OneShotOpticalGenerator,
    codes: torch.Tensor,
    labels: torch.Tensor,
    teacher_camera: torch.Tensor,
    train_cfg: TrainingConfig,
    device: torch.device,
    *,
    variant_name: str,
    front_sigma: float,
    back_sigma: float,
) -> dict:
    """运行一个 PSF 变体.

    通过动态切换 model.front_path.front_path_model 和
    model.back_path.physics.back_path_model 实现正确的 PSF 模拟:
      - front coherent PSF: relay_coherent_psf (sigma 控制复场高斯模糊)
      - back intensity PSF: relay_psf (sigma 控制强度高斯模糊)
    """
    _set_front_psf(model, front_sigma)
    _set_back_psf(model, back_sigma)

    torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model(codes.to(device), labels.to(device))
    student_camera = outputs["camera_plane"].detach().cpu()
    slm_amplitude = outputs["slm_amplitude"].detach().cpu()

    print(f"  front={model.front_path.front_path_model} "
          f"σ_front={front_sigma}, "
          f"back={model.back_path.physics.back_path_model} "
          f"σ_back={back_sigma}")

    # ── Core metrics ──
    metrics = compute_one_shot_distill_eval_metrics(
        torch,
        teacher_camera=teacher_camera,
        student_camera=student_camera,
        slm_amplitude=slm_amplitude,
        class_labels=labels,
    )
    metrics["variant"] = variant_name
    metrics["psf_config"] = {"front_sigma": front_sigma, "back_sigma": back_sigma}

    # ── Label audit ──
    try:
        prototypes = _build_mnist_label_prototype_bank(
            torch,
            class_labels=tuple(range(10)),
            image_size=200,
            root="data",
            max_per_class=20,
        )
        audit_tensors = compute_teacher_distill_label_audit_batched(
            torch,
            student_camera,
            labels,
            prototypes,
            class_labels=tuple(range(10)),
            batch_size=50,
            device=device,
        )
        audit_json = _teacher_label_audit_json(labels, audit_tensors, class_labels=tuple(range(10)))
        add_student_label_audit_metrics(metrics, audit_json)
    except Exception as e:
        print(f"  Label audit failed: {e}")
        overall = metrics.setdefault("overall", {})
        overall["student_label_correct_ratio"] = 0.0
        overall["student_label_correct_count"] = 0
        overall["student_label_margin_mean"] = 0.0
        overall["min_class_student_label_correct_ratio"] = 0.0
        overall["min_class_student_label_margin_mean"] = 0.0

    # ── Strict readability acceptance ──
    try:
        strict = compute_strict_student_readability_acceptance(metrics)
    except Exception as e:
        print(f"  Strict readability acceptance failed: {e}")
        strict = {
            "passed": False,
            "checks": {},
            "values": {},
            "thresholds": dict(STRICT_STUDENT_READABILITY_THRESHOLDS),
        }
    metrics["strict_readability_acceptance"] = strict

    return metrics


# ─────────────────────────────────────────────
# Phase 3: Comparison & Decision Gates
# ─────────────────────────────────────────────

def print_comparison(all_metrics: dict[str, dict]) -> None:
    """打印 4 组 PSF 变体的关键指标对比表."""
    score_keys = [
        ("paired_corr", "overall"),
        ("paired_dice", "overall"),
        ("min_class_paired_dice", "strict_readability_acceptance.values"),
        ("student_label_correct_ratio", "overall"),
        ("min_class_student_label_correct_ratio", "overall"),
        ("min_class_student_label_margin_mean", "overall"),
        ("background_speckle_ratio", "overall"),
        ("foreground_hole_ratio", "overall"),
        ("student_high_saturation_ratio", "overall"),
        ("readable_contrast", "overall"),
        ("mse", "overall"),
        ("dice", "overall"),
    ]

    header = f"{'Metric':<50}" + "".join(f"{v:<24}" for v in all_metrics)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"{'v9.8 Full-Class Controlled Re-verification Comparison':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for key, section in score_keys:
        vals = []
        for m in all_metrics.values():
            v = m
            for part in section.split("."):
                v = v.get(part, {})
            val = v.get(key, "N/A")
            if isinstance(val, float):
                vals.append(f"{val:<24.6f}")
            else:
                vals.append(f"{str(val):<24}")
        print(f"{key:<50}" + "".join(vals))

    # Pass/fail
    print(sep)
    for m_name, m in all_metrics.items():
        sra = m.get("strict_readability_acceptance", {})
        passed = sra.get("passed", False)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{'Strict Acceptance':<50}{status:<24}")
    print(sep)


def evaluate_decision_gates(all_metrics: dict[str, dict]) -> dict:
    """根据计划中的 decision gates 做出判断."""
    gates = {}

    # Gate 1: sigma0/0 strict pass
    s00 = all_metrics.get("front0_back0", {})
    sra_s00 = s00.get("strict_readability_acceptance", {})
    gates["sigma0_0_strict_pass"] = sra_s00.get("passed", False)

    # Gate 2: sigma1/1 strict pass
    s11 = all_metrics.get("front1_back1", {})
    sra_s11 = s11.get("strict_readability_acceptance", {})
    gates["sigma1_1_strict_pass"] = sra_s11.get("passed", False)

    # Gate 3: sigma1/1 foreground_hole_ratio ≤ 0.10
    hole = s11.get("overall", {}).get("foreground_hole_ratio", 1.0) if s11 else 1.0
    gates["sigma1_1_foreground_hole_acceptable"] = (
        isinstance(hole, (int, float)) and hole <= 0.10
    )
    gates["sigma1_1_foreground_hole_ratio"] = hole

    # Gate 4: sigma1/1 corr/dice drop from sigma0/0 ≤ 0.02
    if s00 and s11:
        corr0 = s00.get("overall", {}).get("paired_corr", 0.0)
        corr1 = s11.get("overall", {}).get("paired_corr", 0.0)
        dice0 = s00.get("overall", {}).get("paired_dice", 0.0)
        dice1 = s11.get("overall", {}).get("paired_dice", 0.0)
        gates["corr_drop"] = corr0 - corr1
        gates["dice_drop"] = dice0 - dice1
        gates["corr_dice_drop_acceptable"] = (
            gates["corr_drop"] <= 0.02 and gates["dice_drop"] <= 0.02
        )
    else:
        gates["corr_dice_drop_acceptable"] = False

    all_pass = all(gates[k] for k in gates if k.endswith("_pass") or k.endswith("_acceptable"))
    gates["all_gates_pass"] = all_pass

    print("\n" + "=" * 70)
    print("Decision Gates Evaluation")
    print("=" * 70)
    for k, v in gates.items():
        status = "✓" if isinstance(v, bool) and v else ("✗" if isinstance(v, bool) else "")
        print(f"  {status} {k}: {v}")
    print(f"\n  >>> All gates {'PASS' if all_pass else 'FAIL'} <<<")

    return gates


# ─────────────────────────────────────────────
# Phase 4: Controlled Training (stub)
# ─────────────────────────────────────────────

def run_controlled_training(
    output_dir: str,
    device: torch.device,
    seed: int = 1234,
) -> None:
    """提示如何运行 controlled training."""
    print("\n" + "=" * 70)
    print("Controlled Training")
    print("=" * 70)
    print(f"To run: train.py --stage one_shot_distill "
          f"--train-config <controlled_config.yaml> --device cuda "
          f"--one-shot-generator-checkpoint {output_dir}/v9_8_diagnostic_checkpoint.pt "
          f"--distill-dataset <dataset> --seed {seed}")
    print("=" * 70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="v9.8 full-class controlled re-verification")
    parser.add_argument("--v8-2c-checkpoint", required=True,
                        help="Path to v8.2c one_shot_distill_best.pt")
    parser.add_argument("--distill-dataset", required=True,
                        help="Path to readable test distill dataset (100 samples, 10/class)")
    parser.add_argument("--output-dir", default="/tmp/v9_8_reverification",
                        help="Output directory for results")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--run-training", action="store_true",
                        help="If gates pass, print controlled training instructions")
    parser.add_argument("--skip-checkpoint-build", action="store_true",
                        help="Skip Phase 1, load existing checkpoint from --output-dir")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_dir / "v9_8_diagnostic_checkpoint.pt")

    model: OneShotOpticalGenerator | None = None

    # Phase 1: Build checkpoint
    if not args.skip_checkpoint_build:
        print("\n" + "=" * 70)
        print("Phase 1: Building diagnostic checkpoint")
        print("=" * 70)
        model = build_diagnostic_checkpoint(
            args.v8_2c_checkpoint,
            checkpoint_path,
            device,
            seed=args.seed,
        )

    # Phase 2: Load dataset
    print("\n" + "=" * 70)
    print("Phase 2: Loading distill dataset & running PSF variants")
    print("=" * 70)

    payload = _load_distill_payload(torch, args.distill_dataset)
    codes = payload["codes"].float()
    labels = payload["class_labels"].long()
    teacher_camera = payload["teacher_camera"].float()
    print(f"Distill dataset keys: {list(payload.keys())}")
    print(f"Loaded distill dataset: {len(labels)} samples")
    counts = tuple((labels == i).sum().item() for i in range(10))
    print(f"Per-class distribution: {counts}")

    # Build model if not already built
    if model is None:
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        physics = PhysicsConfig(
            front_path_model="relay_identity",
            back_path_model="relay_identity",
        )
        physics.front_relay_psf_sigma_px = 0.0
        physics.readout.relay_psf_sigma_px = 0.0
        cfg = OneShotConfig(
            code_dim=128, label_embed_dim=32, hidden_dim=256, num_classes=10,
            generator_e_arch="mlp", optical_core_arch="phase_amplitude_d2nn",
            slm_background_floor=0.05, ols_kind="identity", metasurface_grid_size=40,
            d2nn=D2NNConfig(
                wavelength=physics.optics.wavelength, pixel_size=physics.optics.pixel_size,
                grid_size=40, propagation_grid_size=1024,
                encoder_layers=1, decoder_layers=1, layer_spacing=0.005,
                modulation="phase_amplitude", phase_init="small", amplitude_logit_init=4.0,
            ),
            distill_target_source="teacher_camera",
            eval_samples_per_class=20, eval_preview_per_class=2,
        )
        model = OneShotOpticalGenerator(physics, cfg, image_size=200).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    # Build train_cfg for label audit
    train_cfg = _make_training_config(model.config)

    # PSF variants:
    #   front relay_coherent_psf sigma, back relay_psf sigma
    psf_variants = [
        ("front0_back0", 0.0, 0.0),   # relay_identity + relay_identity
        ("front1_back0", 1.0, 0.0),   # relay_coherent_psf σ=1 + relay_identity
        ("front0_back1", 0.0, 1.0),   # relay_identity + relay_psf σ=1
        ("front1_back1", 1.0, 1.0),   # relay_coherent_psf σ=1 + relay_psf σ=1
    ]

    all_metrics: dict[str, dict] = {}
    for variant_name, front_sigma, back_sigma in psf_variants:
        print(f"\n--- Running variant: {variant_name} "
              f"(front σ={front_sigma}, back σ={back_sigma}) ---")
        t0 = time.time()
        metrics = run_psf_variant(
            model, codes, labels, teacher_camera, train_cfg, device,
            variant_name=variant_name,
            front_sigma=front_sigma,
            back_sigma=back_sigma,
        )
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        ov = metrics.get("overall", {})
        print(f"  corr={ov.get('paired_corr', 'N/A'):.4f} "
              f"dice={ov.get('paired_dice', 'N/A'):.4f} "
              f"label_acc={ov.get('student_label_correct_ratio', 'N/A'):.4f} "
              f"speckle={ov.get('background_speckle_ratio', 'N/A'):.4f} "
              f"hole={ov.get('foreground_hole_ratio', 'N/A'):.4f} "
              f"contrast={ov.get('readable_contrast', 'N/A'):.2f}")
        all_metrics[variant_name] = metrics

    # Save results
    for variant_name, metrics in all_metrics.items():
        m_path = output_dir / f"metrics_{variant_name}.json"
        with open(m_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Saved metrics: {m_path}")

    comparison_path = output_dir / "full_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Phase 3: Comparison & Gates
    print("\n" + "=" * 70)
    print("Phase 3: Comparison & Decision Gates")
    print("=" * 70)
    print_comparison(all_metrics)
    gates = evaluate_decision_gates(all_metrics)

    gates_path = output_dir / "decision_gates.json"
    with open(gates_path, "w") as f:
        json.dump(gates, f, indent=2)
    print(f"\nSaved decision gates: {gates_path}")

    # Phase 4: Controlled Training
    if args.run_training and gates.get("all_gates_pass"):
        print("\n" + "=" * 70)
        print("Phase 4: Gates passed — launching controlled training")
        print("=" * 70)
        run_controlled_training(str(output_dir), device, seed=args.seed)
    elif args.run_training:
        print("\nControlled training skipped: gates did not pass.")

    print(f"\nAll results saved to: {output_dir}")
    print(f"Comparison JSON: {comparison_path}")
    print(f"Decision gates:  {gates_path}")


if __name__ == "__main__":
    main()
