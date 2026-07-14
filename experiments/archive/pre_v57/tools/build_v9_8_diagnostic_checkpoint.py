#!/usr/bin/env python3
"""构建 v9.8 全类别受控复验用 diagnostic checkpoint.

加载 v8.2c generator_e.* 参数，将 D2NN 层初始化为 identity（phase=0, amp≈1），
保存为完整 model_state_dict checkpoint，供 one_shot_distill_eval 使用.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ldm_project.config import OneShotConfig, PhysicsConfig, D2NNConfig
from ldm_project.one_shot_model import OneShotOpticalGenerator, ONE_SHOT_MODEL_VERSION


def _load_v8_2c_generator_state(
    model: OneShotOpticalGenerator,
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, int]:
    """从 v8.2c checkpoint 加载 generator_e.* 参数，返回加载统计."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_state = payload["model_state_dict"]
    generator_keys = {k: v for k, v in ckpt_state.items() if k.startswith("generator_e.")}
    print(f"Found {len(generator_keys)} generator_e.* keys in checkpoint:")
    for k, v in generator_keys.items():
        print(f"  {k}: {list(v.shape)}")

    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in generator_keys.items():
        target = model_state.get(k)
        if target is not None and tuple(target.shape) == tuple(v.shape):
            target.copy_(v.to(device=target.device, dtype=target.dtype))
            loaded += 1
        else:
            skipped += 1
            print(f"  SKIPPED (shape mismatch or missing): {k}")
    print(f"generator_e state loaded: {loaded}, skipped: {skipped}")
    return {"loaded": loaded, "skipped": skipped}


def _init_d2nn_identity(model: OneShotOpticalGenerator) -> None:
    """将 D2NN encoder/decoder 初始化为 identity（phase=0, amplitude_logit=20.0 → sigmoid≈1）."""
    with torch.no_grad():
        for name in ("encoder", "decoder"):
            stack = getattr(model, name, None)
            if stack is None or not hasattr(stack, "phase"):
                continue
            stack.phase.zero_()
            print(f"  {name}.phase: zeroed (identity)")
            if hasattr(stack, "amplitude_logits"):
                stack.amplitude_logits.fill_(20.0)
                print(f"  {name}.amplitude_logits: filled 20.0 (transmittance ~1.0)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build v9.8 diagnostic checkpoint")
    parser.add_argument("--v8-2c-checkpoint", required=True,
                        help="Path to v8.2c one_shot_distill_best.pt")
    parser.add_argument("--output", default="outputs/20260620_v9_8/checkpoints/v9_8_diagnostic_checkpoint.pt",
                        help="Output checkpoint path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build v9.8 D2NN config (free_space front/back for identity baseline) ──
    physics = PhysicsConfig(
        front_path_model="relay_identity",
        back_path_model="relay_identity",
    )
    # Set PSF sigma to 0 by default; will be overridden at runtime for variants.
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

    print("Building OneShotOpticalGenerator with v9.8 D2NN config...")
    model = OneShotOpticalGenerator(physics, one_shot_cfg, image_size=200).to(device)
    model.eval()

    # ── Load v8.2c generator_e.* warm-start ──
    summary = _load_v8_2c_generator_state(model, args.v8_2c_checkpoint, device)

    # ── Initialize D2NN as identity ──
    print("Initializing D2NN masks as identity...")
    _init_d2nn_identity(model)

    # ── Build and save checkpoint ──
    checkpoint = {
        "model_version": ONE_SHOT_MODEL_VERSION,
        "model_state_dict": model.state_dict(),
        "config": {
            "optical_core_arch": "phase_amplitude_d2nn",
            "generator_e_arch": "mlp",
            "code_dim": 128,
            "num_classes": 10,
            "d2nn_layers": 1,
            "front_path_model": "free_space",
            "back_path_model": "free_space",
            "seed": 1234,
        },
        "generator_warm_start_summary": summary,
        "description": "v9.8 diagnostic: v8.2c generator_e + identity D2NN (phase=0, amp=1)",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"\nSaved diagnostic checkpoint to: {output_path}")
    print(f"  state dict keys: {len(checkpoint['model_state_dict'])}")
    print(f"  generator_e keys loaded: {summary['loaded']}")


if __name__ == "__main__":
    main()
