from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a PPT-oriented inference flow demo figure.")
    parser.add_argument("--physics-config", default="configs/physics_cfg.yaml")
    parser.add_argument("--train-config", default="configs/train_first_stage_linear_metasurface_2000_grid32.yaml")
    parser.add_argument("--first-stage-checkpoint", default="outputs/final_pt/first_stage_best_optical.pt")
    parser.add_argument(
        "--diffusion-checkpoint",
        default="outputs/final_pt/diffusion_best.pt",
        help="Accepted for CLI compatibility; this SLM encoder-path demo does not load it.",
    )
    parser.add_argument("--output", default="outputs/reports_DEMO-ppt/ppt_inference_flow_demo.png")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; this SLM encoder-path demo does not use guidance.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing --output if it already exists.")
    parser.add_argument(
        "--noise-strength",
        type=float,
        default=0.35,
        help="Deprecated; kept for CLI compatibility. Actual generation uses diffusion noise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_for_image(tensor):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for this visualization script.") from exc

    value = tensor.detach()
    if torch.is_complex(value):
        value = value.abs()
    value = value.float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3 and value.shape[0] > 1:
        value = value.mean(dim=0, keepdim=True)
    if value.ndim == 3:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D/3D/4D tensor for visualization, got shape {tuple(value.shape)}")
    value = value.unsqueeze(0).unsqueeze(0)
    value = F.interpolate(value, size=(200, 200), mode="bilinear", align_corners=False).squeeze()
    min_value = value.amin()
    max_value = value.amax()
    dynamic_range = max_value - min_value
    if dynamic_range < 1e-5:
        return value.clamp(0.0, 1.0).numpy()
    # PPT 展示用分位数拉伸，避免少数亮点把主要结构压黑。
    flat_value = value.flatten()
    low_value = torch.quantile(flat_value, 0.01)
    high_value = torch.quantile(flat_value, 0.995)
    contrast_range = high_value - low_value
    if contrast_range < 1e-5:
        low_value = min_value
        contrast_range = dynamic_range
    return ((value - low_value) / contrast_range.clamp_min(1e-6)).clamp(0.0, 1.0).numpy()


def _noise_latent_to_slm_control(noise_latent, image_size: int):
    import torch.nn.functional as F

    control = noise_latent.detach()
    if control.ndim != 4:
        raise ValueError(f"Expected latent noise with shape BCHW, got {tuple(control.shape)}")
    if control.shape[1] > 1:
        control = control.mean(dim=1, keepdim=True)
    control = F.interpolate(control, size=(image_size, image_size), mode="bilinear", align_corners=False)
    min_value = control.amin(dim=(-2, -1), keepdim=True)
    max_value = control.amax(dim=(-2, -1), keepdim=True)
    return (control - min_value) / (max_value - min_value).clamp_min(1e-6)


def _sample_input_noise(torch_module, train_cfg, device):
    shape = (
        1,
        train_cfg.first_stage.latent_channels,
        train_cfg.first_stage.image_size // 8,
        train_cfg.first_stage.image_size // 8,
    )
    return torch_module.randn(shape, device=device)


def _estimate_after_slm_field(torch_module, first_stage, slm_control):
    slm_transfer = first_stage.to_slm_transfer(slm_control)
    incoming_field = torch_module.ones_like(slm_transfer, dtype=torch_module.complex64)
    return first_stage.front_path.slm(slm_transfer, incoming_field)


def _chip_amplitude_for_display(torch_module, chip_amplitude):
    # 仅用于 PPT 显示；实际模型输出仍是原始 chip_amplitude。
    return torch_module.log1p(chip_amplitude.clamp_min(0.0) * 20.0)


def _resolve_output_path(output_path: Path, overwrite: bool) -> Path:
    if overwrite or not output_path.exists():
        return output_path
    for index in range(1, 1000):
        candidate = output_path.with_name(f"{output_path.stem}_v{index:03d}{output_path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find a free output filename near {output_path}")


def _save_flow_figure(columns: list[tuple[str, str, object]], output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(columns), figsize=(2.7 * len(columns), 3.4), squeeze=False)
    for axis, (label, caption, tensor) in zip(axes[0], columns, strict=True):
        axis.imshow(_normalize_for_image(tensor), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(label, fontsize=11)
        axis.set_xlabel(caption, fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for this visualization script.") from exc

    from ldm_project import (
        OpticalPathVAE,
        build_train_validation_loaders,
        load_first_stage_checkpoint,
        load_physics_config,
        load_training_config,
    )
    from ldm_project.sampling import select_reference_images_for_labels
    from ldm_project.train_utils import resolve_device

    args = parse_args()
    torch.manual_seed(args.seed)
    physics = load_physics_config(args.physics_config)
    train_cfg = load_training_config(args.train_config)
    device = resolve_device(torch, args.device)

    first_stage = OpticalPathVAE(physics, train_cfg.first_stage).to(device)

    first_stage_payload = torch.load(args.first_stage_checkpoint, map_location=device, weights_only=False)
    load_first_stage_checkpoint(first_stage, first_stage_payload)

    first_stage.eval()
    class_labels = torch.tensor([args.class_label], device=device, dtype=torch.long)
    dataset_splits = build_train_validation_loaders(
        data_config=train_cfg.data,
        first_stage_config=train_cfg.first_stage,
        batch_size=train_cfg.batch_size,
    )
    reference_images, _, reference_source = select_reference_images_for_labels(
        torch,
        class_labels,
        dataset_splits.validation,
        dataset_splits.train,
        device,
    )

    with torch.no_grad():
        input_noise = _sample_input_noise(torch, train_cfg, device)
        slm_control = _noise_latent_to_slm_control(input_noise, train_cfg.first_stage.image_size)
        after_slm = _estimate_after_slm_field(torch, first_stage, slm_control)
        encoded = first_stage.encode(slm_control, deterministic=True)
        decoded = first_stage.decode_with_intermediates(encoded["z"])
        decoder_output_display = _chip_amplitude_for_display(torch, decoded["chip_amplitude"])

    columns = [
        ("Reference", f"class {args.class_label} | {reference_source}", reference_images),
        ("Input Noise", "SLM control from noise", slm_control),
        ("After SLM", "estimated field amplitude", after_slm),
        ("Before Encoder", "optical encoder input", encoded["optical_encoder_input"]),
        ("PF2 / Encoder Out", "encoded latent z", encoded["z"]),
        ("Decoder Output", "chip amplitude only, log display", decoder_output_display),
        ("CMOS", "camera plane", decoded["camera_plane"]),
    ]
    title = (
        "PPT SLM encoder-path demo | Reference -> Input Noise -> After SLM -> "
        "Before Encoder -> PF2 / Encoder Out -> Decoder Output -> CMOS"
    )
    output_path = _resolve_output_path(Path(args.output), args.overwrite)
    _save_flow_figure(columns, output_path, title)
    print(f"saved PPT inference flow demo: {output_path}")


if __name__ == "__main__":
    main()
