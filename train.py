from __future__ import annotations

import argparse
import math
import sys


DEFAULT_FINAL_SAMPLE_COUNT = 8
DEFAULT_LATENT_SCALE_BATCHES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Optical-Path VAE LDM scaffold.")
    parser.add_argument("--physics-config", default="configs/physics_cfg.yaml", help="Path to physics_cfg.yaml")
    parser.add_argument("--train-config", default="configs/train_ldm.yaml", help="Path to train_ldm.yaml")
    parser.add_argument("--stage", choices=["first_stage", "diffusion", "full"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--subset-fraction", type=float, default=None, help="Override dataset subset fraction, e.g. 0.1667")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--output-root", default="outputs/runs", help="Root directory for timestamped run outputs")
    parser.add_argument("--run-name", default=None, help="Optional suffix for the timestamped run directory")
    parser.add_argument(
        "--first-stage-checkpoint",
        default=None,
        help="Required for --stage diffusion to load a trained first-stage model.",
    )
    return parser.parse_args()


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def resolve_class_labels(
    torch_module,
    requested_labels: list[int] | None,
    num_samples: int,
    num_classes: int,
    device,
):
    if requested_labels is None or len(requested_labels) == 0:
        return torch_module.randint(0, num_classes, (num_samples,), device=device), "random_uniform"

    for label in requested_labels:
        if label < 0 or label >= num_classes:
            raise ValueError(f"class label must be in [0, {num_classes - 1}], got {label}")

    if len(requested_labels) == 1:
        requested_labels = requested_labels * num_samples
    elif len(requested_labels) != num_samples:
        raise ValueError("class-labels must provide either one label or exactly num_samples labels.")

    return torch_module.tensor(requested_labels, device=device, dtype=torch_module.long), "user_provided"


def should_save_epoch_visual(epoch_index: int, interval: int) -> bool:
    if interval <= 0:
        raise ValueError(f"visual_save_epoch_interval must be positive, got {interval}")
    return epoch_index == 0 or (epoch_index + 1) % interval == 0


def should_log_step(step_index: int, interval: int, is_epoch_last_batch: bool) -> bool:
    if interval <= 0:
        raise ValueError(f"step_log_interval must be positive, got {interval}")
    # 首步、固定间隔和每个 epoch 的最后一步都保留，既方便看趋势，也避免日志过密。
    return step_index == 0 or (step_index + 1) % interval == 0 or is_epoch_last_batch


def resolve_logged_step(batch_index: int, stage_step: int) -> dict[str, int]:
    if batch_index < 0:
        raise ValueError(f"batch_index must be non-negative, got {batch_index}")
    if stage_step < 0:
        raise ValueError(f"stage_step must be non-negative, got {stage_step}")
    # 日志里的 global_step 按当前这轮 train 重新计数，绝对优化步数单独保留，避免跨轮混淆。
    return {
        "global_step": batch_index,
        "stage_step": stage_step,
    }


def build_visualization_labels(torch_module, num_samples: int, num_classes: int, device):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    return torch_module.arange(num_samples, device=device, dtype=torch_module.long) % num_classes


def _normalize_for_visualization(tensor):
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def format_output_title(relative_path: str, label: str) -> str:
    return f"{relative_path} | {label}"


def build_first_stage_visualization_tensors(input_batch, outputs):
    # 物理优先先看相机平面是否承载数字，再用 decoder image 诊断 latent 语义。
    return {
        "input": input_batch,
        "camera_hat": outputs["camera_plane"],
        "image_hat": outputs["image_recon"],
        "pf2": outputs["pf2_intensity"],
        "chip_amplitude": _normalize_for_visualization(outputs["chip_amplitude"]),
        "chip_phase": ((outputs["chip_phase"] / math.pi) + 1.0).clamp(0.0, 2.0) * 0.5,
        "probe": outputs["probe_plane"],
    }


def build_diffusion_visualization_tensors(reference_images, sample_outputs):
    # diffusion 阶段先看生成 latent 经物理后级读出的 camera，再看 decoder image 作为辅助。
    return {
        "reference (参考)": reference_images,
        "generated_camera (生成相机)": sample_outputs["camera_plane"],
        "generated_image (生成图像)": sample_outputs["image_recon"],
        "chip_amplitude (芯片振幅)": _normalize_for_visualization(sample_outputs["chip_amplitude"]),
        "chip_phase (芯片相位)": ((sample_outputs["chip_phase"] / math.pi) + 1.0).clamp(0.0, 2.0) * 0.5,
        "probe (探测平面)": sample_outputs["probe_plane"],
    }


def _resolve_label_value(label) -> int:
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _collect_reference_images_from_dataset(torch_module, dataset, requested_labels: list[int]) -> dict[int, object]:
    requested_set = set(requested_labels)
    found_images: dict[int, object] = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label_value = _resolve_label_value(label)
        if label_value not in requested_set or label_value in found_images:
            continue
        found_images[label_value] = torch_module.as_tensor(image).detach().float().cpu()
        if len(found_images) == len(requested_set):
            break
    return found_images


def select_reference_images_for_labels(
    torch_module,
    class_labels,
    validation_bundle,
    train_bundle,
    device,
):
    label_values = [_resolve_label_value(label) for label in class_labels.detach().cpu()]
    validation_found: dict[int, object] = {}
    if validation_bundle is not None:
        validation_found = _collect_reference_images_from_dataset(
            torch_module,
            validation_bundle.dataset,
            label_values,
        )

    found_images = dict(validation_found)
    missing_labels = sorted({label for label in label_values if label not in found_images})
    if missing_labels:
        found_images.update(
            _collect_reference_images_from_dataset(
                torch_module,
                train_bundle.dataset,
                missing_labels,
            )
        )

    unresolved = [label for label in label_values if label not in found_images]
    if unresolved:
        raise ValueError(f"Unable to find reference images for class labels: {sorted(set(unresolved))}")

    reference_images = torch_module.stack([found_images[label] for label in label_values]).to(device=device)
    reference_labels = torch_module.tensor(label_values, device=device, dtype=torch_module.long)
    if validation_bundle is None:
        reference_source = "train_only"
    elif all(label in validation_found for label in set(label_values)):
        reference_source = "validation_only"
    else:
        reference_source = "validation_then_train_fallback"
    return reference_images, reference_labels, reference_source


def validate_stage_requirements(stage: str, first_stage_checkpoint: str | None) -> None:
    if stage == "diffusion" and first_stage_checkpoint is None:
        raise ValueError("--stage diffusion requires --first-stage-checkpoint to load a trained first stage.")


def resolve_stage_total_steps(loader_length: int, epochs: int) -> int:
    if loader_length <= 0:
        raise ValueError("Training loader is empty.")
    if epochs <= 0:
        raise ValueError(f"stage epochs must be positive, got {epochs}")
    return loader_length * epochs


def resolve_stage_epoch_plan(stage: str, first_stage_epochs: int, diffusion_epochs: int) -> dict[str, int]:
    if first_stage_epochs < 0:
        raise ValueError(f"first_stage_epochs must be non-negative, got {first_stage_epochs}")
    if diffusion_epochs < 0:
        raise ValueError(f"diffusion_epochs must be non-negative, got {diffusion_epochs}")

    if stage == "first_stage":
        return {"first_stage": first_stage_epochs, "diffusion": 0}
    if stage == "diffusion":
        return {"first_stage": 0, "diffusion": diffusion_epochs}
    if stage == "full":
        return {"first_stage": first_stage_epochs, "diffusion": diffusion_epochs}
    raise ValueError(f"Unsupported stage mode: {stage}")


def average_scalar_metrics(metric_totals: dict[str, float], example_count: int) -> dict[str, float]:
    if example_count <= 0:
        raise ValueError("Validation loader is empty.")
    return {name: total / float(example_count) for name, total in metric_totals.items()}


def evaluate_first_stage(
    torch_module,
    first_stage,
    first_stage_loss_fn,
    loader,
    device,
    use_pin_memory: bool,
    stage_step: int,
    total_steps: int,
    train_cfg,
) -> dict[str, float]:
    metric_names = (
        "loss",
        "recon_loss",
        "image_recon_loss",
        "image_fg_mse",
        "image_bg_mse",
        "camera_recon_loss",
        "camera_fg_mse",
        "camera_bg_mse",
        "kl_loss",
        "fg_pixel_ratio",
        "pred_fg_energy_ratio",
        "beta",
    )
    metric_totals = {name: 0.0 for name in metric_names}
    example_count = 0
    validation_step = max(stage_step - 1, 0)
    first_stage.eval()
    with torch_module.no_grad():
        for images, _ in loader:
            input_batch = images.to(device, non_blocking=use_pin_memory)
            outputs = first_stage(input_batch)
            losses = first_stage_loss_fn(
                outputs,
                input_batch,
                global_step=validation_step,
                total_steps=total_steps,
                target_beta=train_cfg.first_stage.kl_beta,
                warmup_ratio=train_cfg.first_stage.kl_warmup_ratio,
                foreground_threshold=train_cfg.first_stage.recon_foreground_threshold,
                foreground_weight=train_cfg.first_stage.recon_foreground_weight,
                background_weight=train_cfg.first_stage.recon_background_weight,
                image_recon_weight=train_cfg.first_stage.image_recon_weight,
                camera_recon_weight=train_cfg.first_stage.camera_recon_weight,
            )
            batch_size = input_batch.shape[0]
            for name in metric_names:
                metric_totals[name] += losses[name].item() * batch_size
            example_count += batch_size
    return average_scalar_metrics(metric_totals, example_count)


def evaluate_diffusion(
    torch_module,
    first_stage,
    diffusion_model,
    scheduler,
    diffusion_loss_fn,
    loader,
    device,
    use_pin_memory: bool,
    latent_scale,
    latent_source: str,
) -> dict[str, float]:
    metric_totals = {"loss": 0.0}
    example_count = 0
    diffusion_model.eval()
    with torch_module.no_grad():
        for images, labels in loader:
            batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            loss = diffusion_loss_fn(
                first_stage,
                diffusion_model,
                scheduler,
                batch,
                class_labels,
                latent_scale,
                latent_source=latent_source,
            )
            batch_size = batch.shape[0]
            metric_totals["loss"] += loss.item() * batch_size
            example_count += batch_size
    averaged = average_scalar_metrics(metric_totals, example_count)
    averaged["latent_scale"] = latent_scale.item()
    return averaged


def estimate_latent_scale_from_loader(
    torch_module,
    first_stage,
    loader,
    device,
    use_pin_memory: bool,
    latent_source: str,
    max_batches: int = DEFAULT_LATENT_SCALE_BATCHES,
):
    from ldm_project.training import select_diffusion_latents

    if max_batches <= 0:
        raise ValueError(f"max_batches must be positive, got {max_batches}")

    total_square = None
    total_count = 0
    first_stage.eval()
    with torch_module.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = images.to(device, non_blocking=use_pin_memory)
            encoded = first_stage.encode(batch)
            latents = select_diffusion_latents(encoded, latent_source).detach()
            square_sum = latents.pow(2).sum()
            total_square = square_sum if total_square is None else total_square + square_sum
            total_count += latents.numel()

    if total_square is None or total_count == 0:
        raise ValueError("Training loader is empty.")
    return (total_square / float(total_count)).sqrt().clamp_min(1e-6)


def save_post_train_samples(
    torch_module,
    run_manager,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    latent_scale,
    device,
    sample_latents_fn,
) -> None:
    # 训练完成后固定生成一组 0-7 类别的可视化样本，用来快速判断 diffusion 是否已经学到类别语义。
    sample_labels = build_visualization_labels(
        torch_module,
        DEFAULT_FINAL_SAMPLE_COUNT,
        train_cfg.diffusion.num_classes,
        device,
    )
    reference_images, reference_labels, reference_source = select_reference_images_for_labels(
        torch_module,
        sample_labels,
        validation_bundle,
        train_bundle,
        device,
    )
    run_manager.save_config_snapshot(
        "post_train_sample_config",
        {
            "num_samples": sample_labels.shape[0],
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "guidance_scale": train_cfg.diffusion.guidance_scale,
            "source": "train.py_post_train_visualization",
        },
    )
    final_sample_batch = sample_latents_fn(
        diffusion_model,
        scheduler,
        (
            sample_labels.shape[0],
            train_cfg.first_stage.latent_channels,
            train_cfg.first_stage.image_size // 8,
            train_cfg.first_stage.image_size // 8,
        ),
        device,
        sample_labels,
        guidance_scale=train_cfg.diffusion.guidance_scale,
    )
    final_sample_outputs = first_stage.decode_with_intermediates(final_sample_batch * latent_scale)
    # 训练完成后的最终 sample 统一保存在 train run 目录，避免再创建独立 sample run。
    run_manager.save_tensor_grid(
        build_diffusion_visualization_tensors(reference_images, final_sample_outputs),
        run_manager.samples_dir / "post_train_sample_grid.png",
        title=format_output_title("images/samples/post_train_sample_grid.png", "post-train sample outputs"),
    )
    run_manager.save_tensor_bundle(
        "post_train_sample_tensors.pt",
        {
            "latents": final_sample_batch.detach().cpu(),
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
            "generated_camera": final_sample_outputs["camera_plane"].detach().cpu(),
            "generated_image": final_sample_outputs["image_recon"].detach().cpu(),
            "chip_amplitude": final_sample_outputs["chip_amplitude"].detach().cpu(),
            "chip_phase": final_sample_outputs["chip_phase"].detach().cpu(),
            "probe_plane": final_sample_outputs["probe_plane"].detach().cpu(),
            "camera_plane": final_sample_outputs["camera_plane"].detach().cpu(),
        },
    )


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run train.py.") from exc

    from ldm_project import (
        FIRST_STAGE_MODEL_VERSION,
        LatentDiffusionUNet,
        LinearNoiseScheduler,
        OpticalPathVAE,
        RunManager,
        build_train_validation_loaders,
        load_first_stage_checkpoint,
        load_physics_config,
        load_training_config,
    )
    from ldm_project.diffusion import DIFFUSION_MODEL_VERSION
    from ldm_project.training import diffusion_loss, first_stage_loss, sample_latents

    args = parse_args()
    physics = load_physics_config(args.physics_config)
    train_cfg = load_training_config(args.train_config)
    if args.stage is not None:
        train_cfg.mode = args.stage
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.subset_fraction is not None:
        train_cfg.data.subset_fraction = args.subset_fraction

    validate_stage_requirements(train_cfg.mode, args.first_stage_checkpoint)

    device = resolve_device(torch, args.device)
    use_pin_memory = train_cfg.data.pin_memory and device.type == "cuda"
    train_cfg.data.pin_memory = use_pin_memory

    run_manager = RunManager.create(
        output_root=args.output_root,
        purpose="train",
        run_name=args.run_name or train_cfg.mode,
    )
    first_stage = OpticalPathVAE(physics, train_cfg.first_stage).to(device)
    diffusion_model = LatentDiffusionUNet(train_cfg.first_stage.latent_channels, train_cfg.diffusion).to(device)
    scheduler = LinearNoiseScheduler(
        timesteps=train_cfg.diffusion.timesteps,
        beta_start=train_cfg.diffusion.beta_start,
        beta_end=train_cfg.diffusion.beta_end,
    ).to(device)

    if args.first_stage_checkpoint is not None:
        first_stage_payload = torch.load(args.first_stage_checkpoint, map_location=device)
        load_first_stage_checkpoint(first_stage, first_stage_payload)

    dataset_splits = build_train_validation_loaders(
        data_config=train_cfg.data,
        first_stage_config=train_cfg.first_stage,
        batch_size=train_cfg.batch_size,
    )
    train_bundle = dataset_splits.train
    validation_bundle = dataset_splits.validation

    run_manager.save_config_snapshot("physics_config", physics)
    run_manager.save_config_snapshot("train_config", train_cfg)
    epoch_plan = resolve_stage_epoch_plan(
        train_cfg.mode,
        train_cfg.first_stage_epochs,
        train_cfg.diffusion_epochs,
    )
    run_manager.write_meta(
        {
            "script": "train.py",
            "argv": sys.argv,
            "device": str(device),
            "mode": train_cfg.mode,
            "physics_config_path": args.physics_config,
            "train_config_path": args.train_config,
            "first_stage_checkpoint": args.first_stage_checkpoint,
            "run_dir": run_manager.run_dir,
            "dataset": train_cfg.data.dataset,
            "dataset_root": train_cfg.data.root,
            "dataset_total_size": dataset_splits.total_size,
            "dataset_effective_size": train_bundle.effective_size,
            "train_split_size": train_bundle.total_size,
            "train_effective_size": train_bundle.effective_size,
            "validation_size": 0 if validation_bundle is None else validation_bundle.effective_size,
            "validation_fraction": train_cfg.data.validation_fraction,
            "validation_split_seed": train_cfg.data.validation_split_seed,
            "subset_fraction": train_cfg.data.subset_fraction,
            "subset_fraction_applies_to": "train_only",
            "planned_first_stage_epochs": epoch_plan["first_stage"],
            "planned_diffusion_epochs": epoch_plan["diffusion"],
            "first_stage_model_version": FIRST_STAGE_MODEL_VERSION,
            "diffusion_latent_source": train_cfg.diffusion.latent_source,
            "diffusion_guidance_scale": train_cfg.diffusion.guidance_scale,
            "conditioning": diffusion_model.conditioning_metadata(),
        }
    )
    print(
        f"planned epochs: first_stage={epoch_plan['first_stage']} "
        f"diffusion={epoch_plan['diffusion']}"
    )

    if train_cfg.mode in {"first_stage", "full"}:
        optimizer = torch.optim.Adam(first_stage.parameters(), lr=train_cfg.learning_rate)
        first_stage.train()
        loader_length = len(train_bundle.loader)
        first_stage_total_steps = resolve_stage_total_steps(
            loader_length,
            train_cfg.first_stage_epochs,
        )
        first_stage_stage_step = 0
        for epoch_index in range(train_cfg.first_stage_epochs):
            epoch_number = epoch_index + 1
            last_batch = None
            last_outputs = None
            for batch_index, (images, _) in enumerate(train_bundle.loader):
                # 物理优先训练把 camera_plane 作为主读出，image_recon 仅保留为辅助诊断。
                input_batch = images.to(device, non_blocking=use_pin_memory)
                optimizer.zero_grad(set_to_none=True)
                outputs = first_stage(input_batch)
                losses = first_stage_loss(
                    outputs,
                    input_batch,
                    global_step=first_stage_stage_step,
                    total_steps=first_stage_total_steps,
                    target_beta=train_cfg.first_stage.kl_beta,
                    warmup_ratio=train_cfg.first_stage.kl_warmup_ratio,
                    foreground_threshold=train_cfg.first_stage.recon_foreground_threshold,
                    foreground_weight=train_cfg.first_stage.recon_foreground_weight,
                    background_weight=train_cfg.first_stage.recon_background_weight,
                    image_recon_weight=train_cfg.first_stage.image_recon_weight,
                    camera_recon_weight=train_cfg.first_stage.camera_recon_weight,
                )
                losses["loss"].backward()
                optimizer.step()
                step_record = resolve_logged_step(batch_index, first_stage_stage_step)
                should_record = should_log_step(
                    step_record["global_step"],
                    train_cfg.step_log_interval,
                    is_epoch_last_batch=batch_index == loader_length - 1,
                )
                if should_record:
                    print(
                        f"[first_stage] epoch={epoch_number} global_step={step_record['global_step']} "
                        f"stage_step={step_record['stage_step']} "
                        f"loss={losses['loss'].item():.6f} recon_w={losses['recon_loss'].item():.6f} "
                        f"image_recon={losses['image_recon_loss'].item():.6f} "
                        f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                        f"image_fg_mse={losses['image_fg_mse'].item():.6f} "
                        f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                        f"kl={losses['kl_loss'].item():.6f}"
                    )
                    run_manager.append_history(
                        "first_stage",
                        {
                            "epoch": epoch_number,
                            **step_record,
                            "loss": losses["loss"].item(),
                            "recon_loss": losses["recon_loss"].item(),
                            "image_recon_loss": losses["image_recon_loss"].item(),
                            "image_fg_mse": losses["image_fg_mse"].item(),
                            "image_bg_mse": losses["image_bg_mse"].item(),
                            "camera_recon_loss": losses["camera_recon_loss"].item(),
                            "camera_fg_mse": losses["camera_fg_mse"].item(),
                            "camera_bg_mse": losses["camera_bg_mse"].item(),
                            "kl_loss": losses["kl_loss"].item(),
                            "fg_pixel_ratio": losses["fg_pixel_ratio"].item(),
                            "pred_fg_energy_ratio": losses["pred_fg_energy_ratio"].item(),
                            "beta": losses["beta"].item(),
                        },
                    )
                last_batch = input_batch
                last_outputs = outputs
                first_stage_stage_step += 1

            # 仅在 epoch 结束时保存一张代表图，避免再暴露 step-based 可视化语义。
            if (
                last_batch is not None
                and last_outputs is not None
                and should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval)
            ):
                run_manager.save_tensor_grid(
                    build_first_stage_visualization_tensors(last_batch, last_outputs),
                    run_manager.first_stage_images_dir / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/first_stage/epoch_{epoch_number:04d}.png",
                        f"first_stage epoch {epoch_number}",
                    ),
                )
            if validation_bundle is not None:
                validation_metrics = evaluate_first_stage(
                    torch,
                    first_stage,
                    first_stage_loss,
                    validation_bundle.loader,
                    device,
                    use_pin_memory,
                    first_stage_stage_step,
                    first_stage_total_steps,
                    train_cfg,
                )
                print(
                    f"[first_stage][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                    f"recon_w={validation_metrics['recon_loss']:.6f} "
                    f"image_recon={validation_metrics['image_recon_loss']:.6f} "
                    f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                    f"image_fg_mse={validation_metrics['image_fg_mse']:.6f} "
                    f"camera_fg_mse={validation_metrics['camera_fg_mse']:.6f} "
                    f"kl={validation_metrics['kl_loss']:.6f}"
                )
                run_manager.append_history(
                    "first_stage_validation",
                    {
                        "epoch": epoch_number,
                        **validation_metrics,
                    },
                )
                first_stage.train()

        run_manager.save_checkpoint(
            "first_stage_latest.pt",
            {
                "model_state_dict": first_stage.state_dict(),
                "stage": "first_stage",
                "model_version": FIRST_STAGE_MODEL_VERSION,
                "physics_config": physics,
                "train_config": train_cfg,
            },
        )

    if train_cfg.mode in {"diffusion", "full"}:
        first_stage.eval()
        if train_cfg.first_stage.freeze_after_pretrain:
            for param in first_stage.parameters():
                param.requires_grad_(False)

        latent_scale = estimate_latent_scale_from_loader(
            torch,
            first_stage,
            train_bundle.loader,
            device,
            use_pin_memory,
            train_cfg.diffusion.latent_source,
        )
        optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_cfg.learning_rate)
        diffusion_model.train()
        loader_length = len(train_bundle.loader)
        diffusion_total_steps = resolve_stage_total_steps(
            loader_length,
            train_cfg.diffusion_epochs,
        )
        diffusion_stage_step = 0
        for epoch_index in range(train_cfg.diffusion_epochs):
            epoch_number = epoch_index + 1
            last_class_labels = None
            last_reference_batch = None
            for batch_index, (images, labels) in enumerate(train_bundle.loader):
                # 第二阶段不再重建输入图，而是学习“给定类别条件时如何在 latent 空间去噪”。
                batch = images.to(device, non_blocking=use_pin_memory)
                class_labels = labels.to(device, non_blocking=use_pin_memory)
                optimizer.zero_grad(set_to_none=True)
                loss = diffusion_loss(
                    first_stage,
                    diffusion_model,
                    scheduler,
                    batch,
                    class_labels,
                    latent_scale,
                    latent_source=train_cfg.diffusion.latent_source,
                )
                loss.backward()
                optimizer.step()
                step_record = resolve_logged_step(batch_index, diffusion_stage_step)
                should_record = should_log_step(
                    step_record["global_step"],
                    train_cfg.step_log_interval,
                    is_epoch_last_batch=batch_index == loader_length - 1,
                )
                if should_record:
                    print(
                        f"[diffusion] epoch={epoch_number} global_step={step_record['global_step']} "
                        f"stage_step={step_record['stage_step']} "
                        f"loss={loss.item():.6f} latent_scale={latent_scale.item():.6f}"
                    )
                    run_manager.append_history(
                        "diffusion",
                        {
                            "epoch": epoch_number,
                            **step_record,
                            "loss": loss.item(),
                            "latent_scale": latent_scale.item(),
                        },
                    )
                last_class_labels = class_labels
                last_reference_batch = batch.detach()
                diffusion_stage_step += 1

            if (
                last_class_labels is not None
                and last_reference_batch is not None
                and should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval)
            ):
                with torch.no_grad():
                    sample_labels = last_class_labels[: min(4, last_class_labels.shape[0])]
                    reference_images = last_reference_batch[: sample_labels.shape[0]]
                    sample_batch = sample_latents(
                        diffusion_model,
                        scheduler,
                        (
                            sample_labels.shape[0],
                            train_cfg.first_stage.latent_channels,
                            train_cfg.first_stage.image_size // 8,
                            train_cfg.first_stage.image_size // 8,
                        ),
                        device,
                        sample_labels,
                        guidance_scale=train_cfg.diffusion.guidance_scale,
                    )
                    sample_outputs = first_stage.decode_with_intermediates(sample_batch * latent_scale)
                    run_manager.save_tensor_grid(
                        build_diffusion_visualization_tensors(reference_images, sample_outputs),
                        run_manager.diffusion_images_dir / f"epoch_{epoch_number:04d}.png",
                        title=format_output_title(
                            f"images/diffusion/epoch_{epoch_number:04d}.png",
                            f"diffusion epoch {epoch_number}",
                        ),
                    )
            if validation_bundle is not None:
                validation_metrics = evaluate_diffusion(
                    torch,
                    first_stage,
                    diffusion_model,
                    scheduler,
                    diffusion_loss,
                    validation_bundle.loader,
                    device,
                    use_pin_memory,
                    latent_scale,
                    train_cfg.diffusion.latent_source,
                )
                print(
                    f"[diffusion][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                    f"latent_scale={validation_metrics['latent_scale']:.6f}"
                )
                run_manager.append_history(
                    "diffusion_validation",
                    {
                        "epoch": epoch_number,
                        **validation_metrics,
                    },
                )
                diffusion_model.train()

        run_manager.save_checkpoint(
            "diffusion_latest.pt",
            {
                "model_state_dict": diffusion_model.state_dict(),
                "latent_scale": latent_scale.detach().cpu(),
                "latent_source": train_cfg.diffusion.latent_source,
                "stage": "diffusion",
                "model_version": DIFFUSION_MODEL_VERSION,
                "first_stage_model_version": FIRST_STAGE_MODEL_VERSION,
                "conditioning": diffusion_model.conditioning_metadata(),
                "physics_config": physics,
                "train_config": train_cfg,
            },
        )

        diffusion_model.eval()
        save_post_train_samples(
            torch,
            run_manager,
            first_stage,
            diffusion_model,
            scheduler,
            train_bundle,
            validation_bundle,
            train_cfg,
            latent_scale,
            device,
            sample_latents,
        )

    print(f"train dataset size: {train_bundle.effective_size}/{train_bundle.total_size}")
    if validation_bundle is None:
        print("validation dataset size: disabled")
    else:
        print(f"validation dataset size: {validation_bundle.effective_size}/{validation_bundle.total_size}")
    print(f"dataset total size: {dataset_splits.total_size}")
    print(f"run outputs saved to: {run_manager.run_dir}")


if __name__ == "__main__":
    main()
