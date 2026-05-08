from __future__ import annotations

from .evaluation import compute_first_stage_balanced_metrics, evaluate_diffusion_with_per_class, evaluate_first_stage_with_per_class
from .sampling import (
    build_diffusion_visualization_tensors,
    build_first_stage_visualization_tensors,
    format_output_title,
    save_post_train_samples,
)
from .train_utils import (
    estimate_latent_scale_from_loader,
    is_best_optical_candidate,
    load_latent_scale_from_diffusion_checkpoint,
    require_strict_optical_best_checkpoint,
    resolve_image_fusion_weight,
    resolve_logged_step,
    resolve_stage_total_steps,
    set_first_stage_image_fusion_weight,
    should_log_step,
    should_save_epoch_visual,
    should_update_best_optical,
    should_validate_epoch,
)


FIRST_STAGE_BEST_OPTICAL_CHECKPOINT = "first_stage_best_optical.pt"
FIRST_STAGE_BEST_OPTICAL_METRIC = "camera_complex_balanced_mnist_score"
DIFFUSION_BEST_CHECKPOINT = "diffusion_best.pt"
DIFFUSION_BEST_METRIC = "diffusion_validation.loss"


def run_sample_stage(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    diffusion_checkpoint: str,
    first_stage_model_version: str,
    load_diffusion_checkpoint_fn,
    sample_latents_fn,
    sample_latents_with_trace_fn,
    resolve_denoising_trace_steps_fn,
) -> None:
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, train_cfg.first_stage.image_fusion_end_weight)
    diffusion_payload = torch_module.load(diffusion_checkpoint, map_location=device, weights_only=False)
    load_diffusion_checkpoint_fn(diffusion_model, diffusion_payload, first_stage_model_version)
    latent_scale = load_latent_scale_from_diffusion_checkpoint(torch_module, diffusion_payload, device)
    diffusion_model.eval()
    run_meta.update(
        {
            "latent_scale_source": "diffusion_checkpoint",
            "diffusion_best_loaded_for_sampling": True,
        }
    )
    run_manager.write_meta(run_meta)
    print(f"loaded diffusion checkpoint for sampling: {diffusion_checkpoint}")
    sample_artifacts = save_post_train_samples(
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
        sample_latents_with_trace_fn,
        resolve_denoising_trace_steps_fn,
    )
    run_meta.update(sample_artifacts)
    run_manager.write_meta(run_meta)


def run_first_stage_training(
    torch_module,
    *,
    first_stage,
    first_stage_loss_fn,
    first_stage_visual_batch,
    run_manager,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    first_stage_model_version: str,
) -> None:
    optimizer = torch_module.optim.Adam(first_stage.parameters(), lr=train_cfg.learning_rate)
    first_stage.train()
    loader_length = len(train_bundle.loader)
    first_stage_total_steps = resolve_stage_total_steps(
        loader_length,
        train_cfg.first_stage_epochs,
    )
    first_stage_stage_step = 0
    best_optical_score = None
    best_optical_checkpoint_path = None
    for epoch_index in range(train_cfg.first_stage_epochs):
        epoch_number = epoch_index + 1
        image_fusion_weight = resolve_image_fusion_weight(train_cfg.first_stage, epoch_index)
        set_first_stage_image_fusion_weight(first_stage, image_fusion_weight)
        for batch_index, (images, _) in enumerate(train_bundle.loader):
            # 物理优先训练把 camera_plane 作为主读出，image_recon 仅保留为辅助诊断。
            input_batch = images.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            outputs = first_stage(input_batch)
            losses = first_stage_loss_fn(
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
                structure_recon_weight=train_cfg.first_stage.structure_recon_weight,
                shape_recon_weight=train_cfg.first_stage.shape_recon_weight,
                field_smoothness_weight=train_cfg.first_stage.field_smoothness_weight,
                phase_smoothness_weight=train_cfg.first_stage.phase_smoothness_weight,
                background_leakage_weight=train_cfg.first_stage.background_leakage_weight,
                field_energy_weight=train_cfg.first_stage.field_energy_weight,
                camera_hotspot_weight=train_cfg.first_stage.camera_hotspot_weight,
                camera_topology_weight=train_cfg.first_stage.camera_topology_weight,
                camera_saturation_weight=train_cfg.first_stage.camera_saturation_weight,
                camera_artifact_weight=train_cfg.first_stage.camera_artifact_weight,
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
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"hotspot={losses['camera_hotspot_ratio'].item():.4f} "
                    f"topology={losses['camera_topology_loss'].item():.6f} "
                    f"structure={losses['structure_loss'].item():.6f} "
                    f"shape={losses['shape_loss'].item():.6f} "
                    f"field={losses['field_smoothness_loss'].item():.6f} "
                    f"phase_tv={losses['phase_tv'].item():.6f} "
                    f"field_energy={losses['field_energy'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"artifact={losses['camera_artifact_loss'].item():.6f} "
                    f"hole={losses['camera_foreground_hole_ratio'].item():.4f} "
                    f"image_fusion={image_fusion_weight:.4f} "
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
                        "camera_foreground_dice_loss": losses["camera_foreground_dice_loss"].item(),
                        "camera_foreground_dice": losses["camera_foreground_dice"].item(),
                        "structure_loss": losses["structure_loss"].item(),
                        "image_structure_loss": losses["image_structure_loss"].item(),
                        "camera_structure_loss": losses["camera_structure_loss"].item(),
                        "shape_loss": losses["shape_loss"].item(),
                        "image_shape_loss": losses["image_shape_loss"].item(),
                        "camera_shape_loss": losses["camera_shape_loss"].item(),
                        "field_smoothness_loss": losses["field_smoothness_loss"].item(),
                        "field_tv": losses["field_tv"].item(),
                        "phase_smoothness_loss": losses["phase_smoothness_loss"].item(),
                        "phase_tv": losses["phase_tv"].item(),
                        "field_energy_loss": losses["field_energy_loss"].item(),
                        "field_energy": losses["field_energy"].item(),
                        "background_leakage_loss": losses["background_leakage_loss"].item(),
                        "camera_background_leakage": losses["camera_background_leakage"].item(),
                        "camera_hotspot_loss": losses["camera_hotspot_loss"].item(),
                        "camera_hotspot_ratio": losses["camera_hotspot_ratio"].item(),
                        "camera_saturation_ratio": losses["camera_saturation_ratio"].item(),
                        "camera_topology_loss": losses["camera_topology_loss"].item(),
                        "camera_topology_base": losses["camera_topology_base"].item(),
                        "camera_saturation_loss": losses["camera_saturation_loss"].item(),
                        "camera_saturation_penalty": losses["camera_saturation_penalty"].item(),
                        "camera_artifact_loss": losses["camera_artifact_loss"].item(),
                        "camera_foreground_hole_loss": losses["camera_foreground_hole_loss"].item(),
                        "camera_foreground_hole_ratio": losses["camera_foreground_hole_ratio"].item(),
                        "camera_foreground_roughness": losses["camera_foreground_roughness"].item(),
                        "kl_loss": losses["kl_loss"].item(),
                        "fg_pixel_ratio": losses["fg_pixel_ratio"].item(),
                        "pred_fg_energy_ratio": losses["pred_fg_energy_ratio"].item(),
                        "beta": losses["beta"].item(),
                        "image_fusion_weight": image_fusion_weight,
                    },
                )
            first_stage_stage_step += 1

        # 仅在 epoch 结束时保存固定验证样本，方便横向观察复杂数字结构是否改善。
        if first_stage_visual_batch is not None and should_save_epoch_visual(
            epoch_index,
            train_cfg.visual_save_epoch_interval,
        ):
            first_stage.eval()
            with torch_module.no_grad():
                visual_outputs = first_stage(first_stage_visual_batch)
                run_manager.save_tensor_grid(
                    build_first_stage_visualization_tensors(first_stage_visual_batch, visual_outputs),
                    run_manager.first_stage_images_dir / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/first_stage/epoch_{epoch_number:04d}.png",
                        f"first_stage epoch {epoch_number}",
                    ),
                    max_items=first_stage_visual_batch.shape[0],
                )
            first_stage.train()
        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.first_stage_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics, per_class_metrics = evaluate_first_stage_with_per_class(
                torch_module,
                first_stage,
                first_stage_loss_fn,
                validation_bundle.loader,
                device,
                use_pin_memory,
                first_stage_stage_step,
                first_stage_total_steps,
                train_cfg,
            )
            validation_metrics.update(compute_first_stage_balanced_metrics(validation_metrics, per_class_metrics))
            print(
                f"[first_stage][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"recon_w={validation_metrics['recon_loss']:.6f} "
                f"image_recon={validation_metrics['image_recon_loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"image_fg_mse={validation_metrics['image_fg_mse']:.6f} "
                f"camera_fg_mse={validation_metrics['camera_fg_mse']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"hotspot={validation_metrics['camera_hotspot_ratio']:.4f} "
                f"topology={validation_metrics['camera_topology_loss']:.6f} "
                f"structure={validation_metrics['structure_loss']:.6f} "
                f"shape={validation_metrics['shape_loss']:.6f} "
                f"field={validation_metrics['field_smoothness_loss']:.6f} "
                f"phase_tv={validation_metrics['phase_tv']:.6f} "
                f"field_energy={validation_metrics['field_energy']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"artifact={validation_metrics['camera_artifact_loss']:.6f} "
                f"hole={validation_metrics['camera_foreground_hole_ratio']:.4f} "
                f"camera_corr={validation_metrics['camera_corr']:.4f} "
                f"camera_edge_corr={validation_metrics['camera_edge_corr']:.4f} "
                f"min_class_corr={validation_metrics['min_class_camera_corr']:.4f} "
                f"balanced={validation_metrics['camera_balanced_score']:.4f} "
                f"mnist_score={validation_metrics['camera_mnist_score']:.4f} "
                f"min_class_mnist={validation_metrics['min_class_camera_mnist_score']:.4f} "
                f"class_balanced_mnist={validation_metrics['camera_class_balanced_mnist_score']:.4f} "
                f"complex_corr={validation_metrics['complex_camera_corr']:.4f} "
                f"complex_mnist={validation_metrics['complex_camera_mnist_score']:.4f} "
                f"complex_balanced_mnist={validation_metrics['camera_complex_balanced_mnist_score']:.4f} "
                f"kl={validation_metrics['kl_loss']:.6f}"
            )
            run_manager.append_history(
                "first_stage_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            run_manager.append_history(
                "first_stage_validation_per_class",
                {
                    "epoch": epoch_number,
                    "classes": per_class_metrics,
                },
            )
            optical_candidate = is_best_optical_candidate(train_cfg.first_stage, image_fusion_weight)
            selection_score = validation_metrics[FIRST_STAGE_BEST_OPTICAL_METRIC]
            updated_best = optical_candidate and should_update_best_optical(selection_score, best_optical_score)
            if updated_best:
                best_optical_score = selection_score
                best_optical_checkpoint_path = run_manager.save_checkpoint(
                    FIRST_STAGE_BEST_OPTICAL_CHECKPOINT,
                    {
                        "model_state_dict": first_stage.state_dict(),
                        "stage": "first_stage",
                        "model_version": first_stage_model_version,
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
                        "selection_score": selection_score,
                        "validation_metrics": validation_metrics,
                        "image_fusion_weight": image_fusion_weight,
                    },
                )
                run_meta.update(
                    {
                        "first_stage_best_optical_checkpoint": best_optical_checkpoint_path,
                        "first_stage_best_optical_epoch": epoch_number,
                        "first_stage_best_optical_score": best_optical_score,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "first_stage_best_optical",
                {
                    "epoch": epoch_number,
                    "eligible": optical_candidate,
                    "updated": updated_best,
                    "selection_metric": FIRST_STAGE_BEST_OPTICAL_METRIC,
                    "selection_score": selection_score,
                    "best_score": best_optical_score,
                    "image_fusion_weight": image_fusion_weight,
                    "checkpoint": best_optical_checkpoint_path,
                },
            )
            first_stage.train()

    run_manager.save_checkpoint(
        "first_stage_latest.pt",
        {
            "model_state_dict": first_stage.state_dict(),
            "stage": "first_stage",
            "model_version": first_stage_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
        },
    )


def run_diffusion_training(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    run_manager,
    physics,
    device,
    use_pin_memory: bool,
    first_stage_model_version: str,
    diffusion_model_version: str,
    diffusion_visual_labels,
    diffusion_visual_reference_images,
    load_first_stage_checkpoint_fn,
    load_diffusion_checkpoint_fn,
    diffusion_loss_fn,
    diffusion_per_sample_losses_fn,
    sample_latents_fn,
    sample_latents_with_trace_fn,
    resolve_denoising_trace_steps_fn,
) -> None:
    if train_cfg.mode == "full":
        best_checkpoint_path = require_strict_optical_best_checkpoint(
            run_meta.get("first_stage_best_optical_checkpoint")
        )
        # full 训练的第二阶段必须使用验证集上最好的纯 optical first stage。
        best_first_stage_payload = torch_module.load(best_checkpoint_path, map_location=device, weights_only=False)
        load_first_stage_checkpoint_fn(first_stage, best_first_stage_payload)
        run_meta["first_stage_best_optical_loaded_for_diffusion"] = True
        run_manager.write_meta(run_meta)
        print(f"loaded best optical first stage for diffusion: {best_checkpoint_path}")
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, train_cfg.first_stage.image_fusion_end_weight)
    if train_cfg.first_stage.freeze_after_pretrain:
        for param in first_stage.parameters():
            param.requires_grad_(False)

    latent_scale = estimate_latent_scale_from_loader(
        torch_module,
        first_stage,
        train_bundle.loader,
        device,
        use_pin_memory,
        train_cfg.diffusion.latent_source,
    )
    optimizer = torch_module.optim.Adam(diffusion_model.parameters(), lr=train_cfg.learning_rate)
    diffusion_model.train()
    loader_length = len(train_bundle.loader)
    resolve_stage_total_steps(
        loader_length,
        train_cfg.diffusion_epochs,
    )
    diffusion_stage_step = 0
    best_diffusion_loss = None
    best_diffusion_checkpoint_path = None
    for epoch_index in range(train_cfg.diffusion_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (images, labels) in enumerate(train_bundle.loader):
            # 第二阶段不再重建输入图，而是学习“给定类别条件时如何在 latent 空间去噪”。
            batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss_fn(
                first_stage,
                diffusion_model,
                scheduler,
                batch,
                class_labels,
                latent_scale,
                latent_source=train_cfg.diffusion.latent_source,
                class_weight_labels=train_cfg.diffusion.class_weight_labels,
                class_weight=train_cfg.diffusion.class_weight,
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
            diffusion_stage_step += 1

        if (
            diffusion_visual_labels is not None
            and diffusion_visual_reference_images is not None
            and should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval)
        ):
            with torch_module.no_grad():
                sample_count = int(diffusion_visual_labels.shape[0])
                sample_batch = sample_latents_fn(
                    diffusion_model,
                    scheduler,
                    (
                        sample_count,
                        train_cfg.first_stage.latent_channels,
                        train_cfg.first_stage.image_size // 8,
                        train_cfg.first_stage.image_size // 8,
                    ),
                    device,
                    diffusion_visual_labels,
                    guidance_scale=train_cfg.diffusion.guidance_scale,
                )
                sample_outputs = first_stage.decode_with_intermediates(sample_batch * latent_scale)
                run_manager.save_tensor_grid(
                    build_diffusion_visualization_tensors(diffusion_visual_reference_images, sample_outputs),
                    run_manager.diffusion_images_dir / f"epoch_{epoch_number:04d}.png",
                    title=format_output_title(
                        f"images/diffusion/epoch_{epoch_number:04d}.png",
                        f"diffusion epoch {epoch_number}",
                    ),
                    max_items=sample_count,
                )
        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.diffusion_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics, per_class_metrics = evaluate_diffusion_with_per_class(
                torch_module,
                first_stage,
                diffusion_model,
                scheduler,
                diffusion_per_sample_losses_fn,
                validation_bundle.loader,
                device,
                use_pin_memory,
                latent_scale,
                train_cfg.diffusion.latent_source,
                train_cfg.diffusion.num_classes,
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
            run_manager.append_history(
                "diffusion_validation_per_class",
                {
                    "epoch": epoch_number,
                    "classes": per_class_metrics,
                },
            )
            selection_loss = validation_metrics["loss"]
            updated_best_diffusion = best_diffusion_loss is None or selection_loss < best_diffusion_loss
            if updated_best_diffusion:
                best_diffusion_loss = selection_loss
                best_diffusion_checkpoint_path = run_manager.save_checkpoint(
                    DIFFUSION_BEST_CHECKPOINT,
                    {
                        "model_state_dict": diffusion_model.state_dict(),
                        "latent_scale": latent_scale.detach().cpu(),
                        "latent_source": train_cfg.diffusion.latent_source,
                        "stage": "diffusion",
                        "model_version": diffusion_model_version,
                        "first_stage_model_version": first_stage_model_version,
                        "conditioning": diffusion_model.conditioning_metadata(),
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": DIFFUSION_BEST_METRIC,
                        "selection_loss": selection_loss,
                        "validation_metrics": validation_metrics,
                    },
                )
                run_meta.update(
                    {
                        "diffusion_best_checkpoint": best_diffusion_checkpoint_path,
                        "diffusion_best_epoch": epoch_number,
                        "diffusion_best_loss": best_diffusion_loss,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "diffusion_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best_diffusion,
                    "selection_metric": DIFFUSION_BEST_METRIC,
                    "selection_loss": selection_loss,
                    "best_loss": best_diffusion_loss,
                    "checkpoint": best_diffusion_checkpoint_path,
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
            "model_version": diffusion_model_version,
            "first_stage_model_version": first_stage_model_version,
            "conditioning": diffusion_model.conditioning_metadata(),
            "physics_config": physics,
            "train_config": train_cfg,
        },
    )

    diffusion_model.eval()
    if best_diffusion_checkpoint_path is not None:
        # 最终样图用验证集上最好的 diffusion 权重，避免最后一个 epoch 回退污染展示。
        best_diffusion_payload = torch_module.load(
            best_diffusion_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        load_diffusion_checkpoint_fn(diffusion_model, best_diffusion_payload, first_stage_model_version)
        run_meta["diffusion_best_loaded_for_sampling"] = True
        run_manager.write_meta(run_meta)
        print(f"loaded best diffusion for sampling: {best_diffusion_checkpoint_path}")
        diffusion_model.eval()
    sample_artifacts = save_post_train_samples(
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
        sample_latents_with_trace_fn,
        resolve_denoising_trace_steps_fn,
    )
    run_meta.update(sample_artifacts)
    run_manager.write_meta(run_meta)
