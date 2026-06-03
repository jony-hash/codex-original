from __future__ import annotations

from .evaluation import compute_first_stage_balanced_metrics, evaluate_diffusion_with_per_class, evaluate_first_stage_with_per_class
from .sampling import (
    build_diffusion_visualization_tensors,
    build_first_stage_visualization_tensors,
    labels_for_visual_rows,
    save_post_train_samples,
    save_visualization_grids,
)
from .teacher_training_utils import (
    _record_diffusion_best_history,
    _record_diffusion_train_step,
    _record_diffusion_validation,
    _record_first_stage_best_history,
    _record_first_stage_train_step,
    _record_first_stage_validation,
    _update_diffusion_best_checkpoint,
    _update_first_stage_best_checkpoint,
)
from .train_utils import (
    estimate_latent_scale_from_loader,
    require_strict_optical_best_checkpoint,
    resolve_image_fusion_weight,
    resolve_logged_step,
    resolve_stage_total_steps,
    set_first_stage_image_fusion_weight,
    should_log_step,
    should_save_epoch_visual,
    should_validate_epoch,
)


def run_first_stage_training(
    torch_module,
    *,
    first_stage,
    first_stage_loss_fn,
    first_stage_visual_batch,
    first_stage_visual_labels,
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
                _record_first_stage_train_step(
                    run_manager,
                    epoch_number=epoch_number,
                    step_record=step_record,
                    losses=losses,
                    image_fusion_weight=image_fusion_weight,
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
                save_visualization_grids(
                    run_manager,
                    build_first_stage_visualization_tensors(first_stage_visual_batch, visual_outputs),
                    run_manager.first_stage_images_dir / f"epoch_{epoch_number:04d}.png",
                    title_label=f"first_stage epoch {epoch_number}",
                    max_items=first_stage_visual_batch.shape[0],
                    row_labels=labels_for_visual_rows(first_stage_visual_labels)
                    if first_stage_visual_labels is not None
                    else None,
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
            _record_first_stage_validation(
                run_manager,
                epoch_number=epoch_number,
                validation_metrics=validation_metrics,
                per_class_metrics=per_class_metrics,
            )
            (
                best_optical_score,
                best_optical_checkpoint_path,
                optical_candidate,
                selection_score,
                updated_best,
            ) = _update_first_stage_best_checkpoint(
                run_manager,
                first_stage=first_stage,
                train_cfg=train_cfg,
                run_meta=run_meta,
                physics=physics,
                epoch_number=epoch_number,
                validation_metrics=validation_metrics,
                image_fusion_weight=image_fusion_weight,
                best_optical_score=best_optical_score,
                best_optical_checkpoint_path=best_optical_checkpoint_path,
                first_stage_model_version=first_stage_model_version,
            )
            _record_first_stage_best_history(
                run_manager,
                epoch_number=epoch_number,
                optical_candidate=optical_candidate,
                updated_best=updated_best,
                selection_score=selection_score,
                best_optical_score=best_optical_score,
                image_fusion_weight=image_fusion_weight,
                best_optical_checkpoint_path=best_optical_checkpoint_path,
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
    if train_cfg.mode == "teacher_full":
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
                _record_diffusion_train_step(
                    run_manager,
                    epoch_number=epoch_number,
                    step_record=step_record,
                    loss=loss,
                    latent_scale=latent_scale,
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
                save_visualization_grids(
                    run_manager,
                    build_diffusion_visualization_tensors(diffusion_visual_reference_images, sample_outputs),
                    run_manager.diffusion_images_dir / f"epoch_{epoch_number:04d}.png",
                    title_label=f"diffusion epoch {epoch_number}",
                    max_items=sample_count,
                    row_labels=labels_for_visual_rows(diffusion_visual_labels),
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
            _record_diffusion_validation(
                run_manager,
                epoch_number=epoch_number,
                validation_metrics=validation_metrics,
                per_class_metrics=per_class_metrics,
            )
            (
                best_diffusion_loss,
                best_diffusion_checkpoint_path,
                selection_loss,
                updated_best_diffusion,
            ) = _update_diffusion_best_checkpoint(
                run_manager,
                diffusion_model=diffusion_model,
                train_cfg=train_cfg,
                run_meta=run_meta,
                physics=physics,
                epoch_number=epoch_number,
                validation_metrics=validation_metrics,
                latent_scale=latent_scale,
                best_diffusion_loss=best_diffusion_loss,
                best_diffusion_checkpoint_path=best_diffusion_checkpoint_path,
                first_stage_model_version=first_stage_model_version,
                diffusion_model_version=diffusion_model_version,
            )
            _record_diffusion_best_history(
                run_manager,
                epoch_number=epoch_number,
                updated_best_diffusion=updated_best_diffusion,
                selection_loss=selection_loss,
                best_diffusion_loss=best_diffusion_loss,
                best_diffusion_checkpoint_path=best_diffusion_checkpoint_path,
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
