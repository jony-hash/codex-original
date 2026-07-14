from __future__ import annotations

from ..generation.codes import _sample_one_shot_codes
from .training_utils import (
    ONE_SHOT_HISTORY_LOSS_KEYS,
    _build_one_shot_optimizer,
    _evaluate_one_shot,
    _mask_amplitude_summary,
    _one_shot_loss_kwargs,
    _phase_grad_summary,
    _scalar_loss_items,
    _select_one_shot_visual_batch,
)
from .artifacts import (
    _save_one_shot_d2nn_diagnostics,
    _save_one_shot_slm_to_encoder_focus_visualization,
)
from .evaluation import _save_strict_optical_field_artifacts
from .visualization import build_one_shot_shared_scale_visualization_tensors
from ..visualization_core import labels_for_visual_rows, save_visualization_grids
from ..stage_constants import ONE_SHOT_BEST_CHECKPOINT, ONE_SHOT_BEST_METRIC
from ..train_utils import (
    resolve_logged_step,
    resolve_stage_total_steps,
    should_log_step,
    should_save_epoch_visual,
    should_validate_epoch,
)


def run_one_shot_training(
    torch_module,
    *,
    one_shot_model,
    one_shot_loss_fn,
    run_manager,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    one_shot_model_version: str,
) -> None:
    optimizer = _build_one_shot_optimizer(torch_module, one_shot_model, train_cfg)
    loader_length = len(train_bundle.loader)
    total_steps = resolve_stage_total_steps(loader_length, train_cfg.one_shot_epochs)
    visual_codes, visual_labels, visual_references = _select_one_shot_visual_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg,
        device,
    )
    one_shot_stage_step = 0
    best_validation_loss = None
    best_checkpoint_path = None
    one_shot_model.train()
    for epoch_index in range(train_cfg.one_shot_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (images, labels) in enumerate(train_bundle.loader):
            target_batch = images.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            codes = _sample_one_shot_codes(
                torch_module, target_batch.shape[0], train_cfg, device
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = one_shot_model(codes, class_labels)
            losses = one_shot_loss_fn(
                outputs,
                target_batch,
                class_labels=class_labels,
                **_one_shot_loss_kwargs(train_cfg),
            )
            losses["loss"].backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, one_shot_stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                phase_info = _phase_grad_summary(one_shot_model)
                mask_info = _mask_amplitude_summary(one_shot_model)
                print(
                    f"[one_shot] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"slm_tv={losses['slm_tv'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"sat_low={losses['camera_low_saturation_ratio'].item():.4f} "
                    f"sat_high={losses['camera_high_saturation_ratio'].item():.4f} "
                    f"div={losses['code_diversity_penalty'].item():.4f} "
                    f"div_ratio={losses['diversity_ratio'].item():.4f} "
                    f"xcls_ratio={losses['cross_class_diversity_ratio'].item():.4f} "
                    f"{phase_info} {mask_info}"
                )
                run_manager.append_history(
                    "one_shot",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        **_scalar_loss_items(losses, ONE_SHOT_HISTORY_LOSS_KEYS),
                    },
                )
            one_shot_stage_step += 1

        if should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval):
            one_shot_model.eval()
            with torch_module.no_grad():
                visual_outputs = one_shot_model(
                    visual_codes,
                    visual_labels,
                    return_optical_fields=True,
                )
                save_visualization_grids(
                    run_manager,
                    {
                        "intensity": build_one_shot_shared_scale_visualization_tensors(
                            visual_references,
                            visual_outputs,
                            codes=visual_codes,
                        ).get("intensity", {}),
                    },
                    run_manager.images_dir
                    / "one_shot"
                    / f"epoch_{epoch_number:04d}.png",
                    title_label=f"one_shot epoch {epoch_number} intensity/control diagnostics",
                    max_items=visual_labels.shape[0],
                    row_labels=labels_for_visual_rows(visual_labels),
                )
                _save_strict_optical_field_artifacts(
                    run_manager,
                    visual_outputs,
                    artifact_prefix=f"one_shot_epoch_{epoch_number:04d}",
                    title_prefix=f"one-shot epoch {epoch_number}",
                    max_items=visual_labels.shape[0],
                    row_labels=labels_for_visual_rows(visual_labels),
                    front_path_model=physics.front_path_model,
                )
                _save_one_shot_d2nn_diagnostics(
                    run_manager,
                    visual_outputs,
                    f"one_shot/epoch_{epoch_number:04d}_d2nn_diagnostics",
                    {
                        "stage": "one_shot",
                        "epoch": epoch_number,
                        "class_labels": visual_labels.detach().cpu().tolist(),
                    },
                )
                _save_one_shot_slm_to_encoder_focus_visualization(
                    run_manager,
                    visual_outputs,
                    run_manager.images_dir
                    / "one_shot"
                    / "d2nn_focus"
                    / f"epoch_{epoch_number:04d}_slm_to_encoder_focus.png",
                    title_label=f"one_shot epoch {epoch_number} SLM to encoder focus",
                    row_label=f"label {int(visual_labels[0].detach().cpu().item())}",
                )
            one_shot_model.train()

        if validation_bundle is not None and should_validate_epoch(
            epoch_index,
            train_cfg.one_shot_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics = _evaluate_one_shot(
                torch_module,
                one_shot_model,
                one_shot_loss_fn,
                validation_bundle.loader,
                train_cfg,
                device,
                use_pin_memory,
            )
            selection_loss = validation_metrics["loss"]
            updated_best = (
                best_validation_loss is None or selection_loss < best_validation_loss
            )
            print(
                f"[one_shot][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"slm_tv={validation_metrics['slm_tv']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"div={validation_metrics['code_diversity_penalty']:.4f} "
                f"xcls_ratio={validation_metrics['cross_class_diversity_ratio']:.4f}"
            )
            run_manager.append_history(
                "one_shot_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            if updated_best:
                best_validation_loss = selection_loss
                best_checkpoint_path = run_manager.save_checkpoint(
                    ONE_SHOT_BEST_CHECKPOINT,
                    {
                        "model_state_dict": one_shot_model.state_dict(),
                        "stage": "one_shot",
                        "model_version": one_shot_model_version,
                        "physics_config": physics,
                        "train_config": train_cfg,
                        "epoch": epoch_number,
                        "selection_metric": ONE_SHOT_BEST_METRIC,
                        "selection_loss": selection_loss,
                        "validation_metrics": validation_metrics,
                    },
                )
                run_meta.update(
                    {
                        "one_shot_best_checkpoint": best_checkpoint_path,
                        "one_shot_best_epoch": epoch_number,
                        "one_shot_best_loss": best_validation_loss,
                    }
                )
                run_manager.write_meta(run_meta)
            run_manager.append_history(
                "one_shot_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best,
                    "selection_metric": ONE_SHOT_BEST_METRIC,
                    "selection_loss": selection_loss,
                    "best_loss": best_validation_loss,
                    "checkpoint": best_checkpoint_path,
                },
            )
            one_shot_model.train()

    run_manager.save_checkpoint(
        "one_shot_latest.pt",
        {
            "model_state_dict": one_shot_model.state_dict(),
            "stage": "one_shot",
            "model_version": one_shot_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
            "total_steps": total_steps,
        },
    )
