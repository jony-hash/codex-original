from __future__ import annotations

from dataclasses import dataclass

from .one_shot_distill_data import (
    _build_distill_loaders,
    _load_distill_payload,
    _subset_distill_payload_for_training,
    _validated_distill_tensors,
)
from .one_shot_eval_artifacts import (
    _save_one_shot_d2nn_diagnostics,
    _save_one_shot_d2nn_focus_visualization,
    _save_one_shot_slm_to_encoder_focus_visualization,
)
from .one_shot_quality import _resolve_one_shot_distill_selection
from .one_shot_training_utils import (
    ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS,
    _build_one_shot_optimizer,
    _evaluate_one_shot_distill,
    _one_shot_d2nn_crop_retention_metrics,
    _one_shot_loss_kwargs,
    _phase_grad_summary,
    _scalar_loss_items,
)
from .sampling import (
    build_one_shot_visualization_tensors,
    labels_for_visual_rows,
    save_visualization_grids,
    select_balanced_visual_preview_indices,
)
from .stage_constants import ONE_SHOT_DISTILL_BEST_CHECKPOINT, ONE_SHOT_DISTILL_LATEST_CHECKPOINT
from .train_utils import (
    resolve_logged_step,
    resolve_stage_total_steps,
    should_log_step,
    should_save_epoch_visual,
    should_validate_epoch,
)


@dataclass(frozen=True)
class _DistillLoaderBundle:
    payload: dict[str, object]
    source_total_size: int
    effective_total_size: int
    train_loader: object
    validation_loader: object
    total_size: int
    train_size: int
    validation_size: int
    effective_train_batch_size: int


@dataclass(frozen=True)
class _DistillVisualBatch:
    codes: object
    labels: object
    references: object
    summary: dict[str, object]


def _maybe_load_one_shot_warm_start(
    torch_module,
    *,
    one_shot_model,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_checkpoint: str | None,
    load_one_shot_checkpoint_fn,
) -> None:
    if one_shot_checkpoint is None:
        return
    if load_one_shot_checkpoint_fn is None:
        raise ValueError("one_shot_distill warm-start requires load_one_shot_checkpoint_fn.")
    payload = torch_module.load(one_shot_checkpoint, map_location=device, weights_only=False)
    load_one_shot_checkpoint_fn(one_shot_model, payload)
    run_meta["one_shot_distill_warm_start_checkpoint"] = one_shot_checkpoint
    run_manager.write_meta(run_meta)
    print(f"warm-started one_shot_distill from checkpoint: {one_shot_checkpoint}")


def _prepare_distill_loaders(torch_module, *, distill_dataset: str, train_cfg, use_pin_memory: bool):
    source_payload = _load_distill_payload(torch_module, distill_dataset)
    payload, source_total_size, effective_total_size = _subset_distill_payload_for_training(
        torch_module,
        source_payload,
        train_cfg,
    )
    train_loader, validation_loader, total_size, train_size, validation_size = _build_distill_loaders(
        torch_module,
        payload,
        train_cfg,
        use_pin_memory,
    )
    uses_custom_batch_sampler = train_cfg.distillation.paired_class_batches or train_cfg.distillation.multiclass_batches
    effective_train_batch_size = (
        max((len(batch) for batch in getattr(train_loader, "batch_sampler", [])), default=train_cfg.batch_size)
        if uses_custom_batch_sampler
        else train_cfg.batch_size
    )
    return _DistillLoaderBundle(
        payload=payload,
        source_total_size=source_total_size,
        effective_total_size=effective_total_size,
        train_loader=train_loader,
        validation_loader=validation_loader,
        total_size=total_size,
        train_size=train_size,
        validation_size=validation_size,
        effective_train_batch_size=effective_train_batch_size,
    )


def _select_distill_visual_batch(torch_module, *, payload, train_cfg, device, use_pin_memory: bool):
    distill_codes, distill_labels, distill_targets = _validated_distill_tensors(torch_module, payload, train_cfg)
    visual_indices, visual_summary = select_balanced_visual_preview_indices(
        distill_labels,
        max_per_class=1,
        class_labels=tuple(range(train_cfg.one_shot.num_classes)),
    )
    if not visual_indices:
        raise ValueError("one_shot_distill visualization selected zero samples.")
    visual_index_tensor = torch_module.tensor(visual_indices, dtype=torch_module.long)
    visual_codes = distill_codes.index_select(0, visual_index_tensor)
    visual_labels = distill_labels.index_select(0, visual_index_tensor)
    visual_references = distill_targets.index_select(0, visual_index_tensor)
    return _DistillVisualBatch(
        codes=visual_codes.to(device, non_blocking=use_pin_memory),
        labels=visual_labels.to(device, non_blocking=use_pin_memory),
        references=visual_references.to(device, non_blocking=use_pin_memory),
        summary=visual_summary,
    )


def _record_distill_run_setup(
    run_manager,
    run_meta: dict[str, object],
    *,
    distill_dataset: str,
    train_cfg,
    loaders,
    visual_summary: dict[str, object],
) -> None:
    run_manager.save_config_snapshot(
        "one_shot_distill_config",
        {
            "distill_dataset": distill_dataset,
            "distill_source_total_size": loaders.source_total_size,
            "distill_train_subset_size": train_cfg.distillation.train_subset_size,
            "distill_effective_total_size": loaders.effective_total_size,
            "dataset_total_size": loaders.total_size,
            "train_size": loaders.train_size,
            "validation_size": loaders.validation_size,
            "paired_class_batches": train_cfg.distillation.paired_class_batches,
            "multiclass_batches": train_cfg.distillation.multiclass_batches,
            "effective_train_batch_size": loaders.effective_train_batch_size,
            "visual_preview": visual_summary,
            "source": "train.py_one_shot_distill",
        },
    )
    run_meta.update(
        {
            "distill_dataset": distill_dataset,
            "distill_source_total_size": loaders.source_total_size,
            "distill_train_subset_size": train_cfg.distillation.train_subset_size,
            "distill_effective_total_size": loaders.effective_total_size,
            "distill_dataset_total_size": loaders.total_size,
            "distill_train_size": loaders.train_size,
            "distill_validation_size": loaders.validation_size,
            "distill_paired_class_batches": train_cfg.distillation.paired_class_batches,
            "distill_multiclass_batches": train_cfg.distillation.multiclass_batches,
            "distill_effective_train_batch_size": loaders.effective_train_batch_size,
        }
    )
    run_manager.write_meta(run_meta)


def _save_latest_distill_checkpoint(
    run_manager,
    run_meta: dict[str, object],
    *,
    one_shot_model,
    train_cfg,
    physics,
    epoch_number: int,
    stage_step: int,
    total_steps: int,
    one_shot_model_version: str,
    distill_dataset: str,
) -> None:
    latest_path = run_manager.save_checkpoint(
        ONE_SHOT_DISTILL_LATEST_CHECKPOINT,
        {
            "model_state_dict": one_shot_model.state_dict(),
            "stage": "one_shot_distill",
            "model_version": one_shot_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
            "epoch": epoch_number,
            "stage_step": stage_step,
            "total_steps": total_steps,
            "distill_dataset": distill_dataset,
        },
    )
    run_meta.update(
        {
            "one_shot_distill_latest_checkpoint": latest_path,
            "one_shot_distill_latest_epoch": epoch_number,
            "one_shot_distill_latest_step": stage_step,
        }
    )
    run_manager.write_meta(run_meta)


def _save_best_distill_checkpoint(
    run_manager,
    run_meta: dict[str, object],
    *,
    one_shot_model,
    train_cfg,
    physics,
    epoch_number: int,
    one_shot_model_version: str,
    selection_metric: str,
    selection_score: float,
    validation_metrics: dict[str, float],
    distill_dataset: str,
):
    best_checkpoint_path = run_manager.save_checkpoint(
        ONE_SHOT_DISTILL_BEST_CHECKPOINT,
        {
            "model_state_dict": one_shot_model.state_dict(),
            "stage": "one_shot_distill",
            "model_version": one_shot_model_version,
            "physics_config": physics,
            "train_config": train_cfg,
            "epoch": epoch_number,
            "selection_metric": selection_metric,
            "selection_loss": selection_score,
            "selection_score": selection_score,
            "validation_metrics": validation_metrics,
            "distill_dataset": distill_dataset,
        },
    )
    run_meta.update(
        {
            "one_shot_best_checkpoint": best_checkpoint_path,
            "one_shot_best_epoch": epoch_number,
            "one_shot_best_loss": selection_score,
            "one_shot_best_score": selection_score,
            "one_shot_distill_best_checkpoint": best_checkpoint_path,
            "one_shot_distill_best_epoch": epoch_number,
            "one_shot_distill_best_loss": selection_score,
            "one_shot_distill_best_score": selection_score,
            "one_shot_distill_best_metric": selection_metric,
        }
    )
    run_manager.write_meta(run_meta)
    return best_checkpoint_path


def run_one_shot_distill_training(
    torch_module,
    *,
    one_shot_model,
    one_shot_loss_fn,
    run_manager,
    train_cfg,
    run_meta: dict[str, object],
    physics,
    device,
    use_pin_memory: bool,
    one_shot_model_version: str,
    distill_dataset: str,
    one_shot_checkpoint: str | None = None,
    load_one_shot_checkpoint_fn=None,
) -> None:
    _maybe_load_one_shot_warm_start(
        torch_module,
        one_shot_model=one_shot_model,
        run_manager=run_manager,
        run_meta=run_meta,
        device=device,
        one_shot_checkpoint=one_shot_checkpoint,
        load_one_shot_checkpoint_fn=load_one_shot_checkpoint_fn,
    )
    loaders = _prepare_distill_loaders(
        torch_module,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        use_pin_memory=use_pin_memory,
    )
    train_loader = loaders.train_loader
    validation_loader = loaders.validation_loader
    optimizer = _build_one_shot_optimizer(torch_module, one_shot_model, train_cfg)
    loader_length = len(train_loader)
    total_steps = resolve_stage_total_steps(loader_length, train_cfg.one_shot_epochs)
    visual_batch = _select_distill_visual_batch(
        torch_module,
        payload=loaders.payload,
        train_cfg=train_cfg,
        device=device,
        use_pin_memory=use_pin_memory,
    )
    stage_step = 0
    best_selection_score = None
    best_checkpoint_path = None
    _record_distill_run_setup(
        run_manager,
        run_meta,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        loaders=loaders,
        visual_summary=visual_batch.summary,
    )

    one_shot_model.train()
    for epoch_index in range(train_cfg.one_shot_epochs):
        epoch_number = epoch_index + 1
        for batch_index, (codes, labels, targets) in enumerate(train_loader):
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            outputs = one_shot_model(code_batch, class_labels)
            losses = one_shot_loss_fn(outputs, target_batch, class_labels=class_labels, **_one_shot_loss_kwargs(train_cfg))
            losses["loss"].backward()
            optimizer.step()
            step_record = resolve_logged_step(batch_index, stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                phase_info = _phase_grad_summary(one_shot_model)
                print(
                    f"[one_shot_distill] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"shape={losses['shape_penalty'].item():.4f} "
                    f"edge={losses['edge_penalty'].item():.4f} "
                    f"slm_tv={losses['slm_tv'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"div_pairs={losses['diversity_pair_count'].item():.0f} "
                    f"div_ratio={losses['diversity_ratio'].item():.4f} "
                    f"xcls_pairs={losses['cross_class_pair_count'].item():.0f} "
                    f"xcls_ratio={losses['cross_class_diversity_ratio'].item():.4f} "
                    f"{phase_info}"
                )
                run_manager.append_history(
                    "one_shot_distill",
                    {
                        "epoch": epoch_number,
                        **step_record,
                        **_scalar_loss_items(losses, ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS),
                        **_one_shot_d2nn_crop_retention_metrics(outputs),
                    },
                )
            stage_step += 1

        _save_latest_distill_checkpoint(
            run_manager,
            run_meta,
            one_shot_model=one_shot_model,
            train_cfg=train_cfg,
            physics=physics,
            epoch_number=epoch_number,
            stage_step=stage_step,
            total_steps=total_steps,
            one_shot_model_version=one_shot_model_version,
            distill_dataset=distill_dataset,
        )

        if should_save_epoch_visual(epoch_index, train_cfg.visual_save_epoch_interval):
            one_shot_model.eval()
            with torch_module.no_grad():
                visual_outputs = one_shot_model(visual_batch.codes, visual_batch.labels)
                save_visualization_grids(
                    run_manager,
                    build_one_shot_visualization_tensors(
                        visual_batch.references,
                        visual_outputs,
                        codes=visual_batch.codes,
                    ),
                    run_manager.images_dir / "one_shot_distill" / f"epoch_{epoch_number:04d}.png",
                    title_label=f"one_shot_distill epoch {epoch_number}",
                    max_items=visual_batch.labels.shape[0],
                    row_labels=labels_for_visual_rows(visual_batch.labels),
                )
                _save_one_shot_d2nn_focus_visualization(
                    run_manager,
                    visual_outputs,
                    run_manager.images_dir
                    / "one_shot_distill"
                    / "d2nn_focus"
                    / f"epoch_{epoch_number:04d}_encoder_L1_focus.png",
                    title_label=f"one_shot_distill epoch {epoch_number} encoder L1 focus",
                    row_label=f"label {int(visual_batch.labels[0].detach().cpu().item())}",
                )
                _save_one_shot_slm_to_encoder_focus_visualization(
                    run_manager,
                    visual_outputs,
                    run_manager.images_dir
                    / "one_shot_distill"
                    / "d2nn_focus"
                    / f"epoch_{epoch_number:04d}_slm_to_encoder_focus.png",
                    title_label=f"one_shot_distill epoch {epoch_number} SLM to encoder focus",
                    row_label=f"label {int(visual_batch.labels[0].detach().cpu().item())}",
                )
                _save_one_shot_d2nn_diagnostics(
                    run_manager,
                    visual_outputs,
                    f"one_shot_distill/epoch_{epoch_number:04d}_d2nn_diagnostics",
                    {
                        "stage": "one_shot_distill",
                        "epoch": epoch_number,
                        "class_labels": visual_batch.labels.detach().cpu().tolist(),
                    },
                )
            one_shot_model.train()

        if validation_loader is not None and should_validate_epoch(
            epoch_index,
            train_cfg.one_shot_epochs,
            train_cfg.validation_epoch_interval,
        ):
            validation_metrics = _evaluate_one_shot_distill(
                torch_module,
                one_shot_model,
                one_shot_loss_fn,
                validation_loader,
                train_cfg,
                device,
                use_pin_memory,
            )
            selection_score, selection_metric = _resolve_one_shot_distill_selection(train_cfg, validation_metrics)
            updated_best = best_selection_score is None or selection_score < best_selection_score
            print(
                f"[one_shot_distill][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"shape={validation_metrics['shape_penalty']:.4f} "
                f"edge={validation_metrics['edge_penalty']:.4f} "
                f"slm_tv={validation_metrics['slm_tv']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"div_ratio={validation_metrics['diversity_ratio']:.4f} "
                f"xcls_ratio={validation_metrics['cross_class_diversity_ratio']:.4f} "
                f"selection={selection_score:.6f} ({selection_metric})"
            )
            run_manager.append_history(
                "one_shot_distill_validation",
                {
                    "epoch": epoch_number,
                    **validation_metrics,
                },
            )
            if updated_best:
                best_selection_score = selection_score
                best_checkpoint_path = _save_best_distill_checkpoint(
                    run_manager,
                    run_meta,
                    one_shot_model=one_shot_model,
                    train_cfg=train_cfg,
                    physics=physics,
                    epoch_number=epoch_number,
                    one_shot_model_version=one_shot_model_version,
                    selection_metric=selection_metric,
                    selection_score=selection_score,
                    validation_metrics=validation_metrics,
                    distill_dataset=distill_dataset,
                )
            run_manager.append_history(
                "one_shot_distill_best",
                {
                    "epoch": epoch_number,
                    "updated": updated_best,
                    "selection_metric": selection_metric,
                    "selection_loss": selection_score,
                    "selection_score": selection_score,
                    "best_loss": best_selection_score,
                    "best_score": best_selection_score,
                    "checkpoint": best_checkpoint_path,
                },
            )
            one_shot_model.train()
