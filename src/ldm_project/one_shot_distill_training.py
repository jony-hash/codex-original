from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .one_shot_distill_data import (
    _build_distill_loaders,
    _load_distill_payload,
    _normalize_distill_target_bank,
    _resolve_aux_detail_targets,
    _resolve_distill_targets,
    _subset_distill_payload_for_training,
    _validated_distill_tensors,
)
from .one_shot_model import load_one_shot_checkpoint_subset
from .one_shot_eval import _save_strict_optical_field_artifacts
from .one_shot_losses import transform_aux_topology_target
from .one_shot_eval_artifacts import (
    _save_one_shot_d2nn_diagnostics,
    _save_one_shot_d2nn_focus_visualization,
    _save_one_shot_slm_to_encoder_focus_visualization,
)
from .one_shot_quality import _resolve_one_shot_distill_selection
from .teacher_distill import _build_mnist_label_prototype_bank
from .one_shot_training_utils import (
    ONE_SHOT_DISTILL_HISTORY_LOSS_KEYS,
    _accumulate_target_code_bridge_trainability_gradients,
    _build_one_shot_optimizer,
    _accumulate_decoder_trainability_gradients,
    _evaluate_one_shot_distill,
    _finalize_target_code_bridge_trainability_audit,
    _finalize_decoder_trainability_audit,
    _init_target_code_bridge_trainability_audit,
    _init_decoder_trainability_audit,
    _one_shot_d2nn_crop_retention_metrics,
    _one_shot_loss_kwargs,
    _mask_amplitude_summary,
    _phase_grad_summary,
    _scalar_loss_items,
    _snapshot_frozen_one_shot_parameters,
    _verify_frozen_one_shot_parameters,
)
from .one_shot_visualization import (
    build_one_shot_distill_residual_visualization_tensors,
    build_one_shot_distill_readability_visualization_tensors,
    build_one_shot_shared_scale_visualization_tensors,
)
from .sampling_reports import select_balanced_visual_preview_indices
from .visualization_core import (
    labels_for_visual_rows,
    save_visualization_grids,
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
    validation_dataset_path: str | None = None
    validation_source: str = "internal_split"


@dataclass(frozen=True)
class _DistillVisualBatch:
    codes: object
    labels: object
    references: object
    summary: dict[str, object]


def _build_student_label_margin_prototypes(torch_module, payload: dict[str, object], train_cfg):
    needs_prototypes = (
        float(train_cfg.one_shot.student_label_margin_weight) > 0.0
        or float(train_cfg.one_shot.student_prototype_ce_weight) > 0.0
        or float(train_cfg.one_shot.student_negative_label_margin_weight) > 0.0
        or float(train_cfg.one_shot.student_roi_prototype_ce_weight) > 0.0
        or float(train_cfg.one_shot.student_roi_negative_label_margin_weight) > 0.0
        or (
            float(train_cfg.one_shot.aux_semantic_weight) > 0.0
            and (
                float(train_cfg.one_shot.aux_semantic_prototype_ce_weight) > 0.0
                or float(train_cfg.one_shot.aux_semantic_negative_label_margin_weight) > 0.0
            )
        )
        or train_cfg.one_shot.checkpoint_selection_metric == "strict_readability"
    )
    if not needs_prototypes:
        return None, None
    source_key = train_cfg.one_shot.student_label_margin_prototype_source
    if source_key == "mnist":
        prototypes = _build_mnist_label_prototype_bank(
            torch_module,
            class_labels=tuple(range(int(train_cfg.one_shot.num_classes))),
            image_size=int(train_cfg.first_stage.image_size),
            root=train_cfg.data.root,
            max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
        )
        return prototypes.contiguous(), source_key
    if source_key == "teacher_image" and "teacher_image" not in payload:
        raise ValueError("student_label_margin_prototype_source='teacher_image' requires teacher_image in dataset.")
    source = payload[source_key].float()
    labels = payload["class_labels"].long()
    prototypes = []
    for label in range(int(train_cfg.one_shot.num_classes)):
        mask = labels == int(label)
        if not bool(mask.any().item()):
            raise ValueError(f"Cannot build student label-margin prototype: class {label} has no samples.")
        prototypes.append(source[mask].mean(dim=0))
    prototype_tensor = torch_module.stack(prototypes, dim=0).contiguous()
    return prototype_tensor, source_key

def _build_distill_target_prototypes(torch_module, train_cfg):
    if (
        train_cfg.one_shot.distill_target_source != "mnist_prototype"
        and train_cfg.one_shot.aux_detail_target_source != "mnist_prototype"
        and not (
            train_cfg.one_shot.generator_e_arch in {"label_template", "label_template_code_residual"}
            and train_cfg.one_shot.generator_template_init == "mnist_prototype"
        )
    ):
        return None
    prototypes = _build_mnist_label_prototype_bank(
        torch_module,
        class_labels=tuple(range(int(train_cfg.one_shot.num_classes))),
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    )
    return prototypes.contiguous()


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


def _validate_frozen_generator_warm_start(
    train_cfg,
    *,
    one_shot_checkpoint: str | None,
    one_shot_generator_checkpoint: str | None,
) -> None:
    if float(train_cfg.one_shot.generator_base_lr_scale) > 0.0:
        return
    if (
        train_cfg.one_shot.generator_e_arch in {"label_template", "label_template_code_residual"}
        and train_cfg.one_shot.generator_template_init == "mnist_prototype"
    ):
        return
    if one_shot_checkpoint is not None or one_shot_generator_checkpoint is not None:
        return
    raise ValueError(
        "one_shot_distill freezes generator_e base parameters "
        "(one_shot.generator_base_lr_scale=0) and therefore requires an explicit "
        "--one-shot-generator-checkpoint or --one-shot-checkpoint."
    )


def _maybe_initialize_label_template(
    torch_module,
    *,
    one_shot_model,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    one_shot_checkpoint: str | None,
    generator_checkpoint: str | None,
) -> None:
    if train_cfg.one_shot.generator_e_arch not in {"label_template", "label_template_code_residual"}:
        return
    if one_shot_checkpoint is not None or generator_checkpoint is not None:
        return
    if train_cfg.one_shot.generator_template_init != "mnist_prototype":
        raise ValueError("label_template requires an explicit supported template initializer")
    prototypes = _build_distill_target_prototypes(torch_module, train_cfg)
    if prototypes is None:
        raise ValueError("label_template initialization requires mnist_prototype targets")
    template_bank = _normalize_distill_target_bank(prototypes).contiguous()
    one_shot_model.generator_e.initialize_label_templates(template_bank)
    template_bytes = template_bank.detach().cpu().numpy().tobytes()
    run_meta.update(
        {
            "one_shot_generator_template_source": "mnist_prototype",
            "one_shot_generator_template_shape": list(template_bank.shape),
            "one_shot_generator_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
            "one_shot_generator_template_min": float(template_bank.min().item()),
            "one_shot_generator_template_max": float(template_bank.max().item()),
            "one_shot_generator_command_transform": train_cfg.one_shot.generator_command_transform,
            "one_shot_generator_code_conditioning_active": (
                train_cfg.one_shot.generator_e_arch == "label_template_code_residual"
            ),
        }
    )
    run_manager.write_meta(run_meta)


def _maybe_load_one_shot_module_warm_starts(
    torch_module,
    *,
    one_shot_model,
    run_manager,
    run_meta: dict[str, object],
    device,
    generator_checkpoint: str | None,
    optical_checkpoint: str | None,
) -> None:
    loaded = False
    if generator_checkpoint is not None:
        payload = torch_module.load(generator_checkpoint, map_location=device, weights_only=False)
        summary = load_one_shot_checkpoint_subset(
            one_shot_model,
            payload,
            include_prefixes=("generator_e.",),
            description="one-shot generator warm-start checkpoint",
        )
        run_meta["one_shot_distill_generator_warm_start_checkpoint"] = generator_checkpoint
        run_meta["one_shot_distill_generator_warm_start_summary"] = summary
        print(f"warm-started one_shot_distill generator_e from checkpoint: {generator_checkpoint}")
        loaded = True
    if optical_checkpoint is not None:
        payload = torch_module.load(optical_checkpoint, map_location=device, weights_only=False)
        summary = load_one_shot_checkpoint_subset(
            one_shot_model,
            payload,
            include_prefixes=("encoder.", "decoder."),
            description="one-shot optical warm-start checkpoint",
            expected_missing_prefixes=("encoder.amplitude_logits", "decoder.amplitude_logits"),
        )
        run_meta["one_shot_distill_optical_warm_start_checkpoint"] = optical_checkpoint
        run_meta["one_shot_distill_optical_warm_start_summary"] = summary
        print(f"warm-started one_shot_distill optical core from checkpoint: {optical_checkpoint}")
        loaded = True
    if loaded:
        run_manager.write_meta(run_meta)


def _effective_distill_train_batch_size(train_loader, train_cfg) -> int:
    uses_custom_batch_sampler = train_cfg.distillation.paired_class_batches or train_cfg.distillation.multiclass_batches
    return (
        max((len(batch) for batch in getattr(train_loader, "batch_sampler", [])), default=train_cfg.batch_size)
        if uses_custom_batch_sampler
        else train_cfg.batch_size
    )


def _prepare_distill_loaders(
    torch_module,
    *,
    distill_dataset: str,
    train_cfg,
    use_pin_memory: bool,
    distill_validation_dataset: str | None = None,
):
    source_payload = _load_distill_payload(torch_module, distill_dataset)
    payload, source_total_size, effective_total_size = _subset_distill_payload_for_training(
        torch_module,
        source_payload,
        train_cfg,
    )
    target_prototypes = _build_distill_target_prototypes(torch_module, train_cfg)
    loader_cfg = train_cfg
    if distill_validation_dataset is not None:
        loader_cfg = replace(train_cfg, data=replace(train_cfg.data, validation_fraction=0.0))
    train_loader, validation_loader, total_size, train_size, validation_size = _build_distill_loaders(
        torch_module,
        payload,
        loader_cfg,
        use_pin_memory,
        mnist_prototypes=target_prototypes,
    )
    validation_source = "internal_split"
    if distill_validation_dataset is not None:
        validation_payload = _load_distill_payload(torch_module, distill_validation_dataset)
        val_codes, val_labels, _ = _validated_distill_tensors(torch_module, validation_payload, train_cfg)
        val_targets = _resolve_distill_targets(
            torch_module,
            validation_payload,
            train_cfg,
            mnist_prototypes=target_prototypes,
        )
        val_aux_targets = _resolve_aux_detail_targets(
            torch_module,
            validation_payload,
            train_cfg,
            mnist_prototypes=target_prototypes,
        )
        if val_aux_targets is None:
            validation_dataset = torch_module.utils.data.TensorDataset(val_codes, val_labels, val_targets)
        else:
            validation_dataset = torch_module.utils.data.TensorDataset(
                val_codes,
                val_labels,
                val_targets,
                val_aux_targets,
            )
        validation_loader = torch_module.utils.data.DataLoader(
            validation_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.data.num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
        validation_size = len(validation_dataset)
        total_size = train_size + validation_size
        validation_source = "external_dataset"
    return _DistillLoaderBundle(
        payload=payload,
        source_total_size=source_total_size,
        effective_total_size=effective_total_size,
        train_loader=train_loader,
        validation_loader=validation_loader,
        total_size=total_size,
        train_size=train_size,
        validation_size=validation_size,
        effective_train_batch_size=_effective_distill_train_batch_size(train_loader, train_cfg),
        validation_dataset_path=distill_validation_dataset,
        validation_source=validation_source,
    )


def _select_distill_visual_batch(torch_module, *, payload, train_cfg, device, use_pin_memory: bool):
    target_prototypes = _build_distill_target_prototypes(torch_module, train_cfg)
    distill_codes, distill_labels, _ = _validated_distill_tensors(torch_module, payload, train_cfg)
    distill_targets = _resolve_distill_targets(
        torch_module,
        payload,
        train_cfg,
        mnist_prototypes=target_prototypes,
    )
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


def _distill_run_setup_metadata(
    *,
    distill_dataset: str,
    distill_validation_dataset: str | None,
    train_cfg,
    loaders,
) -> dict[str, object]:
    return {
        "distill_dataset": distill_dataset,
        "distill_validation_dataset": distill_validation_dataset,
        "distill_validation_source": loaders.validation_source,
        "one_shot_distill_target_source": train_cfg.one_shot.distill_target_source,
        "one_shot_aux_detail_target_source": train_cfg.one_shot.aux_detail_target_source,
        "one_shot_aux_topology_target_transform": train_cfg.one_shot.aux_topology_target_transform,
        "one_shot_decoder_trainability_audit": train_cfg.one_shot.decoder_trainability_audit,
        "one_shot_target_code_bridge_trainability_audit": (
            train_cfg.one_shot.target_code_bridge_trainability_audit
        ),
        "one_shot_target_code_diagnostic_mode": train_cfg.one_shot.target_code_diagnostic_mode,
        "one_shot_target_code_spatial_scale": train_cfg.one_shot.target_code_spatial_scale,
        "one_shot_target_code_bridge_source": train_cfg.one_shot.target_code_bridge_source,
        "one_shot_target_code_bridge_arch": train_cfg.one_shot.target_code_bridge_arch,
    }


def _record_distill_run_setup(
    run_manager,
    run_meta: dict[str, object],
    *,
    distill_dataset: str,
    distill_validation_dataset: str | None,
    train_cfg,
    loaders,
    visual_summary: dict[str, object],
) -> None:
    physical_phase_available = train_cfg.one_shot.optical_core_arch != "camera_direct_clean_oracle"
    setup_metadata = _distill_run_setup_metadata(
        distill_dataset=distill_dataset,
        distill_validation_dataset=distill_validation_dataset,
        train_cfg=train_cfg,
        loaders=loaders,
    )
    run_manager.save_config_snapshot(
        "one_shot_distill_config",
        {
            **setup_metadata,
            "physical_phase_available": physical_phase_available,
            "visualization_mode": "optical_chain" if physical_phase_available else "direct_readable",
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
            **setup_metadata,
            "one_shot_physical_phase_available": physical_phase_available,
            "one_shot_visualization_mode": "optical_chain" if physical_phase_available else "direct_readable",
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


def _distill_checkpoint_payload(
    *,
    one_shot_model,
    train_cfg,
    physics,
    epoch_number: int,
    one_shot_model_version: str,
    **details,
) -> dict[str, object]:
    return {
        "model_state_dict": one_shot_model.state_dict(),
        "stage": "one_shot_distill",
        "model_version": one_shot_model_version,
        "physics_config": physics,
        "train_config": train_cfg,
        "epoch": epoch_number,
        **details,
    }


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
        _distill_checkpoint_payload(
            one_shot_model=one_shot_model,
            train_cfg=train_cfg,
            physics=physics,
            epoch_number=epoch_number,
            one_shot_model_version=one_shot_model_version,
            stage_step=stage_step,
            total_steps=total_steps,
            distill_dataset=distill_dataset,
        ),
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
        _distill_checkpoint_payload(
            one_shot_model=one_shot_model,
            train_cfg=train_cfg,
            physics=physics,
            epoch_number=epoch_number,
            one_shot_model_version=one_shot_model_version,
            selection_metric=selection_metric,
            selection_loss=selection_score,
            selection_score=selection_score,
            validation_metrics=validation_metrics,
            distill_dataset=distill_dataset,
        ),
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


def _save_epoch_distill_checkpoint(
    run_manager,
    run_meta: dict[str, object],
    *,
    one_shot_model,
    train_cfg,
    physics,
    epoch_number: int,
    one_shot_model_version: str,
    distill_dataset: str,
    validation_metrics: dict[str, float],
) -> None:
    interval = getattr(train_cfg, "one_shot_distill_epoch_checkpoint_interval", 0)
    if interval <= 0 or epoch_number % interval != 0:
        return
    name = f"one_shot_distill_epoch_{epoch_number:04d}.pt"
    ckpt_path = run_manager.save_checkpoint(
        name,
        _distill_checkpoint_payload(
            one_shot_model=one_shot_model,
            train_cfg=train_cfg,
            physics=physics,
            epoch_number=epoch_number,
            one_shot_model_version=one_shot_model_version,
            validation_metrics=validation_metrics,
            distill_dataset=distill_dataset,
        ),
    )
    epoch_ckpt_records: list[dict[str, object]] = run_meta.setdefault("epoch_checkpoints", [])
    epoch_ckpt_records.append(
        {
            "epoch": epoch_number,
            "path": str(ckpt_path),
            "diversity_ratio": validation_metrics.get("diversity_ratio"),
            "eval_corr": validation_metrics.get("eval_corr"),
            "eval_dice": validation_metrics.get("eval_dice"),
            "best_loss": validation_metrics.get("best_loss"),
        },
    )
    run_manager.write_meta(run_meta)


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
    distill_validation_dataset: str | None = None,
    one_shot_checkpoint: str | None = None,
    one_shot_generator_checkpoint: str | None = None,
    one_shot_optical_checkpoint: str | None = None,
    load_one_shot_checkpoint_fn=None,
) -> None:
    _validate_frozen_generator_warm_start(
        train_cfg,
        one_shot_checkpoint=one_shot_checkpoint,
        one_shot_generator_checkpoint=one_shot_generator_checkpoint,
    )
    _maybe_load_one_shot_warm_start(
        torch_module,
        one_shot_model=one_shot_model,
        run_manager=run_manager,
        run_meta=run_meta,
        device=device,
        one_shot_checkpoint=one_shot_checkpoint,
        load_one_shot_checkpoint_fn=load_one_shot_checkpoint_fn,
    )
    _maybe_load_one_shot_module_warm_starts(
        torch_module,
        one_shot_model=one_shot_model,
        run_manager=run_manager,
        run_meta=run_meta,
        device=device,
        generator_checkpoint=one_shot_generator_checkpoint,
        optical_checkpoint=one_shot_optical_checkpoint,
    )
    _maybe_initialize_label_template(
        torch_module,
        one_shot_model=one_shot_model,
        train_cfg=train_cfg,
        run_manager=run_manager,
        run_meta=run_meta,
        one_shot_checkpoint=one_shot_checkpoint,
        generator_checkpoint=one_shot_generator_checkpoint,
    )
    frozen_parameter_snapshots = _snapshot_frozen_one_shot_parameters(
        one_shot_model,
        train_cfg,
    )
    loaders = _prepare_distill_loaders(
        torch_module,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        use_pin_memory=use_pin_memory,
        distill_validation_dataset=distill_validation_dataset,
    )
    train_loader = loaders.train_loader
    validation_loader = loaders.validation_loader
    optimizer = _build_one_shot_optimizer(torch_module, one_shot_model, train_cfg)
    decoder_trainability_audit = _init_decoder_trainability_audit(one_shot_model, optimizer, train_cfg)
    bridge_trainability_audit = _init_target_code_bridge_trainability_audit(one_shot_model, optimizer, train_cfg)
    if decoder_trainability_audit is not None:
        run_meta["one_shot_decoder_trainability_audit_enabled"] = True
        run_meta["one_shot_decoder_trainability_optimizer_param_groups"] = decoder_trainability_audit[
            "optimizer_param_groups"
        ]
        run_manager.write_meta(run_meta)
    if bridge_trainability_audit is not None:
        run_meta["one_shot_target_code_bridge_trainability_audit_enabled"] = True
        run_meta["one_shot_target_code_bridge_trainability_optimizer_param_groups"] = bridge_trainability_audit[
            "optimizer_param_groups"
        ]
        run_manager.write_meta(run_meta)
    loader_length = len(train_loader)
    total_steps = resolve_stage_total_steps(loader_length, train_cfg.one_shot_epochs)
    visual_batch = _select_distill_visual_batch(
        torch_module,
        payload=loaders.payload,
        train_cfg=train_cfg,
        device=device,
        use_pin_memory=use_pin_memory,
    )
    student_label_margin_prototypes, prototype_source = _build_student_label_margin_prototypes(
        torch_module,
        loaders.payload,
        train_cfg,
    )
    if student_label_margin_prototypes is not None:
        run_meta.update(
            {
                "one_shot_student_label_margin_prototype_source": prototype_source,
                "one_shot_student_label_margin_prototype_shape": list(student_label_margin_prototypes.shape),
            }
        )
        run_manager.write_meta(run_meta)
    stage_step = 0
    best_selection_score = None
    best_checkpoint_path = None
    _record_distill_run_setup(
        run_manager,
        run_meta,
        distill_dataset=distill_dataset,
        distill_validation_dataset=distill_validation_dataset,
        train_cfg=train_cfg,
        loaders=loaders,
        visual_summary=visual_batch.summary,
    )

    one_shot_model.train()
    for epoch_index in range(train_cfg.one_shot_epochs):
        epoch_number = epoch_index + 1
        for batch_index, batch in enumerate(train_loader):
            codes, labels, targets, *aux_items = batch
            code_batch = codes.to(device, non_blocking=use_pin_memory)
            class_labels = labels.to(device, non_blocking=use_pin_memory)
            target_batch = targets.to(device, non_blocking=use_pin_memory)
            aux_detail_batch = (
                aux_items[0].to(device, non_blocking=use_pin_memory)
                if aux_items
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = one_shot_model(code_batch, class_labels)
            losses = one_shot_loss_fn(
                outputs,
                target_batch,
                class_labels=class_labels,
                aux_detail_target=aux_detail_batch,
                **_one_shot_loss_kwargs(train_cfg, student_label_margin_prototypes),
            )
            losses["loss"].backward()
            _accumulate_decoder_trainability_gradients(decoder_trainability_audit, one_shot_model)
            _accumulate_target_code_bridge_trainability_gradients(bridge_trainability_audit, one_shot_model)
            optimizer.step()
            step_record = resolve_logged_step(batch_index, stage_step)
            should_record = should_log_step(
                step_record["global_step"],
                train_cfg.step_log_interval,
                is_epoch_last_batch=batch_index == loader_length - 1,
            )
            if should_record:
                phase_info = _phase_grad_summary(one_shot_model)
                mask_info = _mask_amplitude_summary(one_shot_model)
                print(
                    f"[one_shot_distill] epoch={epoch_number} global_step={step_record['global_step']} "
                    f"stage_step={step_record['stage_step']} "
                    f"loss={losses['loss'].item():.6f} "
                    f"camera_recon={losses['camera_recon_loss'].item():.6f} "
                    f"camera_fg_mse={losses['camera_fg_mse'].item():.6f} "
                    f"camera_dice={losses['camera_foreground_dice'].item():.4f} "
                    f"speckle={losses['background_speckle_ratio'].item():.4f} "
                    f"contrast={losses['readable_contrast'].item():.3f} "
                    f"label_acc={losses['student_label_correct_ratio'].item():.3f} "
                    f"margin={losses['student_label_margin_mean'].item():.4f} "
                    f"roi_acc={losses['student_roi_prototype_ce_correct_ratio'].item():.3f} "
                    f"roi_margin={losses['student_roi_negative_label_margin_mean'].item():.4f} "
                    f"base_mse={losses['student_base_mse'].item():.6f} "
                    f"shape={losses['shape_penalty'].item():.4f} "
                    f"edge={losses['edge_penalty'].item():.4f} "
                    f"slm_tv={losses['slm_tv'].item():.6f} "
                    f"sat={losses['camera_saturation_ratio'].item():.4f} "
                    f"sat_low={losses['camera_low_saturation_ratio'].item():.4f} "
                    f"sat_high={losses['camera_high_saturation_ratio'].item():.4f} "
                    f"div_pairs={losses['diversity_pair_count'].item():.0f} "
                    f"div_ratio={losses['diversity_ratio'].item():.4f} "
                    f"xcls_pairs={losses['cross_class_pair_count'].item():.0f} "
                    f"xcls_ratio={losses['cross_class_diversity_ratio'].item():.4f} "
                    f"{phase_info} {mask_info}"
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

        frozen_parameter_report = _verify_frozen_one_shot_parameters(
            one_shot_model,
            frozen_parameter_snapshots,
        )
        run_meta["one_shot_frozen_parameter_check"] = frozen_parameter_report
        run_manager.write_meta(run_meta)

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
                visual_outputs = one_shot_model(
                    visual_batch.codes,
                    visual_batch.labels,
                    return_optical_fields=True,
                )
                if train_cfg.one_shot.optical_core_arch != "camera_direct_clean_oracle":
                    save_visualization_grids(
                        run_manager,
                        {
                            "intensity": build_one_shot_shared_scale_visualization_tensors(
                                visual_batch.references,
                                visual_outputs,
                                codes=visual_batch.codes,
                            ).get("intensity", {}),
                        },
                        run_manager.images_dir / "one_shot_distill" / f"epoch_{epoch_number:04d}.png",
                        title_label=f"one_shot_distill epoch {epoch_number} intensity/control diagnostics",
                        max_items=visual_batch.labels.shape[0],
                        row_labels=labels_for_visual_rows(visual_batch.labels),
                    )
                    _save_strict_optical_field_artifacts(
                        run_manager,
                        visual_outputs,
                        artifact_prefix=f"one_shot_distill_epoch_{epoch_number:04d}",
                        title_prefix=f"one-shot distill epoch {epoch_number}",
                        max_items=visual_batch.labels.shape[0],
                        row_labels=labels_for_visual_rows(visual_batch.labels),
                        front_path_model=getattr(physics, "front_path_model", "free_space"),
                    )
                save_visualization_grids(
                    run_manager,
                    build_one_shot_distill_readability_visualization_tensors(
                        visual_batch.references,
                        visual_outputs,
                        codes=visual_batch.codes,
                    ),
                    run_manager.images_dir / "one_shot_distill" / f"epoch_{epoch_number:04d}_readability.png",
                    title_label=f"one_shot_distill epoch {epoch_number} readability",
                    max_items=visual_batch.labels.shape[0],
                    row_labels=labels_for_visual_rows(visual_batch.labels),
                )
                residual_visualization = build_one_shot_distill_residual_visualization_tensors(
                    visual_batch.references,
                    visual_outputs,
                )
                if residual_visualization is not None:
                    save_visualization_grids(
                        run_manager,
                        residual_visualization,
                        run_manager.images_dir / "one_shot_distill" / f"epoch_{epoch_number:04d}_residual.png",
                        title_label=f"one_shot_distill epoch {epoch_number} residual",
                        max_items=visual_batch.labels.shape[0],
                        row_labels=labels_for_visual_rows(visual_batch.labels),
                    )
                if train_cfg.one_shot.aux_topology_target_transform != "none":
                    transformed = transform_aux_topology_target(
                        visual_batch.references,
                        train_cfg.one_shot.aux_topology_target_transform,
                        train_cfg.one_shot.recon_foreground_threshold,
                    )
                    save_visualization_grids(
                        run_manager,
                        {
                            "reference_target": visual_batch.references,
                            f"aux_topology_target_{train_cfg.one_shot.aux_topology_target_transform}": transformed[
                                "target"
                            ],
                            "aux_topology_foreground_mask": transformed["foreground_mask"],
                        },
                        run_manager.images_dir
                        / "one_shot_distill"
                        / f"epoch_{epoch_number:04d}_aux_topology_target.png",
                        title_label=(
                            "one_shot_distill "
                            f"epoch {epoch_number} aux topology target "
                            f"{train_cfg.one_shot.aux_topology_target_transform}"
                        ),
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
                student_label_margin_prototypes,
            )
            selection_score, selection_metric = _resolve_one_shot_distill_selection(train_cfg, validation_metrics)
            updated_best = best_selection_score is None or selection_score < best_selection_score
            print(
                f"[one_shot_distill][val] epoch={epoch_number} loss={validation_metrics['loss']:.6f} "
                f"camera_recon={validation_metrics['camera_recon_loss']:.6f} "
                f"camera_dice={validation_metrics['camera_foreground_dice']:.4f} "
                f"speckle={validation_metrics.get('background_speckle_ratio', 0.0):.4f} "
                f"contrast={validation_metrics.get('readable_contrast', 0.0):.3f} "
                f"label_acc={validation_metrics.get('student_label_correct_ratio', 0.0):.3f} "
                    f"margin={validation_metrics.get('student_label_margin_mean', 0.0):.4f} "
                    f"base_mse={validation_metrics.get('student_base_mse', 0.0):.6f} "
                    f"shape={validation_metrics['shape_penalty']:.4f} "
                f"edge={validation_metrics['edge_penalty']:.4f} "
                f"slm_tv={validation_metrics['slm_tv']:.6f} "
                f"sat={validation_metrics['camera_saturation_ratio']:.4f} "
                f"sat_low={validation_metrics.get('camera_low_saturation_ratio', 0.0):.4f} "
                f"sat_high={validation_metrics.get('camera_high_saturation_ratio', 0.0):.4f} "
                f"paired_corr={validation_metrics.get('paired_corr', 0.0):.4f} "
                f"paired_dice={validation_metrics.get('paired_dice', 0.0):.4f} "
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
            _save_epoch_distill_checkpoint(
                run_manager,
                run_meta,
                one_shot_model=one_shot_model,
                train_cfg=train_cfg,
                physics=physics,
                epoch_number=epoch_number,
                one_shot_model_version=one_shot_model_version,
                distill_dataset=distill_dataset,
                validation_metrics=validation_metrics,
            )
            one_shot_model.train()

    finalized_trainability_audit = _finalize_decoder_trainability_audit(
        decoder_trainability_audit,
        one_shot_model,
    )
    if finalized_trainability_audit is not None:
        audit_path = run_manager.save_artifact_json(
            "one_shot_decoder_trainability_audit",
            finalized_trainability_audit,
        )
        run_meta["one_shot_decoder_trainability_audit"] = finalized_trainability_audit
        run_meta["one_shot_decoder_trainability_audit_path"] = str(audit_path)
        run_manager.write_meta(run_meta)
    finalized_bridge_trainability_audit = _finalize_target_code_bridge_trainability_audit(
        bridge_trainability_audit,
        one_shot_model,
    )
    if finalized_bridge_trainability_audit is not None:
        audit_path = run_manager.save_artifact_json(
            "one_shot_target_code_bridge_trainability_audit",
            finalized_bridge_trainability_audit,
        )
        run_meta["one_shot_target_code_bridge_trainability_audit"] = finalized_bridge_trainability_audit
        run_meta["one_shot_target_code_bridge_trainability_audit_path"] = str(audit_path)
        run_manager.write_meta(run_meta)
