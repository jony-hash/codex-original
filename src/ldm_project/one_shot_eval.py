from __future__ import annotations

import json
from pathlib import Path

from .one_shot_distill_data import (
    _load_distill_payload,
    _resolve_distill_targets,
    _select_distill_eval_indices,
    _validated_distill_tensors,
)
from .one_shot_eval_artifacts import (
    _save_one_shot_d2nn_diagnostics,
    _save_one_shot_d2nn_focus_visualization,
)
from .one_shot_eval_metrics import compute_one_shot_distill_eval_metrics
from .one_shot_eval_metrics import (
    add_student_label_audit_metrics,
    compute_strict_student_readability_acceptance,
)
from .one_shot_stage_common import (
    ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS,
    ONE_SHOT_DISTILL_EVAL_FOCUS_TENSOR_KEYS,
    ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS,
    ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS,
    _detach_cpu_tensors,
    _sample_one_shot_codes,
)
from .one_shot_visualization import (
    build_one_shot_display_diagnostic_tensors,
    build_one_shot_distill_residual_visualization_tensors,
    build_one_shot_distill_readability_visualization_tensors,
    build_one_shot_eval_shared_scale_visualization_tensors,
    build_one_shot_eval_visualization_tensors,
    build_one_shot_shared_scale_visualization_tensors,
    build_one_shot_visual_diagnostics,
)
from .sampling_reports import (
    compute_sample_quality_metrics,
    select_balanced_visual_preview_indices,
    select_post_train_sample_batch,
)
from .teacher_distill import (
    _build_mnist_label_prototype_bank,
    _teacher_label_audit_json,
    compute_teacher_distill_label_audit_batched,
)
from .visualization_core import (
    VISUALIZATION_AMPLITUDE_KEY,
    VISUALIZATION_INTENSITY_KEY,
    VISUALIZATION_PHASE_KEY,
    _load_original_mnist_digits,
    build_code_visualization_tensor,
    enhance_camera_for_display,
    labels_for_visual_rows,
    save_visualization_grids,
)

__all__ = [
    "_save_one_shot_d2nn_focus_visualization",
    "compute_one_shot_distill_eval_metrics",
    "run_one_shot_distill_eval_diagnostics_stage",
    "run_one_shot_distill_eval_stage",
    "run_one_shot_sample_stage",
]


def _compute_student_eval_label_audit(torch_module, *, train_cfg, student_camera, labels, device) -> dict[str, object]:
    class_labels = tuple(range(int(train_cfg.one_shot.num_classes)))
    prototypes = _build_mnist_label_prototype_bank(
        torch_module,
        class_labels=class_labels,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    )
    batch_size = train_cfg.distillation.label_audit_batch_size
    if batch_size is None:
        batch_size = max(1, int(labels.shape[0]))
    audit_tensors = compute_teacher_distill_label_audit_batched(
        torch_module,
        student_camera,
        labels,
        prototypes,
        class_labels=class_labels,
        batch_size=int(batch_size),
        device=device,
    )
    return _teacher_label_audit_json(labels, audit_tensors, class_labels=class_labels)


def _direct_readable_eval_visuals(eval_preview_visuals: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    direct_keys = (
        "digit",
        "teacher intensity target",
        "SLM command (amplitude control)",
        "student camera intensity",
        "intensity abs error",
    )
    intensity_map = eval_preview_visuals.get(VISUALIZATION_INTENSITY_KEY, {})
    return {
        VISUALIZATION_INTENSITY_KEY: {
            key: value
            for key, value in intensity_map.items()
            if key in direct_keys
        },
        VISUALIZATION_PHASE_KEY: {},
    }


def _build_eval_distill_target_prototypes(torch_module, train_cfg):
    if train_cfg.one_shot.distill_target_source != "mnist_prototype":
        return None
    return _build_mnist_label_prototype_bank(
        torch_module,
        class_labels=tuple(range(int(train_cfg.one_shot.num_classes))),
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    ).contiguous()


def run_one_shot_distill_eval_diagnostics_stage(
    torch_module,
    *,
    run_manager,
    run_meta: dict[str, object],
    eval_tensor_bundle: str,
    eval_metrics: str,
    diagnostic_class_labels,
    diagnostic_max_per_class: int,
) -> None:
    tensor_payload = torch_module.load(eval_tensor_bundle, map_location="cpu", weights_only=False)
    if "class_labels" not in tensor_payload:
        raise ValueError("one-shot eval tensor bundle is missing class_labels.")
    missing_keys = [key for key in ONE_SHOT_DISTILL_EVAL_FOCUS_TENSOR_KEYS if key not in tensor_payload]
    if missing_keys:
        raise ValueError(f"one-shot eval tensor bundle is missing diagnostic tensors: {missing_keys}")

    metrics_path = Path(eval_metrics)
    with metrics_path.open("r", encoding="utf-8") as handle:
        source_metrics = json.load(handle)

    labels = tensor_payload["class_labels"].detach().cpu()
    visual_max_per_class = min(diagnostic_max_per_class, ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS)
    preview_indices, preview_summary = select_balanced_visual_preview_indices(
        labels,
        max_per_class=visual_max_per_class,
        class_labels=diagnostic_class_labels,
    )
    if not preview_indices:
        raise ValueError("one-shot eval diagnostics selected zero samples.")
    preview_index_tensor = torch_module.tensor(preview_indices, dtype=torch_module.long)
    preview_labels = labels.index_select(0, preview_index_tensor)

    def _selected_tensor(key: str):
        tensor = tensor_payload[key]
        if tensor.shape[0] != labels.shape[0]:
            raise ValueError(
                f"diagnostic tensor {key} batch size {tensor.shape[0]} does not match labels {labels.shape[0]}"
            )
        return tensor.detach().cpu().index_select(0, preview_index_tensor)

    label_suffix = "_".join(str(label) for label in preview_summary["class_labels"])
    image_name = f"one_shot_distill_eval_focus_{label_suffix}.png"
    run_manager.save_config_snapshot(
        "one_shot_distill_eval_diagnostics_config",
        {
            "eval_tensor_bundle": eval_tensor_bundle,
            "eval_metrics": eval_metrics,
            "diagnostic_class_labels": preview_summary["class_labels"],
            "diagnostic_max_per_class": diagnostic_max_per_class,
            "visual_max_per_class": visual_max_per_class,
            "visual_preview_count": preview_summary["visual_preview_count"],
            "visual_preview_indices": preview_summary["visual_preview_indices"],
            "visual_preview_labels": preview_summary["visual_preview_labels"],
            "visual_preview": preview_summary,
            "focus_target_display_key": (
                "eval_target_display" if "eval_target_display" in tensor_payload else "teacher_camera_display"
            ),
            "source": "train.py_one_shot_distill_eval_diagnostics",
        },
    )
    selected_tensors = {key: _selected_tensor(key) for key in ONE_SHOT_DISTILL_EVAL_FOCUS_TENSOR_KEYS}
    if "eval_target_display" in tensor_payload:
        selected_tensors["eval_target_display"] = _selected_tensor("eval_target_display")
    target_display_key = "eval_target_display" if "eval_target_display" in selected_tensors else "teacher_camera_display"
    target_display_label = "target intensity" if target_display_key == "eval_target_display" else "teacher intensity"
    target_display_tensor = selected_tensors[target_display_key]
    optional_focus_keys = (
        "codes_display",
        "slm_phase_display",
        "encoder_input_amplitude_display",
        "encoder_input_phase_display",
        "encoder_input_grid_amplitude_display",
        "encoder_input_grid_phase_display",
        "encoded_amplitude_display",
        "encoded_phase_display",
        "ols_input_amplitude_display",
        "ols_input_phase_display",
        "ols_output_amplitude_display",
        "ols_output_phase_display",
        "decoder_input_amplitude_display",
        "decoder_input_phase_display",
        "decoder_output_amplitude_display",
        "decoder_output_phase_display",
    )
    selected_tensors.update(
        {
            key: _selected_tensor(key)
            for key in optional_focus_keys
            if key in tensor_payload
        }
    )
    amplitude_tensors = {
        "chip |E|": selected_tensors["chip_field_amplitude_display"],
    }
    intensity_tensors = {
        target_display_label: target_display_tensor,
        "chip I": selected_tensors["chip_intensity_display"],
        "student camera intensity": selected_tensors["student_camera_display"],
        "intensity abs error": selected_tensors["abs_error"],
    }
    if "codes_display" in selected_tensors:
        amplitude_tensors = {
            "encoder in |E|": selected_tensors["encoder_input_amplitude_display"],
            "encoder grid in |E|": selected_tensors["encoder_input_grid_amplitude_display"],
            "encoder out / OLS in |E|": selected_tensors["encoded_amplitude_display"],
            "OLS out / decoder in |E|": selected_tensors["ols_output_amplitude_display"],
            "decoder out |E|": selected_tensors["decoder_output_amplitude_display"],
            "chip |E|": selected_tensors["chip_field_amplitude_display"],
        }
        intensity_tensors = {"noise code": selected_tensors["codes_display"], **intensity_tensors}
    phase_tensors = {"chip phase": selected_tensors["chip_phase_display"]}
    if "encoder_input_phase_display" in selected_tensors:
        phase_tensors = {
            "encoder in phase": selected_tensors["encoder_input_phase_display"],
            "encoder grid in phase": selected_tensors["encoder_input_grid_phase_display"],
            "encoder out / ols in phase": selected_tensors["encoded_phase_display"],
            "ols out / decoder in phase": selected_tensors["ols_output_phase_display"],
            "decoder out phase": selected_tensors["decoder_output_phase_display"],
            "chip phase": selected_tensors["chip_phase_display"],
        }
        if "slm_phase_display" in selected_tensors:
            phase_tensors = {"slm out phase": selected_tensors["slm_phase_display"], **phase_tensors}
    image_paths = save_visualization_grids(
        run_manager,
        {
            VISUALIZATION_AMPLITUDE_KEY: amplitude_tensors,
            VISUALIZATION_INTENSITY_KEY: intensity_tensors,
            VISUALIZATION_PHASE_KEY: phase_tensors,
        },
        run_manager.images_dir / image_name,
        title_label=None,
        max_items=int(preview_summary["visual_preview_count"]),
        row_labels=labels_for_visual_rows(preview_labels),
    )

    source_per_class = source_metrics.get("per_class", {})
    focus_classes = {}
    for label in preview_summary["class_labels"]:
        class_metrics = source_per_class.get(str(label), {})
        focus_classes[str(label)] = {
            "selected": preview_summary["per_class"][str(label)]["selected"],
            "preview_indices": preview_summary["per_class"][str(label)]["preview_indices"],
            "paired_dice": class_metrics.get("paired_dice"),
            "paired_corr": class_metrics.get("paired_corr"),
            "foreground_hole_ratio": class_metrics.get("foreground_hole_ratio"),
            "student_saturation_ratio": class_metrics.get("student_saturation_ratio"),
            "student_low_saturation_ratio": class_metrics.get("student_low_saturation_ratio"),
            "student_high_saturation_ratio": class_metrics.get("student_high_saturation_ratio"),
            "diversity_ratio": class_metrics.get("diversity_ratio"),
        }
    diagnostics_payload = {
        "source_eval_tensor_bundle": eval_tensor_bundle,
        "source_eval_metrics": eval_metrics,
        "images": {
            name: str(path.relative_to(run_manager.run_dir))
            for name, path in image_paths.items()
        },
        "selection": preview_summary,
        "overall": source_metrics.get("overall", {}),
        "classes": focus_classes,
        "focus_target_display_key": target_display_key,
        "visual_diagnostics": build_one_shot_visual_diagnostics(selected_tensors),
    }
    diagnostics_path = run_manager.save_artifact_json("one_shot_distill_eval_focus_diagnostics", diagnostics_payload)
    print(
        "[one_shot_distill_eval_diagnostics] "
        f"selected={preview_summary['visual_preview_count']} "
        f"classes={preview_summary['class_labels']} images={image_paths}"
    )
    run_meta.update(
        {
            "one_shot_distill_eval_diagnostics_source_tensor_bundle": eval_tensor_bundle,
            "one_shot_distill_eval_diagnostics_source_metrics": eval_metrics,
            "one_shot_distill_eval_diagnostics_images": {
                name: str(path.relative_to(run_manager.run_dir))
                for name, path in image_paths.items()
            },
            "one_shot_distill_eval_diagnostics_metric_source": str(
                diagnostics_path.relative_to(run_manager.run_dir)
            ),
            "one_shot_distill_eval_diagnostics_count": preview_summary["visual_preview_count"],
            "one_shot_distill_eval_diagnostics_class_labels": preview_summary["class_labels"],
        }
    )
    run_manager.write_meta(run_meta)

def run_one_shot_sample_stage(
    torch_module,
    *,
    one_shot_model,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_checkpoint: str,
    load_one_shot_checkpoint_fn,
) -> None:
    one_shot_payload = torch_module.load(one_shot_checkpoint, map_location=device, weights_only=False)
    load_one_shot_checkpoint_fn(one_shot_model, one_shot_payload)
    one_shot_model.eval()
    print(f"loaded one-shot checkpoint for sampling: {one_shot_checkpoint}")
    sample_labels, reference_images, reference_labels, reference_source = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.one_shot.num_classes,
        device,
        sample_class_labels=train_cfg.one_shot.sample_class_labels,
        sample_repeats_per_class=train_cfg.one_shot.sample_repeats_per_class,
    )
    sample_count = int(sample_labels.shape[0])
    codes = _sample_one_shot_codes(torch_module, sample_count, train_cfg, device)
    with torch_module.no_grad():
        sample_outputs = one_shot_model(codes, sample_labels)
    sample_shared_scale_visuals = build_one_shot_shared_scale_visualization_tensors(
        reference_images,
        sample_outputs,
        codes=codes,
    )
    sample_display_diagnostics = build_one_shot_display_diagnostic_tensors(sample_outputs)
    sample_visual_diagnostics = build_one_shot_visual_diagnostics(sample_outputs, sample_display_diagnostics)
    run_manager.save_config_snapshot(
        "one_shot_sample_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "sample_class_labels": list(train_cfg.one_shot.sample_class_labels),
            "sample_repeats_per_class": train_cfg.one_shot.sample_repeats_per_class,
            "source": "train.py_one_shot_sample",
        },
    )
    save_visualization_grids(
        run_manager,
        {
            VISUALIZATION_AMPLITUDE_KEY: sample_shared_scale_visuals[VISUALIZATION_AMPLITUDE_KEY],
            VISUALIZATION_INTENSITY_KEY: sample_shared_scale_visuals.get(VISUALIZATION_INTENSITY_KEY, {}),
            VISUALIZATION_PHASE_KEY: sample_shared_scale_visuals[VISUALIZATION_PHASE_KEY],
        },
        run_manager.images_dir / "one_shot_sample.png",
        title_label="one-shot sample physical-chain diagnostics",
        max_items=sample_count,
        row_labels=labels_for_visual_rows(sample_labels),
    )
    _save_one_shot_d2nn_diagnostics(
        run_manager,
        sample_outputs,
        "one_shot_sample_d2nn_diagnostics",
        {
            "stage": "one_shot_sample",
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
        },
    )
    run_manager.save_tensor_bundle(
        "one_shot_sample_tensors.pt",
        {
            "codes": codes.detach().cpu(),
            "codes_display": build_code_visualization_tensor(codes).detach().cpu(),
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "generated_camera": sample_outputs["camera_plane"].detach().cpu(),
            "generated_camera_raw": sample_outputs["camera_plane"].detach().cpu(),
            "generated_camera_display": enhance_camera_for_display(sample_outputs["camera_plane"]).detach().cpu(),
            **_detach_cpu_tensors(sample_outputs, ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS),
            **_detach_cpu_tensors(sample_display_diagnostics, ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS),
        },
    )
    run_manager.save_artifact_json("one_shot_sample_visual_diagnostics", sample_visual_diagnostics)
    run_manager.save_artifact_json(
        "one_shot_sample_shared_scale_diagnostics",
        sample_shared_scale_visuals["diagnostics"],
    )
    sample_metrics = compute_sample_quality_metrics(sample_outputs["camera_plane"], reference_images, sample_labels)
    metrics_path = run_manager.save_artifact_json("one_shot_sample_metrics", sample_metrics)
    run_meta.update(
        {
            "one_shot_checkpoint": one_shot_checkpoint,
            "one_shot_sample_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
        }
    )
    run_manager.write_meta(run_meta)

def run_one_shot_distill_eval_stage(
    torch_module,
    *,
    one_shot_model,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    one_shot_checkpoint: str,
    distill_dataset: str,
    load_one_shot_checkpoint_fn,
) -> None:
    payload = _load_distill_payload(torch_module, distill_dataset)
    codes, labels, teacher_camera = _validated_distill_tensors(torch_module, payload, train_cfg)
    target_prototypes = _build_eval_distill_target_prototypes(torch_module, train_cfg)
    resolved_targets = _resolve_distill_targets(
        torch_module,
        payload,
        train_cfg,
        mnist_prototypes=target_prototypes,
    )
    selected_indices, selection_summary = _select_distill_eval_indices(torch_module, labels, train_cfg)
    if not selected_indices:
        raise ValueError("one_shot_distill_eval selected zero samples.")

    index_tensor = torch_module.tensor(selected_indices, dtype=torch_module.long)
    eval_codes = codes.index_select(0, index_tensor)
    eval_labels = labels.index_select(0, index_tensor)
    eval_raw_teacher = teacher_camera.index_select(0, index_tensor)
    eval_target = resolved_targets.index_select(0, index_tensor)

    one_shot_payload = torch_module.load(one_shot_checkpoint, map_location=device, weights_only=False)
    load_one_shot_checkpoint_fn(one_shot_model, one_shot_payload)
    one_shot_model.eval()
    print(f"loaded one-shot checkpoint for distill eval: {one_shot_checkpoint}")
    with torch_module.no_grad():
        eval_outputs = one_shot_model(eval_codes.to(device), eval_labels.to(device))
    student_camera = eval_outputs["camera_plane"].detach().cpu()
    eval_original_digits = _load_original_mnist_digits(
        eval_labels, train_cfg.first_stage.image_size, root=train_cfg.data.root
    )
    eval_display_diagnostics = build_one_shot_display_diagnostic_tensors(eval_outputs)
    eval_visual_diagnostics = build_one_shot_visual_diagnostics(eval_outputs, eval_display_diagnostics)
    physical_phase_available = train_cfg.one_shot.optical_core_arch != "camera_direct_clean_oracle"
    eval_visual_diagnostics["physical_phase_available"] = physical_phase_available
    eval_visual_diagnostics["visualization_mode"] = "optical_chain" if physical_phase_available else "direct_readable"
    slm_amplitude = eval_outputs["slm_amplitude"].detach().cpu()
    abs_error = (student_camera - eval_target).abs()
    eval_visuals = build_one_shot_eval_visualization_tensors(
        eval_target, eval_outputs, abs_error, codes=eval_codes, original_images=eval_original_digits,
    )
    eval_shared_scale_visuals = build_one_shot_eval_shared_scale_visualization_tensors(
        eval_target,
        eval_outputs,
        abs_error,
        codes=eval_codes,
        original_images=eval_original_digits,
    )
    metrics = compute_one_shot_distill_eval_metrics(
        torch_module,
        teacher_camera=eval_target,
        student_camera=student_camera,
        slm_amplitude=slm_amplitude,
        class_labels=eval_labels,
    )
    metrics["selection"] = selection_summary
    student_label_audit_path = None
    if train_cfg.one_shot.student_label_audit_enabled:
        student_label_audit = _compute_student_eval_label_audit(
            torch_module,
            train_cfg=train_cfg,
            student_camera=student_camera,
            labels=eval_labels,
            device=device,
        )
        add_student_label_audit_metrics(metrics, student_label_audit)
        student_label_audit_path = run_manager.save_artifact_json(
            "one_shot_distill_eval_student_label_audit",
            student_label_audit,
        )
    metrics["strict_readability_acceptance"] = compute_strict_student_readability_acceptance(metrics)
    visual_preview_indices, visual_preview_summary = select_balanced_visual_preview_indices(
        eval_labels,
        max_per_class=int(train_cfg.one_shot.eval_preview_per_class),
    )
    visual_preview_index_tensor = torch_module.tensor(visual_preview_indices, dtype=torch_module.long)

    def _preview_tensor(tensor):
        return tensor.detach().cpu().index_select(0, visual_preview_index_tensor)

    eval_preview_visuals = {
        "amplitude": {
            key: _preview_tensor(value)
            for key, value in eval_visuals["amplitude"].items()
        },
        "intensity": {
            key: _preview_tensor(value)
            for key, value in eval_visuals.get("intensity", {}).items()
        },
        "phase": {
            key: _preview_tensor(value)
            for key, value in eval_visuals["phase"].items()
        },
    }
    eval_preview_shared_scale_visuals = {
        "amplitude": {
            key: _preview_tensor(value)
            for key, value in eval_shared_scale_visuals["amplitude"].items()
        },
        "intensity": {
            key: _preview_tensor(value)
            for key, value in eval_shared_scale_visuals.get("intensity", {}).items()
        },
        "phase": {
            key: _preview_tensor(value)
            for key, value in eval_shared_scale_visuals["phase"].items()
        },
        "diagnostics": eval_shared_scale_visuals["diagnostics"],
    }
    eval_preview_outputs = {}
    for key, value in eval_outputs.items():
        if hasattr(value, "ndim") and value.ndim > 0 and value.shape[0] == eval_labels.shape[0]:
            eval_preview_outputs[key] = _preview_tensor(value)
        else:
            eval_preview_outputs[key] = value
    preview_labels = eval_labels.index_select(0, visual_preview_index_tensor)

    run_manager.save_config_snapshot(
        "one_shot_distill_eval_config",
        {
            "distill_dataset": distill_dataset,
            "one_shot_distill_target_source": train_cfg.one_shot.distill_target_source,
            "physical_phase_available": physical_phase_available,
            "visualization_mode": "optical_chain" if physical_phase_available else "direct_readable",
            "one_shot_checkpoint": one_shot_checkpoint,
            "selection": selection_summary,
            "visual_preview_count": visual_preview_summary["visual_preview_count"],
            "visual_preview_indices": visual_preview_summary["visual_preview_indices"],
            "visual_preview_labels": visual_preview_summary["visual_preview_labels"],
            "visual_preview": visual_preview_summary,
            "student_label_audit_enabled": bool(train_cfg.one_shot.student_label_audit_enabled),
            "source": "train.py_one_shot_distill_eval",
        },
    )
    if physical_phase_available:
        save_visualization_grids(
            run_manager,
            {
                VISUALIZATION_AMPLITUDE_KEY: eval_preview_shared_scale_visuals[VISUALIZATION_AMPLITUDE_KEY],
                VISUALIZATION_INTENSITY_KEY: eval_preview_shared_scale_visuals.get(VISUALIZATION_INTENSITY_KEY, {}),
                VISUALIZATION_PHASE_KEY: eval_preview_shared_scale_visuals[VISUALIZATION_PHASE_KEY],
            },
            run_manager.images_dir / "one_shot_distill_eval.png",
            title_label="one-shot distill paired eval physical-chain diagnostics",
            max_items=int(visual_preview_summary["visual_preview_count"]),
            row_labels=labels_for_visual_rows(preview_labels),
        )
    else:
        save_visualization_grids(
            run_manager,
            _direct_readable_eval_visuals(eval_preview_visuals),
            run_manager.images_dir / "one_shot_distill_eval_direct_readable.png",
            title_label="one-shot distill direct oracle eval",
            max_items=int(visual_preview_summary["visual_preview_count"]),
            row_labels=labels_for_visual_rows(preview_labels),
        )
    eval_readability_visuals = build_one_shot_distill_readability_visualization_tensors(
        eval_target, eval_outputs, codes=eval_codes,
    )
    eval_preview_readability_visuals = {
        VISUALIZATION_INTENSITY_KEY: {
            key: _preview_tensor(value)
            for key, value in eval_readability_visuals[VISUALIZATION_INTENSITY_KEY].items()
        },
        VISUALIZATION_PHASE_KEY: {},
    }
    save_visualization_grids(
        run_manager,
        eval_preview_readability_visuals,
        run_manager.images_dir / "one_shot_distill_eval_readability.png",
        title_label="one-shot distill eval readability",
        max_items=int(visual_preview_summary["visual_preview_count"]),
        row_labels=labels_for_visual_rows(preview_labels),
    )
    residual_visualization = build_one_shot_distill_residual_visualization_tensors(
        _preview_tensor(eval_target),
        eval_preview_outputs,
    )
    if residual_visualization is not None:
        save_visualization_grids(
            run_manager,
            residual_visualization,
            run_manager.images_dir / "one_shot_distill_eval_residual.png",
            title_label="one-shot distill eval residual",
            max_items=int(visual_preview_summary["visual_preview_count"]),
            row_labels=labels_for_visual_rows(preview_labels),
        )
    _save_one_shot_d2nn_diagnostics(
        run_manager,
        eval_preview_outputs,
        "one_shot_distill_eval_d2nn_diagnostics",
        {
            "stage": "one_shot_distill_eval",
            "visual_preview_count": int(visual_preview_summary["visual_preview_count"]),
            "visual_preview_labels": preview_labels.detach().cpu().tolist(),
        },
    )
    tensor_path = run_manager.save_tensor_bundle(
        "one_shot_distill_eval_tensors.pt",
        {
            "indices": index_tensor,
            "visual_preview_indices": visual_preview_index_tensor,
            "codes": eval_codes,
            "codes_display": build_code_visualization_tensor(eval_codes),
            "class_labels": eval_labels,
            "teacher_camera": eval_raw_teacher,
            "teacher_camera_display": enhance_camera_for_display(eval_raw_teacher),
            "eval_target": eval_target,
            "eval_target_display": enhance_camera_for_display(eval_target),
            "one_shot_distill_target_source": train_cfg.one_shot.distill_target_source,
            "student_camera_raw": student_camera,
            "student_camera_display": enhance_camera_for_display(student_camera),
            "original_digits": eval_original_digits,
            "abs_error": abs_error,
            **_detach_cpu_tensors(eval_outputs, ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS),
            **_detach_cpu_tensors(eval_display_diagnostics, ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS),
        },
    )
    run_manager.save_artifact_json("one_shot_distill_eval_visual_diagnostics", eval_visual_diagnostics)
    eval_shared_scale_visuals["diagnostics"]["physical_phase_available"] = physical_phase_available
    eval_shared_scale_visuals["diagnostics"]["visualization_mode"] = (
        "optical_chain" if physical_phase_available else "direct_readable"
    )
    run_manager.save_artifact_json(
        "one_shot_distill_eval_shared_scale_diagnostics",
        eval_shared_scale_visuals["diagnostics"],
    )
    metrics_path = run_manager.save_artifact_json("one_shot_distill_eval_metrics", metrics)
    overall = metrics["overall"]
    print(
        "[one_shot_distill_eval] "
        f"mse={overall['paired_mse']:.6f} corr={overall['paired_corr']:.4f} "
        f"dice={overall['paired_dice']:.4f} sat={overall['student_saturation_ratio']:.4f} "
        f"slm_tv={overall['slm_tv']:.6f} diversity_ratio={overall['diversity_ratio']}"
    )
    run_meta.update(
        {
            "one_shot_checkpoint": one_shot_checkpoint,
            "distill_dataset": distill_dataset,
            "one_shot_distill_target_source": train_cfg.one_shot.distill_target_source,
            "one_shot_physical_phase_available": physical_phase_available,
            "one_shot_visualization_mode": "optical_chain" if physical_phase_available else "direct_readable",
            "one_shot_distill_eval_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
            "one_shot_distill_eval_tensor_source": str(tensor_path.relative_to(run_manager.run_dir)),
            "one_shot_distill_eval_strict_readability_passed": bool(
                metrics["strict_readability_acceptance"]["passed"]
            ),
        }
    )
    if student_label_audit_path is not None:
        run_meta["one_shot_distill_eval_student_label_audit_source"] = str(
            student_label_audit_path.relative_to(run_manager.run_dir)
        )
    run_manager.write_meta(run_meta)
