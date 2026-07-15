from __future__ import annotations

from ..distillation.dataset import (
    load_distill_payload,
    filter_distill_payload_by_indices,
)
from ..visualization_core import (
    format_output_title,
    readable_camera_for_display,
)
from ..train_utils import validate_class_label_values

from .distill_support import (
    balanced_preview_indices as _balanced_preview_indices,
    metric_summary as _metric_summary,
)
from .label_audit import (
    resolve_teacher_label_audit_inputs as _resolve_teacher_label_audit_inputs,
    save_teacher_label_audit_artifacts as _save_teacher_label_audit_artifacts,
)

def _readable_gate_thresholds(train_cfg) -> dict[str, float]:
    return {
        "min_label_margin": float(train_cfg.distillation.readable_min_label_margin),
        "min_readable_contrast": float(train_cfg.distillation.readable_min_contrast),
        "max_background_speckle_ratio": float(
            train_cfg.distillation.readable_max_background_speckle
        ),
        "max_foreground_hole_ratio": float(
            train_cfg.distillation.readable_max_foreground_hole
        ),
    }

def _readable_candidate_score(audit_tensors, indices):
    margin = audit_tensors["label_margin"].index_select(0, indices)
    contrast = audit_tensors["readable_contrast"].index_select(0, indices)
    speckle = audit_tensors["background_speckle_ratio"].index_select(0, indices)
    holes = audit_tensors["foreground_hole_ratio"].index_select(0, indices)
    return margin + 0.25 * contrast - 0.25 * speckle - 0.25 * holes

def _split_readable_teacher_indices(
    torch_module, labels, audit_tensors: dict[str, object], train_cfg
):
    train_per_class = int(train_cfg.distillation.readable_train_per_class)
    val_per_class = int(train_cfg.distillation.readable_val_per_class)
    test_per_class = int(train_cfg.distillation.readable_test_per_class)
    required_per_class = train_per_class + val_per_class + test_per_class
    if (
        train_per_class < 0
        or val_per_class < 0
        or test_per_class < 0
        or required_per_class <= 0
    ):
        raise ValueError(
            "readable split counts must be non-negative and sum to a positive value."
        )

    labels_cpu = labels.detach().long().cpu()
    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.one_shot.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(sorted({int(label) for label in labels_cpu.tolist()}))

    thresholds = _readable_gate_thresholds(train_cfg)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    selected_indices: list[int] = []
    per_class: dict[str, dict[str, object]] = {}
    failed = False
    generator = torch_module.Generator().manual_seed(
        int(train_cfg.distillation.readable_split_seed)
    )

    for label in class_labels:
        class_indices = torch_module.nonzero(
            labels_cpu == int(label), as_tuple=False
        ).flatten()
        class_count = int(class_indices.numel())
        if class_count == 0:
            failed = True
            per_class[str(label)] = {
                "available": 0,
                "readable_available": 0,
                "readability_filter_failed": True,
            }
            continue

        label_correct = audit_tensors["label_correct"].index_select(0, class_indices)
        margin = audit_tensors["label_margin"].index_select(0, class_indices)
        contrast = audit_tensors["readable_contrast"].index_select(0, class_indices)
        speckle = audit_tensors["background_speckle_ratio"].index_select(
            0, class_indices
        )
        holes = audit_tensors["foreground_hole_ratio"].index_select(0, class_indices)
        readable_mask = (
            label_correct
            & (margin >= thresholds["min_label_margin"])
            & (contrast >= thresholds["min_readable_contrast"])
            & (speckle <= thresholds["max_background_speckle_ratio"])
            & (holes <= thresholds["max_foreground_hole_ratio"])
        )
        readable_indices = class_indices[readable_mask]
        readable_count = int(readable_indices.numel())
        class_failed = readable_count < required_per_class
        failed = failed or class_failed

        chosen = readable_indices
        split_counts = {"train": 0, "val": 0, "test": 0}
        if not class_failed:
            score = _readable_candidate_score(audit_tensors, readable_indices)
            order = torch_module.argsort(score, descending=True)
            chosen = readable_indices.index_select(0, order[:required_per_class])
            permutation = torch_module.randperm(required_per_class, generator=generator)
            chosen = chosen.index_select(0, permutation)
            train_part = chosen[:train_per_class]
            val_part = chosen[train_per_class : train_per_class + val_per_class]
            test_part = chosen[train_per_class + val_per_class :]
            train_indices.extend(int(index) for index in train_part.tolist())
            val_indices.extend(int(index) for index in val_part.tolist())
            test_indices.extend(int(index) for index in test_part.tolist())
            selected_indices.extend(int(index) for index in chosen.tolist())
            split_counts = {
                "train": int(train_part.numel()),
                "val": int(val_part.numel()),
                "test": int(test_part.numel()),
            }

        per_class[str(label)] = {
            "available": class_count,
            "label_correct_available": int(label_correct.sum().item()),
            "readable_available": readable_count,
            "required": required_per_class,
            "readability_filter_failed": class_failed,
            "split_counts": split_counts,
            "rejected_label_incorrect": int((~label_correct).sum().item()),
            "rejected_low_margin": int(
                (label_correct & (margin < thresholds["min_label_margin"])).sum().item()
            ),
            "rejected_low_contrast": int(
                (label_correct & (contrast < thresholds["min_readable_contrast"]))
                .sum()
                .item()
            ),
            "rejected_high_speckle": int(
                (label_correct & (speckle > thresholds["max_background_speckle_ratio"]))
                .sum()
                .item()
            ),
            "rejected_high_hole": int(
                (label_correct & (holes > thresholds["max_foreground_hole_ratio"]))
                .sum()
                .item()
            ),
            "readable_label_margin": _metric_summary(
                audit_tensors["label_margin"].index_select(0, readable_indices)
            ),
            "readable_contrast": _metric_summary(
                audit_tensors["readable_contrast"].index_select(0, readable_indices)
            ),
            "readable_background_speckle_ratio": _metric_summary(
                audit_tensors["background_speckle_ratio"].index_select(
                    0, readable_indices
                )
            ),
            "readable_foreground_hole_ratio": _metric_summary(
                audit_tensors["foreground_hole_ratio"].index_select(0, readable_indices)
            ),
        }

    if failed:
        train_indices = []
        val_indices = []
        test_indices = []
        selected_indices = []

    return {
        "failed": failed,
        "thresholds": thresholds,
        "class_labels": list(class_labels),
        "split_seed": int(train_cfg.distillation.readable_split_seed),
        "required_per_class": required_per_class,
        "split_counts_per_class": {
            "train": train_per_class,
            "val": val_per_class,
            "test": test_per_class,
        },
        "indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "selected": selected_indices,
        },
        "per_class": per_class,
    }

def _add_readability_audit_fields(
    filtered_payload, audit_tensors, index_tensor
) -> None:
    filtered_payload["label_audit_requested_corr"] = audit_tensors[
        "requested_label_corr"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_nearest_label"] = audit_tensors[
        "nearest_label"
    ].index_select(0, index_tensor)
    filtered_payload["label_audit_margin"] = audit_tensors["label_margin"].index_select(
        0, index_tensor
    )
    filtered_payload["readability_audit_contrast"] = audit_tensors[
        "readable_contrast"
    ].index_select(0, index_tensor)
    filtered_payload["readability_audit_background_speckle_ratio"] = audit_tensors[
        "background_speckle_ratio"
    ].index_select(0, index_tensor)
    filtered_payload["readability_audit_foreground_hole_ratio"] = audit_tensors[
        "foreground_hole_ratio"
    ].index_select(
        0,
        index_tensor,
    )

def _readable_split_class_counts(
    torch_module, payload: dict[str, object]
) -> dict[str, int]:
    labels = payload["class_labels"].detach().long().cpu()
    return {
        str(label): int((labels == int(label)).sum().item())
        for label in sorted({int(item) for item in labels.tolist()})
    }

def run_teacher_distill_readable_split(
    torch_module,
    *,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    distill_dataset: str,
    device=None,
    prototype_builder=None,
) -> None:
    payload = load_distill_payload(torch_module, distill_dataset)
    (
        codes,
        labels,
        teacher_camera,
        prototypes,
        audit_tensors,
        audit_payload,
        class_labels,
    ) = _resolve_teacher_label_audit_inputs(
        torch_module,
        payload,
        train_cfg,
        device=device,
        prototype_builder=prototype_builder,
    )
    split_summary = _split_readable_teacher_indices(
        torch_module, labels, audit_tensors, train_cfg
    )
    metrics_payload = {
        "teacher_filter": "readability_strict_split",
        "source_dataset": distill_dataset,
        "source_count": int(codes.shape[0]),
        "readability_filter_failed": bool(split_summary["failed"]),
        "audit": {
            key: value for key, value in audit_payload.items() if key != "per_sample"
        },
        **{key: value for key, value in split_summary.items() if key != "indices"},
    }
    run_manager.save_config_snapshot(
        "teacher_distill_readable_split_config",
        {
            "distill_dataset": distill_dataset,
            "audit_device": str(device) if device is not None else "cpu",
            "thresholds": split_summary["thresholds"],
            "split_counts_per_class": split_summary["split_counts_per_class"],
            "split_seed": split_summary["split_seed"],
            "source": "train.py_teacher_distill_readable_split",
        },
    )
    metrics_path = run_manager.save_artifact_json(
        "teacher_distill_readable_split_metrics", metrics_payload
    )
    audit_path, audit_image_path = _save_teacher_label_audit_artifacts(
        torch_module,
        run_manager=run_manager,
        distill_dataset=distill_dataset,
        train_cfg=train_cfg,
        labels=labels,
        prototypes=prototypes,
        audit_tensors=audit_tensors,
        audit_payload=audit_payload,
        class_labels=class_labels,
        artifact_name="teacher_distill_readable_split_audit",
        image_name="teacher_distill_readable_split_audit.png",
    )

    split_paths: dict[str, str | None] = {"train": None, "val": None, "test": None}
    split_class_counts: dict[str, dict[str, int]] = {}
    if not split_summary["failed"]:
        source_config = payload.get("config", {})
        for split_name in ("train", "val", "test"):
            split_indices = split_summary["indices"][split_name]
            filtered_payload, index_tensor = filter_distill_payload_by_indices(
                torch_module,
                payload,
                split_indices,
                source_size=int(codes.shape[0]),
            )
            filtered_config = (
                dict(source_config)
                if isinstance(source_config, dict)
                else {"source_config": source_config}
            )
            filtered_config["filter"] = {
                "teacher_filter": "readability_strict_split",
                "split": split_name,
                "source_dataset": distill_dataset,
                "source_count": int(codes.shape[0]),
                "selected_count": len(split_indices),
                "thresholds": split_summary["thresholds"],
                "split_counts_per_class": split_summary["split_counts_per_class"],
                "split_seed": split_summary["split_seed"],
            }
            filtered_payload["config"] = filtered_config
            filtered_payload["readability_split"] = split_name
            filtered_payload["readability_filter_indices"] = index_tensor
            _add_readability_audit_fields(filtered_payload, audit_tensors, index_tensor)
            artifact_path = run_manager.save_tensor_bundle(
                f"teacher_distill_dataset_readable_{split_name}.pt",
                filtered_payload,
            )
            split_paths[split_name] = str(artifact_path)
            split_class_counts[split_name] = _readable_split_class_counts(
                torch_module, filtered_payload
            )

        source_preview = _balanced_preview_indices(torch_module, labels)
        selected_index_tensor = torch_module.tensor(
            split_summary["indices"]["selected"], dtype=torch_module.long
        )
        selected_preview = _balanced_preview_indices(
            torch_module, labels.index_select(0, selected_index_tensor)
        )
        selected_preview = selected_index_tensor.index_select(0, selected_preview)
        run_manager.save_tensor_grid(
            {
                "source teacher readable": readable_camera_for_display(
                    teacher_camera.index_select(0, source_preview)
                ),
                "readable selected": readable_camera_for_display(
                    teacher_camera.index_select(0, selected_preview)
                ),
            },
            run_manager.images_dir / "teacher_distill_readable_split_before_after.png",
            title=format_output_title(
                "images/teacher_distill_readable_split_before_after.png",
                "teacher readable split filter",
            ),
            max_items=min(20, len(split_summary["indices"]["selected"])),
        )

    print(
        "[teacher_distill_readable_split] "
        f"failed={bool(split_summary['failed'])} "
        f"train={len(split_summary['indices']['train'])} "
        f"val={len(split_summary['indices']['val'])} "
        f"test={len(split_summary['indices']['test'])}"
    )
    run_meta.update(
        {
            "teacher_distill_source_dataset": distill_dataset,
            "teacher_distill_filter": "readability_strict_split",
            "teacher_distill_readability_filter_failed": bool(split_summary["failed"]),
            "teacher_distill_readable_split_metric_source": str(
                metrics_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_split_audit_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_split_audit_image": str(
                audit_image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_distill_readable_train_dataset": split_paths["train"],
            "teacher_distill_readable_val_dataset": split_paths["val"],
            "teacher_distill_readable_test_dataset": split_paths["test"],
            "teacher_distill_readable_split_class_counts": split_class_counts,
        }
    )
    run_manager.write_meta(run_meta)

__all__ = ["run_teacher_distill_readable_split"]
