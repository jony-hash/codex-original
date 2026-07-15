from __future__ import annotations

from typing import Any, cast

from ..generation.codes import _sample_one_shot_codes_with_seed
from .distill_support import (
    build_mnist_label_prototype_bank as _build_mnist_label_prototype_bank,
    distill_class_label_sequence as _distill_class_label_sequence,
    metric_summary as _metric_summary,
    project_codes_to_teacher_noise as _project_codes_to_teacher_noise,
)
from .label_audit import (
    compute_teacher_distill_label_audit_batched,
    save_teacher_label_audit_artifacts as _save_teacher_label_audit_artifacts,
    teacher_label_audit_visual_tensors as _teacher_label_audit_visual_tensors,
)
from ..evaluation.label_audit import label_audit_json as _teacher_label_audit_json
from ..train_utils import (
    load_latent_scale_from_diffusion_checkpoint,
    set_first_stage_image_fusion_weight,
    validate_class_label_values,
)
from ..visualization_core import format_output_title


def _load_mnist_validation_batch(
    torch_module, *, class_labels, samples_per_class: int, image_size: int, root: str
):
    try:
        from torchvision import datasets, transforms  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to load MNIST validation images."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    mnist = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    requested = tuple(int(label) for label in class_labels)
    buckets: dict[int, list[object]] = {label: [] for label in requested}
    for image, label in mnist:
        label = int(label)
        if label not in buckets:
            continue
        if len(buckets[label]) >= int(samples_per_class):
            continue
        buckets[label].append(image)
        if all(len(items) >= int(samples_per_class) for items in buckets.values()):
            break
    missing = [
        label for label, items in buckets.items() if len(items) < int(samples_per_class)
    ]
    if missing:
        raise ValueError(
            f"MNIST validation set missing labels or insufficient samples: {missing}"
        )
    images = []
    labels_out = []
    for label in requested:
        images.extend(buckets[label])
        labels_out.extend([label] * len(buckets[label]))
    return torch_module.stack(images, dim=0), torch_module.tensor(
        labels_out, dtype=torch_module.long
    )


def _run_first_stage_oracle_camera(
    torch_module, first_stage, images, labels, batch_size: int, device
):
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, 0.0)
    image_size = int(images.shape[0])
    camera_parts: list[object] = []
    with torch_module.no_grad():
        for start in range(0, image_size, int(batch_size)):
            end = min(start + int(batch_size), image_size)
            batch_images = images[start:end].to(device=device)
            encoded = first_stage.encode(batch_images)
            decoded = first_stage.decode_with_intermediates(encoded["z"])
            camera_parts.append(decoded["camera_plane"].detach().cpu())
    return torch_module.cat(camera_parts, dim=0), labels


def run_teacher_first_stage_oracle_audit(
    torch_module,
    *,
    first_stage,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
) -> None:
    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.diffusion.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(range(train_cfg.diffusion.num_classes))

    samples_per_class = int(train_cfg.distillation.samples_per_class)
    if samples_per_class <= 0:
        raise ValueError(
            f"distillation.samples_per_class must be positive, got {samples_per_class}"
        )

    images, labels = _load_mnist_validation_batch(
        torch_module,
        class_labels=class_labels,
        samples_per_class=samples_per_class,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
    )
    teacher_camera, labels = _run_first_stage_oracle_camera(
        torch_module,
        first_stage,
        images,
        labels,
        batch_size=int(train_cfg.batch_size),
        device=device,
    )

    prototypes = _build_mnist_label_prototype_bank(
        torch_module,
        class_labels=class_labels,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    )

    audit_batch_size = (
        train_cfg.distillation.label_audit_batch_size or train_cfg.batch_size
    )
    audit_device = device if device is not None else torch_module.device("cpu")
    audit_tensors = compute_teacher_distill_label_audit_batched(
        torch_module,
        teacher_camera,
        labels,
        prototypes,
        class_labels=class_labels,
        batch_size=int(audit_batch_size),
        device=audit_device,
        foreground_threshold=train_cfg.distillation.filter_foreground_threshold,
    )
    audit_payload = _teacher_label_audit_json(
        labels, audit_tensors, class_labels=class_labels
    )

    run_manager.save_config_snapshot(
        "teacher_first_stage_oracle_audit_config",
        {
            "stage": "teacher_first_stage_oracle_audit",
            "class_labels": [int(label) for label in class_labels],
            "samples_per_class": samples_per_class,
            "image_source": "mnist_validation",
            "foreground_threshold": train_cfg.distillation.filter_foreground_threshold,
            "prototype_source": "train_mnist_mean_by_label",
            "prototypes_per_class": int(
                train_cfg.distillation.label_audit_prototypes_per_class
            ),
            "source": "train.py_teacher_first_stage_oracle_audit",
        },
    )

    audit_path, image_path = _save_teacher_label_audit_artifacts(
        torch_module,
        run_manager=run_manager,
        distill_dataset="mnist_validation_images",
        train_cfg=train_cfg,
        labels=labels,
        prototypes=prototypes,
        audit_tensors=audit_tensors,
        audit_payload=audit_payload,
        class_labels=class_labels,
        artifact_name="teacher_first_stage_oracle_audit",
        image_name="teacher_first_stage_oracle_audit.png",
    )

    print(
        "[teacher_first_stage_oracle_audit] "
        f"label_correct={audit_payload['label_correct_count']}/{audit_payload['sample_count']} "
        f"({audit_payload['label_correct_ratio']:.4f})"
    )
    audit_per_class = cast(
        dict[str, dict[str, Any]], audit_payload.get("per_class", {})
    )
    for label_str, stats in audit_per_class.items():
        print(
            f"  class {label_str}: "
            f"correct={stats['label_correct_count']}/{stats['count']} "
            f"({stats['label_correct_ratio']:.4f}) "
            f"margin={stats['label_margin']['mean']:.4f} "
            f"contrast={stats['readable_contrast']['mean']:.4f} "
            f"speckle={stats['background_speckle_ratio']['mean']:.4f}"
        )

    run_meta.update(
        {
            "teacher_first_stage_oracle_audit_metric_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_first_stage_oracle_audit_image": str(
                image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_first_stage_oracle_label_correct_count": audit_payload[
                "label_correct_count"
            ],
            "teacher_first_stage_oracle_label_correct_ratio": audit_payload[
                "label_correct_ratio"
            ],
        }
    )
    run_manager.write_meta(run_meta)


def _build_latent_prototypes_from_mnist(
    torch_module,
    first_stage,
    *,
    class_labels,
    image_size: int,
    root: str,
    max_per_class: int,
    device,
    batch_size: int,
):
    """Encode real MNIST train images through first_stage to get per-class mean latent prototypes."""
    try:
        from torchvision import datasets, transforms  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to build latent prototypes."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    mnist = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    requested = tuple(int(label) for label in class_labels)
    buckets: dict[int, list[object]] = {label: [] for label in requested}
    for image, label in mnist:
        label = int(label)
        if label not in buckets or len(buckets[label]) >= int(max_per_class):
            continue
        buckets[label].append(image)
        if all(len(items) >= int(max_per_class) for items in buckets.values()):
            break

    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, 0.0)
    latent_prototypes = []
    with torch_module.no_grad():
        for label in requested:
            class_images = torch_module.stack(buckets[label], dim=0).to(device=device)
            class_latents = []
            for start in range(0, len(buckets[label]), int(batch_size)):
                end = min(start + int(batch_size), len(buckets[label]))
                batch = class_images[start:end]
                encoded = first_stage.encode(batch)
                class_latents.append(encoded["z"].detach().cpu())
            all_class_latents = torch_module.cat(class_latents, dim=0)
            latent_prototypes.append(all_class_latents.mean(dim=0, keepdim=True))
    return torch_module.cat(latent_prototypes, dim=0)


def _latent_nearest_label(
    torch_module, sampled_latents, latent_prototypes, class_labels
):
    """Find nearest latent prototype for each sampled latent by L2 distance."""
    sampled_flat = sampled_latents.detach().float().flatten(1).cpu()
    proto_flat = latent_prototypes.detach().float().flatten(1).cpu()
    distances = []
    for i in range(sampled_flat.shape[0]):
        dist = (proto_flat - sampled_flat[i]).pow(2).sum(dim=1).sqrt()
        distances.append(dist)
    dist_matrix = torch_module.stack(distances, dim=0)
    nearest_dist, nearest_idx = dist_matrix.min(dim=1)
    nearest_labels = torch_module.tensor(
        [int(class_labels[int(idx)]) for idx in nearest_idx.tolist()],
        dtype=torch_module.long,
    )
    return nearest_labels, nearest_dist


def _latent_prototype_margin(
    torch_module,
    sampled_latents,
    latent_prototypes,
    class_labels,
    requested_label_indices,
):
    """Compute margin between nearest prototype and second-nearest for each sampled latent."""
    sampled_flat = sampled_latents.detach().float().flatten(1).cpu()
    proto_flat = latent_prototypes.detach().float().flatten(1).cpu()
    margins = []
    nearest_labels = []
    for i in range(sampled_flat.shape[0]):
        dist = (proto_flat - sampled_flat[i]).pow(2).sum(dim=1).sqrt()
        sorted_dist, sorted_idx = torch_module.sort(dist)
        nearest = sorted_dist[0]
        second = sorted_dist[1] if sorted_dist.shape[0] > 1 else nearest
        margin = second - nearest
        margins.append(margin)
        nearest_labels.append(int(class_labels[int(sorted_idx[0])]))
    return (
        torch_module.tensor(nearest_labels, dtype=torch_module.long),
        torch_module.tensor(margins),
    )


def _latent_audit_json(
    labels, sampled_latents, latent_prototypes, *, class_labels
) -> dict[str, object]:
    import torch as _torch

    labels_cpu = labels.detach().long().cpu()
    nearest_labels, margins = _latent_prototype_margin(
        _torch,
        sampled_latents,
        latent_prototypes,
        class_labels,
        None,
    )
    label_correct = nearest_labels.cpu() == labels_cpu

    per_class = {}
    for label in class_labels:
        indices = [
            i for i, item in enumerate(labels_cpu.tolist()) if int(item) == int(label)
        ]
        if not indices:
            continue
        correct = label_correct[labels_cpu.new_tensor(indices)].float()
        class_margins = margins[labels_cpu.new_tensor(indices)]
        per_class[str(label)] = {
            "count": len(indices),
            "latent_label_correct_count": int(correct.sum().item()),
            "latent_label_correct_ratio": float(correct.mean().item()),
            "latent_margin": _metric_summary(class_margins),
        }

    correct_all = label_correct.float()
    return {
        "class_labels": [int(label) for label in class_labels],
        "sample_count": int(labels_cpu.numel()),
        "latent_label_correct_count": int(correct_all.sum().item()),
        "latent_label_correct_ratio": float(correct_all.mean().item())
        if int(labels_cpu.numel())
        else 0.0,
        "overall": {
            "latent_margin": _metric_summary(margins),
        },
        "per_class": per_class,
    }


def run_teacher_diffusion_latent_audit(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    diffusion_checkpoint: str,
    first_stage_model_version: str,
    load_diffusion_checkpoint_fn,
    sample_latents_fn,
) -> None:
    first_stage.eval()
    set_first_stage_image_fusion_weight(first_stage, 0.0)

    diffusion_payload = torch_module.load(
        diffusion_checkpoint, map_location=device, weights_only=False
    )
    load_diffusion_checkpoint_fn(
        diffusion_model, diffusion_payload, first_stage_model_version
    )
    latent_scale = load_latent_scale_from_diffusion_checkpoint(
        torch_module, diffusion_payload, device
    )
    diffusion_model.eval()

    requested_labels = tuple(train_cfg.distillation.class_labels)
    if requested_labels:
        class_labels = validate_class_label_values(
            requested_labels,
            train_cfg.diffusion.num_classes,
            "distillation.class_labels",
        )
    else:
        class_labels = tuple(range(train_cfg.diffusion.num_classes))

    sample_labels = _distill_class_label_sequence(torch_module, train_cfg, device)
    sample_count = int(sample_labels.shape[0])

    codes = _sample_one_shot_codes_with_seed(
        torch_module,
        sample_count,
        train_cfg.one_shot.code_dim,
        train_cfg.distillation.code_seed,
        device,
    )
    sample_shape = (
        sample_count,
        train_cfg.first_stage.latent_channels,
        train_cfg.first_stage.image_size // 8,
        train_cfg.first_stage.image_size // 8,
    )

    teacher_latents = []
    teacher_camera = []
    batch_size = train_cfg.batch_size
    with torch_module.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_labels = sample_labels[start:end]
            batch_codes = codes[start:end]
            batch_shape = (end - start, *sample_shape[1:])
            initial_noise = _project_codes_to_teacher_noise(
                torch_module,
                batch_codes,
                batch_shape,
                train_cfg.distillation.code_projection_seed,
                device,
            )
            batch_latents = sample_latents_fn(
                diffusion_model,
                scheduler,
                batch_shape,
                device,
                batch_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
                initial_noise=initial_noise,
                sampling_steps=train_cfg.distillation.sampling_steps,
            )
            outputs = first_stage.decode_with_intermediates(
                batch_latents * latent_scale
            )
            teacher_latents.append(batch_latents.detach().cpu())
            teacher_camera.append(outputs["camera_plane"].detach().cpu())
            print(f"[teacher_diffusion_latent_audit] sampled {end}/{sample_count}")

    all_latents = torch_module.cat(teacher_latents, dim=0)
    all_camera = torch_module.cat(teacher_camera, dim=0)
    labels_cpu = sample_labels.detach().cpu()

    # Camera audit
    prototypes = _build_mnist_label_prototype_bank(
        torch_module,
        class_labels=class_labels,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
    )
    audit_batch_size = (
        train_cfg.distillation.label_audit_batch_size or train_cfg.batch_size
    )
    audit_device = device if device is not None else torch_module.device("cpu")
    camera_audit_tensors = compute_teacher_distill_label_audit_batched(
        torch_module,
        all_camera,
        labels_cpu,
        prototypes,
        class_labels=class_labels,
        batch_size=int(audit_batch_size),
        device=audit_device,
        foreground_threshold=train_cfg.distillation.filter_foreground_threshold,
    )
    camera_audit_payload = _teacher_label_audit_json(
        labels_cpu, camera_audit_tensors, class_labels=class_labels
    )

    # Latent audit: build latent prototypes from real MNIST
    latent_prototypes = _build_latent_prototypes_from_mnist(
        torch_module,
        first_stage,
        class_labels=class_labels,
        image_size=int(train_cfg.first_stage.image_size),
        root=train_cfg.data.root,
        max_per_class=int(train_cfg.distillation.label_audit_prototypes_per_class),
        device=device,
        batch_size=int(train_cfg.batch_size),
    )
    latent_audit_payload = _latent_audit_json(
        labels_cpu,
        all_latents,
        latent_prototypes,
        class_labels=class_labels,
    )

    # Combined JSON
    combined_payload = {
        "guidance_scale": float(train_cfg.diffusion.guidance_scale),
        "samples_per_class": int(train_cfg.distillation.samples_per_class),
        "class_labels": [int(label) for label in class_labels],
        "sample_count": sample_count,
        "camera_audit": camera_audit_payload,
        "latent_audit": latent_audit_payload,
    }

    run_manager.save_config_snapshot(
        "teacher_diffusion_latent_audit_config",
        {
            "stage": "teacher_diffusion_latent_audit",
            "guidance_scale": float(train_cfg.diffusion.guidance_scale),
            "samples_per_class": int(train_cfg.distillation.samples_per_class),
            "class_labels": [int(label) for label in class_labels],
            "sampling_steps": train_cfg.distillation.sampling_steps,
            "latent_channels": train_cfg.first_stage.latent_channels,
            "prototype_source": "train_mnist_mean_encoded_latent",
            "prototypes_per_class": int(
                train_cfg.distillation.label_audit_prototypes_per_class
            ),
            "source": "train.py_teacher_diffusion_latent_audit",
        },
    )

    audit_path = run_manager.save_artifact_json(
        "teacher_diffusion_latent_audit", combined_payload
    )

    # Camera audit PNG
    camera_visual_tensors, camera_row_labels = _teacher_label_audit_visual_tensors(
        torch_module,
        labels_cpu,
        prototypes,
        camera_audit_tensors,
        class_labels=class_labels,
    )
    camera_image_path = run_manager.save_tensor_grid(
        camera_visual_tensors,
        run_manager.images_dir / "teacher_diffusion_latent_audit_camera.png",
        title=format_output_title(
            "images/teacher_diffusion_latent_audit_camera.png",
            "diffusion latent audit – camera readable",
        ),
        max_items=min(20, sample_count),
        row_labels=camera_row_labels,
    )

    print(
        "[teacher_diffusion_latent_audit] "
        f"camera label_correct={camera_audit_payload['label_correct_count']}/{camera_audit_payload['sample_count']} "
        f"({camera_audit_payload['label_correct_ratio']:.4f})"
    )
    print(
        "[teacher_diffusion_latent_audit] "
        f"latent label_correct={latent_audit_payload['latent_label_correct_count']}/{latent_audit_payload['sample_count']} "
        f"({latent_audit_payload['latent_label_correct_ratio']:.4f})"
    )

    camera_audit = cast(dict[str, Any], combined_payload["camera_audit"])
    latent_audit = cast(dict[str, Any], combined_payload["latent_audit"])
    camera_per_class = cast(
        dict[str, dict[str, Any]], camera_audit.get("per_class", {})
    )
    latent_per_class = cast(
        dict[str, dict[str, Any]], latent_audit.get("per_class", {})
    )
    for label_str in sorted(camera_per_class.keys(), key=int):
        cstats = camera_per_class.get(label_str, {})
        lstats = latent_per_class.get(label_str, {})
        print(
            f"  class {label_str}: "
            f"camera_correct={cstats.get('label_correct_ratio', 0):.3f} "
            f"latent_correct={lstats.get('latent_label_correct_ratio', 0):.3f} "
            f"latent_margin={lstats.get('latent_margin', {}).get('mean', 0):.4f}"
        )

    run_meta.update(
        {
            "teacher_diffusion_latent_audit_metric_source": str(
                audit_path.relative_to(run_manager.run_dir)
            ),
            "teacher_diffusion_latent_audit_camera_image": str(
                camera_image_path.relative_to(run_manager.run_dir)
            ),
            "teacher_diffusion_latent_audit_camera_label_correct_ratio": camera_audit_payload[
                "label_correct_ratio"
            ],
            "teacher_diffusion_latent_audit_latent_label_correct_ratio": latent_audit_payload[
                "latent_label_correct_ratio"
            ],
        }
    )
    run_manager.write_meta(run_meta)
