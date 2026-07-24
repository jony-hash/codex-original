from __future__ import annotations

from pathlib import Path

from ..generation.codes import (
    sample_one_shot_codes_with_seed,
    project_codes_to_teacher_noise,
)
from ..visualization_core import (
    enhance_camera_for_display,
    format_output_title,
)
from ..train_utils import (
    load_latent_scale_from_diffusion_checkpoint,
    set_first_stage_image_fusion_weight,
)

from .distill_support import (
    distill_class_label_sequence as _distill_class_label_sequence,
)

def run_teacher_distill_export(
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
    set_first_stage_image_fusion_weight(
        first_stage, train_cfg.first_stage.image_fusion_end_weight
    )
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
    sample_labels = _distill_class_label_sequence(torch_module, train_cfg, device)
    sample_count = int(sample_labels.shape[0])
    codes = sample_one_shot_codes_with_seed(
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
    teacher_initial_noise = []
    teacher_camera = []
    teacher_image = []
    batch_size = train_cfg.batch_size
    with torch_module.no_grad():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            batch_labels = sample_labels[start:end]
            batch_codes = codes[start:end]
            batch_shape = (end - start, *sample_shape[1:])
            initial_noise = project_codes_to_teacher_noise(
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
            teacher_initial_noise.append(initial_noise.detach().cpu())
            teacher_latents.append(batch_latents.detach().cpu())
            teacher_camera.append(outputs["camera_plane"].detach().cpu())
            teacher_image.append(outputs["image_recon"].detach().cpu())
            print(f"[teacher_distill_export] generated {end}/{sample_count}")

    payload = {
        "codes": codes.detach().cpu(),
        "class_labels": sample_labels.detach().cpu(),
        "teacher_camera": torch_module.cat(teacher_camera, dim=0),
        "teacher_image": torch_module.cat(teacher_image, dim=0),
        "teacher_latents": torch_module.cat(teacher_latents, dim=0),
        "teacher_initial_noise": torch_module.cat(teacher_initial_noise, dim=0),
        "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
        "latent_scale": latent_scale.detach().cpu(),
        "config": {
            "samples_per_class": train_cfg.distillation.samples_per_class,
            "total_samples": train_cfg.distillation.total_samples,
            "class_labels": list(train_cfg.distillation.class_labels),
            "code_dim": train_cfg.one_shot.code_dim,
            "code_seed": train_cfg.distillation.code_seed,
            "code_projection_seed": train_cfg.distillation.code_projection_seed,
            "sampling_steps": train_cfg.distillation.sampling_steps,
            "teacher_sample_shape": sample_shape,
        },
    }
    run_manager.save_config_snapshot(
        "teacher_distill_export_config",
        {
            **payload["config"],
            "num_samples": sample_count,
            "source": "train.py_teacher_distill_export",
        },
    )
    artifact_path: Path
    if train_cfg.distillation.output_path:
        artifact_path = Path(train_cfg.distillation.output_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch_module.save(payload, artifact_path)
    else:
        artifact_path = run_manager.save_tensor_bundle(
            "teacher_distill_dataset.pt", payload
        )
    run_manager.save_tensor_grid(
        {
            "teacher_camera_raw": payload["teacher_camera"],
            "teacher_camera_display": enhance_camera_for_display(
                payload["teacher_camera"]
            ),
            "teacher_image": payload["teacher_image"],
        },
        run_manager.images_dir / "teacher_distill_export.png",
        title=format_output_title(
            "images/teacher_distill_export.png", "teacher distillation targets"
        ),
        max_items=min(sample_count, 20),
    )
    run_meta.update(
        {
            "latent_scale_source": "diffusion_checkpoint",
            "diffusion_best_loaded_for_sampling": True,
            "teacher_distill_dataset": str(artifact_path),
            "teacher_distill_sample_count": sample_count,
        }
    )
    run_manager.write_meta(run_meta)

__all__ = ["run_teacher_distill_export"]
